# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time

import nvdlfw_inspect.api as debug_api
import torch
import torchmetrics
import torchmetrics.text
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.distributed.tensor import DTensor
from tqdm import tqdm
from transformers.modeling_outputs import MaskedLMOutput

from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)


# Dense BF16 tensor core peak TFLOPS (without sparsity). Product pages often list
# the 2x sparse number; dense = sparse / 2. Sources: NVIDIA datasheets for each GPU.
_GPU_PEAK_TFLOPS_BF16 = {
    "H100": 989.0,
    "H200": 989.0,
    "A100": 312.0,
    "A6000": 155.0,
    "L40": 181.0,
    "GH200": 989.0,
    "B200": 2250.0,
    "GB200": 2250.0,
    "B300": 2500.0,
    "GB300": 2500.0,
}

# Model types that use gated MLP (SwiGLU/GeGLU) with 3 projections vs. standard FFN with 2.
_GATED_MLP_MODEL_TYPES = frozenset({"llama", "mistral", "qwen2"})


def _detect_peak_tflops_bf16():
    """Auto-detect dense BF16 peak TFLOPS for the local GPU. Returns (peak, device_name)."""
    if not torch.cuda.is_available():
        return None, "unknown"
    name = torch.cuda.get_device_name(0)
    for key, tflops in _GPU_PEAK_TFLOPS_BF16.items():
        if key.lower() in name.lower():
            return tflops, name
    return None, name


def _compute_per_token_flops(model_config_dict: dict, seq_len: int) -> int:
    """Training FLOPs per token for a transformer (forward + backward = 3x forward).

    First-principles matmul count: Q/K/V/O projections (GQA-aware), attention
    logits/values (the S^2 cost expressed per-token as 4*S*H), 2-or-3-projection
    MLP (SwiGLU detected via model_type), and LM head. The returned value is
    multiplied by the actual unpadded token count at log time, so it naturally
    handles BSHD, THD (sequence packing), DP, and CP: unpadded tokens on each
    rank already reflect that rank's share of work.
    """
    h = model_config_dict["hidden_size"]
    n_heads = model_config_dict["num_attention_heads"]
    n_kv = model_config_dict.get("num_key_value_heads", n_heads)
    head_dim = h // n_heads
    kv_dim = n_kv * head_dim
    ffn = model_config_dict["intermediate_size"]
    vocab = model_config_dict.get("vocab_size", 0)
    num_layers = model_config_dict["num_hidden_layers"]
    model_type = model_config_dict.get("model_type", "")
    num_mlp_proj = 3 if model_type in _GATED_MLP_MODEL_TYPES else 2

    per_layer = (
        2 * h * h  # Q projection
        + 4 * h * kv_dim  # K + V projections (GQA-aware)
        + 2 * h * h  # O projection
        + 4 * seq_len * h  # attention logits + values (S^2 -> S per token)
        + 2 * num_mlp_proj * h * ffn  # MLP (2 or 3 projections)
    )
    lm_head = 2 * h * vocab if vocab > 0 else 0
    per_token_fwd = num_layers * per_layer + lm_head
    return 3 * per_token_fwd


class PerfLogger:
    """Class to log performance metrics to stdout and wandb, and print final averaged metrics at the end of training.

    Args:
        dist_config: The distributed configuration.
        args: The arguments.
        model_config_dict: Optional HF-style model config dict. When supplied together with
            ``args.log_mfu`` set to True, the logger computes per-step Model FLOPs Utilization
            (``train/mfu_pct``) and throughput (``train/tflops_per_gpu``) on each logging step.

    Attributes:
        min_loss: The minimum loss seen so far.
    """

    def __init__(self, dist_config: DistributedConfig, args: DictConfig, model_config_dict: dict | None = None):
        """Initialize the logger."""
        self._dist_config = dist_config
        self._run_config = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

        self.min_loss = torch.tensor(float("inf"), device=torch.device(f"cuda:{dist_config.local_rank}"))

        self.logging_frequency = args.logger.frequency
        # Track whether to collect memory stats (disabled by default for max performance)

        # MFU setup: compute per-token FLOPs and peak TFLOPS once at init. Actual FLOPs per
        # step are derived at log time from the current batch's unpadded token count, which
        # already reflects each rank's share under DP/CP and sequence packing.
        self._log_mfu = bool(args.get("log_mfu", False)) and model_config_dict is not None
        self._per_token_flops = 0
        self._peak_tflops: float | None = None
        if self._log_mfu:
            self._per_token_flops = _compute_per_token_flops(model_config_dict, args.dataset.max_seq_length)
            self._peak_tflops, gpu_name = _detect_peak_tflops_bf16()
            if dist_config.local_rank == 0:
                logger.info(
                    "MFU tracking enabled: GPU=%s, peak=%s TFLOPS BF16, per-token FLOPs=%.3e, seq_len=%d",
                    gpu_name,
                    f"{self._peak_tflops:.1f}" if self._peak_tflops else "unknown",
                    float(self._per_token_flops),
                    args.dataset.max_seq_length,
                )

        metrics_dict = {
            "train/loss": torchmetrics.MeanMetric(),
            "train/grad_norm": torchmetrics.MeanMetric(),
            "train/learning_rate": torchmetrics.MeanMetric(),
            "train/step_time": torchmetrics.MeanMetric(),
            "train/tokens_per_second_per_gpu": torchmetrics.MeanMetric(),
            "train/unpadded_tokens_per_second_per_gpu": torchmetrics.MeanMetric(),
            "train/total_unpadded_tokens_per_batch": torchmetrics.SumMetric(),
            "train/perplexity": torchmetrics.text.Perplexity(ignore_index=-100),
            "train/gpu_memory_allocated_max_gb": torchmetrics.MaxMetric(),
            "train/gpu_memory_allocated_mean_gb": torchmetrics.MeanMetric(),
        }
        if self._log_mfu:
            metrics_dict["train/tflops_per_gpu"] = torchmetrics.MeanMetric()
            if self._peak_tflops is not None:
                metrics_dict["train/mfu_pct"] = torchmetrics.MeanMetric()

        self.metrics = torchmetrics.MetricCollection(metrics_dict)
        # We move metrics to a GPU device so we can use torch.distributed to aggregate them before logging.
        self.metrics.to(torch.device(f"cuda:{dist_config.local_rank}"))
        self.previous_step_time = time.perf_counter()

        if self._dist_config.is_main_process():
            # Log the entire args object to wandb for experiment tracking and reproducibility.
            wandb.init(**args.wandb_init_args, config=self._run_config)
            self._progress_bar = tqdm(total=args.num_train_steps, desc="Training")

        # Whether to step debug_api.step() after each step
        self.quant_stats_config = args.quant_stats_config.enabled

    def log_step(
        self,
        step: int,
        batch: dict[str, torch.Tensor],
        outputs: MaskedLMOutput,
        grad_norm: torch.Tensor | DTensor,
        lr: float,
    ):
        """Log a step to the logger and wandb.

        Args:
            step: The step number.
            batch: The batch of data for the step.
            outputs: The outputs of the step.
            grad_norm: The gradient norm of the step.
            lr: The learning rate of the step.
        """
        with torch.no_grad():
            # FSDP2's clip_grad_norm_ returns a DTensor; convert to local tensor for torchmetrics compatibility.
            if isinstance(grad_norm, DTensor):
                grad_norm = grad_norm.to_local()

            if self.quant_stats_config:
                debug_api.step()

            if step % self.logging_frequency == 0 and step > 0:
                num_tokens = batch["input_ids"].numel()
                # 1 is the padding token for ESM-2.
                num_unpadded_tokens = batch["input_ids"][batch["input_ids"] != 1].numel()

                self.min_loss = torch.minimum(self.min_loss, outputs.loss)
                elapsed_time, self.previous_step_time = (
                    time.perf_counter() - self.previous_step_time,
                    time.perf_counter(),
                )
                step_time = elapsed_time / self.logging_frequency

                self.metrics["train/loss"].update(outputs.loss)
                self.metrics["train/learning_rate"].update(lr)
                self.metrics["train/grad_norm"].update(grad_norm)
                self.metrics["train/step_time"].update(step_time)
                self.metrics["train/tokens_per_second_per_gpu"].update(num_tokens / step_time)
                self.metrics["train/unpadded_tokens_per_second_per_gpu"].update(num_unpadded_tokens / step_time)
                self.metrics["train/total_unpadded_tokens_per_batch"].update(num_unpadded_tokens)

                if self._log_mfu:
                    # Current batch's unpadded tokens already reflect this rank's share (CP
                    # shards the batch; DP replicates the model across ranks on distinct
                    # micro-batches). step_time is the per-step average over the logging window.
                    flops_per_step = self._per_token_flops * num_unpadded_tokens
                    tflops_per_gpu = flops_per_step / step_time / 1e12
                    self.metrics["train/tflops_per_gpu"].update(tflops_per_gpu)
                    if self._peak_tflops is not None:
                        self.metrics["train/mfu_pct"].update(tflops_per_gpu / self._peak_tflops * 100.0)

                # Handle sequence packing for torchmetrics calculation.
                if outputs.logits.dim() < 3:
                    outputs.logits = outputs.logits.unsqueeze(0)

                self.metrics["train/perplexity"].update(outputs.logits, batch["labels"])

                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                self.metrics["train/gpu_memory_allocated_max_gb"].update(memory_allocated)
                self.metrics["train/gpu_memory_allocated_mean_gb"].update(memory_allocated)

                metrics = self.metrics.compute()
                self.metrics.reset()
                metrics = {
                    k: v.detach().cpu().item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v
                    for k, v in metrics.items()
                }
                metrics["train/global_step"] = step

                if self._dist_config.is_main_process():
                    wandb.log(metrics, step=step)
                    self._progress_bar.update(self.logging_frequency)
                    self._progress_bar.set_postfix({"loss": outputs.loss.item()})

                if self._dist_config.local_rank == 0:
                    logger.info(", ".join([f"{k.split('/')[1]}: {v:.3g}" for k, v in metrics.items()]))

    def finish(self):
        """Finish the logger and close the progress bar."""
        if self.quant_stats_config:
            debug_api.end_debug()

        if not self._dist_config.is_main_process():
            return

        wandb.finish()
        self._progress_bar.close()
