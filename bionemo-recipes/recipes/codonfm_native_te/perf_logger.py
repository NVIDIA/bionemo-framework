# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Performance logging utilities for CodonFM training."""

import logging
import time

import nvdlfw_inspect.api as debug_api
import torch
import torchmetrics
import torchmetrics.text
import wandb
from distributed_config import DistributedConfig
from omegaconf import DictConfig, OmegaConf
from torch.distributed.tensor import DTensor
from tqdm import tqdm
from transformers.modeling_outputs import MaskedLMOutput


logger = logging.getLogger(__name__)

# CodonFM pad token ID
PAD_TOKEN_ID = 3


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
    logits/values (the S^2 cost expressed per-token as 4*S*H for a uniform
    BSHD batch of length seq_len), 2-or-3-projection MLP (SwiGLU detected via
    model_type), and LM head.

    Kept for back-compat. For accurate per-step accounting use
    ``_compute_non_attn_per_token_flops`` (applied to the total token count)
    together with ``_compute_attn_flop_coeff`` (applied to Σ(Lᵢ²) from
    cu_seq_lens), since a packed THD batch of total length S containing docs
    L₁, L₂, … has actual attention work Σ(Lᵢ²) ≤ S², not B·S².
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


def _compute_non_attn_per_token_flops(model_config_dict: dict) -> int:
    """Per-token FLOPs for everything EXCEPT the S² attention term.

    Q/K/V/O projections (GQA-aware) + MLP + LM head, 3x for fwd+bwd. Multiply by the
    actual total token count of the batch to get per-step non-attention FLOPs. Pairs
    with ``_compute_attn_flop_coeff`` so that
    ``non_attn + coeff·S ≡ _compute_per_token_flops(cfg, S)``.
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
        + 2 * num_mlp_proj * h * ffn  # MLP (2 or 3 projections)
    )
    lm_head = 2 * h * vocab if vocab > 0 else 0
    return 3 * (num_layers * per_layer + lm_head)


def _compute_attn_flop_coeff(model_config_dict: dict) -> int:
    """Coefficient K such that per-step attention FLOPs = K · Σ(Lᵢ²) globally.

    Per CP rank: ``K · Σ(Lᵢ²) / cp_size`` — each CP rank computes 1/cp_size of each
    doc's Lᵢ * Lᵢ score matrix. The 4 counts QK^T (2) + softmax·V (2); the 3 is
    fwd+bwd. Hidden size appears linearly because attention is over heads and each
    contributes head_dim, and heads * head_dim == h.
    """
    h = model_config_dict["hidden_size"]
    num_layers = model_config_dict["num_hidden_layers"]
    return 3 * num_layers * 4 * h


def _attn_work_from_batch(batch: dict, device: torch.device, cp_size: int = 1) -> torch.Tensor:
    """Return GLOBAL Σ(Lᵢ²) for this batch as an int64 scalar tensor.

    The caller divides by cp_size in log_step to convert this global number into
    per-rank attention work, so this helper always returns a pre-CP-shard quantity.

    * THD: uses ``cu_seq_lens_q`` (real per-doc lengths — already global, the collator
      emits boundaries before per-rank sharding). Falls back to ``cu_seq_lens_q_padded``
      if only the padded variant is present. Excludes CP zigzag padding that FA
      processes as cycles but contributes nothing to the training signal.
    * BSHD: ``batch["input_ids"].shape`` per rank is ``(B, S/cp)`` when CP is active
      because ``ContextParallelDataLoaderWrapper`` pre-splits along the seq dim. To
      recover the global B·S² from a per-rank shape, multiply by cp_size². When
      cp_size=1 this is a no-op (per-rank shape == global shape). CodonFM runs FSDP
      without CP, so cp_size=1 always here, but the formula stays correct if CP is
      added later.

    Int32 lens cast to int64 BEFORE squaring (overflow at L ≈ 46k otherwise).
    """
    cu = batch.get("cu_seq_lens_q")
    if cu is None:
        cu = batch.get("cu_seq_lens_q_padded")
    if cu is not None:
        lens = (cu[1:] - cu[:-1]).to(torch.int64)
        return (lens * lens).sum()
    shape = batch["input_ids"].shape
    batch_size, seq_len_per_rank = int(shape[0]), int(shape[-1])
    return torch.tensor(
        batch_size * seq_len_per_rank * seq_len_per_rank * cp_size * cp_size,
        dtype=torch.int64,
        device=device,
    )


class PerfLogger:
    """Performance logger for CodonFM training.

    Logs metrics to stdout and wandb, and prints final averaged metrics at the end of training.

    Args:
        dist_config: The distributed configuration.
        args: The Hydra arguments.
        model_config_dict: Optional HF-style model config dict. When supplied together with
            ``args.log_mfu`` set to True, the logger computes per-step Model FLOPs Utilization
            (``train/mfu_pct``) and throughput (``train/tflops_per_gpu``) on each logging step.
    """

    def __init__(self, dist_config: DistributedConfig, args: DictConfig, model_config_dict: dict | None = None):
        """Initialize the logger."""
        self._dist_config = dist_config
        self._run_config = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

        self._device = torch.device(f"cuda:{dist_config.local_rank}")
        self.min_loss = torch.tensor(float("inf"), device=self._device)

        self.logging_frequency = args.logger.frequency

        # MFU setup: compute per-token FLOPs and peak TFLOPS once at init. Actual FLOPs per
        # step are derived at log time from the tracked unpadded token count, which already
        # reflects each rank's share under DP and sequence packing.
        self._log_mfu = bool(args.get("log_mfu", False)) and model_config_dict is not None
        self._per_token_flops = 0
        self._non_attn_per_token_flops = 0
        self._attn_flop_coeff = 0
        self._cp_size = int(args.get("cp_size", 1))
        self._peak_tflops: float | None = None
        if self._log_mfu:
            self._per_token_flops = _compute_per_token_flops(model_config_dict, args.dataset.max_seq_length)
            self._non_attn_per_token_flops = _compute_non_attn_per_token_flops(model_config_dict)
            self._attn_flop_coeff = _compute_attn_flop_coeff(model_config_dict)
            self._peak_tflops, gpu_name = _detect_peak_tflops_bf16()
            if dist_config.local_rank == 0:
                logger.info(
                    "MFU tracking enabled: GPU=%s, peak=%s TFLOPS BF16, per-token FLOPs=%.3e, seq_len=%d, "
                    "non_attn_per_token=%.3e, attn_coeff=%.3e, cp_size=%d",
                    gpu_name,
                    f"{self._peak_tflops:.1f}" if self._peak_tflops else "unknown",
                    float(self._per_token_flops),
                    args.dataset.max_seq_length,
                    float(self._non_attn_per_token_flops),
                    float(self._attn_flop_coeff),
                    self._cp_size,
                )

        metrics_dict = {
            "train/loss": torchmetrics.MeanMetric(),
            "train/grad_norm": torchmetrics.MeanMetric(),
            "train/learning_rate": torchmetrics.MeanMetric(),
            "train/step_time": torchmetrics.MeanMetric(),
            "train/tokens_per_second_per_gpu": torchmetrics.MeanMetric(),
            "train/unpadded_tokens_per_second_per_gpu": torchmetrics.MeanMetric(),
            "train/perplexity": torchmetrics.text.Perplexity(ignore_index=-100),
            "train/gpu_memory_allocated_max_gb": torchmetrics.MaxMetric(),
            "train/gpu_memory_allocated_mean_gb": torchmetrics.MeanMetric(),
        }
        if self._log_mfu:
            metrics_dict["train/tflops_per_gpu"] = torchmetrics.MeanMetric()
            if self._peak_tflops is not None:
                metrics_dict["train/mfu_pct"] = torchmetrics.MeanMetric()

        self.metrics = torchmetrics.MetricCollection(metrics_dict)
        self.metrics.to(self._device)
        self.previous_step_time = time.perf_counter()

        if self._dist_config.is_main_process():
            wandb.init(**args.wandb_init_args, config=self._run_config)
            self._progress_bar = tqdm(total=args.num_train_steps, desc="Training")

        # Whether to step debug_api.step() after each step
        self.quant_stats_config = args.quant_stats_config.enabled

        # Gradient accumulation tracking
        self.num_tokens = 0
        self.num_unpadded_tokens = torch.tensor(0, dtype=torch.int64, device=self._device)
        # Σ(Lᵢ²) over grad-acc micro-batches — drives the attention-FLOP term at log time.
        self._attn_work_accum = torch.tensor(0, dtype=torch.int64, device=self._device)
        self.running_loss = torch.tensor(0.0, device=self._device)
        self.grad_acc_step_count = 0

    def log_micro_step(self, step: int, batch: dict[str, torch.Tensor], outputs: MaskedLMOutput):
        """Store data on micro step for gradient accumulation metrics.

        Args:
            step: The current optimizer step number.
            batch: The input batch for this micro-step.
            outputs: Model outputs for this micro-step (with unscaled loss).
        """
        assert outputs.loss is not None, "Loss is None"

        with torch.no_grad():
            self.grad_acc_step_count += 1
            self.running_loss += outputs.loss

            if step % self.logging_frequency == 0 and step > 0:
                self.num_tokens += batch["input_ids"].numel()
                num_unpadded_tokens = batch["input_ids"][batch["input_ids"] != PAD_TOKEN_ID].numel()
                self.num_unpadded_tokens += num_unpadded_tokens
                if self._log_mfu:
                    # Σ(Lᵢ²) from cu_seq_lens (THD) or B·S² from input_ids shape (BSHD).
                    # Helper returns a GLOBAL value (pre-CP-shard); log_step divides by cp_size.
                    # Accumulates across grad-acc micro-batches; drained in log_step.
                    self._attn_work_accum += _attn_work_from_batch(batch, self._device, self._cp_size)

                # Update perplexity per micro-batch since it needs logits + labels
                logits = outputs.logits
                if logits.dim() < 3:
                    logits = logits.unsqueeze(0)
                self.metrics["train/perplexity"].update(logits, batch["labels"])

    def log_step(
        self,
        step: int,
        grad_norm: torch.Tensor | DTensor | float,
        lr: float,
    ):
        """Log a training step (called once per optimizer step).

        Args:
            step: Current optimizer step.
            grad_norm: Gradient norm value.
            lr: Current learning rate.
        """
        with torch.no_grad():
            assert self.grad_acc_step_count > 0, (
                f"Gradient accumulation steps ({self.grad_acc_step_count}) must be greater than 0, "
                f"and can be incremented by log_micro_step()."
            )

            if isinstance(grad_norm, DTensor):
                grad_norm = grad_norm.to_local()

            if self.quant_stats_config:
                debug_api.step()

            # Calculate average loss over all micro steps in the logging window
            avg_loss = self.running_loss / self.grad_acc_step_count
            self.min_loss = torch.minimum(self.min_loss, avg_loss)

            if step % self.logging_frequency == 0 and step > 0:
                elapsed_time, self.previous_step_time = (
                    time.perf_counter() - self.previous_step_time,
                    time.perf_counter(),
                )
                step_time = elapsed_time / self.logging_frequency

                self.metrics["train/loss"].update(avg_loss)
                self.metrics["train/learning_rate"].update(lr)
                self.metrics["train/grad_norm"].update(
                    grad_norm if isinstance(grad_norm, torch.Tensor) else torch.tensor(grad_norm)
                )
                self.metrics["train/step_time"].update(step_time)
                self.metrics["train/tokens_per_second_per_gpu"].update(self.num_tokens / step_time)
                self.metrics["train/unpadded_tokens_per_second_per_gpu"].update(self.num_unpadded_tokens / step_time)

                if self._log_mfu:
                    # Two-term FLOP accounting:
                    #   non_attn_flops = non_attn_per_token * num_tokens
                    #   attn_flops     = attn_flop_coeff * Σ(Lᵢ²) / cp_size
                    # num_tokens follows the PaLM/Megatron/MosaicML convention of counting
                    # every slot in input_ids (hardware runs QKV/MLP/LM-head matmuls over
                    # padded positions too). The attention term uses the real per-doc
                    # Σ(Lᵢ²) from cu_seq_lens because FA gates the score matrix on doc
                    # boundaries, so a packed batch with fragmented docs does genuinely
                    # less attention work than a uniform B·S² would imply — the old
                    # single-term formula overstated MFU in exactly that regime.
                    attn_work = int(self._attn_work_accum.item())
                    non_attn_flops = self._non_attn_per_token_flops * self.num_tokens
                    attn_flops = (self._attn_flop_coeff * attn_work) // self._cp_size
                    flops_per_step = non_attn_flops + attn_flops
                    tflops_per_gpu = flops_per_step / step_time / 1e12
                    self.metrics["train/tflops_per_gpu"].update(tflops_per_gpu)
                    if self._peak_tflops is not None:
                        self.metrics["train/mfu_pct"].update(tflops_per_gpu / self._peak_tflops * 100.0)

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
                    self._progress_bar.set_postfix({"loss": avg_loss.item()})

                if self._dist_config.local_rank == 0:
                    logger.info(", ".join([f"{k.split('/')[1]}: {v:.3g}" for k, v in metrics.items()]))

                # Reset running accumulators for next logging window
                self.running_loss.zero_()
                self.num_tokens = 0
                self.num_unpadded_tokens.zero_()
                self._attn_work_accum.zero_()
                self.grad_acc_step_count = 0

    def finish(self):
        """Finish the logger."""
        if self.quant_stats_config:
            debug_api.end_debug()

        if not self._dist_config.is_main_process():
            return
        wandb.finish()
        self._progress_bar.close()
