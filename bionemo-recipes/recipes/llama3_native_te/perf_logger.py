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
from pathlib import Path

import torch
import torchmetrics
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.profiler import profile, schedule, tensorboard_trace_handler
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)


class PerfLogger:
    """Class to log performance metrics to stdout and wandb, and print final averaged metrics at the end of training.

    Args:
        dist_config: The distributed configuration.
        args: The arguments.
        log_tev: Whether to log Token Embedding Variance (TEV) statistics.

    Attributes:
        min_loss: The minimum loss seen so far.
    """

    def __init__(self, dist_config: DistributedConfig, args: DictConfig, log_tev: bool = False):
        """Initialize the logger."""
        self._dist_config = dist_config
        self._run_config = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
        self._log_tev = log_tev

        self.min_loss = float("inf")

        self.logging_frequency = args.logger.frequency

        metrics_dict = {
            "train/loss": torchmetrics.MeanMetric(),
            "train/grad_norm": torchmetrics.MeanMetric(),
            "train/learning_rate": torchmetrics.MeanMetric(),
            "train/step_time": torchmetrics.MeanMetric(),
            "train/tokens_per_second_per_gpu": torchmetrics.MeanMetric(),
            "train/unpadded_tokens_per_second_per_gpu": torchmetrics.MeanMetric(),
            "train/total_unpadded_tokens_per_batch": torchmetrics.SumMetric(),
            "train/gpu_memory_allocated_max_gb": torchmetrics.MaxMetric(),
            "train/gpu_memory_allocated_mean_gb": torchmetrics.MeanMetric(),
        }

        # Add TEV metrics if enabled (matches John's tev_mean/tev_sd logging)
        if log_tev:
            metrics_dict["train/tev_mean"] = torchmetrics.MeanMetric()
            metrics_dict["train/tev_sd"] = torchmetrics.MeanMetric()

        # Add Megatron-style loss metrics
        metrics_dict["train/megatron_loss"] = torchmetrics.MeanMetric()
        metrics_dict["train/total_valid_tokens"] = torchmetrics.SumMetric()

        self.metrics = torchmetrics.MetricCollection(metrics_dict)
        # We move metrics to a GPU device so we can use torch.distributed to aggregate them before logging.
        self.metrics.to(torch.device(f"cuda:{dist_config.local_rank}"))
        self.previous_step_time = time.perf_counter()
        self._profiler = None

        if self._dist_config.is_main_process():
            # Log the entire args object to wandb for experiment tracking and reproducibility.
            self._wandb_run = wandb.init(**args.wandb, config=self._run_config)
            self._progress_bar = tqdm(total=args.num_train_steps, desc="Training")

            if args.profiler.enabled:
                self._profiler = setup_profiler(args, self._wandb_run)
                self._profiler.__enter__()

        # Gradient accumulation tracking
        self.num_tokens = 0
        self.num_unpadded_tokens = 0
        self.running_loss = 0.0
        self.grad_acc_step_count = 0

        # Megatron-style loss accumulation
        self.running_loss_sum = 0.0
        self.running_valid_tokens = 0

    def log_micro_step(
        self,
        batch: dict[str, torch.Tensor],
        outputs: CausalLMOutputWithPast,
        loss_sum: float | None = None,
        num_tokens: int | None = None,
    ):
        """Store data on micro step for gradient accumulation metrics.

        Args:
            batch: The batch of data for the micro step.
            outputs: The outputs of the micro step.
            loss_sum: Optional sum of per-token losses (for Megatron-style loss).
            num_tokens: Optional count of valid tokens (for Megatron-style loss).
        """
        self.grad_acc_step_count += 1
        self.num_tokens += batch["input_ids"].numel()
        # Use attention_mask to count unpadded tokens (works for both BSHD and THD)
        if "attention_mask" in batch:
            self.num_unpadded_tokens += batch["attention_mask"].sum().item()
        else:
            # Fallback for pure sequence packing with no padding: all tokens are unpadded
            self.num_unpadded_tokens += batch["input_ids"].numel()
        self.running_loss += outputs.loss.item()

        # Accumulate Megatron-style loss if provided
        if loss_sum is not None and num_tokens is not None:
            self.running_loss_sum += loss_sum
            self.running_valid_tokens += num_tokens

    def log_step(
        self,
        step: int,
        grad_norm: float,
        lr: float,
        megatron_loss: float | None = None,
        total_tokens: int | None = None,
        tev_mean: float = 0.0,
        tev_sd: float = 0.0,
    ):
        """Log a step to the logger and wandb.

        Args:
            step: The step number.
            grad_norm: The gradient norm of the step.
            lr: The learning rate of the step.
            megatron_loss: Optional Megatron-style per-token loss (sum/total_tokens).
            total_tokens: Optional total valid tokens across gradient accumulation steps.
            tev_mean: Token Embedding Variance mean (0 if TEV logging disabled).
            tev_sd: Token Embedding Variance standard deviation (0 if TEV logging disabled).
        """
        # Use accumulated metrics from gradient accumulation
        assert self.grad_acc_step_count > 0, (
            f"Gradient accumulation steps ({self.grad_acc_step_count}) must be greater than 0, "
            f"and can be incremented by log_micro_step()."
        )

        avg_loss = self.running_loss / self.grad_acc_step_count
        self.min_loss = min(self.min_loss, avg_loss)
        step_time, self.previous_step_time = time.perf_counter() - self.previous_step_time, time.perf_counter()

        self.metrics["train/loss"].update(avg_loss)
        self.metrics["train/learning_rate"].update(lr)
        self.metrics["train/grad_norm"].update(grad_norm)
        self.metrics["train/step_time"].update(step_time)
        self.metrics["train/tokens_per_second_per_gpu"].update(self.num_tokens / step_time)
        self.metrics["train/unpadded_tokens_per_second_per_gpu"].update(self.num_unpadded_tokens / step_time)
        self.metrics["train/total_unpadded_tokens_per_batch"].update(self.num_unpadded_tokens / self.logging_frequency)

        # Log Megatron-style loss if provided
        if megatron_loss is not None:
            self.metrics["train/megatron_loss"].update(megatron_loss)
        if total_tokens is not None:
            self.metrics["train/total_valid_tokens"].update(total_tokens)

        # Log TEV metrics if enabled
        if self._log_tev and (tev_mean > 0 or tev_sd > 0):
            self.metrics["train/tev_mean"].update(tev_mean)
            self.metrics["train/tev_sd"].update(tev_sd)

        if self._profiler is not None:
            self._profiler.step()

        if step % self.logging_frequency == 0:
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            self.metrics["train/gpu_memory_allocated_max_gb"].update(memory_allocated)
            self.metrics["train/gpu_memory_allocated_mean_gb"].update(memory_allocated)

            metrics = self.metrics.compute()
            self.metrics.reset()
            metrics["train/global_step"] = torch.tensor(step, dtype=torch.int64)

            if self._dist_config.is_main_process():
                wandb.log(metrics, step=step)
                self._progress_bar.update(self.logging_frequency)
                # Show megatron_loss if available, otherwise show HF loss
                display_loss = megatron_loss if megatron_loss is not None else avg_loss
                self._progress_bar.set_postfix({"loss": display_loss})

            if self._dist_config.local_rank == 0:
                # Filter out zero metrics for cleaner logging
                log_items = []
                for k, v in metrics.items():
                    name = k.split("/")[1]
                    # Skip metrics that are zero (not logged this step)
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    if val != 0 or name in ["loss", "megatron_loss", "grad_norm", "learning_rate"]:
                        log_items.append(f"{name}: {val:.3g}")
                logger.info(", ".join(log_items))

        # Reset gradient accumulation tracking for next step
        self.num_tokens = 0
        self.num_unpadded_tokens = 0
        self.running_loss = 0.0
        self.grad_acc_step_count = 0
        self.running_loss_sum = 0.0
        self.running_valid_tokens = 0

    def log_validation(self, step: int, val_metrics: dict):
        """Log validation metrics to wandb.

        Args:
            step: The current training step.
            val_metrics: Dictionary with val_loss, val_ppl, val_tokens, val_batches.
        """
        if self._dist_config.is_main_process():
            wandb.log(
                {
                    "val/loss": val_metrics["val_loss"],
                    "val/ppl": val_metrics["val_ppl"],
                    "val/tokens": val_metrics["val_tokens"],
                    "val/batches": val_metrics["val_batches"],
                },
                step=step,
            )

    def finish(self):
        """Finish the logger and close the progress bar."""
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)

        if not self._dist_config.is_main_process():
            return

        wandb.finish()
        self._progress_bar.close()


def setup_profiler(args: DictConfig, wandb_run: wandb.Run):
    """Setup a basic torch profiler for the experiment.

    Args:
        args: The arguments.
        wandb_run: The wandb run.

    Returns:
        The profiler.
    """
    _trace_dir = Path(HydraConfig.get().runtime.output_dir) / "traces"
    _trace_dir.mkdir(parents=True, exist_ok=True)

    def on_trace_ready(prof):
        """Custom callback to save chrome trace, export memory timeline, and log to wandb."""
        # Save chrome trace using tensorboard_trace_handler
        tensorboard_trace_handler(str(_trace_dir))(prof)
        # Export memory timeline
        prof.export_memory_timeline(str(_trace_dir / "memory_timeline.html"), device="cuda:0")
        # Log artifacts to wandb
        profile_art = wandb.Artifact(name=f"{wandb_run.name}_profile", type="profile")
        for file in _trace_dir.glob("*.json"):
            profile_art.add_file(str(file), name=file.name)
        profile_art.add_file(str(_trace_dir / "memory_timeline.html"), name="memory_timeline.html")
        wandb_run.log_artifact(profile_art)

    return profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule(**args.profiler.schedule),
        on_trace_ready=on_trace_ready,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        profile_memory=True,
        record_shapes=True,
    )
