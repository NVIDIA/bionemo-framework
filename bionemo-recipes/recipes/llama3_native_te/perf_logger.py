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

import gc
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

    Attributes:
        min_loss: The minimum loss seen so far.
    """

    def __init__(self, dist_config: DistributedConfig, args: DictConfig):
        """Initialize the logger."""
        self._dist_config = dist_config
        self._run_config = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

        self.min_loss = float("inf")

        self.logging_frequency = args.logger.frequency
        # Track whether to collect memory stats (disabled by default for max performance)

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

        self.metrics = torchmetrics.MetricCollection(metrics_dict)
        # We move metrics to a GPU device so we can use torch.distributed to aggregate them before logging.
        self.metrics.to(torch.device(f"cuda:{dist_config.local_rank}"))
        self.previous_step_time = time.perf_counter()
        self._profiler = None

        # Manually control garbage collection for cleaner profiling.
        self._gc_interval = args.profiler.gc_interval
        gc.disable()
        self._run_garbage_collection()

        if self._dist_config.is_main_process():
            # Log the entire args object to wandb for experiment tracking and reproducibility.
            self._wandb_run = wandb.init(**args.wandb, config=self._run_config)
            self._progress_bar = tqdm(total=args.num_train_steps, desc="Training")

            if args.profiler.enabled:
                self._profiler = setup_profiler(args, self._wandb_run)
                self._profiler.__enter__()

    def log_step(
        self,
        step: int,
        batch: dict[str, torch.Tensor],
        outputs: CausalLMOutputWithPast,
        grad_norm: float,
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
        num_tokens = batch["input_ids"].numel()
        if "attention_mask" in batch:
            num_unpadded_tokens = batch["attention_mask"].sum().item()
        else:
            num_unpadded_tokens = num_tokens

        self.min_loss = min(self.min_loss, outputs.loss.item())
        step_time, self.previous_step_time = time.perf_counter() - self.previous_step_time, time.perf_counter()

        self.metrics["train/loss"].update(outputs.loss)
        self.metrics["train/learning_rate"].update(lr)
        self.metrics["train/grad_norm"].update(grad_norm)
        self.metrics["train/step_time"].update(step_time)
        self.metrics["train/tokens_per_second_per_gpu"].update(num_tokens / step_time)
        self.metrics["train/unpadded_tokens_per_second_per_gpu"].update(num_unpadded_tokens / step_time)
        self.metrics["train/total_unpadded_tokens_per_batch"].update(num_unpadded_tokens / self.logging_frequency)

        if self._profiler is not None:
            self._profiler.step()

        if (step + 1) % self.logging_frequency == 0:
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            self.metrics["train/gpu_memory_allocated_max_gb"].update(memory_allocated)
            self.metrics["train/gpu_memory_allocated_mean_gb"].update(memory_allocated)

            metrics = self.metrics.compute()
            self.metrics.reset()
            metrics["train/global_step"] = torch.tensor(step, dtype=torch.int64)

            if self._dist_config.is_main_process():
                wandb.log(metrics, step=step)
                self._progress_bar.update(self.logging_frequency)
                self._progress_bar.set_postfix({"loss": outputs.loss.item()})

            if self._dist_config.local_rank == 0:
                logger.info(", ".join([f"{k.split('/')[1]}: {v:.3g}" for k, v in metrics.items()]))

        if (step + 1) % self._gc_interval == 0:
            self._run_garbage_collection()

    def finish(self):
        """Finish the logger and close the progress bar."""
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)

        if not self._dist_config.is_main_process():
            return

        wandb.finish()
        self._progress_bar.close()

    def _run_garbage_collection(self):
        """Run garbage collection."""
        gc.collect()
        torch.cuda.empty_cache()


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
