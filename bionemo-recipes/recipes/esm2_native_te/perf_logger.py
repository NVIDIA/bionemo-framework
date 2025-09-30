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

import torch
import torchmetrics
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

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
        self.metrics = torchmetrics.MetricCollection(
            {
                "train/loss": torchmetrics.MeanMetric(),
                "train/grad_norm": torchmetrics.MeanMetric(),
                "train/learning_rate": torchmetrics.MeanMetric(),
                "train/step_time": torchmetrics.MeanMetric(),
                "train/tokens_per_second": torchmetrics.MeanMetric(),
                "train/unpadded_tokens_per_second": torchmetrics.MeanMetric(),
            }
        )
        # We move metrics to a GPU device so we can use torch.distributed to aggregate them before logging.
        self.metrics.to(torch.device(f"cuda:{dist_config.local_rank}"))
        self.previous_step_time = time.perf_counter()

        if self._dist_config.is_main_process():
            # Log the entire args object to wandb for experiment tracking and reproducibility.
            wandb.init(**args.wandb_init_args, config=self._run_config)
            self._progress_bar = tqdm(total=args.num_train_steps, desc="Training")

    def log_step(
        self,
        step: int,
        num_tokens: int,
        num_unpadded_tokens: int,
        loss: float,
        grad_norm: float,
        lr: float,
    ):
        """Log a step to the logger and wandb.

        Args:
            step: The step number.
            num_tokens: The input tokens for the step, used to track token throughput.
            num_unpadded_tokens: The number of non-padded tokens for the step, used to track token throughput.
            loss: The loss of the step.
            grad_norm: The gradient norm of the step.
            lr: The learning rate of the step.
        """
        self.min_loss = min(self.min_loss, loss)
        step_time, self.previous_step_time = time.perf_counter() - self.previous_step_time, time.perf_counter()

        self.metrics["train/loss"].update(loss)
        self.metrics["train/learning_rate"].update(lr)
        self.metrics["train/grad_norm"].update(grad_norm)
        self.metrics["train/step_time"].update(step_time)
        self.metrics["train/tokens_per_second"].update(num_tokens / step_time)
        self.metrics["train/unpadded_tokens_per_second"].update(num_unpadded_tokens / step_time)

        if step % self.logging_frequency == 0 and step > 0:
            metrics = self.metrics.compute()
            self.metrics.reset()
            metrics["train/global_step"] = torch.tensor(step, dtype=torch.int64)

            if self._dist_config.is_main_process():
                wandb.log(metrics, step=step)
                self._progress_bar.update(self.logging_frequency)
                self._progress_bar.set_postfix({"loss": loss})

            if self._dist_config.local_rank == 0:
                logger.info(", ".join([f"{k.split('/')[1]}: {v:.3g}" for k, v in metrics.items()]))

    def finish(self):
        """Finish the logger and close the progress bar."""
        if not self._dist_config.is_main_process():
            return

        wandb.finish()
        self._progress_bar.close()
