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
import pprint
import time
from collections import deque

import numpy as np
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
        if not dist_config.is_main_process():
            return

        # Log the entire args object to wandb for experiment tracking and reproducibility.s
        wandb.init(**args.wandb_init_args, config=self._run_config)

        self._progress_bar = tqdm(total=args.num_train_steps, desc="Training")

        # Store the last `max_step_times` step times to compute the mean values at the end of training.
        self._step_times = deque(maxlen=args.logger.max_step_times)
        self._num_tokens_per_second = deque(maxlen=args.logger.max_step_times)
        self._num_unpadded_tokens_per_second = deque(maxlen=args.logger.max_step_times)

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
        if not self._dist_config.is_main_process():
            return

        # Store these in buffers to compute the mean values at the end of training.
        step_time = self.get_step_duration()
        self._num_tokens_per_second.append(num_tokens / step_time)
        self._num_unpadded_tokens_per_second.append(num_unpadded_tokens / step_time)

        metrics = {
            "train/loss": loss,
            "train/global_step": step,
            "train/learning_rate": lr,
            "train/grad_norm": grad_norm,
            "train/step_time": step_time,
            "train/tokens_per_second": self._num_tokens_per_second[-1],
            "train/unpadded_tokens_per_second": self._num_unpadded_tokens_per_second[-1],
        }

        self._progress_bar.update(1)
        self._progress_bar.set_postfix({"loss": loss})

        wandb.log(metrics)
        logger.info(
            ", ".join(
                [
                    f"{k.split('/')[1]}: {v:.3g}" if isinstance(v, float) else f"{k.split('/')[1]}: {v}"
                    for k, v in metrics.items()
                ]
            )
        )

    def get_step_duration(self):
        """Get the duration of the last step.

        Returns:
            float: The duration of the last step.
        """
        if not self._dist_config.is_main_process():
            raise RuntimeError("Step duration can only be logged on the main process")

        self._step_times.append(time.perf_counter())
        if len(self._step_times) > 1:
            return self._step_times[-1] - self._step_times[-2]
        else:
            return np.nan

    def finish(self):
        """Finish the logger and close the progress bar."""
        if not self._dist_config.is_main_process():
            return

        wandb.finish()
        self._progress_bar.close()

        # Log the run config, distributed config, and final averaged metrics to stdout.
        logger.info("RUN CONFIG:\n%s", pprint.pformat(self._run_config))
        logger.info("DISTRIBUTED CONFIG:\n%s", pprint.pformat(self._dist_config.__dict__))
        logger.info(
            f"FINAL METRICS:\n"
            f"Minimum loss seen: {self.min_loss:.3g}\n"
            f"Mean step time (last {len(self._step_times)} steps): {np.diff(self._step_times).mean():.3g}s\n"
            f"Mean tokens per second (per GPU): {np.mean(self._num_tokens_per_second):.3g}\n"
            f"Mean unpadded tokens per second (per GPU) (last {len(self._num_unpadded_tokens_per_second)} steps): "
            f"{np.mean(self._num_unpadded_tokens_per_second):.3g}\n"
        )
