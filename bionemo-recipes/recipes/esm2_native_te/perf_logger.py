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
from collections import deque

import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)


class PerfLogger:
    """Class to log performance metrics to stdout and wandb.

    Args:
        dist_config: The distributed configuration.
        args: The arguments.

    Attributes:
        min_loss: The minimum loss seen so far.
    """

    def __init__(self, dist_config: DistributedConfig, args: DictConfig):
        """Initialize the logger."""
        self._dist_config = dist_config
        self.min_loss = float("inf")
        if not dist_config.is_main_process():
            return

        # Log the entire args object to wandb for experiment tracking and reproducibility.s
        wandb.init(**args.wandb_init_args, config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True))

        # Initialize the progress bar and step time buffer. We store the last N step times to compute the mean step time
        # reliably.
        self._progress_bar = tqdm(total=args.num_train_steps, desc="Training")
        self._step_times = deque(maxlen=args.logger.max_step_times)

    def log_step(self, step: int, loss: float, grad_norm: float, lr: float):
        """Log a step to the logger and wandb."""
        self.min_loss = min(self.min_loss, loss)
        if not self._dist_config.is_main_process():
            return

        step_time, mean_step_time = self.get_step_duration()

        metrics = {
            "train/loss": loss,
            "train/global_step": step,
            "train/learning_rate": lr,
            "train/grad_norm": grad_norm,
            "train/step_time": step_time,
            "train/mean_step_time": mean_step_time,
        }

        self._progress_bar.update(1)
        self._progress_bar.set_postfix({"loss": loss})

        wandb.log(metrics)
        logger.info(
            ", ".join(
                [
                    f"{k.strip('train/')}: {v:.3g}" if isinstance(v, float) else f"{k.strip('train/')}: {v}"
                    for k, v in metrics.items()
                ]
            )
        )

    def get_step_duration(self):
        """Get the duration of the last step and the mean duration of the last N steps.

        Returns:
            tuple[float, float]: The duration of the last step and the mean duration of the last N steps, where N is set
            via logger.max_step_times.
        """
        if not self._dist_config.is_main_process():
            raise RuntimeError("Step duration can only be logged on the main process")

        self._step_times.append(time.perf_counter())
        if len(self._step_times) > 1:
            step_durations = np.diff(self._step_times)
            return step_durations[-1], step_durations.mean()

        else:
            return None, None

    def finish(self):
        """Finish the logger and close the progress bar."""
        if not self._dist_config.is_main_process():
            return

        wandb.finish()
        self._progress_bar.close()
        logger.info(
            f"FINAL METRICS: Minimum loss seen: {self.min_loss:.3g}, "
            f"Mean step time (last {len(self._step_times)} steps): {np.diff(self._step_times).mean():.3g}s"
        )
