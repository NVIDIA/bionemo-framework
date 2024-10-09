# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from nemo.utils import logging
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

from bionemo.utils.dllogger import DLLogger
from bionemo.utils.tests import save_expected_training_results


class PerfLoggingCallback(Callback):
    """
    Callback logger that measures performance of the model per global batch.
    The logged metrics are throughput and latency.
    """

    def __init__(self, dllogger: DLLogger, global_batch_size: int, mode: str, warmup: int = 10):
        """
        Args:
            dllogger: instance of dllogger
            global_batch_size: the cumulative batch size
            mode: mode of the logging ie test, train, infer
            warmup: number of warmup steps before the logging starts

        TODO: add logging for testing: at on_test_batch_end and on_test_batch_start
        """

        self.dllogger = dllogger
        self.warmup_steps = warmup
        self.global_batch_size = global_batch_size
        self.step = 0
        self.step_val = 0
        self.mode = mode
        self.timestamps = {"start": [], "end": []}
        self.timestamps_val = {"start": [], "end": []}

    def log_metadata(self, postfix: str = ""):
        self.dllogger.log_metadata(f"throughput_mean_{self.mode}{postfix}", {"unit": "samples/s"})
        self.dllogger.log_metadata(f"latency_mean_{self.mode}{postfix}", {"unit": "ms"})
        self.dllogger.log_metadata(f"throughput_mean_{self.mode}_total{postfix}", {"unit": "samples/s"})
        self.dllogger.log_metadata(f"latency_mean_{self.mode}_total{postfix}", {"unit": "ms"})
        for level in [90, 95, 99]:
            self.dllogger.log_metadata(f"latency_{level}_{self.mode}{postfix}", {"unit": "ms"})

        for level in [90, 95, 99]:
            self.dllogger.log_metadata(f"latency_{level}_{self.mode}_total{postfix}", {"unit": "ms"})

    def update(self, phase: str, val_batch: bool = False):
        if phase == "start":
            if val_batch:
                self.step_val += 1
            else:
                self.step += 1

        if self._should_log():
            torch.cuda.synchronize()
            if val_batch:
                self.timestamps_val[phase].append(time.perf_counter())
            else:
                self.timestamps[phase].append(time.perf_counter())

    def _should_log(self):
        return (self.step + self.step_val) > self.warmup_steps

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.update(phase="start")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.update(phase="end")

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.update(phase="start", val_batch=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.update(phase="end", val_batch=True)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.update(phase="start")

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.update(phase="end")

    def _calculate_performance_stats(self, timestamps: Dict[str, List[float]]) -> Dict[str, float]:
        """Given a set of timestamps stored in a dictionary of `{start, end}` -> [timestamps...]
            return a dictionary of timing statistics.

        Args:
            timestamps (Dict[str, List[float]]): Named disctionary of timestamps. Typically the keys are
                "start" or "end".

        Notes:
            Due to an issue with lightning >=2.0,<2.1 and/or no workaround
                in NeMo, we somtimes do not call a final on_*_batch_end since we use the dataloader_iter
                and do not always handle StopIteration properly https://github.com/Lightning-AI/pytorch-lightning/issues/17612
                We currently handle these off by one len(start) == len(end)+1 issues with a warning rather than raising
                an exception.
        """

        def _round3(val):
            return round(val, 3)

        end_ts = np.array(timestamps["end"])
        start_ts = np.array(timestamps["start"])
        if (len(start_ts) - len(end_ts)) == 1:
            logging.warning(
                "Start-end timestamp lengths are off by 1, last batch was likely never hit with the on_*_batch_end() callbacks."
            )
            start_ts = start_ts[:-1]
        if abs(len(start_ts) - len(end_ts)) > 1:
            raise ValueError(
                f"There is a large difference between start and end timesteps, please debug: {len(start_ts)}, {len(end_ts)}"
            )
        elapsed_times = end_ts - start_ts
        elapsed_times_total = np.diff(end_ts)

        throughput = _round3(self.global_batch_size / np.mean(elapsed_times))
        throughput_total = _round3(self.global_batch_size / np.mean(elapsed_times_total))
        timestamps_ms = 1000 * elapsed_times
        timestamps_ms_total = 1000 * elapsed_times_total
        stats = {
            f"throughput_mean_{self.mode}": throughput,
            f"throughput_mean_{self.mode}_total": throughput_total,
            f"latency_mean_{self.mode}": _round3(np.mean(timestamps_ms)),
            f"latency_mean_{self.mode}_total": _round3(np.mean(timestamps_ms_total)),
        }

        if len(timestamps_ms) > 1:
            for level in [90, 95, 99]:
                stats.update({f"latency_{level}_{self.mode}": _round3(np.percentile(timestamps_ms, level))})
                stats.update(
                    {f"latency_{level}_{self.mode}_total": _round3(np.percentile(timestamps_ms_total, level))}
                )
        return stats

    @rank_zero_only
    def process_performance_stats(self):
        stats = self._calculate_performance_stats(self.timestamps)
        self.log_metadata()
        if len(self.timestamps_val["start"]) > 0:
            stats_val = self._calculate_performance_stats(self.timestamps_val)
            self.log_metadata(postfix="_val")
            stats_val = {f"{k}_val": v for k, v in stats_val.items()}
            stats.update(stats_val)
        return stats

    @rank_zero_only
    def _log(self, stats: Dict, step: Optional[Tuple] = None):
        self.dllogger.log_metrics(metrics=stats, step=step)
        self.dllogger.flush()

    def on_train_end(self, trainer, pl_module):
        stats = self.process_performance_stats()
        self._log(stats=stats)

    def on_predict_end(self, trainer, pl_module):
        stats = self.process_performance_stats()
        self._log(stats=stats)

    def on_test_end(self, trainer, pl_module):
        stats = self.process_performance_stats()
        self._log(stats=stats)


class SaveTrainerFinalMetricCallback(Callback):
    """
    Saves values of PTL Trainer's metrics to json file at the end of the training
    """

    def __init__(self, log_path: str = "."):
        """

        Args:
            log_path: destination where the json file should be saved
        """
        self.log_path = log_path
        self.trainer_log_file = "trainer_logs.json"

    def on_train_end(self, trainer, pl_module):
        trainer_results = trainer.logged_metrics
        logging.info(f"Saving expected training results to {self.log_path}/{self.trainer_log_file}")
        save_expected_training_results(
            self.log_path, self.trainer_log_file, {k: trainer_results[k].item() for k in trainer_results.keys()}
        )
