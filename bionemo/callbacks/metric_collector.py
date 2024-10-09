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


#
#
#
#
import csv
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from typing import Callable, List

import psutil
import torch
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)
from pytorch_lightning import Callback, LightningModule, Trainer

from bionemo.utils.logging_utils import (
    log_with_nemo_at_level,
)


SLEEP_TIME_FOR_FILE_IO = 5
SLURM_JOB_ID = "SLURM_JOB_ID"


@dataclass
class Event:
    event_type: str
    instance_name: str
    time_point: float  # seconds
    global_step: int
    global_rank: int
    device_memory_used_from_nvml: int  # bytes
    device_memory_total_from_nvml: int  # bytes
    device_memory_allocated_from_torch_cuda: int  # bytes
    device_memory_total_from_torch_cuda: int  # bytes
    gpu_util_rate_from_nvml: float
    cpu_util_rate_from_psutil: float
    main_virtual_memory_used_from_psutil: int  # bytes
    main_virtual_memory_total_from_psutil: int  # bytes
    sensor_temperatures_from_psutil: str
    other: dict = None


EVENT_FIELDS_AT_LEFT = [
    "instance_name",
    "event_type",
    "time_point",
    "device_memory_used_from_nvml",
    "device_memory_total_from_nvml",
    "device_memory_allocated_from_torch_cuda",
    "device_memory_total_from_torch_cuda",
]

EVENT_FIELDS_BLACKLIST = ["sensor_temperatures_from_psutil"]


@dataclass
class Interval:
    """Will it be heavy to store the address of the events?"""

    instance_name: str
    time_delta: float  # seconds
    global_step: int
    global_rank: int
    start_event_instance_name: str
    end_event_instance_name: str
    start_event_type: str
    end_event_type: str
    other: dict = None


INTERVAL_FIELDS_AT_LEFT = [
    "instance_name",
    "time_delta",
]
INTERVAL_FIELDS_BLACKLIST = []


class MetricCollector(Callback):
    """Callback to record metrics at events, and for intervals.

    The goal for this class is to record the duration of each training step,
    for each process-group (rank) in a distributed-data-parallel multi-run
    training job..

    Multi-run implications:
        Multi-run / multi-session refers to a job that is composed of a
        sequence of sub-jobs, each sub-job a slurm job, terminated by time limit.
        Since the full training job is composed of time-separated processes, the
        results from each
        job must written to file (or db table).  Full training statistics must
        be an aggregation of the contents of files, over runs.

    Distributed-data-parallel implications:
        For each run, the training sub-job is composed of process-groups, one
        per GPU.  Looking ahead, we'll want metrics for each input featurization
        step, and training step, for each process group.  So, we instantiate
        this callback for each process group.

        We can choose to aggregate metrics from certain process-groups
        in post-processing..

    Outputs:
        For each run / session, and for each process-group (global rank),
        the directory
            container: /results/metrics_collector_subdir
            lustre: experiment-dir/artifacts/metrics_collector_subdir

        The expected files are like
            "intervals_gstepatstart_1200_grank_2.csv",

        Each file is overwritten at each occurance of on_validation_end()

        To print the first 4 columns of an output csv file
            cat /results/metric_collector/intervals_gstep3_grank  0.csv | cut -d, -f-4


    See # https://pythonhosted.org/nvidia-ml-py/

    See https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/callbacks/callback.py#L211
    for up-to-date list of callback methods and their prototypes.

    For more see:
    https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.hooks.ModelHooks.html
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.hooks.DataHooks.html

    https://stackoverflow.com/questions/75931387/trying-to-fetch-cpu-gpu-vpu-usage-and-utilization-through-python
    """

    DOCKER_CONTAINER_RESULT_DIR = "/result"  # singular
    PID = "pid"
    ON_FIT_START = "on_fit_start"
    ON_TRAIN_START = "on_train_start"
    ON_TRAIN_BATCH_END = "on_train_batch_end"
    ON_VALIDATION_END = "on_validation_end"
    EVENTS_TO_UPDATE_PREV_EVENT = [ON_TRAIN_START, ON_TRAIN_BATCH_END, ON_VALIDATION_END]
    START_EVENTS_FOR_TRAIN_STEP_INTERVAL = [ON_TRAIN_START, ON_TRAIN_BATCH_END, ON_VALIDATION_END]

    def __init__(
        self,
        time_fn: Callable = time.perf_counter,
        metric_collector_subdir: str = "metric_collector",
        do_rank_zero_only: bool = True,
    ):
        """ """
        # (1) store init parameters
        self.time_fn = time_fn
        self.metric_collector_dir = os.path.join(self.DOCKER_CONTAINER_RESULT_DIR, metric_collector_subdir)
        self._metric_collector_dir_this_rank = None
        self.do_rank_zero_only = do_rank_zero_only

        # (2) event indices and containers
        self.i_for_each_event = {}
        self.events = []
        self.intervals = []

        # (3) state
        self.global_rank = None
        self.did_on_train_start = False
        self.prev_event_for_on_train_batch_end = None
        self.time_point_origin = None

        nvmlInit()
        self.nvml_device_memory_info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
        self.nvml_device_util_rates = nvmlDeviceGetUtilizationRates(nvmlDeviceGetHandleByIndex(0))

        log_with_nemo_at_level("""MetricCollector.__init__, end""")

    def __del__(self):
        nvmlShutdown()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """There will be a single call to on_train_start for each training run, i.e.,
        slurm job."""
        event = self._create_event(trainer, pl_module)
        if event.event_type in self.EVENTS_TO_UPDATE_PREV_EVENT:
            self.prev_event_for_on_train_batch_end = event

        if not self.global_rank:
            self.global_rank = trainer.global_rank

        log_with_nemo_at_level("""MetricCollector.on_fit_start, end""")

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        """There will be a single call to on_train_start for each training run, i.e.,
        slurm job.  This is called after the Sanity Check step."""
        event = self._create_event(trainer, pl_module)
        if event.event_type in self.EVENTS_TO_UPDATE_PREV_EVENT:
            self.prev_event_for_on_train_batch_end = event
        self.did_on_train_start = True
        log_with_nemo_at_level("""MetricCollector.on_train_start, end""")

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx):
        log_with_nemo_at_level("""MetricCollector.on_train_batch_end, start""")
        event = self._create_event(trainer, pl_module)

        # (2) logic / math operations
        if self.prev_event_for_on_train_batch_end.event_type in self.START_EVENTS_FOR_TRAIN_STEP_INTERVAL:
            t_interval = self.time_fn()
            interval = Interval(
                instance_name=event.instance_name,
                time_delta=(event.time_point - self.prev_event_for_on_train_batch_end.time_point),
                global_step=trainer.global_step,
                global_rank=trainer.global_rank,
                start_event_instance_name=self.prev_event_for_on_train_batch_end.instance_name,
                end_event_instance_name=event.instance_name,
                start_event_type=self.prev_event_for_on_train_batch_end.event_type,
                end_event_type=event.event_type,
                other={self.PID: os.getpid()},
            )
            interval.other["time_delta_interval"] = self.time_fn() - t_interval
            self.intervals.append(interval)

        if event.event_type in self.EVENTS_TO_UPDATE_PREV_EVENT:
            self.prev_event_for_on_train_batch_end = event
        log_with_nemo_at_level("""MetricCollector.on_train_batch_end, end""")

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        """Typically training occurs over multiple slurm jobs, where each
        slurm job exits at a time limit.  So, write output to file at
        each on_validation_end.

        on_validation_epoch_end()
            stop_when.is_close_to_slurm_job_end_time() decision to sleep is made.

        on_validation_end()
            Last and EMA checkpoints are written.

        """
        log_with_nemo_at_level(
            f"""
            MetricCollector.on_validation_end, start
            len(self.events)={len(self.events)}
            len(self.intervals)={len(self.intervals)}
            """
        )

        # For the call at the end of Sanity Check step, early return
        if not self.did_on_train_start:
            return

        # write results from training steps before validation step
        if trainer.global_rank == 0 or not self.do_rank_zero_only:
            self._write_intervals_to_file_and_delete(trainer)
            self._write_events_to_file_and_delete(trainer)

        # create an event to start the interval for the next training step
        event = self._create_event(trainer, pl_module)
        if event.event_type in self.EVENTS_TO_UPDATE_PREV_EVENT:
            self.prev_event_for_on_train_batch_end = event

        log_with_nemo_at_level("""MetricCollector.on_validation_end, end""")

    def _create_event(self, trainer: Trainer, pl_module: LightningModule, event_type: str = None) -> Event:
        if event_type is None:
            event_type = sys._getframe(1).f_code.co_name

        if event_type not in self.i_for_each_event:
            self.i_for_each_event[event_type] = 1
        else:
            self.i_for_each_event[event_type] += 1

        if event_type == self.ON_FIT_START:
            self.time_point_origin = self.time_fn()
        t_event = self.time_fn()
        event = Event(
            event_type=event_type,
            instance_name=f"{event_type}_{self.i_for_each_event[event_type]}",
            time_point=(self.time_fn() - self.time_point_origin),
            global_step=trainer.global_step,
            global_rank=trainer.global_rank,
            device_memory_used_from_nvml=self.nvml_device_memory_info.used,
            device_memory_total_from_nvml=self.nvml_device_memory_info.total,
            device_memory_allocated_from_torch_cuda=torch.cuda.memory_allocated(0),
            device_memory_total_from_torch_cuda=torch.cuda.get_device_properties(0).total_memory,
            gpu_util_rate_from_nvml=self.nvml_device_util_rates.gpu,
            cpu_util_rate_from_psutil=psutil.cpu_percent(),
            main_virtual_memory_used_from_psutil=str(psutil.virtual_memory().used),
            main_virtual_memory_total_from_psutil=str(psutil.virtual_memory().total),
            sensor_temperatures_from_psutil=str(psutil.sensors_temperatures()),
            other={self.PID: os.getpid()},
        )
        event.other["time_delta_event"] = self.time_fn() - t_event
        self.events.append(event)

        return event

    @property
    def metric_collector_dir_this_rank(self) -> str:
        if not self._metric_collector_dir_this_rank:
            global_rank_as_padded_str = left_pad_with_zeros(self.global_rank, target_len=3)
            self._metric_collector_dir_this_rank = os.path.join(
                self.metric_collector_dir, f"grank{global_rank_as_padded_str}"
            )

        return self._metric_collector_dir_this_rank

    @property
    def basefilenames_in_metric_collector_dir(self) -> List[str]:
        listdir_out = os.listdir(self.metric_collector_dir_this_ran)
        return [x for x in listdir_out if os.path.isfile(os.path.join(self.metric_collector_dir_this_rank, x))]

    def _write_intervals_to_file_and_delete(self, trainer: Trainer):
        log_with_nemo_at_level("""MetricCollector._write_intervals_to_file, start""")
        # make directory and filename
        os.makedirs(self.metric_collector_dir_this_rank, exist_ok=True)
        intervals_full_filename = self._intervals_full_filename(trainer)

        # write file
        with open(intervals_full_filename, "w", newline="") as file:
            prefix = INTERVAL_FIELDS_AT_LEFT
            blacklist = INTERVAL_FIELDS_BLACKLIST
            fieldnames = prefix + [f.name for f in fields(Interval) if f.name not in list(prefix + blacklist)]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for interval in self.intervals:
                writer.writerow({k: v for k, v in asdict(interval).items() if k not in blacklist})
        del self.intervals[:]
        log_with_nemo_at_level("""MetricCollector._write_intervals_to_file, start""")

    def _write_events_to_file_and_delete(self, trainer: Trainer):
        log_with_nemo_at_level("""MetricCollector._write_events_to_file, start""")
        # make directory and filename
        os.makedirs(self.metric_collector_dir_this_rank, exist_ok=True)
        events_full_filename = self._events_full_filename(trainer)

        # write file
        with open(events_full_filename, "w", newline="") as file:
            prefix = EVENT_FIELDS_AT_LEFT
            blacklist = EVENT_FIELDS_BLACKLIST
            fieldnames = prefix + [f.name for f in fields(Event) if f.name not in list(prefix + blacklist)]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for event in self.events:
                writer.writerow({k: v for k, v in asdict(event).items() if k not in blacklist})

        del self.events[:]
        log_with_nemo_at_level("""MetricCollector._write_events_to_file, end""")

    def _intervals_full_filename(self, trainer: Trainer) -> str:
        intervals_base_filename: str = f"intervals_{self._tag_for_filename(trainer)}.csv"
        intervals_full_filename: str = os.path.join(
            self.metric_collector_dir_this_rank,
            intervals_base_filename,
        )
        return intervals_full_filename

    def _events_full_filename(self, trainer: Trainer) -> str:
        events_base_filename: str = f"events_{self._tag_for_filename(trainer)}.csv"
        events_full_filename: str = os.path.join(
            self.metric_collector_dir_this_rank,
            events_base_filename,
        )
        return events_full_filename

    def _tag_for_filename(self, trainer: Trainer) -> str:
        return f"jobid{os.getenv(SLURM_JOB_ID)}_gstep{trainer.global_step}_grank{trainer.global_rank}"


def left_pad_with_zeros(x_in, target_len=3):
    x_out = str(x_in)
    if target_len > len(x_out):
        x_out = (target_len - len(x_out)) * "0" + x_out

    return x_out
