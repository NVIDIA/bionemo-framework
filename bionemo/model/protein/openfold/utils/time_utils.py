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


import os
import time
from datetime import datetime, timedelta

from omegaconf import DictConfig


# option to obtain this value from the environment
DEFAULT_BUFFER_BEFORE_SLURM_JOB_END_TIME = os.getenv("DEFAULT_BUFFER_BEFORE_SLURM_JOB_END_TIME", "00:05:00")
SLURM_JOB_START_TIME = "SLURM_JOB_START_TIME"
SLURM_JOB_END_TIME = "SLURM_JOB_END_TIME"


def timedelta_from_HHMMSS(walltime: str) -> timedelta:
    """Expected format
    Args:
        walltime: HH:MM:SS, the format expected by slurm
    Return:
        timedelta version of the input string
    """
    return timedelta(
        **dict(
            zip(
                ["hours", "minutes", "seconds"],
                [int(x) for x in walltime.split(":")],
            )
        )
    )


def timedelta_from_seconds(seconds_in: float) -> timedelta:
    return timedelta(microseconds=int(seconds_in * 1_000_000))


def slurm_job_start_time_as_datetime_utc() -> datetime:
    return datetime.utcfromtimestamp(int(os.getenv(SLURM_JOB_START_TIME, default=None)))


def slurm_job_end_time_as_datetime_utc() -> datetime:
    return datetime.utcfromtimestamp(int(os.getenv(SLURM_JOB_END_TIME, default=None)))


class StopWhenCloseToSlurmJobEnd:
    """This class provides method calls to sleep the calling application
    if the current time is within 'buffer_before_slurm_job_end_time' units of
    time of the slurm job end in UTC.

    Motivation: In multi-run training jobs, on a slurm cluster, we noticed
    resume failures attributed to the interruption of model checkpoint writes,
    from the job kill/term signal from the slurm cluster manager.

    The intent of this class is to prevent the initiation of checkpoint writing
    within a time interval of the slurm job end, so that corrupted model
    checkpoints are not created.
    """

    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg

        self.slurm_job_end_time_in_utc: datetime = slurm_job_end_time_as_datetime_utc()

        self.buffer_before_slurm_job_end_time: timedelta = timedelta_from_HHMMSS(
            model_cfg.get("buffer_before_slurm_job_end_time", DEFAULT_BUFFER_BEFORE_SLURM_JOB_END_TIME)
        )
        self.time_at_buffer_before_slurm_job_end_time_in_utc: datetime = (
            self.slurm_job_end_time_in_utc - self.buffer_before_slurm_job_end_time
        )

    def is_close_to_slurm_job_end_time(self):
        out: bool = datetime.utcnow() >= self.time_at_buffer_before_slurm_job_end_time_in_utc
        return out

    def sleep(self):
        """Assuming that this is called if is_close_to_slurm_job_end_time()
        returns true, then this function causes the application to sleep until
        slurm_job_end_time."""
        time.sleep(self.buffer_before_slurm_job_end_time.total_seconds() * 2)
