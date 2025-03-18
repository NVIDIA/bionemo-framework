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


import pytest
import torch

from bionemo.moco.schedules.noise.discrete_noise_schedules import DiscreteCosineNoiseSchedule
from bionemo.moco.schedules.utils import TimeDirection


@pytest.mark.parametrize("timesteps", [5, 10, 20])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cosine_schedule(timesteps, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    scheduler = DiscreteCosineNoiseSchedule(timesteps)
    schedule = scheduler.generate_schedule(device=device)

    # Check if schedule has the correct shape
    assert schedule.shape == (timesteps,)
    # Check if schedule is on the correct device
    assert schedule.device.type == device


@pytest.mark.parametrize("timesteps", [5, 10, 20])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("synchronize", [TimeDirection.DIFFUSION, TimeDirection.UNIFIED])
def test_cosine_schedule_direction(timesteps, device, synchronize):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    scheduler = DiscreteCosineNoiseSchedule(timesteps)
    # import ipdb; ipdb.set_trace()
    schedule = scheduler.generate_schedule(device=device, synchronize=synchronize)

    # Check if schedule has the correct shape
    assert schedule.shape == (timesteps,), f"Expected schedule shape to be {(timesteps,)}, but got {schedule.shape}"
    # Check if schedule is on the correct device
    assert schedule.device.type == device, (
        f"Expected schedule to be on device '{device}', but got '{schedule.device.type}'"
    )
    # Check if the schedule is in the correct direction

    if synchronize == TimeDirection.UNIFIED:
        assert schedule[0] < schedule[-1], (
            f"Expected schedule to be in increasing order when synchronized, but got {schedule[0]} >= {schedule[-1]}"
        )
    else:
        assert schedule[0] > schedule[-1], (
            f"Expected schedule to be in decreasing order when not synchronized, but got {schedule[0]} <= {schedule[-1]}"
        )
