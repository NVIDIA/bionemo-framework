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

from bionemo.moco.schedules.inference_time_schedules import (
    DiscreteLinearInferenceSchedule,
    LinearInferenceSchedule,
    LogInferenceSchedule,
    PowerInferenceSchedule,
)
from bionemo.moco.schedules.utils import TimeDirection


@pytest.mark.parametrize("timesteps", [5, 10, 20])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("direction", [TimeDirection.UNIFIED, TimeDirection.DIFFUSION])
def test_uniform_dt(timesteps, device, direction):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    scheduler = LinearInferenceSchedule(timesteps, direction=direction)
    dt = scheduler.discretize(device=device)
    schedule = scheduler.generate_schedule(device=device)

    # Check if all dt's are equal to 1/timesteps
    assert torch.allclose(dt, torch.ones_like(dt) / timesteps)
    assert dt.device.type == device

    # Check if schedule has the correct shape
    assert schedule.shape == (timesteps,)
    # Check if dt has the correct shape
    assert dt.shape == (timesteps,)
    # Check if schedule is on the correct device
    assert schedule.device.type == device
    if direction == TimeDirection.UNIFIED:
        assert schedule[0] < schedule[-1]
    else:
        assert schedule[0] > schedule[-1]


@pytest.mark.parametrize("timesteps", [5, 10, 20])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("power", [0.5, 1.5, 2.0])
@pytest.mark.parametrize("direction", [TimeDirection.UNIFIED, TimeDirection.DIFFUSION])
def test_power_dt(timesteps, device, power, direction):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    scheduler = PowerInferenceSchedule(timesteps, exponent=power, direction=direction)
    dt = scheduler.discretize(device=device)
    schedule = scheduler.generate_schedule(device=device)

    assert dt.device.type == device

    # Check if schedule has the correct shape
    assert schedule.shape == (timesteps,)
    # Check if dt has the correct shape
    assert dt.shape == (timesteps,)
    # Check if schedule is on the correct device
    assert schedule.device.type == device
    if direction == TimeDirection.UNIFIED:
        assert schedule[0] < schedule[-1]
    else:
        assert schedule[0] > schedule[-1]


@pytest.mark.parametrize("timesteps", [5, 10, 20])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("direction", [TimeDirection.UNIFIED, TimeDirection.DIFFUSION])
def test_log_dt(timesteps, device, direction):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    scheduler = LogInferenceSchedule(timesteps, exponent=-2, direction=direction)
    dt = scheduler.discretize(device=device)
    schedule = scheduler.generate_schedule(device=device)

    assert dt.device.type == device

    # Check if schedule has the correct shape
    assert schedule.shape == (timesteps,)
    # Check if dt has the correct shape
    assert dt.shape == (timesteps,)
    # Check if schedule is on the correct device
    assert schedule.device.type == device
    if direction == TimeDirection.UNIFIED:
        assert schedule[0] < schedule[-1] and schedule[0] == 0
    else:
        assert schedule[0] > schedule[-1] and schedule[0] == 1


@pytest.mark.parametrize("timesteps", [5, 10, 20])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("direction", [TimeDirection.UNIFIED, TimeDirection.DIFFUSION])
def test_discrete_uniform_dt(timesteps, device, direction):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    scheduler = DiscreteLinearInferenceSchedule(timesteps, direction=direction)
    dt = scheduler.discretize(device=device)
    schedule = scheduler.generate_schedule(device=device)

    assert dt.device.type == device

    # Check if schedule has the correct shape
    assert schedule.shape == (timesteps,)
    # Check if dt has the correct shape
    assert dt.shape == (timesteps,)
    # Check if schedule is on the correct device
    assert schedule.device.type == device
    # Additional checks specific to DiscreteUniformInferenceSchedule
    assert torch.all(dt == torch.full((timesteps,), 1 / timesteps, device=device))
    if direction == TimeDirection.UNIFIED:
        assert schedule[0] < schedule[-1]
    else:
        assert schedule[0] > schedule[-1]


@pytest.mark.parametrize("timesteps", [10, 20])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("direction", [TimeDirection.UNIFIED, TimeDirection.DIFFUSION])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("dilation", [0, 1])
def test_uniform_dt_padding_dilation(timesteps, device, direction, padding, dilation):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    scheduler = LinearInferenceSchedule(timesteps, padding=padding, dilation=dilation, direction=direction)
    dt = scheduler.discretize(device=device)
    schedule = scheduler.generate_schedule(device=device)

    # Check if all dt's are equal to 1/timesteps
    assert dt.device.type == device

    # Check if schedule has the correct shape
    assert schedule.shape == (timesteps,)
    # Check if dt has the correct shape
    assert dt.shape == (timesteps,)
    # Check if schedule is on the correct device
    assert schedule.device.type == device
    if direction == TimeDirection.UNIFIED:
        assert schedule[0] < schedule[-1]
        for i in range(padding):
            assert schedule[-1 * (i + 1)] == 1.0
    else:
        assert schedule[0] > schedule[-1]
        for i in range(padding):
            assert schedule[-1 * (i + 1)] == 0
