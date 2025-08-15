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
    EntropicInferenceSchedule,
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

@pytest.mark.parametrize("timesteps", [10, 20])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("direction", [TimeDirection.UNIFIED, TimeDirection.DIFFUSION])
def test_entropic_schedule(timesteps, device, direction):
    """Test the EntropicInferenceSchedule for correctness.
    Using a tractable predictor function to ensure the scheduler 
    produces a non-uniform schedule with the correct properties (shape, device, direction, bounds)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Dummy dim for the scheduler
    dim = 2
    # A simple time-dependent predictor. Divergence is D*(2t-1) 
    # non-uniform entropy profile.
    predictor = lambda t, x: (2 * t - 1) * x
    x_0_sampler = lambda bs: torch.randn(bs, dim, device=device)
    x_1_sampler = lambda bs: torch.randn(bs, dim, device=device)

    scheduler = EntropicInferenceSchedule(
        predictor=predictor,
        x_0_sampler=x_0_sampler,
        x_1_sampler=x_1_sampler,
        nsteps=timesteps,
        n_approx_entropy_points=25,  # Fewer points for faster testing
        batch_size=32,
        direction=direction,
        device=device,
    )

    schedule = scheduler.generate_schedule()

    assert schedule.shape == (timesteps,)
    assert schedule.device.type == device

    # Check that values are within the correct [0, 1] bounds
    assert torch.all(schedule >= 0) and torch.all(schedule <= 1)

    # Check for correct ordering based on direction
    if direction == TimeDirection.UNIFIED:
        assert schedule[0] < schedule[-1]
        assert torch.all(torch.diff(schedule) >= 0)  # Increase 0 to 1
    else:
        assert schedule[0] > schedule[-1]
        assert torch.all(torch.diff(schedule) <= 0)  # Decrease 1 to 0

    # Check that the schedule is non-uniform, confirming the entropic logic is active
    # Round to avoid float precision issues making all diffs unique
    diffs = torch.diff(torch.abs(schedule)).round(decimals=5)
    # Expect more than one unique step size, unlike a linear schedule
    if timesteps > 5:
        assert len(torch.unique(diffs)) > 1

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_entropic_schedule_reproducibility(device):
    """Checks that the the EntropicInferenceSchedule produce reproducible results when a torch.Generator with a fixed seed is provided."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    timesteps = 10
    dim = 2
    predictor = lambda t, x: t * torch.sin(x)

    gen1 = torch.Generator(device=device).manual_seed(42)
    sampler1 = lambda bs: torch.randn(bs, dim, device=device, generator=gen1)
    scheduler1 = EntropicInferenceSchedule(
        predictor=predictor,
        x_0_sampler=sampler1,
        x_1_sampler=sampler1,
        nsteps=timesteps,
        device=device,
        generator=gen1,
    )
    schedule1 = scheduler1.generate_schedule()

    # Run again with the same seed ---
    gen2 = torch.Generator(device=device).manual_seed(42)
    sampler2 = lambda bs: torch.randn(bs, dim, device=device, generator=gen2)
    scheduler2 = EntropicInferenceSchedule(
        predictor=predictor,
        x_0_sampler=sampler2,
        x_1_sampler=sampler2,
        nsteps=timesteps,
        device=device,
        generator=gen2,
    )
    schedule2 = scheduler2.generate_schedule()
    
    # Compare again with another seed
    gen3 = torch.Generator(device=device).manual_seed(99)
    sampler3 = lambda bs: torch.randn(bs, dim, device=device, generator=gen3)
    scheduler3 = EntropicInferenceSchedule(
        predictor=predictor,
        x_0_sampler=sampler3,
        x_1_sampler=sampler3,
        nsteps=timesteps,
        device=device,
        generator=gen3,
    )
    schedule3 = scheduler3.generate_schedule()

    # Schedules from identical seeds should be identical
    assert torch.allclose(schedule1, schedule2)

    # Schedules from different seeds should be different
    assert not torch.allclose(schedule1, schedule3)
