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

from bionemo.moco.distributions.prior.discrete.uniform import DiscreteUniformPrior
from bionemo.moco.distributions.time.uniform import UniformTimeDistribution
from bionemo.moco.interpolants.discrete_time.discrete.d3pm import D3PM
from bionemo.moco.schedules.noise.discrete_noise_schedules import DiscreteCosineNoiseSchedule


@pytest.fixture
def d3pm():
    time_distribution = UniformTimeDistribution(discrete_time=True, nsteps=1000)
    prior = DiscreteUniformPrior(num_classes=20)
    noise_schedule = DiscreteCosineNoiseSchedule(nsteps=1000)
    d3pm = D3PM(time_distribution, prior, noise_schedule)
    return d3pm


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_d3pm_interpolate(d3pm, device):
    data = torch.randint(0, 16, (5, 10)).to(device)
    t = torch.randint(0, 10, (5,)).to(device)
    d3pm.to_device(device)
    result = d3pm.interpolate(data, t)
    assert result.shape == (5, 10)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_d3pm_interpolate_square(d3pm, device):
    data = torch.randint(0, 16, (5, 10, 10)).to(device)
    t = torch.randint(0, 10, (5,)).to(device)
    d3pm.to_device(device)
    result = d3pm.interpolate(data, t)
    assert result.shape == (5, 10, 10)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_d3pm_step(d3pm, device):
    # Create a random data tensor
    num_classes = 20
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    d3pm = d3pm.to_device(device)
    torch.manual_seed(42)  # for reproducibility
    data = torch.randint(0, num_classes, (32, 5)).to(device)
    # Create time tensor
    T = 500
    time = d3pm.sample_time(32, device=device) * 0 + T
    # Create a mock model that outputs logits
    logits = torch.zeros((32, 5, num_classes), device=device)
    # Set the logits to a large value (e.g., 1000) for the correct discrete choices
    logits[:, :, :] = -1000  # initialize with a low value
    # Set the logits to 1000 for the correct discrete choices
    logits = logits.scatter(2, data.unsqueeze(-1), 1000)
    # Sample noise
    noise = d3pm.sample_prior(data.shape, device=device)
    # Create model output and xt
    model_out = logits  # torch.softmax(logits, dim=-1)
    xt = data.clone()
    xt[:, 0] = noise[:, 0]
    # Take a step
    next_xt = d3pm.step(model_out, time, xt)
    # Assert shapes
    assert next_xt.shape == data.shape
    model_out_onehot = torch.nn.functional.one_hot(
        model_out.argmax(-1), num_classes=num_classes
    ).float()  # (B, N, num_classes)
    nll = -torch.sum(torch.log(model_out_onehot.view(-1, num_classes) + 1e-8).gather(1, data.view(-1, 1)).squeeze(1))
    assert nll < 1e-10
    loss = d3pm.loss(logits, data, xt, time).mean()
    assert loss.item() == 0
    loss = d3pm.loss(logits, data, xt, time, vb_scale=0.5).mean()
    assert loss.item() < 1.0e-1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_d3pm_step_square(d3pm, device):
    # Create a random data tensor
    num_classes = 20
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    d3pm = d3pm.to_device(device)
    torch.manual_seed(42)  # for reproducibility
    data = torch.randint(0, num_classes, (32, 5, 6)).to(device)
    # Create time tensor
    T = 500
    time = d3pm.sample_time(32, device=device) * 0 + T
    # Create a mock model that outputs logits
    logits = torch.zeros((32, 5, 6, num_classes), device=device)
    # Set the logits to a large value (e.g., 1000) for the correct discrete choices
    logits[:, :, :, :] = -1000  # initialize with a low value
    # Set the logits to 1000 for the correct discrete choices
    logits = logits.scatter(3, data.unsqueeze(-1), 1000)
    # Sample noise
    noise = d3pm.sample_prior(data.shape, device=device)
    # Create model output and xt
    model_out = logits  # torch.softmax(logits, dim=-1)
    xt = data.clone()
    xt[:, 0] = noise[:, 0]
    # Take a step
    next_xt = d3pm.step(model_out, time, xt)
    # Assert shapes
    assert next_xt.shape == data.shape
    model_out_onehot = torch.nn.functional.one_hot(
        model_out.argmax(-1), num_classes=num_classes
    ).float()  # (B, N, num_classes)
    nll = -torch.sum(torch.log(model_out_onehot.view(-1, num_classes) + 1e-8).gather(1, data.view(-1, 1)).squeeze(1))
    assert nll < 1e-10
    loss = d3pm.loss(
        logits.reshape(logits.shape[0], -1, logits.shape[3]), data.reshape(data.shape[0], -1), xt, time
    ).mean()
    assert loss.item() == 0
    loss = d3pm.loss(
        logits.reshape(logits.shape[0], -1, logits.shape[3]),
        data.reshape(data.shape[0], -1),
        xt.reshape(xt.shape[0], -1),
        time,
        vb_scale=0.5,
    ).mean()
    assert loss.item() < 1.0e-1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("state_space", [20, 10, 5, 3, 2])
def test_d3pm_interpolate_notebook(device, state_space):
    B = 32  # batch size
    D = 10  # dimension
    S = state_space  # state space
    DEVICE = device
    prior = DiscreteUniformPrior(num_classes=S)
    time_distribution = UniformTimeDistribution(discrete_time=True, nsteps=1000)
    noise_schedule = DiscreteCosineNoiseSchedule(nsteps=1000)
    d3pm = D3PM(
        time_distribution=time_distribution, prior_distribution=prior, noise_schedule=noise_schedule, device=DEVICE
    )  # this failed on A100 before init on cpu then shift to GPU
    for _ in range(100):
        num_ones = torch.randint(0, D + 1, (B,))
        x1 = (torch.arange(D)[None, :] < num_ones[:, None]).long().to(DEVICE)
        t = d3pm.sample_time(B)
        xt = d3pm.interpolate(x1, t)
        assert xt.shape == x1.shape
