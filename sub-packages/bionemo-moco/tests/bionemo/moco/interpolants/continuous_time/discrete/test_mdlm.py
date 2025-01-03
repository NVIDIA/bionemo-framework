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

from bionemo.moco.distributions.prior.discrete.mask import DiscreteMaskedPrior
from bionemo.moco.distributions.time.uniform import UniformTimeDistribution
from bionemo.moco.interpolants.continuous_time.discrete.mdlm import MDLM
from bionemo.moco.schedules.noise.continuous_noise_transforms import LogLinearExpNoiseTransform


@pytest.fixture
def mdlm():
    time_distribution = UniformTimeDistribution(discrete_time=False)
    prior = DiscreteMaskedPrior(num_classes=20)
    noise_schedule = LogLinearExpNoiseTransform()
    mdlm = MDLM(time_distribution, prior, noise_schedule)
    return mdlm


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mdlm_interpolate(mdlm, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    data = torch.randint(0, 16, (5, 10)).to(device)
    t = torch.rand((5,)).to(device)
    mdlm.to_device(device)
    result = mdlm.interpolate(data, t)
    assert result.shape == (5, 10)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mdlm_step(mdlm, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # Create a random data tensor
    num_classes = 20
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    mdlm = mdlm.to_device(device)
    torch.manual_seed(42)  # for reproducibility
    data = torch.randint(0, num_classes - 1, (32, 5)).to(device)
    # Create time tensor
    # T = 500
    time = mdlm.sample_time(32, device=device)  # * 0 + T
    # Create a mock model that outputs logits
    logits = torch.zeros((32, 5, num_classes), device=device)
    # Set the logits to a large value (e.g., 1000) for the correct discrete choices
    logits[:, :, :] = -1000  # initialize with a low value
    # Set the logits to 1000 for the correct discrete choices
    logits = logits.scatter(2, data.unsqueeze(-1), 1)
    # Sample noise
    noise = mdlm.sample_prior(data.shape, device=device)
    # Create model output and xt
    model_out = logits  # torch.softmax(logits, dim=-1)
    xt = data.clone()
    xt[:, 0] = noise[:, 0]
    time = time * 0 + 40 / 100
    next_xt = mdlm.step(model_out, time, xt, dt=1 / 100)
    score = mdlm.calculate_score(logits, xt, time)
    assert score.shape == logits.shape
    next_xt = mdlm.step_argmax(model_out)
    # Assert shapes
    assert next_xt.shape == data.shape
    model_out_onehot = torch.nn.functional.one_hot(
        model_out.argmax(-1), num_classes=num_classes
    ).float()  # (B, N, num_classes)
    nll = -torch.sum(torch.log(model_out_onehot.view(-1, num_classes) + 1e-8).gather(1, data.view(-1, 1)).squeeze(1))
    assert nll < 1e-10
    loss = mdlm.loss(logits, data, xt, time)
    assert loss.mean() == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mdlm_step_confidence(mdlm, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # Create a random data tensor
    num_classes = 20
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    mdlm = mdlm.to_device(device)
    torch.manual_seed(42)  # for reproducibility
    data = torch.randint(0, num_classes - 1, (32, 5)).to(device)
    # Create time tensor
    # T = 500
    time = mdlm.sample_time(32, device=device)  # * 0 + T
    # Create a mock model that outputs logits
    logits = torch.zeros((32, 5, num_classes), device=device)
    # Set the logits to a large value (e.g., 1000) for the correct discrete choices
    logits[:, :, :] = -1000  # initialize with a low value
    # Set the logits to 1000 for the correct discrete choices
    logits = logits.scatter(2, data.unsqueeze(-1), 1)
    # Sample noise
    noise = mdlm.sample_prior(data.shape, device=device)
    # Create model output and xt
    model_out = logits  # torch.softmax(logits, dim=-1)
    xt = data.clone()
    xt[:, 0] = noise[:, 0]
    time = time * 0 + 2 / 100
    next_xt = mdlm.step_confidence(model_out, xt, curr_step=90, num_steps=100)
    # Assert shapes
    assert next_xt.shape == data.shape
    model_out_onehot = torch.nn.functional.one_hot(
        model_out.argmax(-1), num_classes=num_classes
    ).float()  # (B, N, num_classes)
    nll = -torch.sum(torch.log(model_out_onehot.view(-1, num_classes) + 1e-8).gather(1, data.view(-1, 1)).squeeze(1))
    assert nll < 1e-10
    loss = mdlm.loss(logits, data, xt, time)
    assert loss.mean() == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mdlm_interpolate_square(mdlm, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    data = torch.randint(0, 16, (5, 10, 10)).to(device)
    t = torch.rand((5,)).to(device)
    mdlm.to_device(device)
    result = mdlm.interpolate(data, t)
    assert result.shape == (5, 10, 10)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mdlm_step_square(mdlm, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # Create a random data tensor

    num_classes = 20
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    mdlm = mdlm.to_device(device)
    torch.manual_seed(42)  # for reproducibility
    data = torch.randint(0, num_classes - 1, (5, 10, 10)).to(device)
    # Create time tensor
    # T = 500
    time = mdlm.sample_time(5, device=device)  # * 0 + T
    # Create a mock model that outputs logits
    logits = torch.zeros((5, 10, 10, num_classes), device=device)
    # Set the logits to a large value (e.g., 1000) for the correct discrete choices
    logits[:, :, :, :] = -1000  # initialize with a low value
    # Set the logits to 1000 for the correct discrete choices
    logits = logits.scatter(3, data.unsqueeze(-1), 1)
    # Sample noise
    noise = mdlm.sample_prior(data.shape, device=device)
    # Create model output and xt
    model_out = logits  # torch.softmax(logits, dim=-1)
    xt = data.clone()
    xt[:, 0, 0] = noise[:, 0, 0]
    time = time * 0 + 40 / 100
    next_xt = mdlm.step(model_out, time, xt, dt=1 / 100)
    next_xt = mdlm.step_argmax(model_out)
    # Assert shapes
    assert next_xt.shape == data.shape
    model_out_onehot = torch.nn.functional.one_hot(
        model_out.argmax(-1), num_classes=num_classes
    ).float()  # (B, H, W, num_classes)
    nll = -torch.sum(torch.log(model_out_onehot.view(-1, num_classes) + 1e-8).gather(1, data.view(-1, 1)).squeeze(1))
    assert nll < 1e-10
    loss = mdlm.loss(
        logits.reshape(logits.shape[0], -1, logits.shape[3]),
        data.reshape(data.shape[0], -1),
        xt.data.reshape(data.shape[0], -1),
        time,
    )
    assert loss.mean() == 0
