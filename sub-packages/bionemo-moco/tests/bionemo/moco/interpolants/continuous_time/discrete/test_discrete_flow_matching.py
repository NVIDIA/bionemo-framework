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
from bionemo.moco.distributions.prior.discrete.uniform import DiscreteUniformPrior
from bionemo.moco.distributions.time.uniform import UniformTimeDistribution
from bionemo.moco.interpolants.continuous_time.discrete.discrete_flow_matching import DiscreteFlowMatcher


@pytest.fixture
def dfm_mask():
    time_distribution = UniformTimeDistribution(discrete_time=False)
    prior = DiscreteMaskedPrior(num_classes=20)  # 19 data classes 1 mask class
    dfm = DiscreteFlowMatcher(time_distribution, prior)
    return dfm


@pytest.fixture
def dfm_mask_non_inclusive():
    time_distribution = UniformTimeDistribution(discrete_time=False)
    prior = DiscreteMaskedPrior(num_classes=20, inclusive=False)
    dfm = DiscreteFlowMatcher(time_distribution, prior)
    return dfm


@pytest.fixture
def dfm_uniform():
    time_distribution = UniformTimeDistribution(discrete_time=False)
    prior = DiscreteUniformPrior(num_classes=20)
    dfm = DiscreteFlowMatcher(time_distribution, prior)
    return dfm


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fixture", ["dfm_mask", "dfm_mask_non_inclusive", "dfm_uniform"])
def test_dfm_interpolate(request, fixture, device):
    # Create an indices tensor
    dfm = request.getfixturevalue(fixture)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    batch_size = 5
    num_residues = 10
    dfm = dfm.to_device(device)
    data = torch.randint(0, 19, (batch_size, num_residues)).to(device)
    t = dfm.sample_time(batch_size)
    noise = dfm.sample_prior(data.shape)
    dfm.to_device(device)
    result = dfm.interpolate(data, t, noise)
    assert result.shape == (batch_size, num_residues)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fixture", ["dfm_mask", "dfm_mask_non_inclusive", "dfm_uniform"])
def test_dfm_step(request, fixture, device):
    # Create an indices tensor
    dfm = request.getfixturevalue(fixture)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    batch_size = 5
    num_residues = 10
    num_classes = 20
    dfm = dfm.to_device(device)
    data = torch.randint(0, 19, (batch_size, num_residues)).to(device)
    t = dfm.sample_time(batch_size)
    logits = torch.zeros((batch_size, num_residues, num_classes), device=device)
    # Set the logits to a large value (e.g., 1000) for the correct discrete choices
    logits[:, :, :] = -1000  # initialize with a low value
    # Set the logits to 1000 for the correct discrete choices
    logits = logits.scatter(2, data.unsqueeze(-1), 1)

    t = dfm.sample_time(batch_size)
    noise = dfm.sample_prior(data.shape)
    dfm.to_device(device)
    xt = dfm.interpolate(data, t, noise)
    if isinstance(dfm.prior_distribution, DiscreteMaskedPrior) and dfm.prior_distribution.mask_dim == 20:  #! exclusive
        logits = dfm.prior_distribution.pad_sample(logits)
        next_xt = dfm.step(logits, 0 * t + 0.5, xt, dt=1 / 100)
        assert next_xt.shape == xt.shape
        next_xt = dfm.step_argmax(logits)
        assert next_xt.shape == xt.shape
        next_xt = dfm.step_simple_sample(logits)
    else:
        next_xt = dfm.step(logits, 0 * t + 0.5, xt, dt=1 / 100)
        assert next_xt.shape == xt.shape
        next_xt = dfm.step_argmax(logits)
        assert next_xt.shape == xt.shape
        next_xt = dfm.step_simple_sample(logits)
    assert next_xt.shape == xt.shape


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fixture", ["dfm_mask", "dfm_mask_non_inclusive"])
def test_dfm_loss(request, fixture, device):
    # Create an indices tensor
    dfm = request.getfixturevalue(fixture)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    batch_size = 5
    num_residues = 10
    num_classes = 20
    dfm = dfm.to_device(device)
    data = torch.randint(0, 19, (batch_size, num_residues)).to(device)
    t = dfm.sample_time(batch_size)
    logits = torch.zeros((batch_size, num_residues, num_classes), device=device)
    # Set the logits to a large value (e.g., 1000) for the correct discrete choices
    logits[:, :, :] = -1000  # initialize with a low value
    # Set the logits to 1000 for the correct discrete choices
    logits = logits.scatter(2, data.unsqueeze(-1), 1)

    t = dfm.sample_time(batch_size)
    noise = dfm.sample_prior(data.shape)
    dfm.to_device(device)
    xt = dfm.interpolate(data, t, noise)
    loss = dfm.loss(logits, data)
    assert loss.mean() == 0
    loss = dfm.loss(logits, data, mask=dfm.prior_distribution.is_masked(xt))
    assert loss.mean() == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fixture", ["dfm_mask", "dfm_mask_non_inclusive"])
def test_dfm_step_purity(request, fixture, device):
    # Create an indices tensor
    dfm = request.getfixturevalue(fixture)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    batch_size = 5
    num_residues = 10
    num_classes = 20
    dfm = dfm.to_device(device)
    data = torch.randint(0, 19, (batch_size, num_residues)).to(device)
    t = dfm.sample_time(batch_size)
    logits = torch.zeros((batch_size, num_residues, num_classes), device=device)
    # Set the logits to a large value (e.g., 1000) for the correct discrete choices
    logits[:, :, :] = -1000  # initialize with a low value
    # Set the logits to 1000 for the correct discrete choices
    logits = logits.scatter(2, data.unsqueeze(-1), 1)

    t = dfm.sample_time(batch_size)
    noise = dfm.sample_prior(data.shape)
    dfm.to_device(device)
    xt = dfm.interpolate(data, t, noise)
    if isinstance(dfm.prior_distribution, DiscreteMaskedPrior) and dfm.prior_distribution.mask_dim == 20:  #! exclusive
        logits = dfm.prior_distribution.pad_sample(logits)
        next_xt = dfm.step_purity(logits, 0 * t + 0.5, xt, dt=1 / 100)
    else:
        next_xt = dfm.step_purity(logits, 0 * t + 0.5, xt, dt=1 / 100)
    assert next_xt.shape == xt.shape


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fixture", ["dfm_mask", "dfm_mask_non_inclusive", "dfm_uniform"])
def test_dfm_interpolate_square(request, fixture, device):
    # Create an indices tensor
    dfm = request.getfixturevalue(fixture)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    batch_size = 5
    num_residues = 10
    dfm = dfm.to_device(device)
    data = torch.randint(0, 19, (batch_size, num_residues, num_residues)).to(device)
    t = dfm.sample_time(batch_size)
    noise = dfm.sample_prior(data.shape)
    dfm.to_device(device)
    result = dfm.interpolate(data, t, noise)
    assert result.shape == (batch_size, num_residues, num_residues)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fixture", ["dfm_mask", "dfm_mask_non_inclusive", "dfm_uniform"])
def test_dfm_step_square(request, fixture, device):
    # Create an indices tensor
    dfm = request.getfixturevalue(fixture)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    batch_size = 5
    num_residues = 10
    num_classes = 20
    dfm = dfm.to_device(device)
    data = torch.randint(0, 19, (batch_size, num_residues, num_residues)).to(device)
    t = dfm.sample_time(batch_size)
    logits = torch.zeros((batch_size, num_residues, num_residues, num_classes), device=device)
    # Set the logits to a large value (e.g., 1000) for the correct discrete choices
    logits[:, :, :, :] = -1000  # initialize with a low value
    # Set the logits to 1000 for the correct discrete choices
    logits = logits.scatter(3, data.unsqueeze(-1), 1)

    t = dfm.sample_time(batch_size)
    noise = dfm.sample_prior(data.shape)
    dfm.to_device(device)
    xt = dfm.interpolate(data, t, noise)
    if isinstance(dfm.prior_distribution, DiscreteMaskedPrior) and dfm.prior_distribution.mask_dim == 20:  #! exclusive
        logits = dfm.prior_distribution.pad_sample(logits)
        next_xt = dfm.step(logits, 0 * t + 0.5, xt, dt=1 / 100)
    else:
        next_xt = dfm.step(logits, 0 * t + 0.5, xt, dt=1 / 100)
    assert next_xt.shape == xt.shape


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fixture", ["dfm_mask", "dfm_mask_non_inclusive"])
def test_dfm_loss_square(request, fixture, device):
    # Create an indices tensor
    dfm = request.getfixturevalue(fixture)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    batch_size = 5
    num_residues = 10
    num_classes = 20
    dfm = dfm.to_device(device)
    data = torch.randint(0, 19, (batch_size, num_residues, num_residues)).to(device)
    t = dfm.sample_time(batch_size)
    logits = torch.zeros((batch_size, num_residues, num_residues, num_classes), device=device)
    # Set the logits to a large value (e.g., 1000) for the correct discrete choices
    logits[:, :, :, :] = -1000  # initialize with a low value
    # Set the logits to 1000 for the correct discrete choices
    logits = logits.scatter(3, data.unsqueeze(-1), 1)

    t = dfm.sample_time(batch_size)
    noise = dfm.sample_prior(data.shape)
    dfm.to_device(device)
    xt = dfm.interpolate(data, t, noise)
    assert xt.shape == data.shape
    loss = dfm.loss(logits.reshape(logits.shape[0], -1, logits.shape[3]), data.reshape(data.shape[0], -1))
    assert loss.mean() == 0
