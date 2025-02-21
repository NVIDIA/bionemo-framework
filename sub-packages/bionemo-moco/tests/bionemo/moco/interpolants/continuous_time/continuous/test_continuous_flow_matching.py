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
import torch.nn.functional as F

from bionemo.moco.distributions.prior.continuous.gaussian import GaussianPrior
from bionemo.moco.distributions.time.uniform import UniformTimeDistribution
from bionemo.moco.interpolants.base_interpolant import PredictionType
from bionemo.moco.interpolants.continuous_time.continuous.continuous_flow_matching import ContinuousFlowMatcher


@pytest.fixture
def flow_matcher():
    time_distribution = UniformTimeDistribution(discrete_time=False)
    prior = GaussianPrior(center=False)
    flow_matcher = ContinuousFlowMatcher(
        time_distribution=time_distribution, prior_distribution=prior, prediction_type="vector_field"
    )
    return flow_matcher


@pytest.fixture
def data_matcher():
    time_distribution = UniformTimeDistribution(discrete_time=False)
    prior = GaussianPrior(center=False)
    flow_matcher = ContinuousFlowMatcher(
        time_distribution=time_distribution, prior_distribution=prior, prediction_type=PredictionType.DATA
    )
    return flow_matcher


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fixture", ["flow_matcher"])
def test_cfm_interpolate(request, fixture, device):
    # Create an indices tensor
    flow_matcher = request.getfixturevalue(fixture)
    assert flow_matcher is not None
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    flow_matcher = flow_matcher.to_device(device)
    indices = torch.arange(3).repeat(32, 10)

    # Create a tensor of shape 32 x 10 x 3 where each element is a 3-dimensional one-hot vector
    data = F.one_hot(indices, 3).float().to(device)
    time = flow_matcher.sample_time(32)
    noise = flow_matcher.sample_prior(data.shape)
    xt = flow_matcher.interpolate(data, time, noise)
    assert xt.shape == data.shape

    # When time is 0, the output should be the noise
    data_time = torch.ones_like(time).to(device) * 0
    xt = flow_matcher.interpolate(data, data_time, noise)
    error = (xt - noise) ** 2
    assert torch.all(error < 1e-7)

    # When time is close to 1, i.e. 0.999, the output should be the data
    data_time = torch.ones_like(time).to(device) * 0.999
    xt = flow_matcher.interpolate(data, data_time, noise)
    error = (xt - (noise + (data - noise) * 0.999)) ** 2
    assert torch.all(error < 1e-7)

    # When time is 0.5, if the data is the reflection of the noise, the output should be zeros
    data_time = torch.ones_like(time).to(device) * 0.5
    new_data = torch.clone(noise) * -1
    xt = flow_matcher.interpolate(new_data, data_time, noise)
    error = (xt - torch.zeros_like(xt)) ** 2
    assert torch.all(error < 1e-7)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fixture", ["flow_matcher"])
def test_cfm_interpolate_newshape(request, fixture, device):
    # Create an indices tensor
    flow_matcher = request.getfixturevalue(fixture)
    assert flow_matcher is not None
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    flow_matcher = flow_matcher.to_device(device)
    data = torch.rand((32, 5, 14, 3)).to(device)
    time = flow_matcher.sample_time(32)
    noise = flow_matcher.sample_prior(data.shape)
    xt = flow_matcher.interpolate(data, time, noise)
    assert xt.shape == data.shape

    # When time is 0, the output should be the noise
    data_time = torch.ones_like(time).to(device) * 0
    xt = flow_matcher.interpolate(data, data_time, noise)
    error = (xt - noise) ** 2
    assert torch.all(error < 1e-7)

    # When time is close to 1, i.e. 0.999, the output should be the data
    data_time = torch.ones_like(time).to(device) * 0.999
    xt = flow_matcher.interpolate(data, data_time, noise)
    error = (xt - (noise + (data - noise) * 0.999)) ** 2
    assert torch.all(error < 1e-7)

    # When time is 0.5, if the data is the reflection of the noise, the output should be zeros
    data_time = torch.ones_like(time).to(device) * 0.5
    new_data = torch.clone(noise) * -1
    xt = flow_matcher.interpolate(new_data, data_time, noise)
    error = (xt - torch.zeros_like(xt)) ** 2
    assert torch.all(error < 1e-7)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fixture", ["flow_matcher"])
def test_cfm_interpolate_newshape_preserve(request, fixture, device):
    # Create an indices tensor
    flow_matcher = request.getfixturevalue(fixture)
    assert flow_matcher is not None
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    flow_matcher = flow_matcher.to_device(device)
    data = torch.rand((32, 14, 5, 3)).to(device)
    b, a, n, d = data.shape
    time = flow_matcher.sample_time(32)
    noise = flow_matcher.sample_prior((b, 1, n, d))
    xt = flow_matcher.interpolate(data, time, noise)
    assert xt.shape == data.shape

    # When time is 0, the output should be the noise
    data_time = torch.ones_like(time).to(device) * 0
    xt = flow_matcher.interpolate(data, data_time, noise)
    error = (xt - noise) ** 2
    assert torch.all(error < 1e-7)

    # When time is close to 1, i.e. 0.999, the output should be the data
    data_time = torch.ones_like(time).to(device) * 0.999
    xt = flow_matcher.interpolate(data, data_time, noise)
    error = (xt - (noise + (data - noise) * 0.999)) ** 2
    assert torch.all(error < 1e-7)

    # When time is 0.5, if the data is the reflection of the noise, the output should be zeros
    data_time = torch.ones_like(time).to(device) * 0.5
    new_data = torch.clone(noise) * -1
    xt = flow_matcher.interpolate(new_data, data_time, noise)
    error = (xt - torch.zeros_like(xt)) ** 2
    assert torch.all(error < 1e-7)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cfm_step(flow_matcher, device):
    # Create an indices tensor
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    flow_matcher = flow_matcher.to_device(device)
    indices = torch.arange(3).repeat(32, 10)
    data = F.one_hot(indices, 3).float().to(device)  # shape = [32, 30, 3]
    time = flow_matcher.sample_time(32, device=device)
    noise = flow_matcher.sample_prior(data.shape, device=device)

    # Check if the last step works
    T = 0.999
    dt = time * 0 + 0.001
    model_out = data - noise
    xt = noise + (data - noise) * T
    next_xt = flow_matcher.step(model_out, xt, dt)
    assert next_xt.shape == data.shape
    error = (next_xt - data) ** 2
    assert torch.all(error < 1e-7)

    # When data is the reflection of the noise, check if sign flips after passing t=0.5
    data = noise * -1
    T = 0.499
    dt = time * 0 + 0.002
    model_out = data - noise
    xt = noise + (data - noise) * T
    assert torch.all(torch.sign(xt) == torch.sign(noise))
    next_xt = flow_matcher.step(model_out, xt, dt)
    next_xt_gt = noise + (data - noise) * 0.501
    assert torch.all(torch.sign(next_xt) == torch.sign(data))
    error = (next_xt - next_xt_gt) ** 2
    assert torch.all(error < 1e-7)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cfm_loss(flow_matcher, device):
    # Check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    flow_matcher = flow_matcher.to_device(device)
    indices = torch.arange(3).repeat(32, 10)
    data = F.one_hot(indices, 3).float().to(device)  # shape = [32, 30, 3]
    noise = flow_matcher.sample_prior(data.shape, device=device)
    # Set the ground truth to be the flow, like rectified flow objective
    gt_flow = flow_matcher.calculate_target(data, noise)  # data - noise
    # Set the model output to be the flow with small noise perturbation
    model_out = (data - noise) + torch.randn_like(data) * 0.001
    # Create a mask to mask the last 4 elements of the sequence
    mask = torch.ones(32, 30, dtype=torch.bool).to(device)
    mask[:, -4:] = False
    # Mask out the model output to test if masking in loss works
    model_out = model_out * mask.unsqueeze(-1)

    # Calculate the loss, only model_out is masked but not gt_flow
    loss = flow_matcher.loss(model_out, gt_flow, mask=None, target_type="velocity")
    # Check the shape of the loss
    assert loss.shape == (32,)
    # When mask input to flow_matcher.loss is None, the loss should be large because gt is not masked
    assert loss.mean() > 0.1

    # Calculate the loss with input argument mask as the mask
    loss = flow_matcher.loss(model_out, gt_flow, mask=mask, target_type=PredictionType.VELOCITY)
    # When mask input to flow_matcher.loss is None, the loss should be small
    assert loss.mean() < 1e-4
    # Calculate the loss with input argument mask as the mask
    time = flow_matcher.sample_time(32)
    xt = flow_matcher.interpolate(data, time, noise)
    loss = flow_matcher.loss(
        model_out, data * mask.unsqueeze(-1), time, xt, mask=mask, target_type=PredictionType.DATA
    )
    # When mask input to flow_matcher.loss is None, the loss should be small
    assert loss.mean() < 1e-4


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fixture", ["data_matcher"])
def test_cfm_interpolate_data(request, fixture, device):
    # Create an indices tensor
    torch.manual_seed(42)
    flow_matcher = request.getfixturevalue(fixture)
    assert flow_matcher is not None
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    flow_matcher = flow_matcher.to_device(device)
    indices = torch.arange(3).repeat(32, 10)

    # Create a tensor of shape 32 x 10 x 3 where each element is a 3-dimensional one-hot vector
    data = F.one_hot(indices, 3).float().to(device)
    time = flow_matcher.sample_time(32)
    noise = flow_matcher.sample_prior(data.shape)
    xt = flow_matcher.interpolate(data, time, noise)
    assert xt.shape == data.shape

    # When time is 0, the output should be the noise
    data_time = torch.ones_like(time).to(device) * 0
    xt = flow_matcher.interpolate(data, data_time, noise)
    error = (xt - noise) ** 2
    assert torch.all(error < 1e-7)

    # When time is close to 1, i.e. 0.999, the output should be the data
    data_time = torch.ones_like(time).to(device) * 0.999
    xt = flow_matcher.interpolate(data, data_time, noise)
    error = (xt - (noise + (data - noise) * 0.999)) ** 2
    assert torch.all(error < 1e-7)

    # When time is 0.5, if the data is the reflection of the noise, the output should be zeros
    data_time = torch.ones_like(time).to(device) * 0.5
    new_data = torch.clone(noise) * -1
    xt = flow_matcher.interpolate(new_data, data_time, noise)
    error = (xt - torch.zeros_like(xt)) ** 2
    assert torch.all(error < 1e-7)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cfm_step_data(data_matcher, device):
    # Create an indices tensor
    torch.manual_seed(42)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    flow_matcher = data_matcher.to_device(device)
    indices = torch.arange(3).repeat(32, 10)
    data = F.one_hot(indices, 3).float().to(device)  # shape = [32, 30, 3]
    time = flow_matcher.sample_time(32, device=device)
    noise = flow_matcher.sample_prior(data.shape, device=device)

    # Check if the last step works
    T = 0.999
    dt = time * 0 + 0.001
    model_out = data
    xt = noise + (data - noise) * T
    next_xt = flow_matcher.step(model_out, xt, dt, time * 0 + T)
    assert next_xt.shape == data.shape
    error = (next_xt - data) ** 2
    assert torch.all(error < 1e-7)

    # When data is the reflection of the noise, check if sign flips after passing t=0.5
    data = noise * -1
    T = 0.499
    dt = time * 0 + 0.002
    model_out = data
    xt = noise + (data - noise) * T
    assert torch.all(torch.sign(xt) == torch.sign(noise))
    next_xt = flow_matcher.step(model_out, xt, dt, time * 0 + T)
    next_xt_gt = noise + (data - noise) * 0.501
    assert torch.all(torch.sign(next_xt) == torch.sign(data))
    error = (next_xt - next_xt_gt) ** 2
    assert torch.all(error < 1e-7)

    next_xt = flow_matcher.step_score_stochastic(model_out, xt, dt, time * 0 + T)
    next_xt = flow_matcher.general_step(
        "step_score_stochastic", {"model_out": model_out, "xt": xt, "dt": dt, "t": time * 0 + T}
    )
    error = (next_xt - next_xt_gt) ** 2
    assert error.mean() < 1e-2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cfm_loss_data(data_matcher, device):
    # Check if CUDA is available
    torch.manual_seed(42)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    flow_matcher = data_matcher.to_device(device)
    indices = torch.arange(3).repeat(32, 10)
    data = F.one_hot(indices, 3).float().to(device)  # shape = [32, 30, 3]
    noise = flow_matcher.sample_prior(data.shape, device=device)
    # Set the ground truth to be the flow, like rectified flow objective
    gt_target = flow_matcher.calculate_target(data, noise)  # data - noise
    # Set the model output to be the flow with small noise perturbation
    model_out = data + torch.randn_like(data) * 0.001
    # Create a mask to mask the last 4 elements of the sequence
    mask = torch.ones(32, 30, dtype=torch.bool).to(device)
    mask[:, -4:] = False
    # Mask out the model output to test if masking in loss works
    model_out = model_out * mask.unsqueeze(-1)
    time = flow_matcher.sample_time(32)
    xt = flow_matcher.interpolate(data, time, noise)
    # Calculate the loss, only model_out is masked but not gt_flow
    loss = flow_matcher.loss(model_out, gt_target, time, xt)
    # Check the shape of the loss
    assert loss.shape == (32,)
    # When mask input to flow_matcher.loss is None, the loss should be large because gt is not masked
    assert loss.mean() > 0.1

    # Calculate the loss with input argument mask as the mask
    loss = flow_matcher.loss(model_out, gt_target * mask.unsqueeze(-1), time, xt, mask=mask)
    assert loss.mean() < 1e-2
