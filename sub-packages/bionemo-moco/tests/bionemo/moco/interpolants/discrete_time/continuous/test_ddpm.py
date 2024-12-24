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
from bionemo.moco.interpolants.discrete_time.continuous.ddpm import DDPM
from bionemo.moco.schedules.noise.discrete_noise_schedules import DiscreteCosineNoiseSchedule


@pytest.fixture
def ddpm():
    time_distribution = UniformTimeDistribution(discrete_time=True, nsteps=1000)
    prior = GaussianPrior(center=False)
    noise_schedule = DiscreteCosineNoiseSchedule(nsteps=1000)
    ddpm = DDPM(time_distribution, prior, noise_schedule)
    return ddpm


@pytest.fixture
def ddpm_centered():
    time_distribution = UniformTimeDistribution(discrete_time=True, nsteps=1000)
    prior = GaussianPrior(center=True)
    noise_schedule = DiscreteCosineNoiseSchedule(nsteps=1000)
    ddpm = DDPM(time_distribution, prior, noise_schedule)
    return ddpm


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fixture", ["ddpm", "ddpm_centered"])
def test_ddpm_interpolate(request, fixture, device):
    # Create an indices tensor
    ddpm = request.getfixturevalue(fixture)
    assert ddpm is not None
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    ddpm = ddpm.to_device(device)
    indices = torch.arange(3).repeat(32, 10)

    # Create a tensor of shape 32 x 10 x 3 where each element is a 3-dimensional one-hot vector
    data = F.one_hot(indices, 3).float().to(device)
    time = ddpm.sample_time(32)
    noise = ddpm.sample_prior(data.shape)
    xt = ddpm.interpolate(data, time, noise)
    assert xt.shape == data.shape

    data_time = torch.ones_like(time).to(device) * 0
    xt = ddpm.interpolate(data, data_time, noise)
    error = (xt - data) ** 2
    assert error.mean() <= 2e-3

    data_time = torch.ones_like(time).to(device) * 999
    xt = ddpm.interpolate(data, data_time, noise)
    error = (xt - noise) ** 2
    assert error.mean() < 1e-7


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ddpm_step(ddpm, device):
    # Create an indices tensor
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    ddpm = ddpm.to_device(device)
    indices = torch.arange(3).repeat(32, 10)
    data = F.one_hot(indices, 3).float().to(device)  # shape = [32, 30, 3]
    T = 1
    time = ddpm.sample_time(32, device=device) * 0 + T
    noise = ddpm.sample_prior(data.shape, device=device)
    model_out = 0.9999 * data + 0.0001 * noise
    xt = 0.99 * data + 0.01 * noise
    next_xt = ddpm.step(model_out, time, xt)
    assert next_xt.shape == data.shape
    error = (model_out - data) ** 2
    assert error.mean() < 1e-6
    error = (xt - data) ** 2
    assert error.mean() < 1e-3
    error = (next_xt - data) ** 2
    assert error.mean() < 1e-4
    T = 100
    time = ddpm.sample_time(32, device=device) * 0 + T
    noise = ddpm.sample_prior(data.shape, device=device)
    model_out = 0.99 * data + 0.01 * noise
    xt = 0.9 * data + 0.1 * noise
    next_xt = ddpm.step(model_out, time, xt)
    assert next_xt.shape == data.shape
    error = (model_out - data) ** 2
    assert error.mean() < 1e-3
    error = (xt - data) ** 2
    assert error.mean() < 1e-1
    error = (next_xt - data) ** 2
    assert error.mean() < 1e-1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ddpm_step_masked(ddpm, device):
    # Create an indices tensor
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    ddpm = ddpm.to_device(device)
    indices = torch.arange(3).repeat(32, 10)
    data = F.one_hot(indices, 3).float().to(device)
    T = 1
    time = ddpm.sample_time(32, device=device) * 0 + T
    noise = ddpm.sample_prior(data.shape, device=device)
    model_out = 0.9999 * data + 0.0001 * noise
    xt = 0.99 * data + 0.01 * noise

    # Create a mask to mask out the last 3-4 elements
    mask = torch.ones(32, 30, dtype=torch.bool).to(device)
    mask[:, -4:] = False
    xt = xt * mask.unsqueeze(-1)
    data = data * mask.unsqueeze(-1)
    model_out = model_out * mask.unsqueeze(-1)
    # import ipdb; ipdb.set_trace()
    next_xt = ddpm.step(model_out, time, xt, mask=mask)

    # Check that the masked elements are unchanged
    assert torch.allclose(next_xt[:, -4:, :], xt[:, -4:, :])

    # Check the shape of the output
    assert next_xt.shape == data.shape

    error = (model_out - data) ** 2
    assert error.mean() < 1e-6
    error = (xt - data) ** 2
    assert error.mean() < 1e-3
    error = (next_xt - data) ** 2
    assert error.mean() < 1e-4


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ddpm_centered_step(ddpm_centered, device):
    # Create an indices tensor
    ddpm = ddpm_centered
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    ddpm = ddpm.to_device(device)
    indices = torch.arange(3).repeat(32, 10)
    data = F.one_hot(indices, 3).float().to(device)  # shape = [32, 30, 3]
    data = ddpm.clean_mask_center(data, center=True)
    T = 1
    time = ddpm.sample_time(32, device=device) * 0 + T
    noise = ddpm.sample_prior(data.shape, device=device)
    model_out = 0.9999 * data + 0.0001 * noise
    xt = 0.99 * data + 0.01 * noise
    next_xt = ddpm.step(model_out, time, xt, center=True)
    assert next_xt.shape == data.shape
    error = (model_out - data) ** 2
    assert error.mean() < 1e-6
    error = (xt - data) ** 2
    assert error.mean() < 1e-3
    error = (next_xt - data) ** 2
    assert error.mean() < 1e-4
    T = 100
    time = ddpm.sample_time(32, device=device) * 0 + T
    noise = ddpm.sample_prior(data.shape, device=device)
    model_out = 0.99 * data + 0.01 * noise
    xt = 0.9 * data + 0.1 * noise
    next_xt = ddpm.step(model_out, time, xt, center=True)
    assert next_xt.shape == data.shape
    error = (model_out - data) ** 2
    assert error.mean() < 1e-3
    error = (xt - data) ** 2
    assert error.mean() < 1e-1
    error = (next_xt - data) ** 2
    assert error.mean() < 1e-1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ddim_step(ddpm, device):
    # Create an indices tensor
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    ddpm = ddpm.to_device(device)
    indices = torch.arange(3).repeat(32, 10)
    data = F.one_hot(indices, 3).float().to(device)  # shape = [32, 30, 3]
    T = 1
    time = ddpm.sample_time(32, device=device) * 0 + T
    noise = ddpm.sample_prior(data.shape, device=device)
    model_out = 0.9999 * data + 0.0001 * noise
    xt = 0.99 * data + 0.01 * noise
    next_xt = ddpm.general_step("step_ddim", {"model_out": model_out, "t": time, "xt": xt})
    assert next_xt.shape == data.shape
    error = (model_out - data) ** 2
    assert error.mean() < 1e-6
    error = (xt - data) ** 2
    assert error.mean() < 1e-3
    error = (next_xt - data) ** 2
    assert error.mean() < 1e-4
    T = 100
    time = ddpm.sample_time(32, device=device) * 0 + T
    noise = ddpm.sample_prior(data.shape, device=device)
    model_out = 0.99 * data + 0.01 * noise
    xt = 0.9 * data + 0.1 * noise
    next_xt = ddpm.step_ddim(model_out, time, xt)
    assert next_xt.shape == data.shape
    error = (model_out - data) ** 2
    assert error.mean() < 1e-3
    error = (xt - data) ** 2
    assert error.mean() < 1e-1
    error = (next_xt - data) ** 2
    assert error.mean() < 1e-1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("weight_type", ["ones", "data_to_noise"])
def test_ddpm_loss(ddpm, device, weight_type):
    # Check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    ddpm = ddpm.to_device(device)
    indices = torch.arange(3).repeat(32, 10)
    data = F.one_hot(indices, 3).float().to(device)  # shape = [32, 30, 3]
    T = 1
    time = ddpm.sample_time(32, device=device) * 0 + T
    noise = ddpm.sample_prior(data.shape, device=device)
    model_out = 0.9999 * data + 0.0001 * noise
    mask = torch.ones(32, 30, dtype=torch.bool).to(device)
    mask[:, -4:] = False
    data = data * mask.unsqueeze(-1)
    model_out = model_out * mask.unsqueeze(-1)

    # Calculate the loss
    loss = ddpm.loss(model_out, data, time, mask=mask, weight_type=weight_type)

    # Check the shape of the loss
    assert loss.shape == (32,)
    assert loss.mean() < 1e-3


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ddpm_2d_step(ddpm, device):
    # Create an indices tensor
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    ddpm = ddpm.to_device(device)
    data = torch.rand((32, 10, 10, 3)).to(device)  # shape = [32, 10, 10, 3]
    T = 1
    time = ddpm.sample_time(32, device=device) * 0 + T
    noise = ddpm.sample_prior(data.shape, device=device)
    model_out = 0.9999 * data + 0.0001 * noise
    _ = ddpm.interpolate(data, time, noise)
    xt = 0.99 * data + 0.01 * noise
    next_xt = ddpm.step(model_out, time, xt)
    assert next_xt.shape == data.shape
    error = (model_out - data) ** 2
    assert error.mean() < 1e-6
    error = (xt - data) ** 2
    assert error.mean() < 1e-3
    error = (next_xt - data) ** 2
    assert error.mean() < 1e-4
    T = 100
    time = ddpm.sample_time(32, device=device) * 0 + T
    noise = ddpm.sample_prior(data.shape, device=device)
    model_out = 0.99 * data + 0.01 * noise
    xt = 0.9 * data + 0.1 * noise
    next_xt = ddpm.step(model_out, time, xt)
    assert next_xt.shape == data.shape
    error = (model_out - data) ** 2
    assert error.mean() < 1e-3
    error = (xt - data) ** 2
    assert error.mean() < 1e-1
    error = (next_xt - data) ** 2
    assert error.mean() < 1e-1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("ndim", [2, 3, 4, 5])
def test_ddpm_ndim_step(ddpm, device, ndim):
    # Create an indices tensor
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    ddpm = ddpm.to_device(device)
    shape = [10] * ndim
    batch_size = 32
    data = torch.rand((batch_size, *shape, 3)).to(device)
    T = 1
    time = ddpm.sample_time(batch_size, device=device) * 0 + T
    noise = ddpm.sample_prior(data.shape, device=device)
    model_out = 0.9999 * data + 0.0001 * noise
    _ = ddpm.interpolate(data, time, noise)
    xt = 0.99 * data + 0.01 * noise
    next_xt = ddpm.step(model_out, time, xt)
    assert next_xt.shape == data.shape
    error = (model_out - data) ** 2
    assert error.mean() < 1e-6
    error = (xt - data) ** 2
    assert error.mean() < 1e-3
    error = (next_xt - data) ** 2
    assert error.mean() < 1e-4
    T = 100
    time = ddpm.sample_time(batch_size, device=device) * 0 + T
    noise = ddpm.sample_prior(data.shape, device=device)
    model_out = 0.99 * data + 0.01 * noise
    xt = 0.9 * data + 0.1 * noise
    next_xt = ddpm.step(model_out, time, xt)
    assert next_xt.shape == data.shape
    error = (model_out - data) ** 2
    assert error.mean() < 1e-3
    error = (xt - data) ** 2
    assert error.mean() < 1e-1
    error = (next_xt - data) ** 2
    assert error.mean() < 1e-1


@pytest.mark.parametrize("devices", [("cpu", "cuda"), ("cuda", "cpu")])
def test_ddpm_to_device_multiple(ddpm, devices):
    # Check if CUDA is available
    if "cuda" in devices and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Move the DDPM instance to the first device
    ddpm.to_device(devices[0])

    # Check that all internal tensors have been moved to the first device
    for attr_name in dir(ddpm):
        if attr_name.startswith("_") and isinstance(getattr(ddpm, attr_name), torch.Tensor):
            assert getattr(ddpm, attr_name).device.type == devices[0]

    # Move the DDPM instance to the second device
    ddpm.to_device(devices[1])

    # Check that all internal tensors have been moved to the second device
    for attr_name in dir(ddpm):
        if attr_name.startswith("_") and isinstance(getattr(ddpm, attr_name), torch.Tensor):
            assert getattr(ddpm, attr_name).device.type == devices[1]

    # Check that the device attribute has been updated
    assert ddpm.device == devices[1]


def test_set_loss_weight_fn(ddpm):
    # Define a test function to set as the loss_weight attribute
    def test_loss_weight_fn(raw_loss, t, weight_type):
        return raw_loss * t * weight_type

    # Set the test function as the loss_weight attribute
    ddpm.set_loss_weight_fn(test_loss_weight_fn)

    # Verify that the loss_weight attribute is set to the test function
    assert ddpm.loss_weight is test_loss_weight_fn

    # Test that the function is callable with the correct arguments
    raw_loss = 1.0
    t = 2.0
    weight_type = 3.0
    expected_output = raw_loss * t * weight_type
    assert ddpm.loss_weight(raw_loss, t, weight_type) == expected_output
