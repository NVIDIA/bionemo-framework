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


# tests/moco/interpolants/discrete_time/continuous/test_vdm.py

import pytest
import torch
import torch.nn.functional as F

from bionemo.moco.distributions.prior.continuous.gaussian import GaussianPrior
from bionemo.moco.distributions.time.uniform import UniformTimeDistribution
from bionemo.moco.interpolants.continuous_time.continuous.vdm import VDM
from bionemo.moco.schedules.noise.continuous_snr_transforms import CosineSNRTransform


@pytest.fixture
def vdm():
    time_distribution = UniformTimeDistribution(discrete_time=False)
    prior = GaussianPrior(center=False)
    noise_schedule = CosineSNRTransform()
    vdm = VDM(time_distribution, prior, noise_schedule)
    return vdm


@pytest.fixture
def vdm_centered():
    time_distribution = UniformTimeDistribution(discrete_time=False)
    prior = GaussianPrior(center=True)
    noise_schedule = CosineSNRTransform()
    vdm = VDM(time_distribution, prior, noise_schedule)
    return vdm


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fixture", ["vdm", "vdm_centered"])
def test_vdm_interpolate(request, fixture, device):
    vdm = request.getfixturevalue(fixture)
    assert vdm is not None
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    vdm = vdm.to_device(device)
    indices = torch.arange(3).repeat(32, 10)
    data = F.one_hot(indices, 3).float().to(device)
    time = vdm.sample_time(32)
    noise = vdm.sample_prior(data.shape)
    xt = vdm.interpolate(data, time, noise)
    assert xt.shape == data.shape

    data_time = torch.ones_like(time).to(device) * 0
    xt = vdm.interpolate(data, data_time, noise)
    error = (xt - data) ** 2
    assert error.mean() <= 2e-3
    data_time = torch.ones_like(time).to(device) * (1 - 1e-7)
    xt = vdm.interpolate(data, data_time, noise)
    error = (xt - noise) ** 2
    assert error.mean() < 1e-7


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_vdm_step(vdm, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    vdm = vdm.to_device(device)
    indices = torch.arange(3).repeat(32, 10)
    data = F.one_hot(indices, 3).float().to(device)
    T = 1 / 1000
    time = vdm.sample_time(32, device=device) * 0 + T
    dt = torch.ones_like(time) * 1 / 1000
    noise = vdm.sample_prior(data.shape, device=device)
    model_out = 0.9999 * data + 0.0001 * noise
    xt = 0.99 * data + 0.01 * noise
    next_xt = vdm.step(model_out, time, xt, dt)
    assert next_xt.shape == data.shape
    error = (model_out - data) ** 2
    assert error.mean() < 1e-6
    error = (xt - data) ** 2
    assert error.mean() < 1e-3
    error = (next_xt - data) ** 2
    assert torch.allclose(error.mean(), torch.tensor(0.0001), atol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_vdm_centered_step(vdm_centered, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    vdm = vdm_centered.to_device(device)
    indices = torch.arange(3).repeat(32, 10)
    data = F.one_hot(indices, 3).float().to(device)
    data = vdm.clean_mask_center(data, center=True)
    T = 1 / 1000
    time = vdm.sample_time(32, device=device) * 0 + T
    dt = torch.ones_like(time) * 1 / 1000
    noise = vdm.sample_prior(data.shape, device=device)
    model_out = 0.9999 * data + 0.0001 * noise
    xt = 0.99 * data + 0.01 * noise
    next_xt = vdm.step(model_out, time, xt, dt, center=True)
    assert next_xt.shape == data.shape
    error = (model_out - data) ** 2
    assert error.mean() < 1e-6
    error = (xt - data) ** 2
    assert error.mean() < 1e-3
    error = (next_xt - data) ** 2
    assert torch.allclose(error.mean(), torch.tensor(0.0001), atol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("weight_type", ["ones", "data_to_noise"])
def test_vdm_loss(vdm, device, weight_type):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    vdm = vdm.to_device(device)
    indices = torch.arange(3).repeat(32, 10)
    data = F.one_hot(indices, 3).float().to(device)
    T = 1
    time = vdm.sample_time(32, device=device) * 0 + T
    noise = vdm.sample_prior(data.shape, device=device)
    model_out = 0.9999 * data + 0.0001 * noise
    mask = torch.ones(32, 30, dtype=torch.bool).to(device)
    mask[:, -4:] = False
    data = data * mask.unsqueeze(-1)
    model_out = model_out * mask.unsqueeze(-1)

    loss = vdm.loss(model_out, data, time, mask=mask, weight_type=weight_type)
    assert loss.shape == (32,)
    assert loss.mean() < 1e-3


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_vdm_2d_step(vdm, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    vdm = vdm.to_device(device)
    data = torch.rand((32, 10, 10, 3)).to(device)
    T = 1 / 1000
    time = vdm.sample_time(32, device=device) * 0 + T
    dt = torch.ones_like(time) * 1 / 1000
    noise = vdm.sample_prior(data.shape, device=device)
    model_out = 0.9999 * data + 0.0001 * noise
    _ = vdm.interpolate(data, time, noise)
    xt = 0.99 * data + 0.01 * noise
    next_xt = vdm.step(model_out, time, xt, dt)
    assert next_xt.shape == data.shape
    error = (model_out - data) ** 2
    assert error.mean() < 1e-6
    error = (xt - data) ** 2
    assert error.mean() < 1e-3
    error = (next_xt - data) ** 2
    assert torch.allclose(error.mean(), torch.tensor(0.0001), atol=1e-4)
    T = 100 / 1000
    time = vdm.sample_time(32, device=device) * 0 + T
    noise = vdm.sample_prior(data.shape, device=device)
    model_out = 0.99 * data + 0.01 * noise
    xt = 0.9 * data + 0.1 * noise
    next_xt = vdm.step(model_out, time, xt, dt)
    assert next_xt.shape == data.shape
    error = (model_out - data) ** 2
    assert error.mean() < 1e-3
    error = (xt - data) ** 2
    assert error.mean() < 1e-1
    error = (next_xt - data) ** 2
    assert error.mean() < 1e-1


@pytest.mark.parametrize("devices", [("cpu", "cuda"), ("cuda", "cpu")])
def test_vdm_to_device_multiple(vdm, devices):
    if "cuda" in devices and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    vdm.to_device(devices[0])

    for attr_name in dir(vdm):
        if attr_name.startswith("_") and isinstance(getattr(vdm, attr_name), torch.Tensor):
            assert getattr(vdm, attr_name).device.type == devices[0]

    vdm.to_device(devices[1])

    for attr_name in dir(vdm):
        if attr_name.startswith("_") and isinstance(getattr(vdm, attr_name), torch.Tensor):
            assert getattr(vdm, attr_name).device.type == devices[1]

    assert vdm.device == devices[1]
