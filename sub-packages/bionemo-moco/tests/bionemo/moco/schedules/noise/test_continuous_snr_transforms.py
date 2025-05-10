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

from bionemo.moco.schedules.noise.continuous_snr_transforms import (
    CosineSNRTransform,
    LinearLogInterpolatedSNRTransform,
    LinearSNRTransform,
    TimeDirection,
)


@pytest.mark.parametrize("timesteps", [5, 10, 20])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("synchronize", [TimeDirection.DIFFUSION, TimeDirection.UNIFIED])
def test_cosine_snr_transform(timesteps, device, synchronize):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    t = torch.linspace(0, 1, timesteps, device=device)
    snr_transform = CosineSNRTransform()

    log_snr = snr_transform.calculate_log_snr(t, device=device, synchronize=synchronize)

    # Check if log_snr has the correct shape
    assert log_snr.shape == (timesteps,)
    # Check if log_snr is on the correct device
    assert log_snr.device.type == device


@pytest.mark.parametrize("timesteps", [5, 10, 20])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("synchronize", [TimeDirection.DIFFUSION, TimeDirection.UNIFIED])
def test_linear_snr_transform(timesteps, device, synchronize):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    t = torch.linspace(0, 1, timesteps, device=device)
    snr_transform = LinearSNRTransform()

    log_snr = snr_transform.calculate_log_snr(t, device=device, synchronize=synchronize)

    # Check if log_snr has the correct shape
    assert log_snr.shape == (timesteps,)
    # Check if log_snr is on the correct device
    assert log_snr.device.type == device


@pytest.mark.parametrize("timesteps", [5, 10, 20])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("synchronize", [TimeDirection.DIFFUSION, TimeDirection.UNIFIED])
def test_linear_log_interpolated_snr_transform(timesteps, device, synchronize):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    t = torch.linspace(0, 1, timesteps, device=device)
    snr_transform = LinearLogInterpolatedSNRTransform()

    log_snr = snr_transform.calculate_log_snr(t, device=device, synchronize=synchronize)

    # Check if log_snr has the correct shape
    assert log_snr.shape == (timesteps,)
    # Check if log_snr is on the correct device
    assert log_snr.device.type == device


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cosine_snr_transform_alpha(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    t = torch.tensor(0.5, device=device)
    snr_transform = CosineSNRTransform()

    log_snr = snr_transform.calculate_log_snr(t, device=device)
    alpha = snr_transform.calculate_alpha_log_snr(log_snr)

    # Check if alpha is a valid value
    assert alpha > 0
    assert alpha <= 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_linear_snr_transform_alpha(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    t = torch.tensor(0.5, device=device)
    snr_transform = LinearSNRTransform()

    log_snr = snr_transform.calculate_log_snr(t, device=device)
    alpha = snr_transform.calculate_alpha_log_snr(log_snr)

    # Check if alpha is a valid value
    assert alpha > 0
    assert alpha <= 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_linear_log_interpolated_snr_transform_alpha(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    t = torch.tensor(0.5, device=device)
    snr_transform = LinearLogInterpolatedSNRTransform()

    log_snr = snr_transform.calculate_log_snr(t, device=device)
    alpha = snr_transform.calculate_alpha_log_snr(log_snr)

    # Check if alpha is a valid value
    assert alpha > 0
    assert alpha <= 1
