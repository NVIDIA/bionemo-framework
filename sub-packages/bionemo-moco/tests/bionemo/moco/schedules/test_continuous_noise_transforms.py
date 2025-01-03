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

from bionemo.moco import TimeDirection
from bionemo.moco.schedules.noise.continuous_noise_transforms import (
    CosineExpNoiseTransform,
    LogLinearExpNoiseTransform,
)


class TestContinuousNoiseTransforms:
    @pytest.mark.parametrize("transform_cls", [CosineExpNoiseTransform, LogLinearExpNoiseTransform])
    def test_init(self, transform_cls):
        transform = transform_cls()
        assert transform.direction == TimeDirection.DIFFUSION

    @pytest.mark.parametrize("transform_cls", [CosineExpNoiseTransform, LogLinearExpNoiseTransform])
    def test_calculate_sigma(self, transform_cls):
        transform = transform_cls()
        t = torch.linspace(0, 1, 10)
        sigma = transform.calculate_sigma(t)
        assert sigma.shape == t.shape
        assert (sigma >= 0).all()

    @pytest.mark.parametrize("transform_cls", [CosineExpNoiseTransform, LogLinearExpNoiseTransform])
    def test_calculate_sigma_invalid_input(self, transform_cls):
        transform = transform_cls()
        t = torch.tensor([1.1, 2.2])  # invalid input, max value > 1
        with pytest.raises(ValueError):
            transform.calculate_sigma(t)

    @pytest.mark.parametrize("transform_cls", [CosineExpNoiseTransform, LogLinearExpNoiseTransform])
    def test_sigma_to_alpha(self, transform_cls):
        transform = transform_cls()
        sigma = torch.linspace(0.1, 1.0, 10)
        alpha = transform.sigma_to_alpha(sigma)
        assert alpha.shape == sigma.shape
        assert (alpha >= 0).all()

    @pytest.mark.parametrize("transform_cls", [CosineExpNoiseTransform, LogLinearExpNoiseTransform])
    def test_d_dt_sigma(self, transform_cls):
        transform = transform_cls()
        t = torch.linspace(0, 1, 10)
        derivative = transform.d_dt_sigma(t)
        assert derivative.shape == t.shape

    def test_cosine_transform(self):
        transform = CosineExpNoiseTransform()
        t = torch.linspace(0, 1, 10)
        sigma = transform.calculate_sigma(t)
        assert torch.allclose(sigma, -torch.log(1e-3 + (1 - 1e-3) * torch.cos(t * torch.pi / 2)))

    def test_loglinear_transform(self):
        transform = LogLinearExpNoiseTransform()
        t = torch.linspace(0, 1, 10)
        sigma = transform.calculate_sigma(t)
        assert torch.allclose(sigma, -torch.log1p(-(1 - 1e-3) * t))
