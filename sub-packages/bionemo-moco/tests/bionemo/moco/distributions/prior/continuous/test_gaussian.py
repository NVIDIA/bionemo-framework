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


import torch

from bionemo.moco.distributions.prior.continuous.gaussian import GaussianPrior


def test_gaussian_sampling():
    """
    Test that the GaussianPrior can sample with various shapes."""
    mean, std = 0.5, 2.0
    prior = GaussianPrior(mean, std)

    # Test sampling with various shapes
    shapes = [(10,), (10, 20), (10, 20, 3)]
    for shape in shapes:
        samples = prior.sample(shape)
        assert samples.shape == shape


def test_gaussian_centering_without_mask():
    """
    Test that the GaussianPrior centers the samples without a mask."""
    mean, std = 0, 1
    prior = GaussianPrior(mean, std, center=True)
    shape = (10, 20, 3)
    samples = prior.sample(shape)

    # Calculate the mean of the samples along the middle dimension
    mask = torch.ones(shape[:-1]).bool()

    # Calculate the mean of the samples along the middle dimension
    sample_mean = (samples * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).unsqueeze(-1)

    # Assert the sum of sample means is close to zero
    assert torch.abs(sample_mean.sum()) < 1e-5


def test_gaussian_centering_with_mask():
    """
    Test that the GaussianPrior centers the samples with a mask."""
    mean, std = 0, 1
    prior = GaussianPrior(mean, std, center=True)
    shape = (30, 4, 50)
    mask = torch.ones(shape[:-1]).bool()
    mask[:, 2:] = False  # Mask out the last 2 elements of the middle dimension

    samples = prior.sample(shape, mask=mask)
    # Calculate the mean of the samples along the middle dimension
    sample_mean = (samples * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).unsqueeze(-1)

    # Assert the sum of sample means is close to zero
    assert torch.abs(sample_mean.sum()) < 1e-5

    # Calculate the sum of all the masked out samples
    masked_out_samples_sum = (samples * (~mask).unsqueeze(-1)).sum()

    # Assert the sum of all the masked out samples is close to zero
    assert torch.abs(masked_out_samples_sum) < 1e-12
