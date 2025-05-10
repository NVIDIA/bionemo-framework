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

from bionemo.moco.distributions.prior.discrete.custom import DiscreteCustomPrior


def test_discrete_custom_prior_init():
    """Test the initialization of the DiscreteCustomPrior class."""
    num_classes = 10
    prior_dist = torch.zeros(num_classes)
    prior_dist[-2:] = 0.5
    prior = DiscreteCustomPrior(prior_dist, num_classes)
    assert prior.num_classes == num_classes
    assert torch.sum(prior.prior_dist).item() - 1.0 < 1e-5


def test_discrete_custom_prior_sample():
    """Test the sample method of the DiscreteCustomPrior class."""
    num_classes = 10
    prior_dist = torch.zeros(num_classes)
    prior_dist[-2:] = 0.5
    prior = DiscreteCustomPrior(prior_dist, num_classes)
    shape = (10, 5)
    samples = prior.sample(shape)
    assert samples.shape == shape
    assert samples.max() <= num_classes - 1
    assert samples.min() >= 0


def test_discrete_custom_prior_sample_with_mask():
    """Test the sample method of the DiscreteCustomPrior class with a mask."""
    num_classes = 10
    prior_dist = torch.zeros(num_classes)
    prior_dist[-2:] = 0.5
    prior = DiscreteCustomPrior(prior_dist, num_classes)
    shape = (10, 5)
    mask = torch.ones((10,) + (1,) * (len(shape) - 1))
    mask[5:] = 0
    samples = prior.sample(shape, mask=mask)
    assert samples.shape == shape
    assert samples.max() <= num_classes - 1
    assert samples.min() >= 0
    assert torch.all(samples[5:] == 0)


def test_discrete_custom_prior_sample_on_gpu():
    """Test the sample method of the DiscreteCustomPrior class on a GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    num_classes = 10
    prior_dist = torch.zeros(num_classes)
    prior_dist[-2:] = 0.5
    prior = DiscreteCustomPrior(prior_dist, num_classes)
    shape = (10, 5)
    device = "cuda:0"
    samples = prior.sample(shape, device=device)
    assert samples.device == torch.device(device)
    assert samples.shape == shape
    assert samples.max() <= num_classes - 1
    assert samples.min() >= 0
