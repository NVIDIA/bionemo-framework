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

from bionemo.moco.distributions.time.beta import BetaTimeDistribution
from bionemo.moco.distributions.time.distribution import MixTimeDistribution
from bionemo.moco.distributions.time.logit_normal import LogitNormalTimeDistribution
from bionemo.moco.distributions.time.uniform import SymmetricUniformTimeDistribution, UniformTimeDistribution


# List of distributions to test
distributions = [
    (BetaTimeDistribution, {"p1": 2.0, "p2": 1.0}),
    (UniformTimeDistribution, {}),
    (SymmetricUniformTimeDistribution, {}),
    (LogitNormalTimeDistribution, {"p1": 0.0, "p2": 1.0}),
]

# Devices to test
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")


@pytest.mark.parametrize("dist_class, dist_kwargs", distributions)
@pytest.mark.parametrize("device", devices)
def test_continuous_time_sampling(dist_class, dist_kwargs, device):
    # Initialize the time distribution
    dist = dist_class(min_t=0.0, max_t=1.0, discrete_time=False, **dist_kwargs)

    # Sample from the distribution
    samples = dist.sample(n_samples=1000, device=device)

    # Check if the samples are within the correct range
    assert torch.all(samples >= 0.0)
    assert torch.all(samples <= 1.0)

    # Check if the shape of the samples is correct
    assert samples.shape == (1000,)


@pytest.mark.parametrize("dist_class, dist_kwargs", distributions)
@pytest.mark.parametrize("device", devices)
def test_discrete_time_sampling(dist_class, dist_kwargs, device):
    # Initialize the time distribution
    dist = dist_class(min_t=0.0, max_t=1.0, discrete_time=True, nsteps=10, **dist_kwargs)

    # Sample from the distribution
    samples = dist.sample(n_samples=1000, device=device)

    # Check if the samples are within the correct range
    assert torch.all(samples >= 0)
    assert torch.all(samples <= 9)

    # Check if the shape of the samples is correct
    assert samples.shape == (1000,)

    # Check if the samples are integers
    assert samples.dtype == torch.int64


@pytest.mark.parametrize("dist_class, dist_kwargs", distributions)
@pytest.mark.parametrize("device", devices)
def test_sample_shape(dist_class, dist_kwargs, device):
    # Initialize the time distribution
    dist = dist_class(min_t=0.0, max_t=1.0, discrete_time=False, **dist_kwargs)

    # Sample from the distribution with different number of samples
    samples100 = dist.sample(n_samples=100, device=device)
    samples1000 = dist.sample(n_samples=1000, device=device)

    # Check if the shape of the samples is correct
    assert samples100.shape == (100,)
    assert samples1000.shape == (1000,)


@pytest.mark.parametrize("dist_class, dist_kwargs", distributions)
@pytest.mark.parametrize("device", devices)
def test_device(dist_class, dist_kwargs, device):
    # Initialize the time distribution
    dist = dist_class(min_t=0.0, max_t=1.0, discrete_time=False, **dist_kwargs)

    # Sample from the distribution
    samples = dist.sample(n_samples=100, device=device)

    # Check if the samples are on the correct device
    assert samples.device.type == device


@pytest.mark.parametrize("dist_class, dist_kwargs", distributions)
@pytest.mark.parametrize("device", devices)
def test_min_max_t(dist_class, dist_kwargs, device):
    # Initialize the time distribution with min_t and max_t
    dist = dist_class(min_t=1e-2, max_t=0.99, discrete_time=False, **dist_kwargs)

    # Sample from the distribution
    samples = dist.sample(n_samples=1000, device=device)

    # Check if the samples are within the correct range
    assert torch.all(samples >= 1e-2)
    assert torch.all(samples <= 0.99)


def test_mix_time_distribution():
    # Create a mix of Uniform and Beta distributions
    uniform_dist = UniformTimeDistribution(min_t=0.0, max_t=1.0, discrete_time=False)
    beta_dist = BetaTimeDistribution(min_t=0.0, max_t=1.0, discrete_time=False, p1=2.0, p2=1.0)
    mix_dist = MixTimeDistribution(uniform_dist, beta_dist, mix_fraction=0.5)

    # Test sampling
    n_samples = 100
    device = "cpu"
    samples = mix_dist.sample(n_samples, device)

    # Check that the samples are within the correct range
    assert (samples >= 0.0).all() and (samples <= 1.0).all()

    # Test that the device is correct
    assert samples.device == torch.device(device)


def test_mix_time_distribution_edge_cases():
    # Test that the mix fraction is validated correctly
    uniform_dist = UniformTimeDistribution(min_t=0.0, max_t=1.0, discrete_time=False)
    beta_dist = BetaTimeDistribution(min_t=0.0, max_t=1.0, discrete_time=False, p1=2.0, p2=1.0)

    with pytest.raises(ValueError):
        MixTimeDistribution(uniform_dist, beta_dist, mix_fraction=-0.1)

    with pytest.raises(ValueError):
        MixTimeDistribution(uniform_dist, beta_dist, mix_fraction=1.1)
