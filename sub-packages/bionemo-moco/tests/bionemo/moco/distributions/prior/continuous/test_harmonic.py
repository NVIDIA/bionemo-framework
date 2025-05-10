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

from bionemo.moco.distributions.prior.continuous.harmonic import LinearHarmonicPrior


def test_harmonic_sampling():
    """
    Test that the LinearHarmonicPrior can sample with various shapes."""
    prior = LinearHarmonicPrior(length=20)
    # Test sampling with various shapes
    shapes = [(20, 3), (10, 20, 10), (10, 20, 3), (5, 10, 20, 3), (10, 40, 3)]
    for shape in shapes:
        samples = prior.sample(shape)
        assert samples.shape == shape


# Devices to test
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")


@pytest.mark.parametrize("device", devices)
def test_harmonic_sampling_gpu(device):
    """
    Test that the LinearHarmonicPrior can sample with various shapes."""
    prior = LinearHarmonicPrior(length=20, device=device)
    # Test sampling with various shapes
    shapes = [(10, 20, 10), (10, 20, 3), (5, 10, 20, 3), (10, 40, 3)]
    for shape in shapes:
        samples = prior.sample(shape)
        assert samples.shape == shape


# TODO figure out a test where the prior not on device but the data is
def test_harmonic_sampling_fixed_gpu():
    """
    Test that the LinearHarmonicPrior can sample with various shapes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    prior = LinearHarmonicPrior(length=20)
    # Test sampling with various shapes
    shapes = [(10, 20, 10), (10, 20, 3), (5, 10, 20, 3), (10, 40, 3)]
    for shape in shapes:
        samples = prior.sample(shape, device="cuda")
        assert samples.shape == shape
        assert samples.device.type == "cuda"
