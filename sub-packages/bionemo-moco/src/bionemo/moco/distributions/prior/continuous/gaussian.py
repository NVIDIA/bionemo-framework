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


from typing import Optional, Tuple, Union

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from bionemo.moco.distributions.prior.continuous.utils import remove_center_of_mass
from bionemo.moco.distributions.prior.distribution import PriorDistribution


class GaussianPrior(PriorDistribution):
    """A subclass representing a Gaussian prior distribution."""

    def __init__(
        self,
        mean: Float = 0.0,
        std: Float = 1.0,
        center: Bool = False,
        rng_generator: Optional[torch.Generator] = None,
    ) -> None:
        """Gaussian prior distribution.

        Args:
            mean (Float): The mean of the Gaussian distribution. Defaults to 0.0.
            std (Float): The standard deviation of the Gaussian distribution. Defaults to 1.0.
            center (bool): Whether to center the samples around the mean. Defaults to False.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
        """
        self.mean = mean
        self.std = std
        self.center = center
        self.rng_generator = rng_generator

    def sample(
        self,
        shape: Tuple,
        mask: Optional[Tensor] = None,
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Generates a specified number of samples from the Gaussian prior distribution.

        Args:
            shape (Tuple): The shape of the samples to generate.
            device (str): cpu or gpu.
            mask (Optional[Tensor]): An optional mask to apply to the samples. Defaults to None.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

        Returns:
            Float: A tensor of samples.
        """
        if rng_generator is None:
            rng_generator = self.rng_generator
        samples = torch.randn(*shape, device=device, generator=rng_generator)
        if self.std != 1:
            samples = samples * self.std
        if self.mean != 0:
            samples = samples + self.mean

        if self.center:
            samples = remove_center_of_mass(samples, mask)
        if mask is not None:
            samples = samples * mask.unsqueeze(-1)
        return samples
