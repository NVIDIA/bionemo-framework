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


class LinearHarmonicPrior(PriorDistribution):
    """A subclass representing a Linear Harmonic prior distribution from Jit et al. https://arxiv.org/abs/2304.02198."""

    def __init__(
        self,
        distance: Float = 3.8,
        length: Optional[int] = None,
        center: Bool = False,
        rng_generator: Optional[torch.Generator] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Linear Harmonic prior distribution.

        Args:
            distance (Float): RMS distance between adjacent points in the line graph.
            length (Optional[int]): The number of points in a batch.
            center (bool): Whether to center the samples around the mean. Defaults to False.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        self.distance = distance
        self.length = length
        self.center = center
        self.rng_generator = rng_generator
        self.device = device
        if length:
            self._calculate_terms(length, device)

    def _calculate_terms(self, N, device):
        a = 3 / (self.distance * self.distance)
        J = torch.zeros(N, N)
        for i, j in zip(torch.arange(N - 1), torch.arange(1, N)):
            J[i, i] += a
            J[j, j] += a
            J[i, j] = J[j, i] = -a
        D, P = torch.linalg.eigh(J)
        D_inv = 1 / D
        D_inv[0] = 0
        self.P, self.D_inv = P.to(device), D_inv.to(device)

    def sample(
        self,
        shape: Tuple,
        mask: Optional[Tensor] = None,
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Generates a specified number of samples from the Harmonic prior distribution.

        Args:
            shape (Tuple): The shape of the samples to generate.
            device (str): cpu or gpu.
            mask (Optional[Tensor]): An optional mask to apply to the samples. Defaults to None.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

        Returns:
            Float: A tensor of samples.
        """
        if len(shape) != 3:
            raise ValueError("Input shape can only work for B x L x D")
        if rng_generator is None:
            rng_generator = self.rng_generator

        samples = torch.randn(*shape, device=device, generator=rng_generator)
        N = shape[1]

        if N != self.length:
            self._calculate_terms(N, device)

        std = torch.sqrt(self.D_inv).unsqueeze(-1)
        samples = self.P @ (std * samples)

        if self.center:
            samples = remove_center_of_mass(samples, mask)

        if mask is not None:
            samples = samples * mask.unsqueeze(-1)
        return samples
