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


from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
from jaxtyping import Bool, Float


class TimeDistribution(ABC):
    """An abstract base class representing a time distribution.

    Args:
        discrete_time (Bool): Whether the time is discrete.
        nsteps (Optional[int]): Number of nsteps for discretization.
        min_t (Optional[Float]): Min continuous time.
        max_t (Optional[Float]): Max continuous time.
        rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
    """

    def __init__(
        self,
        discrete_time: Bool = False,
        nsteps: Optional[int] = None,
        min_t: Optional[Float] = None,
        max_t: Optional[Float] = None,
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Initializes a TimeDistribution object."""
        self.discrete_time = discrete_time
        self.nsteps = nsteps
        self.rng_generator = rng_generator
        if discrete_time:
            min_t = 0.0
            max_t = 1.0
            if nsteps is None:
                raise ValueError("nsteps must not be None and must be specified for discrete time")
        if min_t is not None and isinstance(min_t, float):
            if not 0 <= min_t < 1.0:
                raise ValueError("min_t must be greater than or equal to 0 and less than 1.0")
        self.min_t = min_t
        if max_t is not None and isinstance(max_t, float):
            if not 0 < max_t <= 1.0:
                raise ValueError("max_t must be greater than 0 and less than or equal to 1.0")
        self.max_t = max_t
        if (
            self.min_t is not None
            and self.max_t is not None
            and isinstance(self.min_t, float)
            and isinstance(self.max_t, float)
        ):
            if self.min_t >= self.max_t:
                raise ValueError("min_t must be less than max_t")

    @abstractmethod
    def sample(
        self,
        n_samples: Union[int, Tuple[int, ...], torch.Size],
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> Float:
        """Generates a specified number of samples from the time distribution.

        Args:
        n_samples (int): The number of samples to generate.
        device (str): cpu or gpu.
        rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

        Returns:
            Float: A list or array of samples.
        """
        pass


class MixTimeDistribution:
    """An abstract base class representing a mixed time distribution.

    uniform_dist = UniformTimeDistribution(min_t=0.0, max_t=1.0, discrete_time=False)
    beta_dist = BetaTimeDistribution(min_t=0.0, max_t=1.0, discrete_time=False, p1=2.0, p2=1.0)
    mix_dist = MixTimeDistribution(uniform_dist, beta_dist, mix_fraction=0.5)
    """

    def __init__(self, dist1: TimeDistribution, dist2: TimeDistribution, mix_fraction: Float):
        """Initializes a MixTimeDistribution object.

        Args:
            dist1 (TimeDistribution): The first time distribution.
            dist2 (TimeDistribution): The second time distribution.
            mix_fraction (Float): The fraction of samples to draw from dist1. Must be between 0 and 1.
        """
        if not 0 <= mix_fraction <= 1:
            raise ValueError("mix_fraction must be between 0 and 1")
        self.dist1 = dist1
        self.dist2 = dist2
        self.mix_fraction = mix_fraction

    def sample(
        self, n_samples: int, device: Union[str, torch.device] = "cpu", rng_generator: Optional[torch.Generator] = None
    ) -> Float:
        """Generates a specified number of samples from the mixed time distribution.

        Args:
            n_samples (int): The number of samples to generate.
            device (str): cpu or gpu.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

        Returns:
            Float: A list or array of samples.
        """
        samples_dist1 = self.dist1.sample(n_samples, device)
        samples_dist2 = self.dist2.sample(n_samples, device)
        mix = torch.rand(n_samples, device=device, generator=rng_generator)
        return torch.where(mix < self.mix_fraction, samples_dist1, samples_dist2)
