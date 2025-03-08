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

from bionemo.moco.distributions.time.distribution import TimeDistribution


class UniformTimeDistribution(TimeDistribution):
    """A class representing a uniform time distribution."""

    def __init__(
        self,
        min_t: Float = 0.0,
        max_t: Float = 1.0,
        discrete_time: Bool = False,
        nsteps: Optional[int] = None,
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Initializes a UniformTimeDistribution object.

        Args:
            min_t (Float): The minimum time value.
            max_t (Float): The maximum time value.
            discrete_time (Bool): Whether the time is discrete.
            nsteps (Optional[int]): Number of nsteps for discretization.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
        """
        super().__init__(discrete_time, nsteps, min_t, max_t, rng_generator)

    def sample(
        self,
        n_samples: Union[int, Tuple[int, ...], torch.Size],
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Generates a specified number of samples from the uniform time distribution.

        Args:
            n_samples (int): The number of samples to generate.
            device (str): cpu or gpu.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

        Returns:
            A tensor of samples.
        """
        if rng_generator is None:
            rng_generator = self.rng_generator
        if self.discrete_time:
            if self.nsteps is None:
                raise ValueError("nsteps cannot be None for discrete time sampling")
            time_step = torch.randint(
                0,
                self.nsteps,
                size=(n_samples,) if isinstance(n_samples, int) else n_samples,
                device=device,
                generator=rng_generator,
            )
        else:
            time_step = torch.rand(n_samples, device=device, generator=rng_generator)
            if self.min_t and self.max_t and self.min_t > 0:
                time_step = time_step * (self.max_t - self.min_t) + self.min_t
        return time_step


class SymmetricUniformTimeDistribution(TimeDistribution):
    """A class representing a uniform time distribution."""

    def __init__(
        self,
        min_t: Float = 0.0,
        max_t: Float = 1.0,
        discrete_time: Bool = False,
        nsteps: Optional[int] = None,
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Initializes a UniformTimeDistribution object.

        Args:
            min_t (Float): The minimum time value.
            max_t (Float): The maximum time value.
            discrete_time (Bool): Whether the time is discrete.
            nsteps (Optional[int]): Number of nsteps for discretization.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
        """
        super().__init__(discrete_time, nsteps, min_t, max_t, rng_generator)

    def sample(
        self,
        n_samples: Union[int, Tuple[int, ...], torch.Size],
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Generates a specified number of samples from the uniform time distribution.

        Args:
            n_samples (int): The number of samples to generate.
            device (str): cpu or gpu.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

        Returns:
            A tensor of samples.
        """
        if rng_generator is None:
            rng_generator = self.rng_generator
        if not isinstance(n_samples, int):
            n_samples = n_samples[0]
        if self.discrete_time:
            if self.nsteps is None:
                raise ValueError("nsteps cannot be None for discrete time sampling")
            time_step = torch.randint(
                0, self.nsteps, size=(n_samples // 2 + 1,), device=device, generator=rng_generator
            )
            time_step = torch.cat([time_step, self.nsteps - time_step - 1], dim=0)[:n_samples]
        else:
            time_step = torch.rand(n_samples // 2 + 1, device=device, generator=rng_generator)
            time_step = torch.cat([time_step, 1 - time_step], dim=0)[:n_samples]
            if self.min_t and self.max_t and self.min_t > 0:
                time_step = time_step * (self.max_t - self.min_t) + self.min_t
        return time_step
