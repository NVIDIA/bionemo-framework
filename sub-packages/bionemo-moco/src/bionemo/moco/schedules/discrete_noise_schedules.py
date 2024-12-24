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
from typing import Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor

from bionemo.moco.interpolants.base_interpolant import string_to_enum
from bionemo.moco.schedules.utils import TimeDirection


class DiscreteNoiseSchedule(ABC):
    """A base class for discrete schedules. No matter the definition this class returns objects using a unified direction of time."""

    def __init__(self, nsteps: int, direction: TimeDirection):
        """Initialize the DiscreteNoiseSchedule.

        Args:
            nsteps (Optional[int]): Number of time steps. If None, uses the value from initialization.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).

        """
        self.nsteps = nsteps
        self.direction = string_to_enum(direction, TimeDirection)

    def generate_schedule(
        self,
        nsteps: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        synchronize: Optional[TimeDirection] = None,
    ) -> Tensor:
        """Public wrapper to generate the time schedule as a tensor.

        Args:
            nsteps (Optional[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
            synchronize (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).

        Returns:
            Tensor: A tensor of time steps + 1 unless full is False.
        """
        schedule = self._generate_schedule(nsteps, device)
        if synchronize is None:
            return schedule
        synchronize = string_to_enum(synchronize, TimeDirection)
        if self.direction != synchronize:
            return torch.flip(schedule, dims=[0])
        return schedule

    @abstractmethod
    def _generate_schedule(self, nsteps: Optional[int] = None, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Generate the time schedule as a list.

        Args:
            nsteps (Optional[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").

        Returns:
            Tensor: A tensor of time steps + 1 unless full is False.
        """
        pass

    def calculate_derivative(
        self,
        nsteps: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        synchronize: Optional[TimeDirection] = None,
    ) -> Tensor:
        """Calculate the time derivative of the schedule.

        Args:
            nsteps (Optional[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
            synchronize (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).

        Returns:
            Tensor: A tensor representing the time derivative of the schedule.

        Raises:
            NotImplementedError: If the derivative calculation is not implemented for this schedule.
        """
        raise NotImplementedError("Derivative calculation is not implemented for this schedule.")


class DiscreteCosineNoiseSchedule(DiscreteNoiseSchedule):
    """A cosine noise schedule for Diffusion Models."""

    def __init__(self, nsteps: int, nu: Float = 1.0, s: Float = 0.008):
        """Initialize the CosineNoiseSchedule.

        Args:
            nsteps (int): Number of time steps.
            nu (Optional[Float]): Hyperparameter for the cosine schedule (default is 1.0).
            s (Optional[Float]): Hyperparameter for the cosine schedule (default is 0.008).
        """
        super().__init__(nsteps=nsteps, direction=TimeDirection.DIFFUSION)
        self.nu = nu
        self.s = s

    def _generate_schedule(self, nsteps: Optional[int] = None, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Generate the cosine noise schedule.

        Args:
            nsteps (Optional[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").

        Returns:
            Tensor: A tensor of time steps + 1 unless full is False.
        """
        if nsteps is None:
            nsteps = self.nsteps
        steps = nsteps + 2
        x = torch.linspace(0, steps, steps, device=device)
        alphas_cumprod = torch.cos(0.5 * torch.pi * (((x / steps) ** self.nu) + self.s) / (1 + self.s)) ** 2
        alphas_cumprod_new = alphas_cumprod / alphas_cumprod[0]
        alphas_cumprod_new = self._clip_noise_schedule(alphas_cumprod_new, clip_value=0.05)
        alphas = alphas_cumprod_new[1:] / alphas_cumprod_new[:-1]
        alphas = torch.clamp(alphas, min=0.001)
        betas = 1 - alphas
        betas = torch.clamp(betas, 0.0, 0.999)
        result = 1.0 - betas
        return result[1:]

    def _clip_noise_schedule(self, alphas2: Tensor, clip_value: Float = 0.001) -> Tensor:
        """For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during sampling.

        Args:
            alphas2 (Tensor): The noise schedule given by alpha^2.
            clip_value (Optional[Float]): The minimum value for alpha_t / alpha_t-1 (default is 0.001).

        Returns:
            Tensor: The clipped noise schedule.
        """
        alphas2 = torch.cat([torch.ones(1, device=alphas2.device), alphas2], dim=0)

        alphas_step = alphas2[1:] / alphas2[:-1]

        alphas_step = torch.clamp(alphas_step, min=clip_value, max=1.0)
        alphas2 = torch.cumprod(alphas_step, dim=0)

        return alphas2
