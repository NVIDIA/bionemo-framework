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
    """A base class for discrete noise schedules."""

    def __init__(self, nsteps: int, direction: TimeDirection):
        """Initialize the DiscreteNoiseSchedule.

        Args:
           nsteps (int): number of discrete steps.
           direction (TimeDirection): required this defines in which direction the scheduler was built
        """
        self.nsteps = nsteps
        self.direction = string_to_enum(direction, TimeDirection)

    def generate_schedule(
        self,
        nsteps: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        synchronize: Optional[TimeDirection] = None,
    ) -> Tensor:
        """Generate the noise schedule as a tensor.

        Args:
            nsteps (Optional[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
            synchronize (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction,
                this parameter allows to flip the direction to match the specified one (default is None).
        """
        schedule = self._generate_schedule(nsteps, device)
        if synchronize and self.direction != string_to_enum(synchronize, TimeDirection):
            return torch.flip(schedule, dims=[0])
        else:
            return schedule

    @abstractmethod
    def _generate_schedule(self, nsteps: Optional[int] = None, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Generate the noise schedule tensor.

        Args:
            nsteps (Optional[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
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
            synchronize (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction,
                this parameter allows to flip the direction to match the specified one (default is None).

        Returns:
            Tensor: A tensor representing the time derivative of the schedule.

        Raises:
            NotImplementedError: If the derivative calculation is not implemented for this schedule.
        """
        raise NotImplementedError("Derivative calculation is not implemented for this schedule.")


class DiscreteCosineNoiseSchedule(DiscreteNoiseSchedule):
    """A cosine discrete noise schedule."""

    def __init__(self, nsteps: int, nu: Float = 1.0, s: Float = 0.008):
        """Initialize the CosineNoiseSchedule.

        Args:
            nsteps (int): Number of discrete steps.
            nu (Optional[Float]): Hyperparameter for the cosine schedule exponent (default is 1.0).
            s (Optional[Float]): Hyperparameter for the cosine schedule shift (default is 0.008).
        """
        super().__init__(nsteps=nsteps, direction=TimeDirection.DIFFUSION)
        self.nu = nu
        self.s = s

    def _generate_schedule(self, nsteps: Optional[int] = None, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Generate the cosine noise schedule.

        Args:
            nsteps (Optional[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        if nsteps is None:
            nsteps = self.nsteps
        steps = (
            nsteps + 1
        )  #! matches OpenAI code https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py#L62
        x = torch.linspace(0, nsteps, steps, device=device)
        alphas_cumprod = torch.cos(((x / nsteps) ** self.nu + self.s) / (1 + self.s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.001, 0.999)
        return 1 - betas

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


class DiscreteLinearNoiseSchedule(DiscreteNoiseSchedule):
    """A linear discrete noise schedule."""

    def __init__(self, nsteps: int, beta_start: Float = 1e-4, beta_end: Float = 0.02):
        """Initialize the CosineNoiseSchedule.

        Args:
            nsteps (Optional[int]): Number of time steps. If None, uses the value from initialization.
            beta_start (Optional[int]): starting beta value. Defaults to 1e-4.
            beta_end (Optional[int]): end beta value. Defaults to 0.02.
        """
        super().__init__(nsteps=nsteps, direction=TimeDirection.DIFFUSION)
        self.beta_start = beta_start
        self.beta_end = beta_end

    def _generate_schedule(self, nsteps: Optional[int] = None, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Generate the cosine noise schedule.

        Args:
            nsteps (Optional[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        if nsteps is None:
            nsteps = self.nsteps
        betas = torch.linspace(self.beta_start, self.beta_end, nsteps, dtype=torch.float32, device=device)
        return 1 - betas
