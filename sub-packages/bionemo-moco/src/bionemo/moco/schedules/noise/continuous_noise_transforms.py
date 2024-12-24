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


class ContinuousExpNoiseTransform(ABC):
    """A base class for continuous schedules.

    alpha = exp(- sigma) where 1 - alpha controls the masking fraction.
    """

    def __init__(self, direction: TimeDirection):
        """Initialize the DiscreteNoiseSchedule.

        Args:
            direction : TimeDirection, required this defines in which direction the scheduler was built
        """
        self.direction = string_to_enum(direction, TimeDirection)

    def calculate_sigma(
        self,
        t: Tensor,
        device: Union[str, torch.device] = "cpu",
        synchronize: Optional[TimeDirection] = None,
    ) -> Tensor:
        """Calculate the sigma for the given time steps.

        Args:
            t (Tensor): The input tensor representing the time steps, with values ranging from 0 to 1.
            device (Optional[str]): The device to place the schedule on. Defaults to "cpu".
            synchronize (optional[TimeDirection]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction,
                this parameter allows to flip the direction to match the specified one. Defaults to None.

        Returns:
            Tensor: A tensor representing the sigma values for the given time steps.

        Raises:
            ValueError: If the input time steps exceed the maximum allowed value of 1.
        """
        if t.max() > 1:
            raise ValueError(f"Invalid value: max continuous time is 1, but got {t.max().item()}")

        if synchronize and self.direction != string_to_enum(synchronize, TimeDirection):
            t = 1 - t
        return self._calculate_sigma(t, device)

    @abstractmethod
    def _calculate_sigma(self, t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Calculate the -log of the clean data value for the given time steps.

        Args:
            t (Tensor): The input tensor representing the time steps.
            device (Optional[str]): The device to place the schedule on. Defaults to "cpu".

        Returns:
            Tensor: A tensor representing the sigma values for the given time steps.
        """
        pass

    def sigma_to_alpha(self, sigma: Tensor) -> Tensor:
        """Converts sigma to alpha values by alpha = exp(- sigma).

        Args:
            sigma (Tensor): The input sigma tensor.

        Returns:
            Tensor: A tensor containing the alpha values.
        """
        return torch.exp(-1 * sigma)


class CosineExpNoiseTransform(ContinuousExpNoiseTransform):
    """A cosine Exponential noise schedule."""

    def __init__(self, eps: Float = 1.0e-3):
        """Initialize the CosineNoiseSchedule.

        Args:
            eps (Float): small number to prevent numerical issues.
        """
        self.direction = TimeDirection.DIFFUSION
        self.eps = eps

    def _calculate_sigma(self, t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Calculate negative log of data interpolant fraction.

        Args:
            t (Tensor): The input tensor representing the time steps.
            device (Optional[str]): The device to place the schedule on. Defaults to "cpu".

        Returns:
            Tensor: A tensor representing the sigma values for the given time steps.
        """
        cos = torch.cos(t * torch.pi / 2).to(device)
        return -torch.log(self.eps + (1 - self.eps) * cos)

    def d_dt_sigma(self, t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Compute the derivative of sigma with respect to time.

        Args:
            t (Tensor): The input tensor representing the time steps.
            device (Optional[str]): The device to place the schedule on. Defaults to "cpu".

        Returns:
            Tensor: A tensor representing the derivative of sigma with respect to time.

        Notes:
            The derivative of sigma as a function of time is given by:

            d/dt sigma(t) = d/dt (-log(cos(t * pi / 2) + eps))

            Using the chain rule, we get:

            d/dt sigma(t) = (-1 / (cos(t * pi / 2) + eps)) * (-sin(t * pi / 2) * pi / 2)

            This is the derivative that is computed and returned by this method.
        """
        cos = (1 - self.eps) * torch.cos(t * torch.pi / 2)
        sin = (1 - self.eps) * torch.sin(t * torch.pi / 2)
        scale = torch.pi / 2
        derivative = scale * sin / (cos + self.eps)
        return derivative.to(device)


class LogLinearExpNoiseTransform(ContinuousExpNoiseTransform):
    """A log linear exponential schedule."""

    def __init__(self, eps: Float = 1.0e-3):
        """Initialize the CosineNoiseSchedule.

        Args:
            eps (Float): small value to prevent numerical issues.
        """
        self.direction = TimeDirection.DIFFUSION
        self.eps = eps

    def _calculate_sigma(self, t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Calculate negative log of data interpolant fraction.

        Args:
            t (Tensor): The input tensor representing the time steps.
            device (Optional[str]): The device to place the schedule on. Defaults to "cpu".

        Returns:
            Tensor: A tensor representing the sigma values for the given time steps.
        """
        return -torch.log1p(-(1 - self.eps) * t).to(device)

    def d_dt_sigma(self, t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Compute the derivative of sigma with respect to time.

        Args:
            t (Tensor): The input tensor representing the time steps.
            device (Optional[str]): The device to place the schedule on. Defaults to "cpu".

        Returns:
            Tensor: A tensor representing the derivative of sigma with respect to time.
        """
        derivative = (1 - self.eps) / (1 - (1 - self.eps) * t)
        return derivative.to(device)
