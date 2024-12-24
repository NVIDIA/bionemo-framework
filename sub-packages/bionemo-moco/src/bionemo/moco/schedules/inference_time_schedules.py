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
from torch import Tensor
from jaxtyping import Bool, Float

from bionemo.moco.interpolants.base_interpolant import string_to_enum
from bionemo.moco.schedules.utils import TimeDirection


class InferenceSchedule(ABC):
    """A base class for inference time schedules."""

    def __init__(
        self,
        nsteps: int,
        min_t: Float = 0,
        direction: TimeDirection = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the InferenceSchedule.

        Args:
            nsteps (int): Number of time steps.
            min_t (Float): minimum time value defaults to 0.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
            device (Optional[str]): Device to place the schedule on (default is "cpu").

        """
        self.nsteps = nsteps
        self.min_t = min_t
        self.direction = string_to_enum(direction, TimeDirection)
        self.device = device

    @abstractmethod
    def generate_schedule(
        self, nsteps: Optional[int] = None, full: Bool = False, device: Optional[Union[str, torch.device]] = None
    ) -> Tensor:
        """Generate the time schedule as a tensor.

        Args:
            nsteps (Optioanl[int]): Number of time steps. If None, uses the value from initialization.
            full (Bool): Boolean to return entire schedule or just the needed components.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        pass


class ContinuousInferenceSchedule(InferenceSchedule):
    """A base class for continuous time inference schedules."""

    def discretize(
        self,
        nsteps: Optional[int] = None,
        schedule: Optional[Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tensor:
        """Discretize the time schedule into a list of time deltas.

        Args:
            nsteps (Optioanl[int]): Number of time steps. If None, uses the value from initialization.
            schedule (Optional[Tensor]): Time scheudle if None will generate it with generate_schedule.
            device (Optional[str]): Device to place the schedule on (default is "cpu").

        Returns:
            Tensor: A tensor of time deltas.
        """
        if device is None:
            device = self.device
        if schedule is None:
            schedule = self.generate_schedule(nsteps, full=True, device=device)
        if self.direction == TimeDirection.UNIFIED:
            dt = schedule[1:] - schedule[:-1]
        else:
            dt = -1 * (schedule[1:] - schedule[:-1])
        return dt


class DiscreteInferenceSchedule(InferenceSchedule):
    """A base class for discrete time inference schedules."""

    def discretize(
        self,
        nsteps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tensor:
        """Discretize the time schedule into a list of time deltas.

        Args:
            nsteps (Optioanl[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").

        Returns:
            Tensor: A tensor of time deltas.
        """
        if device is None:
            device = self.device
        return torch.full(
            (nsteps if nsteps is not None else self.nsteps,),
            1 / (nsteps if nsteps is not None else self.nsteps),
            device=device,
        )


class DiscreteLinearInferenceSchedule(DiscreteInferenceSchedule):
    """A linear time schedule for discrete time inference."""

    def __init__(
        self,
        nsteps: int,
        min_t: Float = 0,
        direction: TimeDirection = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the DiscreteLinearInferenceSchedule.

        Args:
            nsteps (int): Number of time steps.
            min_t (Float): minimum time value defaults to 0.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        super().__init__(nsteps, min_t, direction, device)

    def generate_schedule(
        self, nsteps: Optional[int] = None, full: Bool = False, device: Optional[Union[str, torch.device]] = None
    ) -> Tensor:
        """Generate the linear time schedule as a tensor.

        Args:
            nsteps (Optional[int]): Number of time steps. If None uses the value from initialization.
            full (Bool): Whether to return the full scheudle defaults to False.
            device (Optional[str]): Device to place the schedule on (default is "cpu").

        Returns:
            Tensor: A tensor of time steps.
        """
        if device is None:
            device = self.device
        if nsteps is None:
            nsteps = self.nsteps
        if full:
            nsteps += 1
        schedule = torch.arange(nsteps).to(device=device)
        if self.direction == TimeDirection.DIFFUSION:
            schedule = schedule.flip(0)
        return schedule


class LinearInferenceSchedule(ContinuousInferenceSchedule):
    """A linear time schedule for continuous time inference."""

    def __init__(
        self,
        nsteps: int,
        min_t: Float = 0,
        direction: TimeDirection = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the LinearInferenceSchedule.

        Args:
            nsteps (int): Number of time steps.
            min_t (Float): minimum time value defaults to 0.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        super().__init__(nsteps, min_t, direction, device)

    def generate_schedule(
        self,
        nsteps: Optional[int] = None,
        full: Bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tensor:
        """Generate the linear time schedule as a tensor.

        Args:
            nsteps (Optional[int]): Number of time steps. If None uses the value from initialization.
            full (Bool): Whether to return the full scheudle defaults to False.
            device (Optional[str]): Device to place the schedule on (default is "cpu").

        Returns:
            Tensor: A tensor of time steps.
        """
        if device is None:
            device = self.device
        if nsteps is None:
            nsteps = self.nsteps
        schedule = torch.linspace(0, 1, nsteps + 1).to(device=device)
        if self.min_t > 0:
            schedule = torch.clamp(schedule, min=self.min_t)
        if self.direction == TimeDirection.DIFFUSION:
            schedule = 1 - schedule  # schedule.flip(0)
        if full:
            return schedule
        return schedule[:-1]


class PowerInferenceSchedule(ContinuousInferenceSchedule):
    """A power time schedule for inference, where time steps are generated by raising a uniform schedule to a specified power."""

    def __init__(
        self,
        nsteps: int,
        min_t: Float = 0,
        p1: Float = 1.0,
        direction: TimeDirection = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the PowerInferenceSchedule.

        Args:
            nsteps (int): Number of time steps.
            min_t (Float): minimum time value defaults to 0.
            p1 (Float): Power parameter defaults to 1.0.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        super().__init__(nsteps, min_t, direction, device)
        self.p1 = p1

    def generate_schedule(
        self,
        nsteps: Optional[int] = None,
        full: Bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tensor:
        """Generate the power time schedule as a tensor.

        Args:
            nsteps (Optional[int]): Number of time steps. If None uses the value from initialization.
            full (Bool): Whether to return the full scheudle defaults to False.
            device (Optional[str]): Device to place the schedule on (default is "cpu").

        Returns:
            Tensor: A tensor of time steps.
        """
        if device is None:
            device = self.device
        if nsteps is None:
            nsteps = self.nsteps
        schedule = torch.linspace(0, 1, nsteps + 1).to(device=device) ** self.p1
        if self.min_t > 0:
            schedule = torch.clamp(schedule, min=self.min_t)
        if self.direction == TimeDirection.DIFFUSION:
            schedule = 1 - schedule  # schedule.flip(0)
        if full:
            return schedule
        return schedule[:-1]


class LogInferenceSchedule(ContinuousInferenceSchedule):
    """A log time schedule for inference, where time steps are generated by taking the logarithm of a uniform schedule."""

    def __init__(
        self,
        nsteps: int,
        min_t: Float = 0,
        p1: Float = 2,
        direction: TimeDirection = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the LogInferenceSchedule.

        Args:
            nsteps (int): Number of time steps.
            min_t (Float): minimum time value defaults to 0.
            p1 (Float): log space parameter defaults to 2.0.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        super().__init__(nsteps, min_t, direction, device)
        if p1 is None:
            raise ValueError("p1 cannot be None for the log schedule")
        if p1 <= 0:
            raise ValueError(f"p1 must be >0 for the log schedule, got {p1}")
        self.p1 = p1

    def generate_schedule(
        self,
        nsteps: Optional[int] = None,
        full: Bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tensor:
        """Generate the log time schedule as a tensor.

        Args:
            nsteps (Optional[int]): Number of time steps. If None uses the value from initialization.
            full (Bool): Whether to return the full scheudle defaults to False.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        if device is None:
            device = self.device
        if nsteps is None:
            nsteps = self.nsteps
        t = 1.0 - torch.logspace(-self.p1, 0, nsteps + 1).flip(0).to(device=device)
        t = t - torch.min(t)
        schedule = t / torch.max(t)
        if self.min_t > 0:
            schedule = torch.clamp(schedule, min=self.min_t)
        if self.direction == TimeDirection.DIFFUSION:
            schedule = 1 - schedule  # schedule.flip(0)
        if full:
            return schedule
        return schedule[:-1]
