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


class InferenceSchedule(ABC):
    """A base class for inference time schedules."""

    def __init__(
        self,
        nsteps: int,
        min_t: Float = 0,
        padding: Float = 0,
        dilation: Float = 0,
        direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the InferenceSchedule.

        Args:
            nsteps (int): Number of time steps.
            min_t (Float): minimum time value defaults to 0.
            padding (Float): padding time value defaults to 0.
            dilation (Float): dilation time value defaults to 0 ie the number of replicates.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
            device (Optional[str]): Device to place the schedule on (default is "cpu").

        """
        self.nsteps = nsteps
        self.min_t = min_t
        self.padding = padding
        self.dilation = dilation
        self.direction = string_to_enum(direction, TimeDirection)
        self.device = device

    @abstractmethod
    def generate_schedule(
        self, nsteps: Optional[int] = None, device: Optional[Union[str, torch.device]] = None
    ) -> Tensor:
        """Generate the time schedule as a tensor.

        Args:
            nsteps (Optioanl[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        pass

    def pad_time(
        self, n_samples: int, scalar_time: Float, device: Optional[Union[str, torch.device]] = None
    ) -> Tensor:
        """Creates a tensor of shape (n_samples,) filled with a scalar time value.

        Args:
            n_samples (int): The desired dimension of the output tensor.
            scalar_time (Float): The scalar time value to fill the tensor with.
            device (Optional[Union[str, torch.device]], optional):
                The device to place the tensor on. Defaults to None, which uses the default device.

        Returns:
            Tensor: A tensor of shape (n_samples,) filled with the scalar time value.
        """
        return torch.full((n_samples,), fill_value=scalar_time).to(device)


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
            schedule = self.generate_schedule(nsteps, device=device)
        if self.direction == TimeDirection.UNIFIED:
            schedule = torch.cat((schedule, torch.ones((1,), device=schedule.device)))
            dt = schedule[1:] - schedule[:-1]
        else:
            schedule = torch.cat((schedule, torch.zeros((1,), device=schedule.device)))
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
        if self.padding > 0 or self.dilation > 0:
            raise NotImplementedError("discreteize is not implemented for discrete schedules with padding or dilation")
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
        padding: Float = 0,
        dilation: Float = 0,
        direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the DiscreteLinearInferenceSchedule.

        Args:
            nsteps (int): Number of time steps.
            min_t (Float): minimum time value defaults to 0.
            padding (Float): padding time value defaults to 0.
            dilation (Float): dilation time value defaults to 0 ie the number of replicates.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        super().__init__(nsteps, min_t, padding, dilation, direction, device)

    def generate_schedule(
        self, nsteps: Optional[int] = None, device: Optional[Union[str, torch.device]] = None
    ) -> Tensor:
        """Generate the linear time schedule as a tensor.

        Args:
            nsteps (Optional[int]): Number of time steps. If None uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").

        Returns:
            Tensor: A tensor of time steps.
            Tensor: A tensor of time steps.
        """
        if device is None:
            device = self.device
        if nsteps is None:
            nsteps = self.nsteps
        nsteps -= self.padding
        dilation = self.dilation + 1
        if dilation > 1:
            if nsteps % dilation != 0:
                raise ValueError(f"nsteps ({nsteps}) is not divisible by dilation + 1 ({dilation})")
            nsteps = int(nsteps / self.dilation)
        if nsteps is None:
            raise ValueError("nsteps cannot be None")
        schedule = torch.arange(nsteps).to(device=device)
        if dilation > 1:
            schedule = schedule.repeat_interleave(dilation)
        if self.direction == TimeDirection.DIFFUSION:
            schedule = schedule.flip(0)
        if self.padding > 0:
            schedule = torch.cat((schedule, schedule[-1] * torch.ones(self.padding, device=device)))
        return schedule


class LinearInferenceSchedule(ContinuousInferenceSchedule):
    """A linear time schedule for continuous time inference."""

    def __init__(
        self,
        nsteps: int,
        min_t: Float = 0,
        padding: Float = 0,
        dilation: Float = 0,
        direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the LinearInferenceSchedule.

        Args:
            nsteps (int): Number of time steps.
            min_t (Float): minimum time value defaults to 0.
            padding (Float): padding time value defaults to 0.
            dilation (Float): dilation time value defaults to 0 ie the number of replicates.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        super().__init__(nsteps, min_t, padding, dilation, direction, device)

    def generate_schedule(
        self,
        nsteps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tensor:
        """Generate the linear time schedule as a tensor.

        Args:
            nsteps (Optional[int]): Number of time steps. If None uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").

        Returns:
            Tensor: A tensor of time steps.
            Tensor: A tensor of time steps.
        """
        if device is None:
            device = self.device
        if nsteps is None:
            nsteps = self.nsteps
        nsteps -= self.padding
        dilation = self.dilation + 1
        if dilation > 1:
            if nsteps % dilation != 0:
                raise ValueError(f"nsteps ({nsteps}) is not divisible by dilation + 1 ({dilation})")
            nsteps = int(nsteps / dilation)
        if nsteps is None:
            raise ValueError("nsteps cannot be None")
        schedule = torch.linspace(0, 1, nsteps + 1).to(device=device)

        if self.min_t > 0:
            schedule = torch.clamp(schedule, min=self.min_t)

        schedule = schedule[:-1]
        if dilation > 1:
            schedule = schedule.repeat_interleave(dilation)
        if self.padding > 0:
            schedule = torch.cat((schedule, torch.ones(self.padding, device=device)))
        if self.direction == TimeDirection.DIFFUSION:
            schedule = 1 - schedule  # schedule.flip(0)
        return schedule


class PowerInferenceSchedule(ContinuousInferenceSchedule):
    """A power time schedule for inference, where time steps are generated by raising a uniform schedule to a specified power."""

    def __init__(
        self,
        nsteps: int,
        min_t: Float = 0,
        padding: Float = 0,
        dilation: Float = 0,
        p1: Float = 1.0,
        direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the PowerInferenceSchedule.

        Args:
            nsteps (int): Number of time steps.
            min_t (Float): minimum time value defaults to 0.
            padding (Float): padding time value defaults to 0.
            dilation (Float): dilation time value defaults to 0 ie the number of replicates.
            p1 (Float): Power parameter defaults to 1.0.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        super().__init__(nsteps, min_t, padding, dilation, direction, device)
        self.p1 = p1

    def generate_schedule(
        self,
        nsteps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tensor:
        """Generate the power time schedule as a tensor.

        Args:
            nsteps (Optional[int]): Number of time steps. If None uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").

        Returns:
            Tensor: A tensor of time steps.
            Tensor: A tensor of time steps.
        """
        if device is None:
            device = self.device
        if nsteps is None:
            nsteps = self.nsteps
        nsteps -= self.padding
        dilation = self.dilation + 1
        if dilation > 1:
            if nsteps % dilation != 0:
                raise ValueError(f"nsteps ({nsteps}) is not divisible by dilation + 1 ({dilation})")
            nsteps = int(nsteps / dilation)
        if nsteps is None:
            raise ValueError("nsteps cannot be None")
        schedule = torch.linspace(0, 1, nsteps + 1).to(device=device) ** self.p1
        if self.min_t > 0:
            schedule = torch.clamp(schedule, min=self.min_t)
        schedule = schedule[:-1]
        if dilation > 1:
            schedule = schedule.repeat_interleave(dilation)
        if self.padding > 0:
            schedule = torch.cat((schedule, torch.ones(self.padding, device=device)))
        if self.direction == TimeDirection.DIFFUSION:
            schedule = 1 - schedule  # schedule.flip(0)
        return schedule


class LogInferenceSchedule(ContinuousInferenceSchedule):
    """A log time schedule for inference, where time steps are generated by taking the logarithm of a uniform schedule."""

    def __init__(
        self,
        nsteps: int,
        min_t: Float = 0,
        padding: Float = 0,
        dilation: Float = 0,
        p1: Float = 2,
        direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the LogInferenceSchedule.

        Args:
            nsteps (int): Number of time steps.
            min_t (Float): minimum time value defaults to 0.
            padding (Float): padding time value defaults to 0.
            dilation (Float): dilation time value defaults to 0 ie the number of replicates.
            p1 (Float): log space parameter defaults to 2.0.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        super().__init__(nsteps, min_t, padding, dilation, direction, device)
        if p1 is None:
            raise ValueError("p1 cannot be None for the log schedule")
        if p1 <= 0:
            raise ValueError(f"p1 must be >0 for the log schedule, got {p1}")
        self.p1 = p1

    def generate_schedule(
        self,
        nsteps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tensor:
        """Generate the log time schedule as a tensor.

        Args:
            nsteps (Optional[int]): Number of time steps. If None uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        if device is None:
            device = self.device
        if nsteps is None:
            nsteps = self.nsteps
        nsteps -= self.padding
        dilation = self.dilation + 1
        if dilation > 1:
            if nsteps % dilation != 0:
                raise ValueError(f"nsteps ({nsteps}) is not divisible by dilation + 1 ({dilation})")
            nsteps = int(nsteps / self.dilation)
        if nsteps is None:
            raise ValueError("nsteps cannot be None")
        t = 1.0 - torch.logspace(-self.p1, 0, nsteps + 1).flip(0).to(device=device)
        t = t - torch.min(t)
        schedule = t / torch.max(t)
        if self.min_t > 0:
            schedule = torch.clamp(schedule, min=self.min_t)
        schedule = schedule[:-1]
        if dilation > 1:
            schedule = schedule.repeat_interleave(dilation)
        if self.padding > 0:
            schedule = torch.cat((schedule, torch.ones(self.padding, device=device)))
        if self.direction == TimeDirection.DIFFUSION:
            schedule = 1 - schedule  # schedule.flip(0)
        return schedule
