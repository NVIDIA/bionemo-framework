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
from typing import Callable, Optional, Union

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

    def __init__(
        self,
        nsteps: int,
        inclusive_end: bool = False,
        min_t: Float = 0,
        padding: Float = 0,
        dilation: Float = 0,
        direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the ContinuousInferenceSchedule.

        Args:
            nsteps (int): Number of time steps.
            inclusive_end (bool): If True, include the end value (1.0) in the schedule otherwise ends at 1.0-1/nsteps (default is False).
            min_t (Float): minimum time value defaults to 0.
            padding (Float): padding time value defaults to 0.
            dilation (Float): dilation time value defaults to 0 ie the number of replicates.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
            device (Optional[str]): Device to place the schedule on (default is "cpu").

        """
        super().__init__(nsteps, min_t, padding, dilation, direction, device)
        self.inclusive_end = inclusive_end

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
        inclusive_end: bool = False,
        min_t: Float = 0,
        padding: Float = 0,
        dilation: Float = 0,
        direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the LinearInferenceSchedule.

        Args:
            nsteps (int): Number of time steps.
            inclusive_end (bool): If True, include the end value (1.0) in the schedule otherwise ends at 1.0-1/nsteps (default is False).
            min_t (Float): minimum time value defaults to 0.
            padding (Float): padding time value defaults to 0.
            dilation (Float): dilation time value defaults to 0 ie the number of replicates.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        super().__init__(nsteps, inclusive_end, min_t, padding, dilation, direction, device)

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
        if not self.inclusive_end:
            schedule = torch.linspace(self.min_t, 1, nsteps + 1).to(device=device)
            schedule = schedule[:-1]
        else:
            schedule = torch.linspace(self.min_t, 1, nsteps).to(device=device)
        if dilation > 1:
            schedule = schedule.repeat_interleave(dilation)
        if self.padding > 0:
            schedule = torch.cat((schedule, torch.ones(self.padding, device=device)))
        if self.direction == TimeDirection.DIFFUSION:
            schedule = 1 - schedule
        return schedule


class PowerInferenceSchedule(ContinuousInferenceSchedule):
    """A power time schedule for inference, where time steps are generated by raising a uniform schedule to a specified power."""

    def __init__(
        self,
        nsteps: int,
        inclusive_end: bool = False,
        min_t: Float = 0,
        padding: Float = 0,
        dilation: Float = 0,
        exponent: Float = 1.0,
        direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the PowerInferenceSchedule.

        Args:
            nsteps (int): Number of time steps.
            inclusive_end (bool): If True, include the end value (1.0) in the schedule otherwise ends at <1.0 (default is False).
            min_t (Float): minimum time value defaults to 0.
            padding (Float): padding time value defaults to 0.
            dilation (Float): dilation time value defaults to 0 ie the number of replicates.
            exponent (Float): Power parameter defaults to 1.0.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        super().__init__(nsteps, inclusive_end, min_t, padding, dilation, direction, device)
        self.exponent = exponent

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
        if not self.inclusive_end:
            schedule = torch.linspace(self.min_t, 1, nsteps + 1).to(device=device) ** self.exponent
            schedule = schedule[:-1]
        else:
            schedule = torch.linspace(self.min_t, 1, nsteps).to(device=device) ** self.exponent
        if dilation > 1:
            schedule = schedule.repeat_interleave(dilation)
        if self.padding > 0:
            schedule = torch.cat((schedule, torch.ones(self.padding, device=device)))
        if self.direction == TimeDirection.DIFFUSION:
            schedule = 1 - schedule
        return schedule


class LogInferenceSchedule(ContinuousInferenceSchedule):
    """A log time schedule for inference, where time steps are generated by taking the logarithm of a uniform schedule."""

    def __init__(
        self,
        nsteps: int,
        inclusive_end: bool = False,
        min_t: Float = 0,
        padding: Float = 0,
        dilation: Float = 0,
        exponent: Float = -2.0,
        direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the LogInferenceSchedule.

        Returns a log space time schedule.

        Which for 100 steps with default parameters is:
            tensor([0.0000, 0.0455, 0.0889, 0.1303, 0.1699, 0.2077, 0.2439, 0.2783, 0.3113,
                    0.3427, 0.3728, 0.4015, 0.4288, 0.4550, 0.4800, 0.5039, 0.5266, 0.5484,
                    0.5692, 0.5890, 0.6080, 0.6261, 0.6434, 0.6599, 0.6756, 0.6907, 0.7051,
                    0.7188, 0.7319, 0.7444, 0.7564, 0.7678, 0.7787, 0.7891, 0.7991, 0.8086,
                    0.8176, 0.8263, 0.8346, 0.8425, 0.8500, 0.8572, 0.8641, 0.8707, 0.8769,
                    0.8829, 0.8887, 0.8941, 0.8993, 0.9043, 0.9091, 0.9136, 0.9180, 0.9221,
                    0.9261, 0.9299, 0.9335, 0.9369, 0.9402, 0.9434, 0.9464, 0.9492, 0.9520,
                    0.9546, 0.9571, 0.9595, 0.9618, 0.9639, 0.9660, 0.9680, 0.9699, 0.9717,
                    0.9734, 0.9751, 0.9767, 0.9782, 0.9796, 0.9810, 0.9823, 0.9835, 0.9847,
                    0.9859, 0.9870, 0.9880, 0.9890, 0.9899, 0.9909, 0.9917, 0.9925, 0.9933,
                    0.9941, 0.9948, 0.9955, 0.9962, 0.9968, 0.9974, 0.9980, 0.9985, 0.9990,
                    0.9995])

        Args:
            nsteps (int): Number of time steps.
            inclusive_end (bool): If True, include the end value (1.0) in the schedule otherwise ends at <1.0 (default is False).
            min_t (Float): minimum time value defaults to 0.
            padding (Float): padding time value defaults to 0.
            dilation (Float): dilation time value defaults to 0 ie the number of replicates.
            exponent (Float): log space exponent parameter defaults to -2.0. The lower number the more aggressive the acceleration of 0 to 0.9 will be thus having more steps from 0.9 to 1.0.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        super().__init__(nsteps, inclusive_end, min_t, padding, dilation, direction, device)
        if exponent is None:
            raise ValueError("exponent cannot be None for the log schedule")
        if exponent >= 0:
            raise ValueError(f"exponent input must be <0, got {exponent}")
        self.exponent = exponent

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

        if not self.inclusive_end:
            t = 1.0 - torch.logspace(self.exponent, 0, nsteps + 1).flip(0).to(device=device)
            t = t - torch.min(t)
            schedule = t / torch.max(t)
            schedule = schedule[:-1]
        else:
            t = 1.0 - torch.logspace(self.exponent, 0, nsteps).flip(0).to(device=device)
            t = t - torch.min(t)
            schedule = t / torch.max(t)

        if self.min_t > 0:
            schedule = torch.clamp(schedule, min=self.min_t)

        if dilation > 1:
            schedule = schedule.repeat_interleave(dilation)
        if self.padding > 0:
            schedule = torch.cat((schedule, torch.ones(self.padding, device=device)))
        if self.direction == TimeDirection.DIFFUSION:
            schedule = 1 - schedule
        return schedule


class EntropicInferenceSchedule(ContinuousInferenceSchedule):
    """Generates an entropic time schedule.

    It remapping time based on the remaps cumulative information gain provided by a predictor function.
    """

    def __init__(
        self,
        predictor: Callable[[Tensor, Tensor], Tensor],
        x_0_sampler: Callable[[int], Tensor],
        x_1_sampler: Callable[[int], Tensor],
        nsteps: int,
        n_approx_entropy_points: int = 100,
        batch_size: int = 128,
        inclusive_end: bool = False,
        min_t: float = 0,
        direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
        generator: Optional[torch.Generator] = None,
    ):
        """Inspired by the work from Dejan Stancevic, Florian Handke, & Luca Ambrogioni. (2025).

        Entropic Time Schedulers for Generative Diffusion Models.
        https://arxiv.org/abs/2504.13612

        This entropy rate is used to create an optimized, data-dependent time-stepping schedule for the generative process.

        Approach benefits:
        - Can improve inference performance
        - Prevents oversampling of less-informative time steps
        - Prevents undersampling of critical time windows
        - Easily adapted into any model architecture leveraging flow-matching
        - Sample-eficient way to generate data

        Args:
            predictor (Callable[[Tensor, Tensor], Tensor]): A callable (e.g., a function
                or functools.partial) that takes a time tensor `t` and a data
                tensor `x` and returns the predicted vector field `v`.

            x_0_sampler (Callable[[int], Tensor]): A function that takes a batch size
                and returns a tensor of samples from the initial distribution p0.

            x_1_sampler (Callable[[int], Tensor]): A function that takes a batch size
                and returns a tensor of samples from the target distribution p1.

            nsteps (int): The final number of time steps for the inference schedule.

            n_approx_entropy_points (int): The number of points used to approximate the
                cumulative entropy curve. Higher is more accurate but slower.

            batch_size (int): Batch size for calculating divergence at each time step.

            inclusive_end (bool): If True, include 1.0, otherwise end just before.

            min_t (Float): The minimum time value for the schedule.

            direction (Union[TimeDirection, str]): 'UNIFIED' for forward (0->1) or
                'DIFFUSION' for reverse (1->0).

            device (Union[str, torch.device]): The device for computation.

            generator (Optional[torch.Generator]): A PyTorch generator for reproducible
                random number generation.
        """
        super().__init__(nsteps, inclusive_end, min_t=min_t, direction=direction, device=device)
        self.predictor = predictor
        self.x_0_sampler = x_0_sampler
        self.x_1_sampler = x_1_sampler
        self.n_approx_entropy_points = n_approx_entropy_points
        self.batch_size = batch_size
        self.generator = generator

    def _hutchinson_divergence(self, t: Tensor, x: Tensor) -> Tensor:
        """Estimates the divergence of a vector field defined by a model using Hutchinson's method.

        Args:
            t (Tensor): A tensor of time values, shape [B, 1].
            x (Tensor): A tensor of positions, shape [B, D].

        Returns:
            Tensor: The estimated divergence for each sample in the batch, shape [B].
        """
        # Gradients with respect to x
        x = x.detach().requires_grad_(True)

        # random vector from the Rademacher distribution
        if self.generator:
            epsilon = (torch.randint(0, 2, x.shape, generator=self.generator, device=x.device) * 2 - 1).to(x.dtype)
        else:
            epsilon = (torch.randint_like(x, 0, 2) * 2 - 1).to(x.dtype)

        v = self.predictor(t, x)

        # Jacobian-vector product (JVP)
        jvp = torch.autograd.grad(outputs=v, inputs=x, grad_outputs=epsilon, create_graph=False)[0]

        # Divergence estimator
        divergence_est = (jvp * epsilon).view(jvp.shape[0], -1).sum(dim=-1)

        return divergence_est

    def _calculate_entropy_rate(self, t_val: float) -> float:
        """Calculates the mean entropy rate at a specific time t."""
        t_batch = torch.full((self.batch_size, 1), t_val, device=self.device)
        x_0 = self.x_0_sampler(self.batch_size).to(self.device)
        x_1 = self.x_1_sampler(self.batch_size).to(self.device)

        x_t = (1 - t_val) * x_0 + t_val * x_1  ## TODO: This can be modified in the future

        div = self._hutchinson_divergence(t_batch, x_t)
        return div.mean().item()

    def generate_schedule(
        self, nsteps: Optional[int] = None, device: Optional[Union[str, torch.device]] = None
    ) -> Tensor:
        """Generates the entropic time schedule."""
        dev = device if device is not None else self.device
        n_final_steps = nsteps if nsteps is not None else self.nsteps

        # Calculating entropic profile over evaluation points
        standard_time = torch.linspace(0, 1, self.n_approx_entropy_points, device=dev)
        entropy_rates = torch.empty(self.n_approx_entropy_points, device=dev)

        for i, t_val in enumerate(standard_time):
            entropy_rates[i] = self._calculate_entropy_rate(t_val.item())

        entropy_rates = entropy_rates.clamp(min=0)
        cumulative_entropy = torch.cumsum(entropy_rates, dim=0)

        if cumulative_entropy[-1] > 1e-6:
            cumulative_entropy = cumulative_entropy / cumulative_entropy[-1]

        num_interp_points = n_final_steps + 1 if not self.inclusive_end else n_final_steps

        uniform_time = torch.linspace(0, 1, num_interp_points, device=dev)

        # projection of cumulative entropy into a linear space for creating the schedule
        import numpy as np

        entropic_schedule = np.interp(
            uniform_time.cpu().numpy(), cumulative_entropy.cpu().numpy(), standard_time.cpu().numpy()
        )
        schedule = torch.from_numpy(entropic_schedule).to(dtype=torch.float32, device=dev)

        if not self.inclusive_end:
            schedule = schedule[:-1]

        schedule = torch.clamp(schedule, min=self.min_t)
        # Flipping the schedule at the end correctly handles the diffusion direction
        if self.direction == TimeDirection.DIFFUSION:
            schedule = 1.0 - schedule

        return schedule
