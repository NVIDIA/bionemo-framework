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


import math
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor

from bionemo.moco.interpolants.base_interpolant import string_to_enum
from bionemo.moco.schedules.utils import TimeDirection


def log(t, eps=1e-20):
    """Compute the natural logarithm of a tensor, clamping values to avoid numerical instability.

    Args:
        t (Tensor): The input tensor.
        eps (float, optional): The minimum value to clamp the input tensor (default is 1e-20).

    Returns:
        Tensor: The natural logarithm of the input tensor.
    """
    return torch.log(t.clamp(min=eps))


class ContinuousSNRTransform(ABC):
    """A base class for continuous SNR schedules."""

    def __init__(self, direction: TimeDirection):
        """Initialize the DiscreteNoiseSchedule.

        Args:
            direction (TimeDirection): required this defines in which direction the scheduler was built
        """
        self.direction = string_to_enum(direction, TimeDirection)

    def calculate_log_snr(
        self,
        t: Tensor,
        device: Union[str, torch.device] = "cpu",
        synchronize: Optional[TimeDirection] = None,
    ) -> Tensor:
        """Public wrapper to generate the time schedule as a tensor.

        Args:
            t (Tensor): The input tensor representing the time steps, with values ranging from 0 to 1.
            device (Optional[str]): The device to place the schedule on. Defaults to "cpu".
            synchronize (optional[TimeDirection]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction,
                this parameter allows to flip the direction to match the specified one. Defaults to None.

        Returns:
            Tensor: A tensor representing the log signal-to-noise (SNR) ratio for the given time steps.
        """
        if t.max() > 1:
            raise ValueError(f"Invalid value: max continuous time is 1, but got {t.max().item()}")

        if synchronize and self.direction != string_to_enum(synchronize, TimeDirection):
            t = 1 - t
        return self._calculate_log_snr(t, device)

    @abstractmethod
    def _calculate_log_snr(self, t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Generate the log signal-to-noise (SNR) ratio.

        Args:
            t (Tensor): The input tensor representing the time steps.
            device (Optional[str]): The device to place the schedule on. Defaults to "cpu".

        Returns:
            Tensor: A tensor representing the log SNR values for the given time steps.
        """
        pass

    def log_snr_to_alphas_sigmas(self, log_snr: Tensor) -> Tuple[Tensor, Tensor]:
        """Converts log signal-to-noise ratio (SNR) to alpha and sigma values.

        Args:
            log_snr (Tensor): The input log SNR tensor.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the squared root of alpha and sigma values.
        """
        squared_alpha = log_snr.sigmoid()
        squared_sigma = (-log_snr).sigmoid()
        return squared_alpha.sqrt(), squared_sigma.sqrt()

    def derivative(self, t: Tensor, func: Callable) -> Tensor:
        """Compute derivative of a function, it supports bached single variable inputs.

        Args:
            t (Tensor): time variable at which derivatives are taken
            func (Callable): function for derivative calculation

        Returns:
            Tensor: derivative that is detached from the computational graph
        """
        with torch.enable_grad():
            t.requires_grad_(True)
            derivative = torch.autograd.grad(func(t).sum(), t, create_graph=False)[0].detach()
            t.requires_grad_(False)
        return derivative

    def calculate_general_sde_terms(self, t):
        """Compute the general SDE terms for a given time step t.

        Args:
            t (Tensor): The input tensor representing the time step.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the drift term f_t and the diffusion term g_t_2.

        Notes:
            This method computes the drift and diffusion terms of the general SDE, which can be used to simulate the stochastic process.
            The drift term represents the deterministic part of the process, while the diffusion term represents the stochastic part.
        """
        t = t.clone()
        t.requires_grad_(True)

        # Compute log SNR
        log_snr = self.calculate_log_snr(t, device=t.device)

        # Alpha^2 and Sigma^2
        alpha_squared = torch.sigmoid(log_snr)
        sigma_squared = torch.sigmoid(-log_snr)

        # Log Alpha
        log_alpha = 0.5 * torch.log(alpha_squared)

        # Compute derivatives
        log_alpha_deriv = torch.autograd.grad(log_alpha.sum(), t, create_graph=False)[0].detach()
        sigma_squared_deriv = torch.autograd.grad(sigma_squared.sum(), t, create_graph=False)[0].detach()

        # Compute drift and diffusion terms
        f_t = log_alpha_deriv  # Drift term
        g_t_2 = sigma_squared_deriv - 2 * log_alpha_deriv * sigma_squared  # Diffusion term

        return f_t, g_t_2

    def calculate_beta(self, t):
        r"""Compute the drift coefficient for the OU process of the form $dx = -\frac{1}{2} \beta(t) x dt + sqrt(beta(t)) dw_t$.

        beta = d/dt log(alpha**2) = 2 * 1/alpha * d/dt(alpha)

        Args:
            t (Union[float, Tensor]): t in [0, 1]

        Returns:
            Tensor: beta(t)
        """
        t = t.clone()
        t.requires_grad_(True)
        log_snr = self.calculate_log_snr(t, device=t.device)
        alpha = self.calculate_alpha_log_snr(log_snr).detach()
        alpha_deriv_t = self.derivative(t, self.calculate_alpha_t).detach()
        beta = 2.0 * alpha_deriv_t / alpha
        # Chroma has a negative here but when removing the negative we get f = d/dt log (alpha**2) and the step_ode function works as expected
        return beta

    def calculate_alpha_log_snr(self, log_snr: Tensor) -> Tensor:
        """Compute alpha values based on the log SNR.

        Args:
            log_snr (Tensor): The input tensor representing the log signal-to-noise ratio.

        Returns:
            Tensor: A tensor representing the alpha values for the given log SNR.

        Notes:
            This method computes alpha values as the square root of the sigmoid of the log SNR.
        """
        return torch.sigmoid(log_snr).sqrt()

    def calculate_alpha_t(self, t: Tensor) -> Tensor:
        """Compute alpha values based on the log SNR schedule.

        Parameters:
            t (Tensor): The input tensor representing the time steps.

        Returns:
            Tensor: A tensor representing the alpha values for the given time steps.

        Notes:
            This method computes alpha values as the square root of the sigmoid of the log SNR.
        """
        log_snr = self.calculate_log_snr(t, device=t.device)
        alpha = torch.sigmoid(log_snr).sqrt()
        return alpha


class CosineSNRTransform(ContinuousSNRTransform):
    """A cosine SNR schedule.

    Args:
        nu (Optional[Float]): Hyperparameter for the cosine schedule exponent (default is 1.0).
        s (Optional[Float]): Hyperparameter for the cosine schedule shift (default is 0.008).
    """

    def __init__(self, nu: Float = 1.0, s: Float = 0.008):
        """Initialize the CosineNoiseSchedule."""
        self.direction = TimeDirection.DIFFUSION
        self.nu = nu
        self.s = s

    def _calculate_log_snr(self, t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Calculate the log signal-to-noise ratio (SNR) for the cosine noise schedule i.e. -gamma.

        The SNR is the equivalent to alpha_bar**2 / (1 - alpha_bar**2) from DDPM.
        This method computes the log SNR as described in the paper "Improved Denoising Diffusion Probabilistic Models" (https://arxiv.org/pdf/2107.00630).
        Note 1 / (1 + exp(- log_snr)) returns this cosine**2 for alpha_bar**2
        See  https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material and https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/continuous_time_gaussian_diffusion.py

        Args:
            t (Tensor): The input tensor representing the time steps.
            device (str): Device to place the schedule on (default is "cpu").

        Returns:
            Tensor: A tensor representing the log SNR for the given time steps.
        """
        return -log((torch.cos((t**self.nu + self.s) / (1 + self.s) * math.pi * 0.5) ** -2) - 1, eps=1e-5).to(device)


class LinearSNRTransform(ContinuousSNRTransform):
    """A Linear SNR schedule."""

    def __init__(self, min_value: Float = 1.0e-4):
        """Initialize the Linear SNR Transform.

        Args:
            min_value (Float): min vaue of SNR defaults to 1.e-4.
        """
        self.direction = TimeDirection.DIFFUSION
        self.min_value = min_value

    def _calculate_log_snr(self, t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Calculate the log signal-to-noise ratio (SNR) for the cosine noise schedule i.e. -gamma.

        The SNR is the equivalent to alpha_bar**2 / (1 - alpha_bar**2) from DDPM.
        See  https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material and https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/continuous_time_gaussian_diffusion.py

        Args:
            t (Tensor): The input tensor representing the time steps.
            device (Optional[str]): The device to place the schedule on. Defaults to "cpu".

        Returns:
            Tensor: A tensor representing the log SNR for the given time steps.
        """
        # This is equivalanet to the interpolated one from -10 to 9.2
        return -log(torch.expm1(self.min_value + 10 * (t**2))).to(device)


class LinearLogInterpolatedSNRTransform(ContinuousSNRTransform):
    """A Linear Log space interpolated SNR schedule."""

    def __init__(self, min_value: Float = -7.0, max_value=13.5):
        """Initialize the Linear log space interpolated SNR Schedule from Chroma.

        Args:
            min_value (Float): The min log SNR value.
            max_value (Float): the max log SNR value.
        """
        self.direction = TimeDirection.DIFFUSION
        self.min_value = min_value
        self.max_value = max_value

    def _calculate_log_snr(self, t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Calculate the log signal-to-noise ratio (SNR) for the cosine noise schedule i.e. -gamma.

        See https://github.com/generatebio/chroma/blob/929407c605013613941803c6113adefdccaad679/chroma/layers/structure/diffusion.py#L316C23-L316C50

        Args:
            t (Tensor): The input tensor representing the time steps.
            device (Optional[str]): The device to place the schedule on. Defaults to "cpu".

        Returns:
            Tensor: A tensor representing the log SNR for the given time steps.
        """
        log_snr = (1 - t) * self.max_value + t * self.min_value
        return log_snr.to(device)
