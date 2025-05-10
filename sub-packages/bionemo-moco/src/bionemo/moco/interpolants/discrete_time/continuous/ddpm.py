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


import warnings
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Bool, Float
from torch import Tensor

from bionemo.moco.distributions.prior.continuous.gaussian import GaussianPrior
from bionemo.moco.distributions.prior.distribution import PriorDistribution
from bionemo.moco.distributions.time.distribution import TimeDistribution
from bionemo.moco.interpolants.base_interpolant import Interpolant, PredictionType, pad_like, string_to_enum
from bionemo.moco.interpolants.discrete_time.utils import safe_index
from bionemo.moco.schedules.noise.discrete_noise_schedules import DiscreteNoiseSchedule


class DDPM(Interpolant):
    """A Denoising Diffusion Probabilistic Model (DDPM) interpolant.

     -------

    Examples:
    ```python
    >>> import torch
    >>> from bionemo.moco.distributions.prior.continuous.gaussian import GaussianPrior
    >>> from bionemo.moco.distributions.time.uniform import UniformTimeDistribution
    >>> from bionemo.moco.interpolants.discrete_time.continuous.ddpm import DDPM
    >>> from bionemo.moco.schedules.noise.discrete_noise_schedules import DiscreteCosineNoiseSchedule
    >>> from bionemo.moco.schedules.inference_time_schedules import DiscreteLinearInferenceSchedule


    ddpm = DDPM(
        time_distribution = UniformTimeDistribution(discrete_time = True,...),
        prior_distribution = GaussianPrior(...),
        noise_schedule = DiscreteCosineNoiseSchedule(...),
        )
    model = Model(...)

    # Training
    for epoch in range(1000):
        data = data_loader.get(...)
        time = ddpm.sample_time(batch_size)
        noise = ddpm.sample_prior(data.shape)
        xt = ddpm.interpolate(data, noise, time)

        x_pred = model(xt, time)
        loss = ddpm.loss(x_pred, data, time)
        loss.backward()

    # Generation
    x_pred = ddpm.sample_prior(data.shape)
    for t in DiscreteLinearTimeSchedule(...).generate_schedule():
        time = torch.full((batch_size,), t)
        x_hat = model(x_pred, time)
        x_pred = ddpm.step(x_hat, time, x_pred)
    return x_pred

    ```
    """

    def __init__(
        self,
        time_distribution: TimeDistribution,
        prior_distribution: PriorDistribution,
        noise_schedule: DiscreteNoiseSchedule,
        prediction_type: Union[PredictionType, str] = PredictionType.DATA,
        device: Union[str, torch.device] = "cpu",
        last_time_idx: int = 0,
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Initializes the DDPM interpolant.

        Args:
            time_distribution (TimeDistribution): The distribution of time steps, used to sample time points for the diffusion process.
            prior_distribution (PriorDistribution): The prior distribution of the variable, used as the starting point for the diffusion process.
            noise_schedule (DiscreteNoiseSchedule): The schedule of noise, defining the amount of noise added at each time step.
            prediction_type (PredictionType): The type of prediction, either "data" or another type. Defaults to "data".
            device (str): The device on which to run the interpolant, either "cpu" or a CUDA device (e.g. "cuda:0"). Defaults to "cpu".
            last_time_idx (int, optional): The last time index for discrete time. Set to 0 if discrete time is T-1, ..., 0 or 1 if T, ..., 1. Defaults to 0.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
        """
        super().__init__(time_distribution, prior_distribution, device, rng_generator)
        if not isinstance(prior_distribution, GaussianPrior):
            warnings.warn("Prior distribution is not a GaussianPrior, unexpected behavior may occur")
        self.noise_schedule = noise_schedule
        self._initialize_schedules(device)
        self.prediction_type = string_to_enum(prediction_type, PredictionType)
        self._loss_function = nn.MSELoss(reduction="none")
        self.last_time_idx = last_time_idx

    def _initialize_schedules(self, device: Union[str, torch.device] = "cpu"):
        """Sets up the Denoising Diffusion Probabilistic Model (DDPM) equations.

        This method initializes the schedules for the forward and reverse processes of the DDPM. It calculates the
        alphas, betas, and log variances required for the diffusion process.

        Specifically, it computes:

        * `alpha_bar`: the cumulative product of `alpha_t`
        * `alpha_bar_prev`: the previous cumulative product of `alpha_t`
        * `posterior_variance`: the variance of the posterior distribution
        * `posterior_mean_c0_coef` and `posterior_mean_ct_coef`: the coefficients for the posterior mean
        * `log_var`: the log variance of the posterior distribution

        These values are then used to set up the forward and reverse schedules for the DDPM.
        Specifically this is equation (6) (7) from https://arxiv.org/pdf/2006.11239
        """
        if self.noise_schedule is None:
            raise ValueError("noise_schedule cannot be None for DDPM")
        alphas = self.noise_schedule.generate_schedule(device=device)
        betas = 1 - alphas
        log_alpha = torch.log(alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        alpha_bar = alphas_cumprod = torch.exp(log_alpha_bar)
        alpha_bar_prev = alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_mean_c0_coef = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alpha_bar)
        posterior_mean_ct_coef = (1.0 - alpha_bar_prev) * torch.sqrt(alphas) / (1.0 - alpha_bar)
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        posterior_logvar = torch.log(
            torch.nn.functional.pad(posterior_variance[:-1], (1, 0), value=posterior_variance[0].item())
        )
        self._forward_data_schedule = torch.sqrt(alpha_bar)
        self._forward_noise_schedule = torch.sqrt(1 - alpha_bar)
        self._reverse_data_schedule = posterior_mean_c0_coef
        self._reverse_noise_schedule = posterior_mean_ct_coef
        self._log_var = posterior_logvar
        self._alpha_bar = alpha_bar
        self._alpha_bar_prev = alpha_bar_prev
        self._betas = betas
        self._posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    @property
    def forward_data_schedule(self) -> torch.Tensor:
        """Returns the forward data schedule."""
        return self._forward_data_schedule

    @property
    def forward_noise_schedule(self) -> torch.Tensor:
        """Returns the forward noise schedule."""
        return self._forward_noise_schedule

    @property
    def reverse_data_schedule(self) -> torch.Tensor:
        """Returns the reverse data schedule."""
        return self._reverse_data_schedule

    @property
    def reverse_noise_schedule(self) -> torch.Tensor:
        """Returns the reverse noise schedule."""
        return self._reverse_noise_schedule

    @property
    def log_var(self) -> torch.Tensor:
        """Returns the log variance."""
        return self._log_var

    @property
    def alpha_bar(self) -> torch.Tensor:
        """Returns the alpha bar values."""
        return self._alpha_bar

    @property
    def alpha_bar_prev(self) -> torch.Tensor:
        """Returns the previous alpha bar values."""
        return self._alpha_bar_prev

    def interpolate(self, data: Tensor, t: Tensor, noise: Tensor):
        """Get x(t) with given time t from noise and data.

        Args:
            data (Tensor): target
            t (Tensor): time
            noise (Tensor): noise from prior()
        """
        psi = safe_index(self._forward_data_schedule, t - self.last_time_idx, data.device)
        omega = safe_index(self._forward_noise_schedule, t - self.last_time_idx, data.device)
        psi = pad_like(psi, data)
        omega = pad_like(omega, data)
        x_t = data * psi + noise * omega
        return x_t

    def forward_process(self, data: Tensor, t: Tensor, noise: Optional[Tensor] = None):
        """Get x(t) with given time t from noise and data.

        Args:
            data (Tensor): target
            t (Tensor): time
            noise (Tensor, optional): noise from prior(). Defaults to None.
        """
        if noise is None:
            noise = self.sample_prior(data.shape)
        return self.interpolate(data, t, noise)

    def process_data_prediction(self, model_output: Tensor, sample: Tensor, t: Tensor):
        """Converts the model output to a data prediction based on the prediction type.

        This conversion stems from the Progressive Distillation for Fast Sampling of Diffusion Models https://arxiv.org/pdf/2202.00512.
        Given the model output and the sample, we convert the output to a data prediction based on the prediction type.
        The conversion formulas are as follows:
        - For "noise" prediction type: `pred_data = (sample - noise_scale * model_output) / data_scale`
        - For "data" prediction type: `pred_data = model_output`
        - For "v_prediction" prediction type: `pred_data = data_scale * sample - noise_scale * model_output`

        Args:
            model_output (Tensor): The output of the model.
            sample (Tensor): The input sample.
            t (Tensor): The time step.

        Returns:
            The data prediction based on the prediction type.

        Raises:
            ValueError: If the prediction type is not one of "noise", "data", or "v_prediction".
        """
        data_scale = safe_index(self._forward_data_schedule, t - self.last_time_idx, model_output.device)
        noise_scale = safe_index(self._forward_noise_schedule, t - self.last_time_idx, model_output.device)
        data_scale = pad_like(data_scale, model_output)
        noise_scale = pad_like(noise_scale, model_output)
        if self.prediction_type == PredictionType.NOISE:
            pred_data = (sample - noise_scale * model_output) / data_scale
        elif self.prediction_type == PredictionType.DATA:
            pred_data = model_output
        elif self.prediction_type == PredictionType.VELOCITY:
            pred_data = data_scale * sample - noise_scale * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of PredictionType.NOISE, PredictionType.DATA or"
                f" PredictionType.VELOCITY for DDPM."
            )
        return pred_data

    def process_noise_prediction(self, model_output, sample, t):
        """Do the same as process_data_prediction but take the model output and convert to nosie.

        Args:
            model_output: The output of the model.
            sample: The input sample.
            t: The time step.

        Returns:
            The input as noise if the prediction type is "noise".

        Raises:
            ValueError: If the prediction type is not "noise".
        """
        data_scale = safe_index(self._forward_data_schedule, t - self.last_time_idx, model_output.device)
        noise_scale = safe_index(self._forward_noise_schedule, t - self.last_time_idx, model_output.device)
        data_scale = pad_like(data_scale, model_output)
        noise_scale = pad_like(noise_scale, model_output)
        if self.prediction_type == PredictionType.NOISE:
            pred_noise = model_output
        elif self.prediction_type == PredictionType.DATA:
            pred_noise = (sample - data_scale * model_output) / noise_scale
        elif self.prediction_type == PredictionType.VELOCITY:
            pred_data = data_scale * sample - noise_scale * model_output
            pred_noise = (sample - data_scale * pred_data) / noise_scale
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `noise`, `data` or"
                " `v_prediction`  for DDPM."
            )
        return pred_noise

    def calculate_velocity(self, data: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Calculate the velocity term given the data, time step, and noise.

        Args:
            data (Tensor): The input data.
            t (Tensor): The current time step.
            noise (Tensor): The noise term.

        Returns:
            Tensor: The calculated velocity term.
        """
        data_scale = safe_index(self._forward_data_schedule, t - self.last_time_idx, data.device)
        noise_scale = safe_index(self._forward_noise_schedule, t - self.last_time_idx, data.device)
        data_scale = pad_like(data_scale, data)
        noise_scale = pad_like(noise_scale, data)
        v = data_scale * noise - noise_scale * data
        return v

    @torch.no_grad()
    def step(
        self,
        model_out: Tensor,
        t: Tensor,
        xt: Tensor,
        mask: Optional[Tensor] = None,
        center: Bool = False,
        temperature: Float = 1.0,
    ):
        """Do one step integration.

        Args:
        model_out (Tensor): The output of the model.
        t (Tensor): The current time step.
        xt (Tensor): The current data point.
        mask (Optional[Tensor], optional): An optional mask to apply to the data. Defaults to None.
        center (bool, optional): Whether to center the data. Defaults to False.
        temperature (Float, optional): The temperature parameter for low temperature sampling. Defaults to 1.0.

        Note:
        The temperature parameter controls the level of randomness in the sampling process. A temperature of 1.0 corresponds to standard diffusion sampling, while lower temperatures (e.g. 0.5, 0.2) result in less random and more deterministic samples. This can be useful for tasks that require more control over the generation process.

        Note for discrete time we sample from [T-1, ..., 1, 0] for T steps so we sample t = 0 hence the mask.
        For continuous time we start from [1, 1 -dt, ..., dt] for T steps where s = t - 1 when t = 0 i.e dt is then 0

        """
        if mask is not None:
            model_out = model_out * mask.unsqueeze(-1)
        x_hat = self.process_data_prediction(model_out, xt, t)
        psi_r = safe_index(self._reverse_data_schedule, t - self.last_time_idx, x_hat.device)
        omega_r = safe_index(self._reverse_noise_schedule, t - self.last_time_idx, x_hat.device)
        log_var = safe_index(self._log_var, t - self.last_time_idx, x_hat.device)  # self._log_var[t.long()]
        nonzero_mask = (t > self.last_time_idx).float()
        psi_r = pad_like(psi_r, x_hat)
        omega_r = pad_like(omega_r, x_hat)
        log_var = pad_like(log_var, x_hat)
        nonzero_mask = pad_like(nonzero_mask, x_hat)

        mean = psi_r * x_hat + omega_r * xt
        eps = torch.randn_like(mean).to(model_out.device)

        x_next = mean + nonzero_mask * (0.5 * log_var).exp() * eps * temperature
        x_next = self.clean_mask_center(x_next, mask, center)
        return x_next

    def step_noise(
        self,
        model_out: Tensor,
        t: Tensor,
        xt: Tensor,
        mask: Optional[Tensor] = None,
        center: Bool = False,
        temperature: Float = 1.0,
    ):
        """Do one step integration.

        Args:
        model_out (Tensor): The output of the model.
        t (Tensor): The current time step.
        xt (Tensor): The current data point.
        mask (Optional[Tensor], optional): An optional mask to apply to the data. Defaults to None.
        center (bool, optional): Whether to center the data. Defaults to False.
        temperature (Float, optional): The temperature parameter for low temperature sampling. Defaults to 1.0.

        Note:
        The temperature parameter controls the level of randomness in the sampling process.
        A temperature of 1.0 corresponds to standard diffusion sampling, while lower temperatures (e.g. 0.5, 0.2)
        result in less random and more deterministic samples. This can be useful for tasks
        that require more control over the generation process.

        Note for discrete time we sample from [T-1, ..., 1, 0] for T steps so we sample t = 0 hence the mask.
        For continuous time we start from [1, 1 -dt, ..., dt] for T steps where s = t - 1 when t = 0 i.e dt is then 0

        """
        if mask is not None:
            model_out = model_out * mask.unsqueeze(-1)
        eps_hat = self.process_noise_prediction(model_out, xt, t)
        beta_t = safe_index(self._betas, t - self.last_time_idx, model_out.device)
        recip_sqrt_alpha_t = torch.sqrt(1 / (1 - beta_t))
        eps_factor = (
            safe_index(self._betas, t - self.last_time_idx, model_out.device)
            / (1 - safe_index(self._alpha_bar, t - self.last_time_idx, model_out.device)).sqrt()
        )
        var = safe_index(self._posterior_variance, t - self.last_time_idx, model_out.device)  # self._log_var[t.long()]

        nonzero_mask = (t > self.last_time_idx).float()
        nonzero_mask = pad_like(nonzero_mask, model_out)
        eps_factor = pad_like(eps_factor, xt)
        recip_sqrt_alpha_t = pad_like(recip_sqrt_alpha_t, xt)
        var = pad_like(var, xt)

        x_next = (
            recip_sqrt_alpha_t * (xt - eps_factor * eps_hat)
            + nonzero_mask * var.sqrt() * torch.randn_like(eps_hat).to(model_out.device) * temperature
        )
        x_next = self.clean_mask_center(x_next, mask, center)
        return x_next

    def score(self, x_hat: Tensor, xt: Tensor, t: Tensor):
        """Converts the data prediction to the estimated score function.

        Args:
            x_hat (Tensor): The predicted data point.
            xt (Tensor): The current data point.
            t (Tensor): The time step.

        Returns:
            The estimated score function.
        """
        alpha = safe_index(self._forward_data_schedule, t - self.last_time_idx, x_hat.device)
        beta = safe_index(self._forward_noise_schedule, t - self.last_time_idx, x_hat.device)
        alpha = pad_like(alpha, x_hat)
        beta = pad_like(beta, x_hat)
        score = alpha * x_hat - xt
        score = score / (beta * beta)
        return score

    def step_ddim(
        self,
        model_out: Tensor,
        t: Tensor,
        xt: Tensor,
        mask: Optional[Tensor] = None,
        eta: Float = 0.0,
        center: Bool = False,
    ):
        """Do one step of DDIM sampling.

        Args:
            model_out (Tensor): output of the model
            t (Tensor): current time step
            xt (Tensor): current data point
            mask (Optional[Tensor], optional): mask for the data point. Defaults to None.
            eta (Float, optional): DDIM sampling parameter. Defaults to 0.0.
            center (Bool, optional): whether to center the data point. Defaults to False.
        """
        if mask is not None:
            model_out = model_out * mask.unsqueeze(-1)
        data_pred = self.process_data_prediction(model_out, xt, t)
        noise_pred = self.process_noise_prediction(model_out, xt, t)
        eps = torch.randn_like(data_pred).to(model_out.device)
        sigma = (
            eta
            * torch.sqrt((1 - self._alpha_bar_prev) / (1 - self._alpha_bar))
            * torch.sqrt(1 - self._alpha_bar / self._alpha_bar_prev)
        )
        sigma_t = safe_index(sigma, t - self.last_time_idx, model_out.device)
        psi_r = safe_index(torch.sqrt(self._alpha_bar_prev), t - self.last_time_idx, model_out.device)
        omega_r = safe_index(torch.sqrt(1 - self._alpha_bar_prev - sigma**2), t - self.last_time_idx, model_out.device)
        sigma_t = pad_like(sigma_t, model_out)
        psi_r = pad_like(psi_r, model_out)
        omega_r = pad_like(omega_r, model_out)
        mean = data_pred * psi_r + omega_r * noise_pred
        x_next = mean + sigma_t * eps
        x_next = self.clean_mask_center(x_next, mask, center)
        return x_next

    def set_loss_weight_fn(self, fn):
        """Sets the loss_weight attribute of the instance to the given function.

        Args:
            fn: The function to set as the loss_weight attribute. This function should take three arguments: raw_loss, t, and weight_type.
        """
        self.loss_weight = fn

    def loss_weight(self, raw_loss: Tensor, t: Optional[Tensor], weight_type: str) -> Tensor:
        """Calculates the weight for the loss based on the given weight type.

        These data_to_noise loss weights is derived in Equation (9) of https://arxiv.org/pdf/2202.00512.

        Args:
            raw_loss (Tensor): The raw loss calculated from the model prediction and target.
            t (Tensor): The time step.
            weight_type (str): The type of weight to use. Can be "ones" or "data_to_noise" or "noise_to_data".

        Returns:
            Tensor: The weight for the loss.

        Raises:
            ValueError: If the weight type is not recognized.
        """
        if weight_type == "ones":
            schedule = torch.ones_like(raw_loss).to(raw_loss.device)
        elif weight_type == "data_to_noise":
            if t is None:
                raise ValueError("Time cannot be None when using the data_to_noise loss weight")
            schedule = (safe_index(self._forward_data_schedule, t - self.last_time_idx, raw_loss.device) ** 2) / (
                safe_index(self._forward_noise_schedule, t - self.last_time_idx, raw_loss.device) ** 2
            )
            schedule = pad_like(schedule, raw_loss)
        elif weight_type == "noise_to_data":
            if t is None:
                raise ValueError("Time cannot be None when using the data_to_noise loss weight")
            schedule = (safe_index(self._forward_noise_schedule, t - self.last_time_idx, raw_loss.device) ** 2) / (
                safe_index(self._forward_data_schedule, t - self.last_time_idx, raw_loss.device) ** 2
            )
            schedule = pad_like(schedule, raw_loss)
        else:
            raise ValueError("Invalid loss weight keyword")
        return schedule

    def loss(
        self,
        model_pred: Tensor,
        target: Tensor,
        t: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weight_type: Literal["ones", "data_to_noise", "noise_to_data"] = "ones",
    ):
        """Calculate the loss given the model prediction, data sample, and time.

        The default weight_type is "ones" meaning no change / multiplying by all ones.
        data_to_noise is available to scale the data MSE loss into the appropriate loss that is theoretically equivalent
        to noise prediction. noise_to_data is provided for a similar reason for completeness.

        Args:
            model_pred (Tensor): The predicted output from the model.
            target (Tensor): The target output for the model prediction.
            t (Tensor): The time at which the loss is calculated.
            mask (Optional[Tensor], optional): The mask for the data point. Defaults to None.
            weight_type (Literal["ones", "data_to_noise", "noise_to_data"]): The type of weight to use for the loss. Defaults to "ones".

        Returns:
            Tensor: The calculated loss batch tensor.
        """
        raw_loss = self._loss_function(model_pred, target)
        if weight_type != "ones":
            update_weight = self.loss_weight(raw_loss, t, weight_type)
            loss = raw_loss * update_weight
        else:
            loss = raw_loss
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            n_elem = torch.sum(mask, dim=-1)
            loss = torch.sum(loss, dim=tuple(range(1, raw_loss.ndim))) / n_elem
        else:
            loss = torch.sum(loss, dim=tuple(range(1, raw_loss.ndim))) / model_pred.size(1)
        return loss
