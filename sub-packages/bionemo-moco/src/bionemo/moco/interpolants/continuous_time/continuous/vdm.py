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
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor

from bionemo.moco.distributions.prior.continuous.gaussian import GaussianPrior
from bionemo.moco.distributions.prior.distribution import PriorDistribution
from bionemo.moco.distributions.time.distribution import TimeDistribution
from bionemo.moco.interpolants.base_interpolant import Interpolant, PredictionType, pad_like, string_to_enum
from bionemo.moco.schedules.noise.continuous_snr_transforms import ContinuousSNRTransform


class VDM(Interpolant):
    """A Variational Diffusion Models (VDM) interpolant.

     -------

    Examples:
    ```python
    >>> import torch
    >>> from bionemo.moco.distributions.prior.continuous.gaussian import GaussianPrior
    >>> from bionemo.moco.distributions.time.uniform import UniformTimeDistribution
    >>> from bionemo.moco.interpolants.discrete_time.continuous.vdm import VDM
    >>> from bionemo.moco.schedules.noise.continuous_snr_transforms import CosineSNRTransform
    >>> from bionemo.moco.schedules.inference_time_schedules import LinearInferenceSchedule


    vdm = VDM(
        time_distribution = UniformTimeDistribution(...),
        prior_distribution = GaussianPrior(...),
        noise_schedule = CosineSNRTransform(...),
        )
    model = Model(...)

    # Training
    for epoch in range(1000):
        data = data_loader.get(...)
        time = vdm.sample_time(batch_size)
        noise = vdm.sample_prior(data.shape)
        xt = vdm.interpolate(data, noise, time)

        x_pred = model(xt, time)
        loss = vdm.loss(x_pred, data, time)
        loss.backward()

    # Generation
    x_pred = vdm.sample_prior(data.shape)
    for t in LinearInferenceSchedule(...).generate_schedule():
        time = torch.full((batch_size,), t)
        x_hat = model(x_pred, time)
        x_pred = vdm.step(x_hat, time, x_pred)
    return x_pred

    ```
    """

    def __init__(
        self,
        time_distribution: TimeDistribution,
        prior_distribution: PriorDistribution,
        noise_schedule: ContinuousSNRTransform,
        prediction_type: Union[PredictionType, str] = PredictionType.DATA,
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Initializes the DDPM interpolant.

        Args:
            time_distribution (TimeDistribution): The distribution of time steps, used to sample time points for the diffusion process.
            prior_distribution (PriorDistribution): The prior distribution of the variable, used as the starting point for the diffusion process.
            noise_schedule (ContinuousSNRTransform): The schedule of noise, defining the amount of noise added at each time step.
            prediction_type (PredictionType, optional): The type of prediction, either "data" or another type. Defaults to "data".
            device (str, optional): The device on which to run the interpolant, either "cpu" or a CUDA device (e.g. "cuda:0"). Defaults to "cpu".
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
        """
        super().__init__(time_distribution, prior_distribution, device, rng_generator)
        if not isinstance(prior_distribution, GaussianPrior):
            warnings.warn("Prior distribution is not a GaussianPrior, unexpected behavior may occur")
        self.noise_schedule = noise_schedule
        self.prediction_type = string_to_enum(prediction_type, PredictionType)
        self._loss_function = nn.MSELoss(reduction="none")

    def interpolate(self, data: Tensor, t: Tensor, noise: Tensor):
        """Get x(t) with given time t from noise and data.

        Args:
            data (Tensor): target
            t (Tensor): time
            noise (Tensor): noise from prior()
        """
        log_snr = self.noise_schedule.calculate_log_snr(t, device=self.device)
        psi, omega = self.noise_schedule.log_snr_to_alphas_sigmas(log_snr)
        psi = pad_like(psi, data)
        omega = pad_like(omega, data)
        x_t = data * psi + noise * omega
        return x_t

    def forward_process(self, data: Tensor, t: Tensor, noise: Optional[Tensor] = None):
        """Get x(t) with given time t from noise and data.

        Args:
            data (Tensor): target
            t (Tensor): time
            noise (Tensor, optional): noise from prior(). Defaults to None
        """
        if noise is None:
            noise = self.sample_prior(data.shape)
        return self.interpolate(data, t, noise)

    def process_data_prediction(self, model_output: Tensor, sample, t):
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
        log_snr = self.noise_schedule.calculate_log_snr(t, device=self.device)
        data_scale, noise_scale = self.noise_schedule.log_snr_to_alphas_sigmas(log_snr)
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
                f" PredictionType.VELOCITY for vdm."
            )
        return pred_data

    def process_noise_prediction(self, model_output: Tensor, sample: Tensor, t: Tensor):
        """Do the same as process_data_prediction but take the model output and convert to nosie.

        Args:
            model_output (Tensor): The output of the model.
            sample (Tensor): The input sample.
            t (Tensor): The time step.

        Returns:
            The input as noise if the prediction type is "noise".

        Raises:
            ValueError: If the prediction type is not "noise".
        """
        log_snr = self.noise_schedule.calculate_log_snr(t, device=self.device)
        data_scale, noise_scale = self.noise_schedule.log_snr_to_alphas_sigmas(log_snr)
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
                " `v_prediction`  for vdm."
            )
        return pred_noise

    def step(
        self,
        model_out: Tensor,
        t: Tensor,
        xt: Tensor,
        dt: Tensor,
        mask: Optional[Tensor] = None,
        center: Bool = False,
        temperature: Float = 1.0,
    ):
        """Do one step integration.

        Args:
            model_out (Tensor): The output of the model.
            xt (Tensor): The current data point.
            t (Tensor): The current time step.
            dt (Tensor): The time step increment.
            mask (Optional[Tensor], optional): An optional mask to apply to the data. Defaults to None.
            center (bool): Whether to center the data. Defaults to False.
            temperature (Float): The temperature parameter for low temperature sampling. Defaults to 1.0.

        Note:
            The temperature parameter controls the trade off between diversity and sample quality.
            Decreasing the temperature sharpens the sampling distribtion to focus on more likely samples.
            The impact of low temperature sampling must be ablated analytically.
        """
        if mask is not None:
            model_out = model_out * mask.unsqueeze(-1)
        x_hat = self.process_data_prediction(model_out, xt, t)

        log_snr = self.noise_schedule.calculate_log_snr(t, device=self.device)
        alpha_t, sigma_t = self.noise_schedule.log_snr_to_alphas_sigmas(log_snr)

        if (t - dt < 0).any():
            raise ValueError(
                "Error in inference schedule: t - dt < 0. Please ensure that your inference time schedule has shape T with the final t = dt to make s = 0"
            )

        log_snr_s = self.noise_schedule.calculate_log_snr(t - dt, device=self.device)
        alpha_s, sigma_s = self.noise_schedule.log_snr_to_alphas_sigmas(log_snr_s)
        sigma_s_2 = sigma_s * sigma_s
        sigma_t_2 = sigma_t * sigma_t
        alpha_t_s = alpha_t / alpha_s
        sigma_2_t_s = -torch.expm1(F.softplus(-log_snr_s) - F.softplus(-log_snr))  # Equation 63

        omega_r = alpha_t_s * sigma_s_2 / sigma_t_2  # Equation 28
        psi_r = alpha_s * sigma_2_t_s / sigma_t_2
        std = sigma_2_t_s.sqrt() * sigma_s / sigma_t
        nonzero_mask = (
            t > 0
        ).float()  # based on the time this is always just ones. can leave for now to see if ever want to take extra step and only grab mean

        psi_r = pad_like(psi_r, x_hat)
        omega_r = pad_like(omega_r, x_hat)
        std = pad_like(std, x_hat)
        nonzero_mask = pad_like(nonzero_mask, x_hat)

        mean = psi_r * x_hat + omega_r * xt
        eps = torch.randn_like(mean).to(model_out.device)
        x_next = mean + nonzero_mask * std * eps * temperature
        x_next = self.clean_mask_center(x_next, mask, center)
        return x_next

    def score(self, x_hat: Tensor, xt: Tensor, t: Tensor):
        """Converts the data prediction to the estimated score function.

        Args:
            x_hat (tensor): The predicted data point.
            xt (Tensor): The current data point.
            t (Tensor): The time step.

        Returns:
            The estimated score function.
        """
        log_snr = self.noise_schedule.calculate_log_snr(t, device=self.device)
        psi, omega = self.noise_schedule.log_snr_to_alphas_sigmas(log_snr)
        psi = pad_like(psi, x_hat)
        omega = pad_like(omega, x_hat)
        score = psi * x_hat - xt
        score = score / (omega * omega)
        return score

    def step_ddim(
        self,
        model_out: Tensor,
        t: Tensor,
        xt: Tensor,
        dt: Tensor,
        mask: Optional[Tensor] = None,
        eta: Float = 0.0,
        center: Bool = False,
    ):
        """Do one step of DDIM sampling.

        From the ddpm equations alpha_bar = alpha**2 and  1 - alpha**2 = sigma**2

        Args:
            model_out (Tensor): output of the model
            t (Tensor): current time step
            xt (Tensor): current data point
            dt (Tensor): The time step increment.
            mask (Optional[Tensor], optional): mask for the data point. Defaults to None.
            eta (Float, optional): DDIM sampling parameter. Defaults to 0.0.
            center (Bool, optional): whether to center the data point. Defaults to False.
        """
        if mask is not None:
            model_out = model_out * mask.unsqueeze(-1)
        data_pred = self.process_data_prediction(model_out, xt, t)
        noise_pred = self.process_noise_prediction(model_out, xt, t)
        eps = torch.randn_like(data_pred).to(model_out.device)
        log_snr = self.noise_schedule.calculate_log_snr(t, device=self.device)
        squared_alpha = log_snr.sigmoid()
        squared_sigma = (-log_snr).sigmoid()
        log_snr_prev = self.noise_schedule.calculate_log_snr(t - dt, device=self.device)
        squared_alpha_prev = log_snr_prev.sigmoid()
        squared_sigma_prev = (-log_snr_prev).sigmoid()
        sigma_t_2 = squared_sigma_prev / squared_sigma * (1 - squared_alpha / squared_alpha_prev)
        psi_r = torch.sqrt(squared_alpha_prev)
        omega_r = torch.sqrt(1 - squared_alpha_prev - eta * eta * sigma_t_2)

        sigma_t_2 = pad_like(sigma_t_2, model_out)
        psi_r = pad_like(psi_r, model_out)
        omega_r = pad_like(omega_r, model_out)

        mean = data_pred * psi_r + omega_r * noise_pred
        x_next = mean + eta * sigma_t_2.sqrt() * eps
        x_next = self.clean_mask_center(x_next, mask, center)
        return x_next

    def set_loss_weight_fn(self, fn: Callable):
        """Sets the loss_weight attribute of the instance to the given function.

        Args:
            fn: The function to set as the loss_weight attribute. This function should take three arguments: raw_loss, t, and weight_type.
        """
        self.loss_weight = fn

    def loss_weight(self, raw_loss: Tensor, t: Tensor, weight_type: str, dt: Float = 0.001) -> Tensor:
        """Calculates the weight for the loss based on the given weight type.

        This function computes the loss weight according to the specified `weight_type`.
        The available weight types are:
        - "ones": uniform weight of 1.0
        - "data_to_noise": derived from Equation (9) of https://arxiv.org/pdf/2202.00512
        - "variational_objective": based on the variational objective, see https://arxiv.org/pdf/2202.00512

        Args:
            raw_loss (Tensor): The raw loss calculated from the model prediction and target.
            t (Tensor): The time step.
            weight_type (str): The type of weight to use. Can be "ones", "data_to_noise", or "variational_objective".
            dt (Float, optional): The time step increment. Defaults to 0.001.

        Returns:
            Tensor: The weight for the loss.

        Raises:
            ValueError: If the weight type is not recognized.
        """
        if weight_type == "ones":
            schedule = torch.ones_like(raw_loss).to(raw_loss.device)
        elif weight_type == "data_to_noise":  #
            log_snr = self.noise_schedule.calculate_log_snr(t, device=self.device)
            psi, omega = self.noise_schedule.log_snr_to_alphas_sigmas(log_snr)
            schedule = (psi**2) / (omega**2)
            for _ in range(raw_loss.ndim - 1):
                schedule = schedule.unsqueeze(-1)
        elif weight_type == "variational_objective":
            # (1-SNR(t-1)/SNR(t)),
            snr = torch.exp(self.noise_schedule.calculate_log_snr(t, device=self.device))
            snr_m1 = torch.exp(self.noise_schedule.calculate_log_snr(t - dt, device=self.device))
            schedule = 1 - snr_m1 / snr
            for _ in range(raw_loss.ndim - 1):
                schedule = schedule.unsqueeze(-1)
        else:
            raise ValueError("Invalid loss weight keyword")
        return schedule

    def loss(
        self,
        model_pred: Tensor,
        target: Tensor,
        t: Tensor,
        dt: Optional[Float] = 0.001,
        mask: Optional[Tensor] = None,
        weight_type: str = "ones",
    ):
        """Calculates the loss given the model prediction, target, and time.

        Args:
            model_pred (Tensor): The predicted output from the model.
            target (Tensor): The target output for the model prediction.
            t (Tensor): The time at which the loss is calculated.
            dt (Optional[Float], optional): The time step increment. Defaults to 0.001.
            mask (Optional[Tensor], optional): The mask for the data point. Defaults to None.
            weight_type (str, optional): The type of weight to use for the loss. Can be "ones", "data_to_noise", or "variational_objective". Defaults to "ones".

        Returns:
            Tensor: The calculated loss batch tensor.
        """
        raw_loss = self._loss_function(model_pred, target)
        update_weight = self.loss_weight(raw_loss, t, weight_type, dt)
        loss = raw_loss * update_weight
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            n_elem = torch.sum(mask, dim=-1)
            loss = torch.sum(loss, dim=tuple(range(1, raw_loss.ndim))) / n_elem
        else:
            loss = torch.sum(loss, dim=tuple(range(1, raw_loss.ndim))) / model_pred.size(1)
        return loss

    def step_hybrid_sde(
        self,
        model_out: Tensor,
        t: Tensor,
        xt: Tensor,
        dt: Tensor,
        mask: Optional[Tensor] = None,
        center: Bool = False,
        temperature: Float = 1.0,
        equilibrium_rate: Float = 0.0,
    ) -> Tensor:
        """Do one step integration of Hybrid Langevin-Reverse Time SDE.

        See section B.3 page 37 https://www.biorxiv.org/content/10.1101/2022.12.01.518682v1.full.pdf.
        and https://github.com/generatebio/chroma/blob/929407c605013613941803c6113adefdccaad679/chroma/layers/structure/diffusion.py#L730

        Args:
            model_out (Tensor): The output of the model.
            xt (Tensor): The current data point.
            t (Tensor): The current time step.
            dt (Tensor): The time step increment.
            mask (Optional[Tensor], optional): An optional mask to apply to the data. Defaults to None.
            center (bool, optional): Whether to center the data. Defaults to False.
            temperature (Float, optional): The temperature parameter for low temperature sampling. Defaults to 1.0.
            equilibrium_rate (Float, optional): The rate of Langevin equilibration.  Scales the amount of Langevin dynamics per unit time. Best values are in the range [1.0, 5.0]. Defaults to 0.0.

        Note:
        For all step functions that use the SDE formulation its important to note that we are moving backwards in time which corresponds to an apparent sign change.
        A clear example can be seen in slide 29 https://ernestryu.com/courses/FM/diffusion1.pdf.
        """
        if mask is not None:
            model_out = model_out * mask.unsqueeze(-1)
        x_hat = self.process_data_prediction(model_out, xt, t)
        log_snr = self.noise_schedule.calculate_log_snr(t, device=self.device)
        alpha, sigma = self.noise_schedule.log_snr_to_alphas_sigmas(log_snr)
        # Schedule coeffiecients
        beta = self.noise_schedule.calculate_beta(t)
        inverse_temperature = 1 / temperature  # lambda_0
        langevin_factor = equilibrium_rate
        # Temperature coefficients
        lambda_t = (
            inverse_temperature * (sigma.pow(2) + alpha.pow(2)) / (inverse_temperature * sigma.pow(2) + alpha.pow(2))
        )
        # langevin_isothermal = True
        lambda_langevin = inverse_temperature  # if langevin_isothermal else lambda_t

        score_scale_t = lambda_t + lambda_langevin * langevin_factor / 2.0

        eps = torch.randn_like(x_hat).to(model_out.device)
        score = self.score(x_hat, xt, t)
        beta = pad_like(beta, model_out)
        score_scale_t = pad_like(score_scale_t, model_out)

        gT = beta * ((-1 / 2) * xt - score_scale_t * score)
        gW = torch.sqrt((1.0 + langevin_factor) * beta.abs()) * eps

        x_next = xt + dt * gT + dt.sqrt() * gW
        x_next = self.clean_mask_center(x_next, mask, center)
        return x_next

    def step_ode(
        self,
        model_out: Tensor,
        t: Tensor,
        xt: Tensor,
        dt: Tensor,
        mask: Optional[Tensor] = None,
        center: Bool = False,
        temperature: Float = 1.0,
    ) -> Tensor:
        """Do one step integration of ODE.

        See section B page 36 https://www.biorxiv.org/content/10.1101/2022.12.01.518682v1.full.pdf.
        and https://github.com/generatebio/chroma/blob/929407c605013613941803c6113adefdccaad679/chroma/layers/structure/diffusion.py#L730

        Args:
            model_out (Tensor): The output of the model.
            xt (Tensor): The current data point.
            t (Tensor): The current time step.
            dt (Tensor): The time step increment.
            mask (Optional[Tensor], optional): An optional mask to apply to the data. Defaults to None.
            center (bool, optional): Whether to center the data. Defaults to False.
            temperature (Float, optional): The temperature parameter for low temperature sampling. Defaults to 1.0.
        """
        if mask is not None:
            model_out = model_out * mask.unsqueeze(-1)
        x_hat = self.process_data_prediction(model_out, xt, t)
        log_snr = self.noise_schedule.calculate_log_snr(t, device=self.device)
        alpha, sigma = self.noise_schedule.log_snr_to_alphas_sigmas(log_snr)
        # Schedule coeffiecients
        beta = self.noise_schedule.calculate_beta(t)
        inverse_temperature = 1 / temperature
        # Temperature coefficients
        lambda_t = (
            inverse_temperature * (sigma.pow(2) + alpha.pow(2)) / (inverse_temperature * sigma.pow(2) + alpha.pow(2))
        )

        score = self.score(x_hat, xt, t)
        beta = pad_like(beta, model_out)
        lambda_t = pad_like(lambda_t, model_out)

        gT = (-1 / 2) * beta * (xt + lambda_t * score)

        x_next = xt + gT * dt
        x_next = self.clean_mask_center(x_next, mask, center)
        return x_next
