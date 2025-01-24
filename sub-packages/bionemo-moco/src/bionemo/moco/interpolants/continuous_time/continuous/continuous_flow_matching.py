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


from typing import Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Bool, Float
from torch import Tensor

from bionemo.moco.distributions.prior.continuous.gaussian import GaussianPrior
from bionemo.moco.distributions.prior.distribution import PriorDistribution
from bionemo.moco.distributions.time.distribution import TimeDistribution
from bionemo.moco.interpolants.base_interpolant import Interpolant, PredictionType, pad_like, string_to_enum
from bionemo.moco.interpolants.batch_augmentation import BatchAugmentation
from bionemo.moco.interpolants.continuous_time.continuous.optimal_transport.ot_types import OptimalTransportType


class ContinuousFlowMatcher(Interpolant):
    """A Continuous Flow Matching interpolant.

     -------

    Examples:
    ```python
    >>> import torch
    >>> from bionemo.moco.distributions.prior.continuous.gaussian import GaussianPrior
    >>> from bionemo.moco.distributions.time.uniform import UniformTimeDistribution
    >>> from bionemo.moco.interpolants.continuous_time.continuous.continuous_flow_matching import ContinuousFlowMatcher
    >>> from bionemo.moco.schedules.inference_time_schedules import LinearInferenceSchedule

    flow_matcher = ContinuousFlowMatcher(
        time_distribution = UniformTimeDistribution(...),
        prior_distribution = GaussianPrior(...),
        )
    model = Model(...)

    # Training
    for epoch in range(1000):
        data = data_loader.get(...)
        time = flow_matcher.sample_time(batch_size)
        noise = flow_matcher.sample_prior(data.shape)
        data, time, noise = flow_matcher.apply_ot(noise, data) # Optional, only for OT
        xt = flow_matcher.interpolate(data, time, noise)
        flow = flow_matcher.calculate_target(data, noise)

        u_pred = model(xt, time)
        loss = flow_matcher.loss(u_pred, flow)
        loss.backward()

    # Generation
    x_pred = flow_matcher.sample_prior(data.shape)
    inference_sched = LinearInferenceSchedule(...)
    for t in inference_sched.generate_schedule():
        time = inference_sched.pad_time(x_pred.shape[0], t)
        u_hat = model(x_pred, time)
        x_pred = flow_matcher.step(u_hat, x_pred, time)
    return x_pred

    ```
    """

    def __init__(
        self,
        time_distribution: TimeDistribution,
        prior_distribution: PriorDistribution,
        prediction_type: Union[PredictionType, str] = PredictionType.DATA,
        sigma: Float = 0,
        ot_type: Optional[Union[OptimalTransportType, str]] = None,
        ot_num_threads: int = 1,
        data_scale: Float = 1.0,
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
        eps: Float = 1e-5,
    ):
        """Initializes the Continuous Flow Matching interpolant.

        Args:
            time_distribution (TimeDistribution): The distribution of time steps, used to sample time points for the diffusion process.
            prior_distribution (PriorDistribution): The prior distribution of the variable, used as the starting point for the diffusion process.
            prediction_type (PredictionType, optional): The type of prediction, either "flow" or another type. Defaults to PredictionType.DATA.
            sigma (Float, optional): The standard deviation of the Gaussian noise added to the interpolated data. Defaults to 0.
            ot_type (Optional[Union[OptimalTransportType, str]], optional): The type of optimal transport, if applicable. Defaults to None.
            ot_num_threads:  Number of threads to use for OT solver. If "max", uses the maximum number of threads. Default is 1.
            data_scale (Float, optional): The scale factor for the data. Defaults to 1.0.
            device (Union[str, torch.device], optional): The device on which to run the interpolant, either "cpu" or a CUDA device (e.g. "cuda:0"). Defaults to "cpu".
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
            eps: Small float to prevent divide by zero
        """
        super().__init__(time_distribution, prior_distribution, device, rng_generator)
        self.prediction_type = string_to_enum(prediction_type, PredictionType)
        self.sigma = sigma
        self.ot_type = ot_type
        self.data_scale = data_scale
        self.eps = eps
        if data_scale <= 0:
            raise ValueError("Data Scale must be > 0")
        if ot_type is not None:
            self.ot_type = ot_type = string_to_enum(ot_type, OptimalTransportType)
            self.ot_sampler = self._build_ot_sampler(method_type=ot_type, num_threads=ot_num_threads)
        self._loss_function = nn.MSELoss(reduction="none")

    def _build_ot_sampler(self, method_type: OptimalTransportType, num_threads: int = 1):
        """Build the optimal transport sampler for the given optimal transport type.

        Args:
            method_type (OptimalTransportType): The type of augmentation.
            num_threads (int): The number of threads to use for the OT sampler, default to 1.

        Returns:
            The augmentation object.
        """
        return BatchAugmentation(self.device, num_threads).create(method_type)

    def apply_ot(self, x0: Tensor, x1: Tensor, mask: Optional[Tensor] = None, **kwargs) -> tuple:
        """Sample and apply the optimal transport plan between batched (and masked) x0 and x1.

        Args:
            x0 (Tensor): shape (bs, *dim), noise from source minibatch.
            x1 (Tensor): shape (bs, *dim), data from source minibatch.
            mask (Optional[Tensor], optional): mask to apply to the output, shape (batchsize, nodes), if not provided no mask is applied. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to self.ot_sampler.apply_ot or handled within this method.


        Returns:
            Tuple: tuple of 2 tensors, represents the noise and data samples following OT plan pi.
        """
        if self.ot_sampler is None:
            raise ValueError("Optimal Transport Sampler is not defined")
        return self.ot_sampler.apply_ot(x0, x1, mask=mask, **kwargs)

    def undo_scale_data(self, data: Tensor) -> Tensor:
        """Downscale the input data by the data scale factor.

        Args:
            data (Tensor): The input data to downscale.

        Returns:
            The downscaled data.
        """
        return 1 / self.data_scale * data

    def scale_data(self, data: Tensor) -> Tensor:
        """Upscale the input data by the data scale factor.

        Args:
            data (Tensor): The input data to upscale.

        Returns:
            The upscaled data.
        """
        return self.data_scale * data

    def interpolate(self, data: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Get x_t with given time t from noise (x_0) and data (x_1).

        Currently, we use the linear interpolation as defined in:
            1. Rectified flow: https://arxiv.org/abs/2209.03003.
            2. Conditional flow matching: https://arxiv.org/abs/2210.02747 (called conditional optimal transport).

        Args:
            noise (Tensor): noise from prior(), shape (batchsize, nodes, features)
            t (Tensor): time, shape (batchsize)
            data (Tensor): target, shape (batchsize, nodes, features)
        """
        assert data.size() == noise.size()
        # Expand t to the same shape as noise: ones([b,n,f]) * t([b,1,1])
        t = pad_like(t, data)
        # Calculate x_t as the linear interpolation between noise and data
        x_t = data * t + noise * (1.0 - t)
        # Add Gaussian Noise
        if self.sigma > 0:
            x_t += self.sigma * torch.randn(x_t.shape, device=x_t.device, generator=self.rng_generator)
        return x_t

    def calculate_target(self, data: Tensor, noise: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Get the target vector field at time t.

        Args:
            noise (Tensor): noise from prior(), shape (batchsize, nodes, features)
            data (Tensor): target, shape (batchsize, nodes, features)
            mask (Optional[Tensor], optional): mask to apply to the output, shape (batchsize, nodes), if not provided no mask is applied. Defaults to None.

        Returns:
            Tensor: The target vector field at time t.
        """
        assert data.size() == noise.size()
        # Calculate the target vector field u_t(x_t|x_1) as the difference between data and noise because t~[0,1]
        if self.prediction_type == PredictionType.VELOCITY:
            u_t = data - noise
        elif self.prediction_type == PredictionType.DATA:
            u_t = data
        else:
            raise ValueError(
                f"Given prediction_type {self.prediction_type} is not supproted for Continuous Flow Matching."
            )
        if mask is not None:
            u_t = u_t * mask.unsqueeze(-1)
        return u_t

    def process_vector_field_prediction(
        self,
        model_output: Tensor,
        xt: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        """Process the model output based on the prediction type to calculate vecotr field.

        Args:
            model_output (Tensor): The output of the model.
            xt (Tensor): The input sample.
            t (Tensor): The time step.
            mask (Optional[Tensor], optional): An optional mask to apply to the model output. Defaults to None.

        Returns:
            The vector field prediction based on the prediction type.

        Raises:
            ValueError: If the prediction type is not "flow" or "data".
        """
        if self.prediction_type == PredictionType.VELOCITY:
            pred_vector_field = model_output
        elif self.prediction_type == PredictionType.DATA:
            if xt is None or t is None:
                raise ValueError("Xt and Time cannpt be None for vector field conversion")
            t = pad_like(t, model_output)
            pred_vector_field = (model_output - xt) / (1 - t + self.eps)
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be `flow` or `data` "
                "for Continuous Flow Matching."
            )
        if mask is not None:
            pred_vector_field = pred_vector_field * mask.unsqueeze(-1)
        return pred_vector_field

    def process_data_prediction(
        self,
        model_output: Tensor,
        xt: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        """Process the model output based on the prediction type to generate clean data.

        Args:
            model_output (Tensor): The output of the model.
            xt (Tensor): The input sample.
            t (Tensor): The time step.
            mask (Optional[Tensor], optional): An optional mask to apply to the model output. Defaults to None.

        Returns:
            The data prediction based on the prediction type.

        Raises:
            ValueError: If the prediction type is not "flow".
        """
        if self.prediction_type == PredictionType.VELOCITY:
            if xt is None or t is None:
                raise ValueError("Xt and time cannot be None")
            t = pad_like(t, model_output)
            pred_data = xt + (1 - t) * model_output
        elif self.prediction_type == PredictionType.DATA:
            pred_data = model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be `flow` " "for Continuous Flow Matching."
            )
        if mask is not None:
            pred_data = pred_data * mask.unsqueeze(-1)
        return pred_data

    def step(
        self,
        model_out: Tensor,
        xt: Tensor,
        dt: Tensor,
        t: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        center: Bool = False,
    ):
        """Perform a single ODE step integration using Euler method.

        Args:
            model_out (Tensor): The output of the model at the current time step.
            xt (Tensor): The current intermediate state.
            dt (Tensor): The time step size.
            t (Tensor, optional): The current time. Defaults to None.
            mask (Optional[Tensor], optional): A mask to apply to the model output. Defaults to None.
            center (Bool, optional): Whether to center the output. Defaults to False.

        Returns:
            x_next (Tensor): The updated state of the system after the single step, x_(t+dt).

        Notes:
        - If a mask is provided, it is applied element-wise to the model output before scaling.
        - The `clean` method is called on the updated state before it is returned.
        """
        if mask is not None:
            model_out = model_out * mask.unsqueeze(-1)
        v_t = self.process_vector_field_prediction(model_out, xt, t, mask)
        dt = pad_like(dt, model_out)
        delta_x = v_t * dt
        x_next = xt + delta_x
        x_next = self.clean_mask_center(x_next, mask, center)
        return x_next

    def step_score_stochastic(
        self,
        model_out: Tensor,
        xt: Tensor,
        dt: Tensor,
        t: Tensor,
        mask: Optional[Tensor] = None,
        gt_mode: str = "tan",
        gt_p: Float = 1.0,
        gt_clamp: Optional[Float] = None,
        score_temperature: Float = 1.0,
        noise_temperature: Float = 1.0,
        t_lim_ode: Float = 0.99,
        center: Bool = False,
    ):
        r"""Perform a single SDE step integration using a score-based Langevin update.

        d x_t = [v(x_t, t) + g(t) * s(x_t, t) * score_temperature] dt + \sqrt{2 * g(t) * noise_temperature} dw_t.

        Args:
            model_out (Tensor): The output of the model at the current time step.
            xt (Tensor): The current intermediate state.
            dt (Tensor): The time step size.
            t (Tensor, optional): The current time. Defaults to None.
            mask (Optional[Tensor], optional): A mask to apply to the model output. Defaults to None.
            gt_mode (str, optional): The mode for the gt function. Defaults to "tan".
            gt_p (Float, optional): The parameter for the gt function. Defaults to 1.0.
            gt_clamp: (Float, optional): Upper limit of gt term. Defaults to None.
            score_temperature (Float, optional): The temperature for the score part of the step. Defaults to 1.0.
            noise_temperature (Float, optional): The temperature for the stochastic part of the step. Defaults to 1.0.
            t_lim_ode (Float, optional): The time limit for the ODE step. Defaults to 0.99.
            center (Bool, optional): Whether to center the output. Defaults to False.

        Returns:
            x_next (Tensor): The updated state of the system after the single step, x_(t+dt).

        Notes:
            - If a mask is provided, it is applied element-wise to the model output before scaling.
            - The `clean` method is called on the updated state before it is returned.
        """
        if self.ot_type is not None:
            raise ValueError("Optimal Transport violates the vector field to score conversion")
        if not isinstance(self.prior_distribution, GaussianPrior):
            raise ValueError(
                "Prior distribution must be an instance of GaussianPrior to learn a proper score function"
            )
        if t.min() >= t_lim_ode:
            return self.step(model_out, xt, dt, t, mask, center)
        if mask is not None:
            model_out = model_out * mask.unsqueeze(-1)
        v_t = self.process_vector_field_prediction(model_out, xt, t, mask)
        dt = pad_like(dt, model_out)
        t = pad_like(t, model_out)
        score = self.vf_to_score(xt, v_t, t)
        gt = self.get_gt(t, gt_mode, gt_p, gt_clamp)
        eps = torch.randn(xt.shape, dtype=xt.dtype, device=xt.device, generator=self.rng_generator)
        std_eps = torch.sqrt(2 * gt * noise_temperature * dt)
        delta_x = (v_t + gt * score * score_temperature) * dt + std_eps * eps
        x_next = xt + delta_x
        x_next = self.clean_mask_center(x_next, mask, center)
        return x_next

    def loss(
        self,
        model_pred: Tensor,
        target: Tensor,
        t: Optional[Tensor] = None,
        xt: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        target_type: Union[PredictionType, str] = PredictionType.DATA,
    ):
        """Calculate the loss given the model prediction, data sample, time, and mask.

        If target_type is FLOW loss = ||v_hat - (x1-x0)||**2
        If target_type is DATA loss = ||x1_hat - x1||**2 * 1 / (1 - t)**2 as the target vector field = x1 - x0 = (1/(1-t)) * x1 - xt where xt = tx1 - (1-t)x0.
        This functions supports any cominbation of prediction_type and target_type in {DATA, FLOW}.

        Args:
            model_pred (Tensor): The predicted output from the model.
            target (Tensor): The target output for the model prediction.
            t (Optional[Tensor], optional): The time for the model prediction. Defaults to None.
            xt (Optional[Tensor], optional): The interpolated data. Defaults to None.
            mask (Optional[Tensor], optional): The mask for the data point. Defaults to None.
            target_type (PredictionType, optional): The type of the target output. Defaults to PredictionType.DATA.

        Returns:
            Tensor: The calculated loss batch tensor.
        """
        target_type = string_to_enum(target_type, PredictionType)
        if target_type == PredictionType.DATA:
            model_pred = self.process_data_prediction(model_pred, xt, t, mask)
        else:
            model_pred = self.process_vector_field_prediction(model_pred, xt, t, mask)
        raw_loss = self._loss_function(model_pred, target)

        if mask is not None:
            loss = raw_loss * mask.unsqueeze(-1)
            n_elem = torch.sum(mask, dim=-1)
            loss = torch.sum(loss, dim=tuple(range(1, raw_loss.ndim))) / n_elem
        else:
            loss = torch.sum(raw_loss, dim=tuple(range(1, raw_loss.ndim))) / model_pred.size(1)
        if target_type == PredictionType.DATA:
            if t is None:
                raise ValueError("Time cannot be None when using a time-based weighting")
            loss_weight = 1.0 / ((1.0 - t) ** 2 + self.eps)
            loss = loss_weight * loss
        return loss

    def vf_to_score(
        self,
        x_t: Tensor,
        v: Tensor,
        t: Tensor,
    ) -> Tensor:
        """From Geffner et al. Computes score of noisy density given the vector field learned by flow matching.

        With our interpolation scheme these are related by

        v(x_t, t) = (1 / t) (x_t + scale_ref ** 2 * (1 - t) * s(x_t, t)),

        or equivalently,

        s(x_t, t) = (t * v(x_t, t) - x_t) / (scale_ref ** 2 * (1 - t)).

        with scale_ref = 1

        Args:
            x_t: Noisy sample, shape [*, dim]
            v: Vector field, shape [*, dim]
            t: Interpolation time, shape [*] (must be < 1)

        Returns:
            Score of intermediate density, shape [*, dim].
        """
        assert torch.all(t < 1.0), "vf_to_score requires t < 1 (strict)"
        t = pad_like(t, v)
        num = t * v - x_t  # [*, dim]
        den = 1.0 - t  # [*, 1]
        score = num / den
        return score  # [*, dim]

    def get_gt(
        self,
        t: Tensor,
        mode: str = "tan",
        param: float = 1.0,
        clamp_val: Optional[float] = None,
        eps: float = 1e-2,
    ) -> Tensor:
        """From Geffner et al. Computes gt for different modes.

        Args:
            t: times where we'll evaluate, covers [0, 1), shape [nsteps]
            mode: "us" or "tan"
            param: parameterized transformation
            clamp_val: value to clamp gt, no clamping if None
            eps: small value leave as it is
        """

        # Function to get variants for some gt mode
        def transform_gt(gt, f_pow=1.0):
            # 1.0 means no transformation
            if f_pow == 1.0:
                return gt

            # First we somewhat normalize between 0 and 1
            log_gt = torch.log(gt)
            mean_log_gt = torch.mean(log_gt)
            log_gt_centered = log_gt - mean_log_gt
            normalized = torch.nn.functional.sigmoid(log_gt_centered)
            # Transformation here
            normalized = normalized**f_pow
            # Undo normalization with the transformed variable
            log_gt_centered_rec = torch.logit(normalized, eps=1e-6)
            log_gt_rec = log_gt_centered_rec + mean_log_gt
            gt_rec = torch.exp(log_gt_rec)
            return gt_rec

        # Numerical reasons for some schedule
        t = torch.clamp(t, 0, 1 - self.eps)

        if mode == "us":
            num = 1.0 - t
            den = t
            gt = num / (den + eps)
        elif mode == "tan":
            num = torch.sin((1.0 - t) * torch.pi / 2.0)
            den = torch.cos((1.0 - t) * torch.pi / 2.0)
            gt = (torch.pi / 2.0) * num / (den + eps)
        elif mode == "1/t":
            num = 1.0
            den = t
            gt = num / (den + eps)
        elif mode == "1/t2":
            num = 1.0
            den = t**2
            gt = num / (den + eps)
        elif mode == "1/t1p5":
            num = 1.0
            den = t**1.5
            gt = num / (den + eps)
        elif mode == "2/t":
            num = 2.0
            den = t
            gt = num / (den + eps)
        elif mode == "2/t2":
            num = 2.0
            den = t**2
            gt = num / (den + eps)
        elif mode == "2/t1p5":
            num = 2.0
            den = t**1.5
            gt = num / (den + eps)
        elif mode == "1mt":
            gt = 1 - t
        elif mode == "t":
            gt = t
        elif mode == "ones":
            gt = 0 * t + 1
        else:
            raise NotImplementedError(f"gt not implemented {mode}")
        gt = transform_gt(gt, f_pow=param)
        gt = torch.clamp(gt, 0, clamp_val)  # If None no clamping
        return gt  # [s]
