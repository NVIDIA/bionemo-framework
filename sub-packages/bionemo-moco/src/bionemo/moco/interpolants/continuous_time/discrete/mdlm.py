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


from typing import Optional

import torch
from torch import Tensor

from bionemo.moco.distributions.prior.discrete.mask import DiscreteMaskedPrior
from bionemo.moco.distributions.time.distribution import TimeDistribution
from bionemo.moco.interpolants.base_interpolant import Interpolant, pad_like
from bionemo.moco.schedules.noise.continuous_noise_transforms import ContinuousExpNoiseTransform


class MDLM(Interpolant):
    """A Masked discrete Diffusion Language Model (MDLM) interpolant.

     -------

    Examples:
    ```python
    >>> import torch
    >>> from bionemo.moco.distributions.prior.discrete.mask import DiscreteMaskedPrior
    >>> from bionemo.moco.distributions.time.uniform import UniformTimeDistribution
    >>> from bionemo.moco.interpolants.continuous_time.discrete.mdlm import MDLM
    >>> from bionemo.moco.schedules.noise.continuous_noise_transforms import CosineExpNoiseTransform
    >>> from bionemo.moco.schedules.inference_time_schedules import LinearTimeSchedule


    mdlm = MDLM(
        time_distribution = UniformTimeDistribution(discrete_time = False,...),
        prior_distribution = DiscreteMaskedPrior(...),
        noise_schedule = CosineExpNoiseTransform(...),
        )
    model = Model(...)

    # Training
    for epoch in range(1000):
        data = data_loader.get(...)
        time = mdlm.sample_time(batch_size)
        xt = mdlm.interpolate(data, time)

        logits = model(xt, time)
        loss = mdlm.loss(logits, data, xt, time)
        loss.backward()

    # Generation
    x_pred = mdlm.sample_prior(data.shape)
    schedule = LinearTimeSchedule(...)
    inference_time = schedule.generate_schedule()
    dts = schedue.discreteize()
    for t, dt in zip(inference_time, dts):
        time = torch.full((batch_size,), t)
        logits = model(x_pred, time)
        x_pred = mdlm.step(logits, time, x_pred, dt)
    return x_pred

    ```
    """

    def __init__(
        self,
        time_distribution: TimeDistribution,
        prior_distribution: DiscreteMaskedPrior,
        noise_schedule: ContinuousExpNoiseTransform,
        device: str = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Initialize the Masked Discrete Language Model (MDLM) interpolant.

        Args:
            time_distribution (TimeDistribution): The distribution governing the time variable in the diffusion process.
            prior_distribution (DiscreteMaskedPrior): The prior distribution over the discrete token space, including masked tokens.
            noise_schedule (ContinuousExpNoiseTransform): The noise schedule defining the noise intensity as a function of time.
            device (str, optional): The device to use for computations. Defaults to "cpu".
            rng_generator (Optional[torch.Generator], optional): The random number generator for reproducibility. Defaults to None.
        """
        super().__init__(time_distribution, prior_distribution, device, rng_generator)
        if not isinstance(prior_distribution, DiscreteMaskedPrior):
            raise ValueError("DiscreteMaskedPrior required for MDLM")
        if not isinstance(noise_schedule, ContinuousExpNoiseTransform):
            raise ValueError("ContinuousExpNoiseTransform required for MDLM")
        self.noise_schedule = noise_schedule
        self.num_classes = prior_distribution.num_classes
        self.mask_index = prior_distribution.mask_dim
        # Gumbel used for confidence sampling. Note rng_generator not compatible with torch.Distribution.
        # self.gumbel_dist = torch.distributions.Gumbel(torch.tensor(0.0), torch.tensor(1.0))

    def interpolate(self, data: Tensor, t: Tensor):
        """Get x(t) with given time t from noise and data.

        Args:
            data (Tensor): target discrete ids
            t (Tensor): time
        """
        if data.dtype == torch.float and data.ndim > 2:
            x0 = data.argmax(-1)
        else:
            x0 = data
        sigma = self.noise_schedule.calculate_sigma(t, data.device)
        alpha = self.noise_schedule.sigma_to_alpha(sigma)
        p_mask = 1 - alpha
        p_mask = pad_like(p_mask, x0)
        mask_indices = torch.rand(*x0.shape, device=x0.device, generator=self.rng_generator) < p_mask
        xt = torch.where(mask_indices, self.mask_index, x0)
        return xt

    def forward_process(self, data: Tensor, t: Tensor) -> Tensor:
        """Apply the forward process to the data at time t.

        Args:
            data (Tensor): target discrete ids
            t (Tensor): time

        Returns:
            Tensor: x(t) after applying the forward process
        """
        return self.interpolate(data, t)

    def loss(
        self,
        logits: Tensor,
        target: Tensor,
        xt: Tensor,
        time: Tensor,
        mask: Optional[Tensor] = None,
        use_weight=True,
    ):
        """Calculate the cross-entropy loss between the model prediction and the target output.

        The loss is calculated between the batch x node x class logits and the target batch x node,
        considering the current state of the discrete sequence `xt` at time `time`.

        If `use_weight` is True, the loss is weighted by the reduced form of the MDLM time weight for continuous NELBO,
        as specified in equation 11 of https://arxiv.org/pdf/2406.07524. This weight is proportional to the derivative
        of the noise schedule with respect to time, and is used to emphasize the importance of accurate predictions at
        certain times in the diffusion process.

        Args:
            logits (Tensor): The predicted output from the model, with shape batch x node x class.
            target (Tensor): The target output for the model prediction, with shape batch x node.
            xt (Tensor): The current state of the discrete sequence, with shape batch x node.
            time (Tensor): The time at which the loss is calculated.
            mask (Optional[Tensor], optional): The mask for the data point. Defaults to None.
            use_weight (bool, optional): Whether to use the MDLM time weight for the loss. Defaults to True.

        Returns:
            Tensor: The calculated loss batch tensor.
        """
        logprobs = self._subs_parameterization(logits, xt)
        log_p_theta = torch.gather(input=logprobs, dim=-1, index=target[..., None]).squeeze(-1)

        sigma = self.noise_schedule.calculate_sigma(time, target.device)
        dsigma = self.noise_schedule.d_dt_sigma(time, target.device)  # type: ignore
        loss = -log_p_theta
        if use_weight:
            loss = loss * (dsigma / torch.expm1(sigma))[:, None]

        if mask is not None:
            loss = loss * mask
            num_non_masked_elements = torch.sum(mask, dim=-1)
            loss = torch.sum(loss, dim=(-1)) / num_non_masked_elements
        else:
            loss = torch.sum(loss, dim=(-1)) / logits.size(1)
        return loss

    def _subs_parameterization(self, logits: Tensor, xt: Tensor) -> Tensor:
        """Apply subsititution parameterization to the logits.

        This function enforces that the model can never predict a mask token by lowering the mask logits.
        Then, for all unmasked tokens, it copies over from xt to enable carry over unmasked.
        Once a token is unmasked, it stays the same.
        See Sec. 3.2.3 https://arxiv.org/pdf/2406.07524.

        Note that recent work has shown that allowing the model to rethink
        carry over unmasking is beneficial https://arxiv.org/abs/2410.06264.

        Args:
            logits (Tensor): The logits tensor with shape batch x node x class.
            xt (Tensor): The tensor of unmasked tokens with shape batch x node.

        Returns:
            Tensor: The modified logits tensor with substitution parameterization applied.
        """
        logits[..., self.mask_index] += -1000000.0  # clean input is never masked
        logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # normalize
        unmasked_indices = xt != self.mask_index
        logprobs[unmasked_indices] = -1000000.0
        logprobs[unmasked_indices, xt[unmasked_indices]] = 0  # Unmasked token remains unchanged
        return logprobs

    def step(self, logits, t, xt, dt) -> Tensor:
        """Perform a single step of MDLM DDPM step.

        Parameters:
        logits (Tensor): The input logits.
        t (float): The current time step.
        xt (Tensor): The current state.
        dt (float): The time step increment.

        Returns:
        Tensor: The updated state.
        """
        sigma_t = self.noise_schedule.calculate_sigma(t, logits.device)
        sigma_s = self.noise_schedule.calculate_sigma(t - dt, logits.device)
        alpha_t = torch.exp(-sigma_t)
        alpha_s = torch.exp(-sigma_s)
        p_mask_s = 1 - alpha_s
        alpha_t = pad_like(alpha_t, logits)
        alpha_s = pad_like(alpha_s, logits)
        p_mask_s = pad_like(p_mask_s, logits)
        # Apply subs parameterization
        log_p_x0 = self._subs_parameterization(logits, xt)
        if p_mask_s.ndim != log_p_x0.ndim:
            raise ValueError(f"Dimension Mistmatch {p_mask_s.shape} {log_p_x0.shape}")
        # Equation 6 from MDLM
        prob_s_given_t = log_p_x0.exp() * (alpha_s - alpha_t)  # righthand side (alpha_s - alpha_t)*x
        prob_s_given_t[..., self.mask_index] = p_mask_s[..., 0]  # lefthand side (1 - alpha_s)*M
        sampled_x = self._sample_categorical(prob_s_given_t)
        carry_over_unmask = (xt != self.mask_index).to(xt.dtype)
        return carry_over_unmask * xt + (1 - carry_over_unmask) * sampled_x

    def _sample_categorical(self, categorical_probs: Tensor) -> Tensor:
        """Sample from a categorical distribution using the Gumbel trick.

        Args:
            categorical_probs (Tensor): The probabilities of each category, shape batch x node x class.

        Returns:
            Tensor: The sampled category indices, shape batch x node.
        """
        gumbel_norm = (
            1e-10
            - (
                torch.rand(*categorical_probs.shape, device=categorical_probs.device, generator=self.rng_generator)
                + 1e-10
            ).log()
        )
        scaled_proability = categorical_probs / gumbel_norm
        return scaled_proability.argmax(dim=-1)

    def step_confidence(
        self,
        logits: Tensor,
        xt: Tensor,
        curr_step: int,
        num_steps: int,
        logit_temperature: float = 1.0,
        randomness: float = 1.0,
        confidence_temperature: float = 1.0,
    ) -> Tensor:
        """Update the input sequence xt by sampling from the predicted logits and adding Gumbel noise.

        Method taken from GenMol Seul et al.

        Args:
            logits: Predicted logits
            xt: Input sequence
            curr_step: Current step
            num_steps: Total number of steps
            logit_temperature: Temperature for softmax over logits
            randomness: Scale for Gumbel noise
            confidence_temperature: Temperature for Gumbel confidence

        Returns:
            Updated input sequence xt
        """
        if xt.ndim > 3:
            raise NotImplementedError(
                "step_confidence is implemented for Batch x Sequence x State Space shaped tensors."
            )
        xt = xt.clone()
        log_p_x0 = self._subs_parameterization(logits, xt)
        # sample the code from the softmax prediction
        probs = torch.softmax(log_p_x0 / logit_temperature, dim=-1)
        preds = torch.distributions.Categorical(probs=probs).sample()

        confidence = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)
        # add Gumbel noise decreasing over the sampling process
        ratio = curr_step / (num_steps - 1)
        # Using manual definition of 0,1 Gumbel to pass in generator
        gumbel_sample = -torch.log(-torch.log(torch.rand(xt.shape, generator=self.rng_generator))).to(logits.device)
        # gumbel_sample = self.gumbel_dist.sample(xt.shape).to(logits.device)
        gumbel_noise = gumbel_sample * randomness * (1 - ratio)  # type: ignore
        confidence = (
            (torch.log(confidence) + gumbel_noise) / confidence_temperature
        )  # stems from tau of https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

        # do not predict on already predicted tokens
        mask = xt == self.mask_index
        confidence[~mask] = -torch.inf

        # choose the predicted token with the highest confidence
        confidence_threshold, idx_mask = torch.topk(confidence, k=1, dim=-1)
        confidence_threshold = confidence_threshold[:, -1].unsqueeze(-1)

        # replace the chosen tokens
        to_replace = confidence >= confidence_threshold
        to_replace = (mask.float() * to_replace.float()).bool()
        xt[to_replace] = preds[to_replace]
        return xt

    def step_argmax(self, model_out: Tensor):
        """Returns the index of the maximum value in the last dimension of the model output.

        Args:
            model_out (Tensor): The output of the model.

        Returns:
            Tensor: The index of the maximum value in the last dimension of the model output.
        """
        return model_out.argmax(dim=-1)

    def calculate_score(self, logits, x, t):
        """Returns score of the given sample x at time t with the corresponding model output logits.

        Args:
            logits (Tensor): The output of the model.
            x (Tensor): The current data point.
            t (Tensor): The current time.

        Returns:
            Tensor: The score defined in Appendix C.3 Equation 76 of MDLM.
        """
        sigma_t = self.noise_schedule.calculate_sigma(t, logits.device)
        log_ratio = -torch.log(
            torch.expm1(sigma_t)
        )  # log ( exp(-sigma) / (1 - exp(-sigma))) = log(1/ (exp(sigma) - 1))

        # Create masked and unmasked log scores
        masked_log_score = logits + pad_like(log_ratio, logits)  # xt is masked and prediction is not
        masked_log_score[..., self.mask_index] = 0  # xt and prediction are mask

        unmasked_log_score = torch.full_like(logits, -1000000.0)
        unmasked_log_score.scatter_(-1, x[..., None], 0)  # place zeros where current predictions are
        unmasked_log_score[..., self.mask_index] = -pad_like(log_ratio, logits[..., 0])

        # Combine masked and unmasked log scores
        masked_indices = (x == self.mask_index).to(logits.dtype)[..., None]
        log_score = masked_log_score * masked_indices + unmasked_log_score * (1 - masked_indices)

        return log_score.exp()
