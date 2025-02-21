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
from typing import Literal, Optional

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

    def step(self, logits: Tensor, t: Tensor, xt: Tensor, dt: Tensor, temperature: float = 1.0) -> Tensor:
        """Perform a single step of MDLM DDPM step.

        Parameters:
        logits (Tensor): The input logits.
        t (Tensor): The current time step.
        xt (Tensor): The current state.
        dt (Tensor): The time step increment.
        temperature (float): Softmax temperature defaults to 1.0.

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
        log_p_x0 = self._subs_parameterization(logits, xt) / temperature
        if p_mask_s.ndim != log_p_x0.ndim:
            raise ValueError(f"Dimension Mistmatch {p_mask_s.shape} {log_p_x0.shape}")
        # Equation 7 from MDLM
        prob_s_given_t = log_p_x0.exp() * (
            alpha_s - alpha_t
        )  # righthand side (alpha_s - alpha_t)*x = (1 - alpha_t - (1 - alpha_s)) * x
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

    def get_num_steps_confidence(self, xt: Tensor, num_tokens_unmask: int = 1):
        """Calculate the maximum number of steps with confidence.

        This method computes the maximum count of occurrences where the input tensor `xt` matches the `mask_index`
        along the last dimension (-1). The result is returned as a single float value.

        Args:
            xt (Tensor): Input tensor to evaluate against the mask index.
            num_tokens_unmask (int): number of tokens to unamsk at each step.

        Returns:
            float: The maximum number of steps with confidence (i.e., matching the mask index).
        """
        nsteps = (xt == self.mask_index).sum(-1).max().item()
        if num_tokens_unmask == 1:
            return int(nsteps)
        else:
            return int(max(math.ceil(nsteps // num_tokens_unmask), 1))

    def step_confidence(
        self,
        logits: Tensor,
        xt: Tensor,
        curr_step: int,
        num_steps: int,
        logit_temperature: float = 1.0,
        randomness: float = 1.0,
        confidence_temperature: float = 1.0,
        num_tokens_unmask: int = 1,
    ) -> Tensor:
        """Update the input sequence xt by sampling from the predicted logits and adding Gumbel noise.

        Method taken from GenMol Lee et al. https://arxiv.org/abs/2501.06158

        Args:
            logits: Predicted logits
            xt: Input sequence
            curr_step: Current step
            num_steps: Total number of steps
            logit_temperature: Temperature for softmax over logits
            randomness: Scale for Gumbel noise
            confidence_temperature: Temperature for Gumbel confidence
            num_tokens_unmask: number of tokens to unmask each step

        Returns:
            Updated input sequence xt unmasking num_tokens_unmask token each step.
        """
        if xt.ndim > 3:
            raise NotImplementedError(
                "step_confidence is implemented for Batch x Sequence x State Space shaped tensors."
            )
        if curr_step < 0 or num_steps < 1 or num_tokens_unmask < 1:
            raise ValueError("Invalid input values for curr_step, num_steps, or num_tokens_unmask.")
        xt = xt.clone()
        log_p_x0 = self._subs_parameterization(logits, xt)
        # sample the code from the softmax prediction
        probs = torch.softmax(log_p_x0 / logit_temperature, dim=-1)
        preds = torch.distributions.Categorical(probs=probs).sample()

        confidence = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)
        # add Gumbel noise decreasing over the sampling process
        ratio = curr_step / (num_steps - 1)
        # Using manual definition of 0,1 Gumbel to pass in generator, manually specifying the device is faster than transfer
        gumbel_sample = -torch.log(
            -torch.log(torch.rand(xt.shape, device=logits.device, generator=self.rng_generator))
        )
        # gumbel_sample = self.gumbel_dist.sample(xt.shape).to(logits.device)
        gumbel_noise = gumbel_sample * randomness * (1 - ratio)  # type: ignore
        confidence = (
            (torch.log(confidence) + gumbel_noise) / confidence_temperature
        )  # stems from tau of https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

        # do not predict on already predicted tokens
        mask = xt == self.mask_index
        confidence[~mask] = -torch.inf

        # choose the predicted token with the highest confidence
        confidence_threshold, idx_mask = torch.topk(confidence, k=num_tokens_unmask, dim=-1)
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

    def step_self_path_planning(
        self,
        logits: Tensor,
        xt: Tensor,
        t: Tensor,
        curr_step: int,
        num_steps: int,
        logit_temperature: float = 1.0,
        randomness: float = 1.0,
        confidence_temperature: float = 1.0,
        score_type: Literal["confidence", "random"] = "confidence",
        fix_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Self Path Planning (P2) Sampling from Peng et al. https://arxiv.org/html/2502.03540v1.

        Args:
            logits (Tensor): Predicted logits for sampling.
            xt (Tensor): Input sequence to be updated.
            t (Tensor): Time tensor (e.g., time steps or temporal info).
            curr_step (int): Current iteration in the planning process.
            num_steps (int): Total number of planning steps.
            logit_temperature (float): Temperature for logits (default: 1.0).
            randomness (float): Introduced randomness level (default: 1.0).
            confidence_temperature (float): Temperature for confidence scoring (default: 1.0).
            score_type (Literal["confidence", "random"]): Sampling score type (default: "confidence").
            fix_mask (Optional[Tensor]): inital mask where True when not a mask tokens (default: None).

        Returns:
            Tensor: Updated input sequence xt after iterative unmasking.
        """
        if xt.ndim > 3:
            raise NotImplementedError(
                "step_confidence is implemented for Batch x Sequence x State Space shaped tensors."
            )
        if curr_step < 0 or num_steps < 1:
            raise ValueError("Invalid input values for curr_step, num_steps.")
        xt = xt.clone()
        if fix_mask is None:
            fix_mask = torch.zeros_like(xt).bool()  #! if any sequenes are fixed from the start of trajectory
        last_mask = xt == self.mask_index
        unmask_candidates = (
            last_mask & ~fix_mask
        )  #! I want to consider tokens to un mask that are currently masked and not fixed. This fixes a typo in pseudo code
        x1_pred, logp = self.stochastic_sample_from_categorical(
            logits, temperature=logit_temperature, noise_scale=confidence_temperature
        )
        if curr_step == num_steps - 1:
            xt[last_mask] = x1_pred[last_mask]
        else:
            if score_type == "confidence":
                score = logp
            elif score_type == "random":
                score = torch.rand_like(logp).log()

            score = score.masked_fill(fix_mask.squeeze(-1), float("inf"))
            score[unmask_candidates.squeeze(-1)] *= randomness
            num_to_mask = torch.clamp(
                ((~fix_mask).sum(dim=1, keepdim=True).float() * t.unsqueeze(-1)).long(), max=xt.shape[-1] - 1
            )  #! here is is t since diffusion time is 1 to 0. Clamp is to set it to 0 N-1 since topk uses it as indices
            mask = self.topk_lowest_masking(score, num_to_mask)
            xt[mask] = self.mask_index
            mask_to_x1 = last_mask & ~mask
            xt[mask_to_x1] = x1_pred[mask_to_x1]
        return xt

    def topk_lowest_masking(self, scores: Tensor, cutoff_len: Tensor):
        """Generates a mask for the lowest scoring elements up to a specified cutoff length.

        Args:
            scores (Tensor): Input scores tensor with shape (... , num_elements)
            cutoff_len (Tensor): Number of lowest-scoring elements to mask (per batch element)

        Returns:
            Tensor: Boolean mask tensor with same shape as `scores`, where `True` indicates
                    the corresponding element is among the `cutoff_len` lowest scores.

        Example:
            >>> scores = torch.tensor([[0.9, 0.8, 0.1, 0.05], [0.7, 0.4, 0.3, 0.2]])
            >>> cutoff_len = 2
            >>> mask = topk_lowest_masking(scores, cutoff_len)
            >>> print(mask)
            tensor([[False, False, True, True],
                    [False, True, True, False]])
        """
        sorted_scores, _ = scores.sort(dim=-1)
        threshold = sorted_scores.gather(dim=-1, index=cutoff_len)
        return scores < threshold

    def stochastic_sample_from_categorical(self, logits: Tensor, temperature: float = 1.0, noise_scale: float = 1.0):
        """Stochastically samples from a categorical distribution defined by input logits, with optional temperature and noise scaling for diverse sampling.

        Args:
            logits (Tensor): Input logits tensor with shape (... , num_categories)
            temperature (float, optional): Softmax temperature. Higher values produce more uniform samples. Defaults to 1.0.
            noise_scale (float, optional): Scale for Gumbel noise. Higher values produce more diverse samples. Defaults to 1.0.

        Returns:
            tuple:
                - **tokens** (LongTensor): Sampling result (category indices) with shape (... , )
                - **scores** (Tensor): Corresponding log-softmax scores for the sampled tokens, with shape (... , )
        """
        if temperature > 0:
            gumbel = -torch.log(
                -torch.log(torch.rand(logits.shape, device=logits.device, generator=self.rng_generator) + 1e-8) + 1e-8
            )  #! avoid device transfers
            logits = logits / temperature + noise_scale * gumbel
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
        return tokens, scores
