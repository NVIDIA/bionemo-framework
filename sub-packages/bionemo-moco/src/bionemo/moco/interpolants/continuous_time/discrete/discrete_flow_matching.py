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
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor

from bionemo.moco.distributions.prior.discrete.mask import DiscreteMaskedPrior
from bionemo.moco.distributions.prior.distribution import DiscretePriorDistribution
from bionemo.moco.distributions.time.distribution import TimeDistribution
from bionemo.moco.interpolants.base_interpolant import Interpolant, pad_like


class DiscreteFlowMatcher(Interpolant):
    """A Discrete Flow Model (DFM) interpolant."""

    def __init__(
        self,
        time_distribution: TimeDistribution,
        prior_distribution: DiscretePriorDistribution,
        device: str = "cpu",
        eps: Float = 1e-5,
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Initialize the DFM interpolant.

        Args:
            time_distribution (TimeDistribution): The time distribution for the diffusion process.
            prior_distribution (DiscretePriorDistribution): The prior distribution for the discrete masked tokens.
            device (str, optional): The device to use for computations. Defaults to "cpu".
            eps: small Float to prevent dividing by zero.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
        """
        super().__init__(time_distribution, prior_distribution, device, rng_generator)
        self.num_classes = prior_distribution.num_classes
        self.eps = eps
        self.use_mask = isinstance(self.prior_distribution, DiscreteMaskedPrior)
        if self.use_mask:
            self.mask_index = prior_distribution.mask_dim  # type: ignore
        self._loss_function = nn.CrossEntropyLoss(reduction="none")

    def interpolate(self, data: Tensor, t: Tensor, noise: Tensor):
        """Get x(t) with given time t from noise and data.

        Args:
            data (Tensor): target discrete ids
            t (Tensor): time
            noise: tensor noise ids
        """
        if data.dtype == torch.float and data.ndim > 2:
            x1 = data.argmax(-1)
        else:
            x1 = data
        x0 = noise
        t = pad_like(t, x1)
        threshold = torch.rand_like(x1.float())
        xt = torch.where((threshold < 1 - t), x0, x1)
        return xt

    def loss(
        self,
        logits: Tensor,
        target: Tensor,
        time: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        use_weight: Bool = False,
    ):
        """Calculate the cross-entropy loss between the model prediction and the target output.

        The loss is calculated between the batch x node x class logits and the target batch x node.
        If using a masked prior please pass in the correct mask to calculate loss values on only masked states.
        i.e. mask = data_mask * is_masked_state which is calculated with self.prior_dist.is_masked(xt))

        If `use_weight` is True, the loss is weighted by 1/(1-t) defined in equation 24 in Appndix C. of https://arxiv.org/pdf/2402.04997

        Args:
            logits (Tensor): The predicted output from the model, with shape batch x node x class.
            target (Tensor): The target output for the model prediction, with shape batch x node.
            time (Tensor): The time at which the loss is calculated.
            mask (Optional[Tensor], optional): The mask for the data point. Defaults to None.
            use_weight (bool, optional): Whether to use the DFM time weight for the loss. Defaults to True.

        Returns:
            Tensor: The calculated loss batch tensor.
        """
        assert target.ndim + 1 == logits.ndim
        loss = self._loss_function(logits.transpose(-1, 1), target.long())
        if mask is not None:
            loss = loss * mask
            num_non_masked_elements = torch.sum(mask, dim=-1)
            num_non_masked_elements[num_non_masked_elements == 0] = (
                1.0  #! prevents divide by zero since if the row is all zero the sum of loss = 0
            )
            loss = torch.sum(loss, dim=(-1)) / num_non_masked_elements
        else:
            loss = torch.sum(loss, dim=(-1)) / logits.size(1)
        if use_weight:
            if time is None:
                raise ValueError("Time is required to compute the DFM liklehood weighting of 1/(1-t + self.eps)")
            loss = loss * 1 / (1 - time + self.eps)
        return loss

    def step(
        self,
        logits: Tensor,
        t: Tensor,
        xt: Tensor,
        dt: Tensor | float,
        temperature: Float = 1.0,
        stochasticity: Float = 1.0,
    ) -> Tensor:
        """Perform a single step of DFM euler updates.

        Args:
            logits (Tensor): The input logits.
            t (Tensor): The current time step.
            xt (Tensor): The current state.
            dt (Tensor | float): The time step increment.
            temperature (Float, optional): The temperature for the softmax calculation. Defaults to 1.0.
            stochasticity (Float, optional): The stochasticity value for the step calculation. Defaults to 1.0.

        Returns:
            Tensor: The updated state.
        """
        x_1_pred_logits = logits
        S = x_1_pred_logits.shape[-1]
        t = pad_like(t, logits)
        if isinstance(dt, float):
            dt = torch.Tensor([dt] * t.shape[0]).to(self.device)
        dt = pad_like(dt, logits)  # type: ignore

        if self.use_mask:
            if self.mask_index >= S:
                raise ValueError(
                    "If using a non inclusive DiscreteMaskedPrior initialization, please pad the logits input with DiscreteMaskedPrior.pad_sample(logits)"
                )

            mask_one_hot = torch.zeros((S,), device=self.device)
            mask_one_hot[self.mask_index] = 1.0
            x_1_pred_logits[..., self.mask_index] = -1.0e9

            x_1_pred_prob = F.softmax(x_1_pred_logits / temperature, dim=-1)

            xt_is_mask = (xt == self.mask_index).unsqueeze(-1).float()  # b x n x 1
            step_prob = (
                dt * x_1_pred_prob * ((1 + stochasticity * t) / (1 - t)) * xt_is_mask
                + dt
                * (1 - xt_is_mask)
                * mask_one_hot.view(1, 1, -1)
                * stochasticity
                * (
                    t + dt < 1
                ).float()  # No remasking if on final step. NOTE should probably use step_argmax or step_sample instead
            )  # (b, n, S)
            step_prob = self._regularize_step_probs(step_prob, xt)
        else:
            x_1_pred_prob = torch.nn.functional.softmax(x_1_pred_logits / temperature, dim=-1)  # (b, n, S)

            pt_x1_eq_xt_prob = torch.gather(x_1_pred_prob, dim=-1, index=xt.long().unsqueeze(-1))  # (b, n, 1)

            step_prob = (
                dt * x_1_pred_prob * ((1 + stochasticity + stochasticity * (S - 1) * t) / (1 - t))
                + dt * pt_x1_eq_xt_prob * stochasticity
            )
            step_prob = self._regularize_step_probs(step_prob, xt)

        x_next = torch.multinomial(step_prob.view(-1, S), num_samples=1, generator=self.rng_generator).view(xt.shape)
        return x_next

    def _regularize_step_probs(self, step_prob: Tensor, xt: Tensor) -> Tensor:
        """Regularize the step probabilities to ensure that the probability of the current state xt is set to the remaining probability mass after clipping and scattering.

        Args:
            step_prob (Tensor): The input step probabilities with shape (batch, node, class).
            xt (Tensor): The current state with shape (batch, node).

        Returns:
            Tensor: The regularized step probabilities with shape (batch, node, class).
        """
        device = step_prob.device
        # Clamp the step probabilities to ensure they are within the valid range [0.0, 1.0]
        step_prob = torch.clamp(step_prob, min=0.0, max=1.0)
        # Set the probability of the current state xt to 0
        step_prob.scatter_(
            dim=-1,
            index=xt.unsqueeze(-1),
            src=torch.zeros((*xt.shape, 1), dtype=torch.float, device=device),
        )
        # Set the probability of the current state xt to the remaining probability mass
        step_prob.scatter_(
            dim=-1,
            index=xt[..., None],
            src=1 - torch.sum(step_prob, dim=-1, keepdim=True),
        )
        step_prob = torch.clamp(step_prob, min=0.0, max=1.0)
        # Clamp the step probabilities again to ensure they are within the valid range [0.0, 1.0]
        return step_prob

    def step_purity(
        self,
        logits: Tensor,
        t: Tensor,
        xt: Tensor,
        dt: Tensor | float,
        temperature: Float = 1.0,
        stochasticity: Float = 1.0,
    ) -> Tensor:
        """Perform a single step of purity sampling.

        https://github.com/jasonkyuyim/multiflow/blob/6278899970523bad29953047e7a42b32a41dc813/multiflow/data/interpolant.py#L346
        Here's a high-level overview of what the function does:
        TODO: check if the -1e9 and 1e-9 are small enough or using torch.inf would be better

        1. Preprocessing:
            Checks if dt is a float and converts it to a tensor if necessary.
            Pads t and dt to match the shape of xt.
            Checks if the mask_index is valid (i.e., within the range of possible discrete values).
        2. Masking:
            Sets the logits corresponding to the mask_index to a low value (-1e9) to effectively mask out those values.
            Computes the softmax probabilities of the logits.
            Sets the probability of the mask_index to a small value (1e-9) to avoid numerical issues.
        3.Purity sampling:
            Computes the maximum log probabilities of the softmax distribution.
            Computes the indices of the top-number_to_unmask samples with the highest log probabilities.
            Uses these indices to sample new values from the original distribution.
        4. Unmasking and updating:
            Creates a mask to select the top-number_to_unmask samples.
            Uses this mask to update the current state xt with the new samples.
        5. Re-masking:
            Generates a new mask to randomly re-mask some of the updated samples.
            Applies this mask to the updated state xt.

        Args:
            logits (Tensor): The input logits.
            t (Tensor): The current time step.
            xt (Tensor): The current state.
            dt (Tensor): The time step increment.
            temperature (Float, optional): The temperature for the softmax calculation. Defaults to 1.0.
            stochasticity (Float, optional): The stochasticity value for the step calculation. Defaults to 1.0.

        Returns:
            Tensor: The updated state.
        """
        if logits.ndim > 3:
            raise ValueError("Purity Sampling is only implmented for logits shape batch x sequence x state space.")
        if isinstance(dt, float):
            dt = torch.Tensor([dt] * t.shape[0]).to(self.device)
        x_1_pred_logits = logits
        B, N, S = x_1_pred_logits.shape

        if not self.use_mask:
            raise ValueError("Purity Sampling only works with a DiscreteMaskPrior")

        if self.mask_index >= S:
            raise ValueError(
                "If using a non inclusive DiscreteMaskedPrior initialization, please pad the logits input with DiscreteMaskedPrior.pad_sample(logits)"
            )
        x_1_pred_logits[..., self.mask_index] = -1.0e9
        x_1_pred_prob = F.softmax(x_1_pred_logits / temperature, dim=-1)
        x_1_pred_prob[..., self.mask_index] = 1e-9
        max_logprob = torch.max(torch.log(x_1_pred_prob), dim=-1)[0]  # (b, n)
        max_logprob = max_logprob - (xt != self.mask_index).float() * 1e9
        sorted_max_logprobs_idcs = torch.argsort(max_logprob, dim=-1, descending=True)  # (b, n)
        unmask_probs = (dt * (1 + stochasticity * t) / (1 - t)).clamp(max=1)
        # For M mask tokens we have p chance to unmask so we try for each one and see how many to do
        number_to_unmask = torch.binomial(
            count=torch.count_nonzero(xt == self.mask_index, dim=-1).float(), prob=unmask_probs
        )
        unmasked_samples = torch.multinomial(x_1_pred_prob.view(-1, S), num_samples=1).view(xt.shape)

        # Taken from MultiFlow
        # Vectorized version of:
        # for b in range(B):
        #     for d in range(D):
        #         if d < number_to_unmask[b]:
        #             aatypes_t[b, d] = unmasked_samples[b, sorted_max_logprobs_idcs[b, d]]

        D_grid = torch.arange(N, device=self.device).view(1, -1).repeat(B, 1)
        mask1 = (D_grid < number_to_unmask.view(-1, 1)).float()
        initial_val_max_logprob_idcs = sorted_max_logprobs_idcs[:, 0].view(-1, 1).repeat(1, N)
        masked_sorted_max_logprobs_idcs = (
            mask1 * sorted_max_logprobs_idcs + (1 - mask1) * initial_val_max_logprob_idcs
        ).long()
        mask2 = torch.zeros((B, N), dtype=torch.long, device=self.device)
        mask2.scatter_(
            dim=1,
            index=masked_sorted_max_logprobs_idcs,
            src=torch.ones((B, N), dtype=torch.long, device=self.device),
        )
        unmask_zero_row = (number_to_unmask == 0).view(-1, 1).repeat(1, N).long()
        mask2 = mask2 * (1 - unmask_zero_row)
        x_next = xt * (1 - mask2) + unmasked_samples * mask2

        # re-mask
        u = torch.rand((B, N), device=self.device, generator=self.rng_generator)
        dt = pad_like(dt, u)  # type: ignore
        re_mask_mask = (u < dt * stochasticity).long()
        x_next = x_next * (1 - re_mask_mask) + self.mask_index * re_mask_mask

        return x_next

    def step_argmax(self, model_out: Tensor):
        """Returns the index of the maximum value in the last dimension of the model output.

        Args:
            model_out (Tensor): The output of the model.

        """
        if self.use_mask:
            model_out[..., self.mask_index] = -1.0e9
        return model_out.argmax(dim=-1)

    def step_simple_sample(self, model_out: Tensor, temperature: float = 1.0, num_samples: int = 1):
        """Samples from the model output logits. Leads to more diversity than step_argmax.

        Args:
            model_out (Tensor): The output of the model.
            temperature (Float, optional): The temperature for the softmax calculation. Defaults to 1.0.
            num_samples (int): Number of samples to return

        """
        if self.use_mask:
            model_out[..., self.mask_index] = -1.0e9
        samples = torch.multinomial(
            torch.nn.functional.softmax(model_out / temperature, dim=-1).view(-1, self.num_classes),
            num_samples=num_samples,
            generator=self.rng_generator,
        )  # batch * seq_len x num_samples
        if num_samples == 1:
            samples = samples.view(*model_out.shape[:-1])
            # batch x seq_len
        else:
            samples = samples.view((*model_out.shape[:-1], num_samples))
            # batch x seq_len x num_samples
        return samples
