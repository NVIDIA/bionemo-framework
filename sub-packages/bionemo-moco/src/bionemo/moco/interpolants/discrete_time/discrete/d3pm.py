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

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from bionemo.moco.distributions.prior.distribution import DiscretePriorDistribution
from bionemo.moco.distributions.time.distribution import TimeDistribution
from bionemo.moco.interpolants.base_interpolant import Interpolant
from bionemo.moco.interpolants.discrete_time.utils import safe_index
from bionemo.moco.schedules.noise.discrete_noise_schedules import DiscreteNoiseSchedule


def _is_one_hot(data, num_classes):
    """Check if data is one-hot encoded.

    Parameters:
    - data (Tensor): Input data to check.
    - num_classes (int): Expected number of classes for one-hot encoding.

    Returns:
    - bool: True if data is one-hot encoded, False otherwise.
    """
    if len(data.shape) < 2 or data.shape[-1] != num_classes:
        return False  # Not one-hot if last dim doesn't match num_classes or less than 2D

    # Check if all vectors are one-hot
    return (data.sum(dim=-1) == 1).all() and (data.flatten().shape[0] / num_classes) % 1 == 0


class D3PM(Interpolant):
    """A Discrete Denoising Diffusion Probabilistic Model (D3PM) interpolant."""

    def __init__(
        self,
        time_distribution: TimeDistribution,
        prior_distribution: DiscretePriorDistribution,
        noise_schedule: DiscreteNoiseSchedule,
        device: str = "cpu",
        last_time_idx: int = 0,
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Initializes the D3PM interpolant.

        Args:
            time_distribution (TimeDistribution): The distribution of time steps, used to sample time points for the diffusion process.
            prior_distribution (PriorDistribution): The prior distribution of the variable, used as the starting point for the diffusion process.
            noise_schedule (DiscreteNoiseSchedule): The schedule of noise, defining the amount of noise added at each time step.
            device (str, optional): The device on which to run the interpolant, either "cpu" or a CUDA device (e.g. "cuda:0"). Defaults to "cpu".
            last_time_idx (int, optional): The last time index to consider in the interpolation process. Defaults to 0.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
        """
        # We initialize with CPU due to numerical precision issues on A100 that are not observed on A6000
        super().__init__(time_distribution, prior_distribution, "cpu", rng_generator)
        self.noise_schedule = noise_schedule
        self._loss_function = nn.CrossEntropyLoss(reduction="none")
        self.timesteps = noise_schedule.nsteps
        self.num_classes = prior_distribution.num_classes
        self.terminal_distribution = prior_distribution.prior_dist.to(self.device)
        self._initialize_schedules(self.device)
        self.last_time_idx = last_time_idx
        self.to_device(device)

    def _get_Qt(self, alphas: Tensor) -> Tensor:
        """Calculate the transition matrix Qt based on the terminal distribution.

        The transition matrix Qt represents the probabilities of transitioning from one state to another at a given time step.
        It is calculated based on the terminal distribution, which can be either uniform, a mask, or a custom distribution.
        See Appendix A.2 D3PM https://arxiv.org/pdf/2107.03006 which shows what happens for various prior distributions.

        The terminal distribution can be:
        - Uniform: a uniform distribution over all states.
        - Mask: a mask where the last dimension is 1 and the rest are 0.
        - Custom: a custom distribution provided by the user.

        Args:
            alphas (Tensor): A tensor of probabilities, where each alpha represents the probability of staying in a state at a given time step.

        Returns:
            Tensor: The transition matrix Qt.
        """
        QT = []
        for alpha_t in alphas:
            stay_prob = torch.eye(len(self.terminal_distribution), device=self.device) * alpha_t
            diffuse_prob = (1.0 - alpha_t) * (
                torch.ones(1, len(self.terminal_distribution), device=self.device)
                * (self.terminal_distribution.unsqueeze(0))
            )
            QT.append(stay_prob + diffuse_prob)
        return torch.stack(QT, dim=0)

    def _calculate_transition_matrix(self, alphas: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Calculates the rate transition matrix `Qt`, its cumulative variant `Qt_bar`, and the cumulative variant of the previous time step `Qt_bar_prev`.

        Args:
            alphas (Tensor): A tensor of probabilities, where each alpha represents the probability of staying in a state at a given time step.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the rate transition matrix `Qt`, its cumulative variant `Qt_bar`, and the cumulative variant of the previous time step `Qt_bar_prev`.
        """
        Qt = self._get_Qt(alphas)
        Qt_prev = torch.eye(self.num_classes, device=self.device)
        Qt_bar = []
        for i in range(len(alphas)):
            Qtb = Qt_prev @ Qt[i]
            if torch.any((Qtb.sum(-1) - 1.0).abs() > 1e-4):
                raise ValueError(f"Invalid Distribution for Qt_bar at step {i}")
            Qt_bar.append(Qtb)
            Qt_prev = Qtb
        Qt_bar = torch.stack(Qt_bar)
        Qt_bar_prev = Qt_bar[:-1]
        Qt_prev_pad = torch.eye(self.num_classes, device=self.device)
        Qt_bar_prev = torch.concat([Qt_prev_pad.unsqueeze(0), Qt_bar_prev], dim=0)
        return Qt, Qt_bar, Qt_bar_prev

    def _initialize_schedules(self, device):
        """Initializes the transition matrices for the discrete diffusion process.

        This method computes the rate transition matrix `Qt` and its cumulative variants `Qt_bar` and `Qt_prev_bar`
        based on the provided noise schedule.

        Note:
            `Qt` represents the rate transition matrix, where `Qt[t]` is the transition matrix at time step `t`.
            `Qt_bar` and `Qt_prev_bar` are the cumulative variants of `Qt`, where `Qt_bar[t]` represents the cumulative
            transition matrix from time step `0` to `t`, and `Qt_prev_bar[t]` represents the cumulative transition matrix
            from time step `0` to `t-1`.

        Args:
            device (str): The device on which to compute the transition matrices.
        """
        if self.noise_schedule is None:
            raise ValueError("noise_schedule cannot be None for D3PM")
        alphas = self.noise_schedule.generate_schedule(device=device)
        log_alpha = torch.log(alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self._alpha_bar = torch.exp(log_alpha_bar)
        #! Note to users that the tranditional cosine schedule is a very quick convergence of alpha. Pay close attention to the scheduler here
        Qt, Qt_bar, Qt_prev_bar = self._calculate_transition_matrix(alphas)
        self._Qt = Qt[-self.timesteps :]
        self._Qt_transposed = self._Qt.transpose(1, 2)
        self._Qt_bar = Qt_bar[-self.timesteps :]
        self._Qt_prev_bar = Qt_prev_bar[-self.timesteps :]

    def interpolate(self, data: Tensor, t: Tensor):
        """Interpolate using discrete interpolation method.

        This method implements Equation 2 from the D3PM paper (https://arxiv.org/pdf/2107.03006), which
        calculates the interpolated discrete state `xt` at time `t` given the input data and noise
        via q(xt|x0) = Cat(xt; p = x0*Qt_bar).

        Args:
            data (Tensor): The input data to be interpolated.
            t (Tensor): The time step at which to interpolate.

        Returns:
            Tensor: The interpolated discrete state `xt` at time `t`.
        """
        if not _is_one_hot(data, self.num_classes):
            x1_hot = F.one_hot(data, self.num_classes)
        else:
            x1_hot = data
        ford = safe_index(self._Qt_bar, t - self.last_time_idx, data.device)
        if x1_hot.ndim > 3:  # einsum precision issues on A100 not A6000 for 2D inputs
            ford_prep = ford
            for _ in range(x1_hot.ndim - 2):
                ford_prep = ford_prep.unsqueeze(1)
            probs = (x1_hot.float().unsqueeze(-2) * ford_prep).sum(dim=(-2))
        else:
            probs = torch.einsum("b...j, bji -> b...i", [x1_hot.float(), ford])
        if torch.any((probs.sum(-1) - 1.0).abs() > 1e-4):
            raise ValueError(
                f"**INVALID BEHAVIOR** Probability Distribution does not sum to 1.0 for time {t}. "
                f"**INVESTIGATE YOUR DEVICE PRECISION**: This error has been triggered before on A100 by initializing the Qt terms on gpu. "
                f"Normalized to ensure validity. Original sums: {probs.sum(-1)}",
            )
        xt = self._sample_categorical(torch.log(probs) + 1.0e-6)
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

    def _sample_categorical(self, logits, mask: Optional[Tensor] = None, temperature: Float = 1.0) -> Tensor:
        """Sample a categorical distribution using the Gumbel-Softmax trick.

        This method samples a categorical distribution from the given logits,
        optionally applying a mask and using a specified temperature.

        Args:
            logits (Tensor): The logits of the categorical distribution.
            mask (Optional[Tensor], optional): An optional mask to apply to the noise added to logits. Defaults to None.
            temperature (float, optional): The temperature to use for the Gumbel-Softmax trick. Defaults to 1.0.

        Returns:
            Tensor: A sample from the categorical distribution.
        """
        noise = torch.rand_like(logits)
        noise = torch.clip(noise, 1.0e-6, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        if mask is not None:
            sample = torch.argmax((logits / temperature) + gumbel_noise * mask, dim=-1)
        else:
            sample = torch.argmax((logits / temperature) + gumbel_noise, dim=-1)
        return sample

    def _q_posterior_logits(
        self, model_out: Tensor, t: Tensor, xt: Tensor, model_out_is_logits: bool = True
    ) -> Tensor:
        """Calculate the q-posterior logits using the predicted x0 and the current state xt at time t.

        This method implements Equation 3 from the D3PM paper (https://arxiv.org/pdf/2107.03006), which calculates the q-posterior
        distribution over the previous state x0 given the current state xt and the model output.

        Args:
            model_out (Tensor): The output of the model at the current time step.
            t (Tensor): The current time step.
            xt (Tensor): The current discrete state at time t.
            model_out_is_logits (bool, optional): A flag indicating whether the model output is already in logits form. If True, the output is assumed to be logits; otherwise, it is converted to logits. Defaults to True.

        Returns:
            Tensor: The q-posterior logits.
        """
        if not model_out_is_logits:  # model_out.dtype == torch.int64 or model_out.dtype == torch.int32:
            # Convert model output to logits if it's a categorical distribution
            x0_logits = torch.log(torch.nn.functional.one_hot(model_out, self.num_classes).float() + 1.0e-6)
        else:
            # Otherwise, assume model output is already logits
            x0_logits = model_out.clone()

        # Calculate xt_guess: the predicted probability of xt given x0 and t
        xt_guess = torch.einsum(
            "b...j, bji -> b...i",
            [
                torch.nn.functional.one_hot(xt, self.num_classes).float(),
                safe_index(self._Qt_transposed, t - self.last_time_idx, model_out.device),
            ],
        )

        # Calculate softmaxed x0_logits
        softmaxed = torch.softmax(x0_logits, dim=-1)  # bs, ..., num_classes

        # Calculate x0_guess: the predicted probability of x0 given xt and t-1
        x0_guess = torch.einsum(
            "b...c,bcd->b...d",
            softmaxed,
            safe_index(self._Qt_prev_bar, t - self.last_time_idx, model_out.device),
        )

        # Calculate q-posterior logits
        out = torch.log(xt_guess + 1.0e-6) + torch.log(x0_guess + 1.0e-6)
        t_broadcast = t.reshape((t.shape[0], *[1] * (xt.dim())))
        q_posterior_logits = torch.where(t_broadcast == self.last_time_idx, x0_logits, out)
        return q_posterior_logits

    def step(
        self,
        model_out: Tensor,
        t: Tensor,
        xt: Tensor,
        mask: Optional[Tensor] = None,
        temperature: Float = 1.0,
        model_out_is_logits: bool = True,
    ):
        """Perform a single step in the discrete interpolant method, transitioning from the current discrete state `xt` at time `t` to the next state.

        This step involves:

        1. Computing the predicted q-posterior logits using the model output `model_out` and the current state `xt` at time `t`.
        2. Sampling the next state from the predicted q-posterior distribution using the Gumbel-Softmax trick.

        Args:
            model_out (Tensor): The output of the model at the current time step, which is used to compute the predicted q-posterior logits.
            t (Tensor): The current time step, which is used to index into the transition matrices and compute the predicted q-posterior logits.
            xt (Tensor): The current discrete state at time `t`, which is used to compute the predicted q-posterior logits and sample the next state.
            mask (Optional[Tensor], optional): An optional mask to apply to the next state, which can be used to mask out certain tokens or regions. Defaults to None.
            temperature (Float, optional): The temperature to use for the Gumbel-Softmax trick, which controls the randomness of the sampling process. Defaults to 1.0.
            model_out_is_logits (bool, optional): A flag indicating whether the model output is already in logits form. If True, the output is assumed to be logits; otherwise, it is converted to logits. Defaults to True.

        Returns:
            Tensor: The next discrete state at time `t-1`.
        """
        pred_q_posterior_logits = self._q_posterior_logits(model_out, t, xt, model_out_is_logits)
        nonzero_mask = (t != self.last_time_idx).to(xt.dtype).reshape(xt.shape[0], *([1] * (len(xt.shape))))
        x_next = self._sample_categorical(pred_q_posterior_logits, nonzero_mask, temperature=temperature)
        # # Apply mask if provided
        if mask is not None:
            x_next = x_next * mask
        return x_next

    def loss(
        self,
        logits: Tensor,
        target: Tensor,
        xt: Tensor,
        time: Tensor,
        mask: Optional[Tensor] = None,
        vb_scale: Float = 0.0,
    ):
        """Calculate the cross-entropy loss between the model prediction and the target output.

        The loss is calculated between the batch x node x class logits and the target batch x node. If a mask is provided, the loss is
        calculated only for the non-masked elements. Additionally, if vb_scale is greater than 0, the variational lower bound loss is
        calculated and added to the total loss.

        Args:
            logits (Tensor): The predicted output from the model, with shape batch x node x class.
            target (Tensor): The target output for the model prediction, with shape batch x node.
            xt (Tensor): The current data point.
            time (Tensor): The time at which the loss is calculated.
            mask (Optional[Tensor], optional): The mask for the data point. Defaults to None.
            vb_scale (Float, optional): The scale factor for the variational lower bound loss. Defaults to 0.0.

        Returns:
            Tensor: The calculated loss tensor. If aggregate is True, the loss and variational lower bound loss are aggregated and
            returned as a single tensor. Otherwise, the loss and variational lower bound loss are returned as separate tensors.
        """
        assert target.ndim + 1 == logits.ndim
        loss = self._loss_function(logits.transpose(-1, 1), target.long())
        if mask is not None:
            loss = loss * mask
            num_non_masked_elements = torch.sum(mask, dim=-1)
            loss = torch.sum(loss, dim=(-1)) / num_non_masked_elements
        else:
            loss = torch.sum(loss, dim=(-1)) / logits.size(1)
        if vb_scale > 0:
            target = F.one_hot(target, num_classes=self.num_classes).float()
            true_q_posterior_logits = self._q_posterior_logits(target, time, xt)
            pred_q_posterior_logits = self._q_posterior_logits(logits, time, xt)
            vb_loss = self._variational_lower_bound(true_q_posterior_logits, pred_q_posterior_logits)
            vb_loss = vb_scale * vb_loss
        else:
            vb_loss = 0
        if vb_scale > 0:
            loss += vb_loss
        return loss

    def _variational_lower_bound(self, dist1: Tensor, dist2: Tensor) -> Tensor:
        """Calculate the variational lower bound (VLB) between two distributions.

        The VLB measures the difference between the true and approximate posterior distributions.
        It is used to regularize the model and encourage it to produce more accurate predictions.

        Args:
            dist1 (Tensor): The true posterior distribution.
            dist2 (Tensor): The approximate posterior distribution.

        Returns:
            Tensor: The variational lower bound loss.
        """
        # Flatten dist1 and dist2 to simplify calculations
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)

        # Calculate the VLB
        out = torch.softmax(dist1 + 1.0e-6, dim=-1) * (
            torch.log_softmax(dist1 + 1.0e-6, dim=-1) - torch.log_softmax(dist2 + 1.0e-6, dim=-1)
        )
        # Return the mean of the VLB across all elements
        return out.sum(dim=-1).mean()
