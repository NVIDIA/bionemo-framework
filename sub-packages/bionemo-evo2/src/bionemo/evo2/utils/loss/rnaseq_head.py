# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Multi-Modal Loss Functions for Parallel Head Training.

This module provides flexible loss abstractions for training multi-modal biological
sequence models with parallel prediction heads (e.g., DNA language modeling, RNA
expression prediction, peptide binding).

Key Features:
    - Base abstraction for custom loss functions
    - Borzoi-style loss combining Multinomial and Poisson NLL
    - Support for sophisticated masking strategies
    - Modular configuration for different data modalities

Usage Example:
    >>> from parallel_head_losses import BorzoiLoss, MultiModalLossConfig
    >>>
    >>> # Configure losses for each modality
    >>> loss_config = MultiModalLossConfig(
    ...     rna_loss_fn=BorzoiLoss(multinomial_weight=5.0, multinomial_resolution=1),
    ...     pep_loss_fn=BorzoiLoss(multinomial_weight=5.0, multinomial_resolution=1),
    ...     rna_loss_weight=1.0,
    ...     pep_loss_weight=1.0
    ... )
    >>>
    >>> # Use in forward pass
    >>> rna_loss = loss_config.compute_rna_loss(predictions, targets, mask)
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _absolute_difference_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Compute per-position absolute difference loss.

    Args:
        predictions: Predicted values
            - Shape: [batch, seq_len, channels]
        targets: Ground truth values
            - Shape: [batch, seq_len, channels]
        epsilon: Small constant for numerical stability
            - default: 1e-7

    Returns:
        Per-position absolute difference loss
            - Shape: [batch, seq_len, channels]
    """
    loss = torch.abs(predictions - targets)
    return loss


def _scale_loss(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """Penalize differences in overall magnitude."""
    pred_mean = predictions.mean(dim=1, keepdim=True).clamp(min=epsilon)
    target_mean = targets.mean(dim=1, keepdim=True).clamp(min=epsilon)

    # Log-space ratio loss
    scale_loss = (torch.log(pred_mean) - torch.log(target_mean)) ** 2
    return scale_loss.expand_as(predictions)


def _borzoi_nll_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    sum_axis: int = 1,
    poisson_weight: float = 3.0,
    multinomial_weight: float = 1.0,
    multinomial_resolution: int = 1,
    epsilon: float = 1e-7,
    poisson_absolute: bool = False,
) -> torch.Tensor:
    """Compute Borzoi negative log-likelihood loss.

    Args:
        predictions: Predicted counts (non-negative)
            - Shape: [batch, seq_len, channels]
        targets: Ground truth counts (non-negative)
            - Shape: [batch, seq_len, channels]
        sum_axis: Axis to sum over for multinomial computation
            - default: 1, seq_len
        poisson_weight: Weight for poisson component
            - default: 1.0
        multinomial_weight: Weight for multinomial component
            - default: 3.0
        multinomial_resolution: Resolution for binning predictions
            - default: 1, no binning
        epsilon: Small constant for numerical stability
            - default: 1e-7
        poisson_absolute: Whether to use absolute value for Poisson loss
            - default: False

    Returns:
        Scalar Borzoi NLL loss
    """
    # Poisson NLL (per-position)
    pred_stable = torch.clamp_min(predictions, epsilon)
    target_stable = torch.clamp_min(targets, epsilon)

    # Per-position Poisson NLL
    poisson_pos = (predictions - targets * torch.log(pred_stable)) / multinomial_resolution

    # Per-position optimal Poisson (shift)
    optimal_poisson_pos = (targets - targets * torch.log(target_stable)) / multinomial_resolution

    # Shifted per-position Poisson
    poisson_loss = poisson_pos - optimal_poisson_pos  # shape: [B, S, C]

    # Get absolute value to prevent negative losses
    if poisson_absolute:
        poisson_loss = torch.abs(poisson_loss)

    # Compute sum over sequence to get single scalar value
    sum_pred = torch.sum(predictions, dim=sum_axis, keepdim=True)  # [B, 1, C]

    sum_pred_stable = torch.clamp_min(sum_pred, epsilon)

    # Compute multinomial probabilities
    multinomial_prob = predictions / sum_pred_stable  # [B, S, C]
    multinomial_prob_stable = torch.clamp_min(multinomial_prob, epsilon)

    # Per-position Multinomial NLL
    positional_loss = -targets * torch.log(multinomial_prob_stable)  # [B, S, C]

    # Total Borzoi Loss = Poisson NLL + (weight x Multinomial NLL) per-position
    region_loss = poisson_weight * poisson_loss + multinomial_weight * positional_loss  # shape: [B, S, C]
    return region_loss


class BaseRegressionLoss(ABC):
    """Base class for regression losses on sequential biological data.

    This abstract class defines the interface that all loss functions must implement
    for use in the multi-modal training pipeline.
    """

    @abstractmethod
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute loss between predictions and targets.

        Args:
            predictions: Model predictions
                - Shape: [batch, seq_len] or [batch, seq_len, channels]
            targets: Ground truth targets
                - Shape: [batch, seq_len] or [batch, seq_len, channels]
            mask: Optional binary mask where 1 = include in loss, 0 = exclude
                - Shape: [batch, seq_len]

        Returns:
            Scalar loss tensor
        """
        pass

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Convenience method to call compute."""
        return self.compute(predictions, targets, mask)


class BorzoiLoss(BaseRegressionLoss):
    """Borzoi-style loss combining Multinomial and Poisson negative log-likelihoods.

    This loss function is designed for count-based biological predictions (e.g.,
    RNA-seq counts, peptide binding intensities) and captures both:

    1. **Multinomial NLL**: Ensures the model learns the correct relative distribution
       across positions (i.e., the "shape" of the signal). This component is weighted
       heavily (default: 5.0) to emphasize learning the distribution pattern.

    2. **Poisson NLL**: Ensures the model learns the correct total magnitude/count
       (i.e., the overall "scale" of the signal). This is scaled by the multinomial
       resolution to balance its contribution.

    Mathematical Formulation:
        For predictions x and targets y:

        sum_pred = sum(x over resolution)
        sum_target = sum(y over resolution)

        poisson_loss = sum_pred - sum_target * log(sum_pred + ε)
        multinomial_prob = x / (sum_pred + ε)
        positional_loss = -sum(y * log(multinomial_prob + ε))

        total_loss = (poisson_loss / resolution) + (weight * positional_loss)

    Args:
        multinomial_weight: Weight for multinomial component (default: 5.0, as in Borzoi)
        multinomial_resolution: Resolution for binning predictions (default: 1, no binning)
        epsilon: Small constant for numerical stability (default: 1e-7)
        reduction: How to reduce the loss ('mean', 'sum', or 'none')
        clamp_predictions: Whether to clamp predictions to non-negative values

    Reference:
        Linder et al. "Predicting RNA-seq coverage from DNA sequence as a
        unifying model of gene regulation" (Borzoi, 2023)
    """

    def __init__(
        self,
        multinomial_weight: float = 5.0,
        multinomial_resolution: int | None = None,
        epsilon: float = 1e-7,
        clamp_predictions: bool = True,
        max_per_position_loss: float = 2000.0,
        dynamic_max_loss: bool = False,
        auxiliary_loss: bool = False,
    ):
        """Initialize BorzoiLoss.

        Args:
            multinomial_weight: Weight for multinomial component (default: 5.0, as in Borzoi)
            multinomial_resolution: Resolution for binning predictions (default: 1, no binning)
            epsilon: Small constant for numerical stability (default: 1e-7)
            clamp_predictions: Whether to clamp predictions to non-negative values (default: True)
            max_per_position_loss: Maximum allowed loss per position to prevent extreme values (default: 500.0)
            dynamic_max_loss: Whether to dynamically adjust max loss based on batch statistics (default: True)
            auxiliary_loss: Whether to include auxiliary absolute difference loss (default: True)
        """
        self.multinomial_weight = multinomial_weight
        self.multinomial_resolution = multinomial_resolution
        self.epsilon = epsilon
        self.clamp_predictions = clamp_predictions
        self.max_per_position_loss = max_per_position_loss
        self.dynamic_max_loss = dynamic_max_loss
        self.auxiliary_loss = auxiliary_loss

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute Borzoi loss between predictions and targets.

        Args:
            predictions: Model predictions
                - Shape: [batch, seq_len] or [batch, seq_len, channels]
            targets: Ground truth targets
                - Shape: [batch, seq_len] or [batch, seq_len, channels]
            mask: Optional binary mask where 1 = include in loss, 0 = exclude
                - Shape: [batch, seq_len]

        Returns:
            Per-position Borzoi loss tensor
                - Shape: [batch, seq_len] or [batch, seq_len, channels]
        """
        # Clamp predictions and targets to non-negative values
        if self.clamp_predictions:
            predictions = torch.clamp(predictions, min=0.0)
        targets = torch.clamp(targets, min=0.0)

        # Standardize to 3D
        if predictions.dim() == 2:
            predictions = predictions.unsqueeze(-1)
            targets = targets.unsqueeze(-1)
            if mask is not None:
                mask = mask.unsqueeze(-1)

        # Extract shapes
        batch_size, seq_len, channels = predictions.shape

        # Set multinomial resolution to 1 for no binning if None
        if self.multinomial_resolution is None:
            self.multinomial_resolution = 1

        # Resolution binning
        if self.multinomial_resolution > 1:
            pad_len = (
                self.multinomial_resolution - (seq_len % self.multinomial_resolution)
            ) % self.multinomial_resolution
            if pad_len > 0:
                predictions = F.pad(predictions, (0, 0, 0, pad_len))
                targets = F.pad(targets, (0, 0, 0, pad_len))
                if mask is not None:
                    mask = F.pad(mask, (0, 0, 0, pad_len), value=0)
                seq_len = predictions.shape[1]

            num_bins = seq_len // self.multinomial_resolution
            predictions = predictions.reshape(batch_size, num_bins, self.multinomial_resolution, channels)
            targets = targets.reshape(batch_size, num_bins, self.multinomial_resolution, channels)
            if mask is not None:
                mask = mask.reshape(batch_size, num_bins, self.multinomial_resolution, channels)
            sum_axis = 2
        else:
            sum_axis = 1

        # Apply mask if provided
        if mask is not None:
            predictions = predictions * mask
            targets = targets * mask
        borzoi_loss = _borzoi_nll_loss(predictions, targets, sum_axis, self.multinomial_weight)
        scale_penalty = _scale_loss(predictions, targets) * 0.5  # weight as needed
        borzoi_loss = borzoi_loss + scale_penalty

        if channels == 1:
            # Reshape to [batch, seq_len] to match DNA loss
            batch_size = borzoi_loss.shape[0]
            # Flatten all dimensions except batch
            borzoi_loss = borzoi_loss.view(batch_size, -1)
            # Reshape targets to [batch, seq_len] to match for loss normalization
            targets = targets.view(batch_size, -1)
            # Reshape predictions to [batch, seq_len] to match for loss normalization
            predictions = predictions.view(batch_size, -1)
        else:
            # TODO: support multichannel. Currently not supported.
            raise NotImplementedError("BorzoiLoss currently only supports single-channel outputs.")

        # Prevent extreme per-position losses
        max_loss = self.max_per_position_loss
        if self.dynamic_max_loss:
            # Dynamically set max loss to 90th percentile of current batch
            max_loss = torch.quantile(borzoi_loss.clone().float(), 0.75).item()
            # Statistically set max loss to median + 2*std of current batch
            # max_loss = borzoi_loss.mean().item() + 2 * borzoi_loss.std().item()
        borzoi_loss = torch.clamp(borzoi_loss, min=0.0, max=max_loss)

        # Add auxiliary total count loss
        if self.auxiliary_loss:
            absolute_diff_loss = _absolute_difference_loss(
                predictions=predictions,
                targets=targets,
            )
            borzoi_loss = borzoi_loss * absolute_diff_loss

        # For debugging print mean and std of loss
        # print(f"BorzoiLoss mean: {borzoi_loss.mean().item():.4f}, std: {borzoi_loss.std().item():.4f}")

        return borzoi_loss


class HuberLoss(BaseRegressionLoss):
    """Smooth L1 loss, robust to outliers."""

    def __init__(self, delta: float = 1.0):
        """Initialize HuberLoss."""
        self.delta = delta

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute Huber loss between predictions and targets."""
        loss = F.huber_loss(predictions, targets, reduction="none", delta=self.delta)

        if mask is not None:
            loss = loss * mask

        return loss


class HybridLoss(BaseRegressionLoss):
    """Combines magnitude loss with distribution loss."""

    def __init__(self, magnitude_weight: float = 1.0, distribution_weight: float = 0.5):
        """Initialize HybridLoss."""
        self.magnitude_weight = magnitude_weight
        self.distribution_weight = distribution_weight

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute Hybrid loss combining magnitude and distribution losses."""
        # L1 for absolute magnitude
        l1_loss = F.l1_loss(predictions, targets, reduction="none")

        # KL div for distribution/shape
        pred_dist = F.softmax(predictions, dim=1)
        target_dist = targets / (targets.sum(dim=1, keepdim=True) + 1e-7)
        kl_loss = F.kl_div(torch.log(pred_dist + 1e-7), target_dist, reduction="none", log_target=False)

        loss = self.magnitude_weight * l1_loss + self.distribution_weight * kl_loss

        if mask is not None:
            loss = loss * mask

        return loss


class PoissonWithDistributionLoss(BaseRegressionLoss):
    """Poisson NLL for magnitude + simple distribution matching."""

    def __init__(
        self,
        poisson_weight: float = 1.0,
        distribution_weight: float = 0.5,
        log_input: bool = False,
        full: bool = False,
        eps: float = 1e-6,
    ):
        """Initialize PoissonWithDistributionLoss."""
        self.poisson_weight = poisson_weight
        self.distribution_weight = distribution_weight
        self.eps = eps

        self.poisson_loss = nn.PoissonNLLLoss(log_input=log_input, full=full, eps=eps, reduction="none")

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute Poisson with distribution loss."""
        # Ensure positivity
        predictions = torch.clamp(predictions, min=self.eps)
        targets = torch.clamp(targets, min=0.0)

        # Flatten if needed
        if predictions.dim() == 3 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)
            targets = targets.squeeze(-1)
            if mask is not None and mask.dim() == 3:
                mask = mask.squeeze(-1)

        # 1. Poisson loss for magnitude
        poisson_loss = self.poisson_loss(predictions, targets)

        # KL div for distribution/shape
        pred_dist = F.softmax(predictions, dim=1)
        target_dist = targets / (targets.sum(dim=1, keepdim=True) + 1e-7)
        kl_loss = F.kl_div(torch.log(pred_dist + 1e-7), target_dist, reduction="none", log_target=False)

        # Combine
        total_loss = self.poisson_weight * poisson_loss + self.distribution_weight * kl_loss

        # Apply mask
        if mask is not None:
            total_loss = total_loss * mask

        return total_loss
