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


def _borzoi_nll_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    sum_axis: int = 1,
    multinomial_weight: float = 5.0,
    multinomial_resolution: int = 1,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Compute Borzoi negative log-likelihood loss.

    Args:
        predictions: Predicted counts (non-negative)
            - Shape: [batch, seq_len, channels]
        targets: Ground truth counts (non-negative)
            - Shape: [batch, seq_len, channels]
        sum_axis: Axis to sum over for multinomial computation
            - default: 1, seq_len
        multinomial_weight: Weight for multinomial component
            - default: 5.0
        multinomial_resolution: Resolution for binning predictions
            - default: 1, no binning
        epsilon: Small constant for numerical stability
            - default: 1e-7

    Returns:
        Scalar Borzoi NLL loss
    """
    # Poisson NLL
    # =========================================================================================
    # PER POSITION EXAMPLE
    # Position:                        0     1     2     3     4     5     6     7     8
    #                                  ─────────────────────────────────────────────────
    # Predictions:                     2     3    45    75    85    65     1     1     0
    # Targets:                         0     0    50    80    90    70     0     0     0
    # STEP 1: Compute sum_pred and sum_target x log(sum_pred + ε)) per position
    # Position:                        0     1     2     3     4     5     6     7     8
    #                                  ─────────────────────────────────────────────────
    # sum_pred:                        2     3    45    75    85    65     1     1     0
    # sum_target x log(sum_pred + ε)): 0     0    82   150   173   126     0     0     0
    # STEP 2: Compute poisson loss     --------------------------------------------------
    # Compute poisson loss             2     3   -37   -75   -88   -61     1     1     0
    # =========================================================================================
    # Poisson NLL (per-position)
    pred_stable = torch.clamp_min(predictions, epsilon)
    target_stable = torch.clamp_min(targets, epsilon)

    # Per-position Poisson NLL
    poisson_pos = (predictions - targets * torch.log(pred_stable)) / multinomial_resolution

    # Per-position optimal Poisson (shift)
    optimal_poisson_pos = (targets - targets * torch.log(target_stable)) / multinomial_resolution

    # Shifted per-position Poisson
    poisson_loss = poisson_pos - optimal_poisson_pos  # shape: [B, S, C]

    # Multinomial NLL
    # =========================================================================================
    # EXAMPLE from Poisson section above
    # STEP 1: Compute multinomial probabilities
    # Position:     0      1      2      3      4      5      6      7      8
    #             ───────────────────────────────────────────────────────────
    # Pred:         2      3     45     75     85     65      1      1      0
    # Sum_pred:                         277
    #             ────────────────────────────────────────────────────────────
    # Prob:      0.007  0.011  0.162  0.271  0.307  0.235  0.004  0.004  0.000
    #            (2/277)(3/277)(45/277)(75/277)(85/277)(65/277)(1/277)(1/277)
    #
    # STEP 2: Compute positional loss
    # Position:     0      1      2      3      4      5      6      7      8
    #             ───────────────────────────────────────────────────────────
    # Target:       0      0     50     80     90     70      0      0      0
    # Prob:      0.007  0.011  0.162  0.271  0.307  0.235  0.004  0.004  0.00
    #             ───────────────────────────────────────────────────────────
    # -log(prob): 4.96   4.51   1.82   1.31   1.18   1.45   5.52   5.52     ∞
    #             ───────────────────────────────────────────────────────────
    # Loss:        0      0     91.0   104.8  106.2  101.5    0      0      0
    #            (0x4.96)(0x4.51)(50x1.82)(80x1.31)(90x1.18)(70x1.45)
    # =========================================================================================
    # Compute sum over sequence to get single scalar value
    sum_pred = torch.sum(predictions, dim=sum_axis, keepdim=True)  # [B, 1, C]

    sum_pred_stable = torch.clamp_min(sum_pred, epsilon)

    # Compute multinomial probabilities
    multinomial_prob = predictions / sum_pred_stable  # [B, S, C]
    multinomial_prob_stable = torch.clamp_min(multinomial_prob, epsilon)

    # Per-position Multinomial NLL
    positional_loss = -targets * torch.log(multinomial_prob_stable)  # [B, S, C]

    # Total Borzoi Loss = Poisson NLL + (weight x Multinomial NLL) per-position
    region_loss = poisson_loss + multinomial_weight * positional_loss  # shape: [B, S, C]
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
        max_per_position_loss: float = 100.0,
    ):
        """Initialize BorzoiLoss.

        Args:
            multinomial_weight: Weight for multinomial component (default: 5.0, as in Borzoi)
            multinomial_resolution: Resolution for binning predictions (default: 1, no binning)
            epsilon: Small constant for numerical stability (default: 1e-7)
            clamp_predictions: Whether to clamp predictions to non-negative values (default: True)
            max_per_position_loss: Maximum allowed loss per position to prevent extreme values (default: 50.0)
        """
        self.multinomial_weight = multinomial_weight
        self.multinomial_resolution = multinomial_resolution
        self.epsilon = epsilon
        self.clamp_predictions = clamp_predictions
        self.max_per_position_loss = max_per_position_loss

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

        # 🔍 DEBUG: Check targets BEFORE masking
        if mask is not None:
            print("=" * 80)
            print("BEFORE MASKING:")
            print(f"Targets shape: {targets.shape}")
            print(f"Targets min: {targets.min()}, max: {targets.max()}, mean: {targets.mean()}")
            print(f"Number of non-zero targets: {torch.count_nonzero(targets)}")
            print(f"First 10 target values: {targets[0, :10, 0]}")
            print(f"\nMask shape: {mask.shape}")
            print(f"Mask min: {mask.min()}, max: {mask.max()}, mean: {mask.mean()}")
            print(f"Number of non-zero mask values: {torch.count_nonzero(mask)}")
            print(f"First 10 mask values: {mask[0, :10, 0] if mask.dim() == 3 else mask[0, :10]}")
            print("=" * 80)

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

        if channels == 1:
            # Reshape to [batch, seq_len] to match DNA loss
            batch_size = borzoi_loss.shape[0]
            # Flatten all dimensions except batch
            borzoi_loss = borzoi_loss.view(batch_size, -1)
            # Reshape targets to [batch, seq_len] to match for loss normalization
            targets = targets.view(batch_size, -1)
        else:
            # If multichannel, raise error (not supported)
            # Users need to extend this method for multichannel support
            raise NotImplementedError("BorzoiLoss currently only supports single-channel outputs.")

        # Prevent extreme per-position losses
        borzoi_loss = torch.clamp(borzoi_loss, min=0.0, max=self.max_per_position_loss)

        # High coverage regions naturally have higher loss - normalize by this
        total_coverage = torch.clamp_min(targets, self.epsilon).sum(dim=1, keepdim=True)  # shape: [B, 1]

        # Normalize by log(coverage) to compress range
        coverage_factor = torch.log1p(total_coverage)  # log(1 + coverage)

        # Normalize loss keeping shape [batch, seq_len]
        borzoi_loss = borzoi_loss / coverage_factor

        # Return loss per position
        return borzoi_loss
