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

"""Loss functions for Evo2 fine-tuning."""

from typing import Dict, Tuple

import torch
from megatron.core.models.bert.bert_lm_head import BERTMLMLossWithReduction


class RegressorLossReduction(BERTMLMLossWithReduction):
    """Loss reduction for regression tasks."""

    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute regression loss.

        Args:
            batch: Batch dictionary containing labels
            forward_out: Model output dictionary containing regression_output

        Returns:
            Tuple of (loss_sum, num_valid_tokens, metrics_dict)
        """
        regression_output = forward_out["regression_output"]
        targets = batch["labels"].to(dtype=regression_output.dtype)
        num_valid_tokens = torch.tensor(targets.numel(), dtype=torch.int, device=targets.device)
        loss_sum = ((regression_output - targets) ** 2).sum()
        loss_sum_and_ub_size = torch.cat([loss_sum.clone().detach().view(1), num_valid_tokens.view(1)])
        return loss_sum, num_valid_tokens, {"loss_sum_and_ub_size": loss_sum_and_ub_size}


class ClassifierLossReduction(BERTMLMLossWithReduction):
    """Loss reduction for classification tasks."""

    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute classification loss.

        Args:
            batch: Batch dictionary containing labels
            forward_out: Model output dictionary containing classifier_output

        Returns:
            Tuple of (loss_sum, num_valid_tokens, metrics_dict)
        """
        classifier_output = forward_out["classifier_output"]  # [b, num_classes]
        targets = batch["labels"].long()  # [b]

        # Flatten for loss calculation
        classifier_output_flat = classifier_output.view(-1, classifier_output.size(-1))
        targets_flat = targets.view(-1)

        loss = torch.nn.functional.cross_entropy(classifier_output_flat, targets_flat, reduction="sum")
        num_valid_tokens = torch.tensor(targets.numel(), dtype=torch.int, device=targets.device)

        loss_sum_and_ub_size = torch.cat([loss.clone().detach().view(1), num_valid_tokens.view(1)])
        return loss, num_valid_tokens, {"loss_sum_and_ub_size": loss_sum_and_ub_size}


class TokenClassifierLossReduction(BERTMLMLossWithReduction):
    """Loss reduction for token classification tasks."""

    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute token classification loss.

        Args:
            batch: Batch dictionary containing labels and optional labels_mask
            forward_out: Model output dictionary containing token_classifier_output

        Returns:
            Tuple of (loss_sum, num_valid_tokens, metrics_dict)
        """
        token_classifier_output = forward_out["token_classifier_output"]  # [b, s, num_classes]
        targets = batch["labels"].long()  # [b, s]

        # Get mask if available
        if "labels_mask" in batch:
            mask = batch["labels_mask"]  # [b, s]
        else:
            mask = torch.ones_like(targets, dtype=torch.bool)

        # Flatten everything
        output_flat = token_classifier_output.view(-1, token_classifier_output.size(-1))
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        # Calculate loss only on masked positions
        if mask_flat.any():
            loss = torch.nn.functional.cross_entropy(output_flat[mask_flat], targets_flat[mask_flat], reduction="sum")
            num_valid_tokens = mask_flat.sum()
        else:
            loss = torch.tensor(0.0, device=token_classifier_output.device)
            num_valid_tokens = torch.tensor(0, dtype=torch.int, device=token_classifier_output.device)

        loss_sum_and_ub_size = torch.cat([loss.clone().detach().view(1), num_valid_tokens.view(1)])
        return loss, num_valid_tokens, {"loss_sum_and_ub_size": loss_sum_and_ub_size}
