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


from typing import Dict, Sequence, Tuple

import torch
from megatron.core import parallel_state
from torch import Tensor

from bionemo.llm.model.loss import BERTMLMLossWithReduction, PerTokenLossDict, SameSizeLossDict


__all__: Sequence[str] = (
    "RegressorLossReduction",
    "ClassifierLossReduction",
)


class RegressorLossReduction(BERTMLMLossWithReduction):
    """A class for calculating the MSE loss of regression output.

    This class used for calculating the loss, and for logging the reduced loss across micro batches.
    """

    def forward(
        self, batch: Dict[str, Tensor], forward_out: Dict[str, Tensor]
    ) -> Tuple[Tensor, PerTokenLossDict | SameSizeLossDict]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside classification head.

        Returns:
            A tuple containing [<loss_tensor>, ReductionT] where the loss tensor will be used for
                backpropagation and the ReductionT will be passed to the reduce method
                (which currently only works for logging.).
        """
        regression_output = forward_out["regression_output"]
        targets = batch["labels"].to(dtype=regression_output.dtype)  # [b, 1]

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size == 1:
            loss = torch.nn.functional.mse_loss(regression_output, targets)
        else:
            raise NotImplementedError("Context Parallel support is not implemented for this loss")

        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[SameSizeLossDict]) -> Tensor:
        """Works across micro-batches. (data on single gpu).

        Note: This currently only works for logging and this loss will not be used for backpropagation.

        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """
        losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return losses.mean()


class ClassifierLossReduction(BERTMLMLossWithReduction):
    """A class for calculating the cross entropy loss of classification output.

    This class used for calculating the loss, and for logging the reduced loss across micro batches.
    """

    def forward(
        self, batch: Dict[str, Tensor], forward_out: Dict[str, Tensor]
    ) -> Tuple[Tensor, PerTokenLossDict | SameSizeLossDict]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside classification head.

        Returns:
            A tuple where the loss tensor will be used for backpropagation and the dict will be passed to
            the reduce method, which currently only works for logging.
        """
        targets = batch["labels"].squeeze()  # [b] or [b, s] for sequence-level or token-level classification

        classification_output = forward_out["classification_output"]  # [b, num_class] or [b, s, num_class]
        # [b, s, num_class] -> [b, num_class, s] to satisfy toke-level input dims for cross_entropy loss
        if classification_output.dim() == 3:
            classification_output = classification_output.permute(0, 2, 1)

        loss_mask = batch["loss_mask"]  # [b, s]

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size == 1:
            losses = torch.nn.functional.cross_entropy(classification_output, targets, reduction="none")
            # token-level losses may contain NaNs at masked locations. We use masked_select to filter out these NaNs
            if classification_output.dim() == 3:
                masked_loss = torch.masked_select(losses, loss_mask)
                loss = masked_loss.sum() / loss_mask.sum()
            else:
                loss = losses.mean()  # sequence-level single value classification
        else:
            raise NotImplementedError("Context Parallel support is not implemented for this loss")

        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[SameSizeLossDict]) -> Tensor:
        """Works across micro-batches. (data on single gpu).

        Note: This currently only works for logging and this loss will not be used for backpropagation.

        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """
        losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return losses.mean()
