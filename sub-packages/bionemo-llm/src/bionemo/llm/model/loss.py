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
"""A collection of megatron compatable losses"""  # noqa: D415

from typing import Dict, List, Sequence, Tuple, TypedDict, Union

import torch
from megatron.core import parallel_state, tensor_parallel
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.lightning.megatron_parallel import (
    MegatronLossReduction,
    masked_token_loss,
    masked_token_loss_context_parallel,
)


__all__: Sequence[str] = (
    "BERTMLMLossWithReduction",
    "PerTokenLossDict",
    "SameSizeLossDict",
)


class PerTokenLossDict(TypedDict):
    """This is the return type for a loss that is computed per token in the batch, supporting microbatches of varying sizes."""

    loss_sum_and_microbatch_size: torch.Tensor


class SameSizeLossDict(TypedDict):
    """This is the return type for a loss that is computed for the entire batch, where all microbatches are the same size."""

    avg: torch.Tensor


class _Nemo2CompatibleLossReduceMixin:
    """This is a mixin class that provides a general purpose reduce function that is compatible with NeMo2.0 and Megatron-LM.
    Mix this into your loss class to satisfy the abstract `reduce` method, unless you need more
    customization. Before you import this to another file, please refactor to remove the private `_` prefix.
    For now we assume that this is local to this file and not something a user would want to import elsewhere.
    If you do need it, then this assumption was incorrect so please refactor accordingly.

    Since this overrides an abstract parent class, this needs to be put first in the inheritance list to ensure that the correct method is called.
    """

    def old_reduce(
        self, losses_reduced_per_micro_batch: List[Union[PerTokenLossDict, SameSizeLossDict]]
    ) -> torch.Tensor:
        if losses_reduced_per_micro_batch:
            if "avg" in losses_reduced_per_micro_batch[0]:
                loss_tensors_list = [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch]
                loss_tensor = torch.concat(loss_tensors_list)

                return loss_tensor.mean()

            loss_sum_tensors_list: List[torch.Tensor] = [
                loss_sum["loss_sum_and_microbatch_size"]
                for loss_sum in losses_reduced_per_micro_batch
                if loss_sum["loss_sum_and_microbatch_size"][1] > 0
            ]
            dummy_tensor = torch.tensor([0.0, 0.0]).cuda()
            loss_sum = (
                torch.vstack(loss_sum_tensors_list).sum(dim=0) if len(loss_sum_tensors_list) > 0 else dummy_tensor
            )
            return loss_sum

        # If losses_reduced_per_micro_batch is empty, return a dummy tensor.
        dummy_tensor = torch.tensor(0.0).cuda()
        return dummy_tensor

    def reduce(self, losses_reduced_per_micro_batch: List[Union[PerTokenLossDict, SameSizeLossDict]]) -> torch.Tensor:
        # NOTE(SKH) This requires two passes over the data instead of one in the `loss_sum_and_microbatch_size` case.

        # Expect two elements: losses, num_tokens. We only care about the num_tokens index.
        NUM_TOKENS_IDX = 1

        if not losses_reduced_per_micro_batch:
            dummy_tensor = torch.tensor(0.0).cuda()
            return dummy_tensor

        # do the gather
        [only_key] = list(losses_reduced_per_micro_batch[0].keys())
        loss_tensors_list = [loss_reduced[only_key] for loss_reduced in losses_reduced_per_micro_batch]
        # switch on the keys
        if only_key == "avg":
            return torch.concat(loss_tensors_list).mean()
        elif only_key == "loss_sum_and_microbatch_size":
            loss_sum_tensors_list = [
                loss_sum for loss_sum in losses_reduced_per_micro_batch if loss_tensors_list[NUM_TOKENS_IDX] > 0
            ]
            if len(loss_sum_tensors_list) == 0:
                # If we get no result, return zero.
                dummy_tensor = torch.tensor([0.0, 0.0]).cuda()
                return dummy_tensor
            else:
                # otherwise do a sum reduction.
                loss_sum = torch.vstack(loss_sum_tensors_list).sum(dim=0)
                return loss_sum
        else:
            raise ValueError(f"Got a key of {only_key=} but expected one of 'avg, losses_reduced_per_micro_batch'")


class BERTMLMLossWithReduction(_Nemo2CompatibleLossReduceMixin, MegatronLossReduction):
    def __init__(self, validation_step: bool = False, val_drop_last: bool = True) -> None:
        """Initializes the Model class.

        Args:
            validation_step (bool, optional): Whether this object is being applied to the validation step. Defaults to False.
            val_drop_last (bool, optional): Whether the last batch is configured to be dropped during validation. Defaults to True.
        """
        # TODO(@jomitchell): Track down how we handle test. This is a common pattern in NeMo2, but these parameters seem likely
        #  to change in the future.
        super().__init__()
        self.validation_step = validation_step
        self.val_drop_last = val_drop_last

    def unreduced_token_loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes the unreduced token loss given the logits and labels without regard to the loss mask.

        Args:
            logits (Tensor): The predicted logits of shape [batch_size, sequence_length, num_classes].
            labels (Tensor): The true labels of shape [batch_size, sequence_length].

        Returns:
            Tensor: The unreduced token loss of shape [batch_size, sequence_length].
        """
        # [b s] => [s b]  # for both of these for the vocab parallel cross entropy calculation. Is this necessary?
        labels = labels.transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous().float()
        loss = tensor_parallel.vocab_parallel_cross_entropy(logits, labels)

        # [s b] => [b, s]
        loss = loss.transpose(0, 1).contiguous()
        return loss

    def unreduced_sequence_loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO (@jstjohn): implement this function to handle the next sequence prediction task
        # TODO (@jstjohn): determine expected shapes of logits/labels in this case and add that to the docstring
        raise NotImplementedError("Sequence loss not implemented yet.")

    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Union[PerTokenLossDict, SameSizeLossDict]]:
        """Computes loss of `labels` in the batch vs `token_logits` in the forward output currently. In the future this will be extended
            to handle other loss types like sequence loss if it is present in the forward_out and batch.

        Args:
            batch (Dict[str, torch.Tensor]): The batch of data. Each tensor should be of shape [batch_size, *, *],
                and match the corresponding dimension for that particular key in the batch output.
                For example, the "labels" and "token_logits" key should have a tensor of shape [batch_size, sequence_length].
            forward_out (Dict[str, torch.Tensor]): The forward output from the model. Each tensor should be of shape [batch_size, *, *]

        Taken from:
        https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L951-L976 .
        """
        if "labels" not in batch:
            raise ValueError("Labels not provided in the batch. These are required for this loss computation.")

        unreduced_token_loss = self.unreduced_token_loss_fn(forward_out["token_logits"], batch["labels"])

        # TODO(@jstjohn) also handle different output keys, like the sequence loss.

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size == 1:
            # reduce the loss across the micro batch
            loss_for_microbatch = masked_token_loss(unreduced_token_loss, batch["loss_mask"])
        else:
            # reduce the loss across the micro batch.
            # TODO(@jomitchell): Figure out who defines "num_valid_tokens_in_ub" in the batch and document/understand this.
            #  This has something to do with context parallel, and there is probably a megatron or nemo function that adds this and
            #  other necessary keys to the batch. Thanks!
            loss_for_microbatch = masked_token_loss_context_parallel(
                unreduced_token_loss, batch["loss_mask"], batch["num_valid_tokens_in_ub"]
            )

        # If we do not drop the last partial batch of validation, we need to do fancy reduction handling to support
        #  reducing the loss across the data parallel group.
        if self.validation_step and not self.val_drop_last:
            num_valid_tokens_in_microbatch = batch["loss_mask"].sum()
            if loss_for_microbatch.isnan():
                # TODO(@jomitchell): Add a unit test for this. This is the case where there are no valid tokens in the microbatch for the loss
                #  to be computed over, so we expect a NaN loss (divide by zero for a mean) but we make this an expected and non-breaking case,
                #  re-defining it as a 0 loss. This is standard in NeMo/NeMo2.
                if batch["loss_mask"].count_nonzero() != 0:
                    raise ValueError("Got NaN loss with non-empty input")
                loss_sum_for_microbatch = torch.zeros_like(num_valid_tokens_in_microbatch)
            else:
                loss_sum_for_microbatch = num_valid_tokens_in_microbatch * loss_for_microbatch

            # In this case we need to store the loss sum as well as the number of valid tokens in the microbatch.
            loss_sum_and_microbatch_size_all_gpu = torch.cat(
                [
                    loss_sum_for_microbatch.clone().detach().view(1),
                    torch.tensor([num_valid_tokens_in_microbatch]).cuda().clone().detach(),
                ]
            )
            torch.distributed.all_reduce(
                loss_sum_and_microbatch_size_all_gpu, group=parallel_state.get_data_parallel_group()
            )
            return loss_for_microbatch * cp_size, {
                "loss_sum_and_microbatch_size": loss_sum_and_microbatch_size_all_gpu
            }

        # average the losses across the data parallel group, but also return the unreduced loss
        reduced_loss = average_losses_across_data_parallel_group([loss_for_microbatch])
        return loss_for_microbatch * cp_size, {"avg": reduced_loss}
