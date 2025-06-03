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

from typing import Dict, List, Literal, Sequence, Tuple, TypedDict

import torch
from megatron.core import tensor_parallel
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from nemo.lightning.megatron_parallel import (
    MegatronLossReduction,
    masked_token_loss,
)
from torch import Tensor


__all__: Sequence[str] = (
    "BERTMLMLossWithReduction",
    "DataParallelGroupLossAndIO",
    "PerTokenLossDict",
    "SameSizeLossDict",
)


# TODO(@sichu) update typing
class PerTokenLossDict(TypedDict):
    """Tensor dictionary for loss.

    This is the return type for a loss that is computed per token in the batch, supporting microbatches of varying sizes.
    """

    loss_sum_and_microbatch_size: Tensor


class SameSizeLossDict(TypedDict):
    """Tensor dictionary for loss.

    This is the return type for a loss that is computed for the entire batch, where all microbatches are the same size.
    """

    avg: Tensor


class DataParallelGroupLossAndIO(TypedDict):
    """Average losses across the data parallel group + the original batch and inference output."""

    avg: Tensor
    batch: dict[str, Tensor]
    forward_out: dict[str, Tensor]


class _Nemo2CompatibleLossReduceMixin:
    """This is a mixin class that provides a general purpose reduce function that is compatible with NeMo2.0 and Megatron-LM.
    Mix this into your loss class to satisfy the abstract `reduce` method, unless you need more
    customization. Before you import this to another file, please refactor to remove the private `_` prefix.
    For now we assume that this is local to this file and not something a user would want to import elsewhere.
    If you do need it, then this assumption was incorrect so please refactor accordingly.

    Since this overrides an abstract parent class, this needs to be put first in the inheritance list to ensure that the correct method is called.

    NOTE (SKH) - This is now dead code.
    """  # noqa: D205

    def old_reduce(self, losses_reduced_per_micro_batch: List[PerTokenLossDict | SameSizeLossDict]) -> Tensor:
        if losses_reduced_per_micro_batch:
            if "avg" in losses_reduced_per_micro_batch[0]:
                loss_tensors_list: list[Tensor] = [
                    loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch
                ]
                loss_tensor = torch.concat(loss_tensors_list)

                return loss_tensor.mean()

            loss_sum_tensors_list: List[Tensor] = [
                loss_sum["loss_sum_and_microbatch_size"]
                for loss_sum in losses_reduced_per_micro_batch
                if loss_sum["loss_sum_and_microbatch_size"][1] > 0
            ]
            dummy_tensor = Tensor([0.0, 0.0]).cuda()
            loss_sum = (
                torch.vstack(loss_sum_tensors_list).sum(dim=0) if len(loss_sum_tensors_list) > 0 else dummy_tensor
            )
            return loss_sum

        # If losses_reduced_per_micro_batch is empty, return a dummy tensor.
        dummy_tensor = Tensor(0.0).cuda()
        return dummy_tensor

    # NOTE: this method reduces across microbatches and cross-device reduction is handled in forward method
    def reduce(self, losses_reduced_per_micro_batch: List[PerTokenLossDict | SameSizeLossDict]) -> Tensor:
        # NOTE(SKH) This requires two passes over the data instead of one in the `loss_sum_and_microbatch_size` case.

        # Expect two elements: losses, num_tokens. We only care about the num_tokens index.
        NUM_TOKENS_IDX = 1

        if not losses_reduced_per_micro_batch:  # model returns zero by default in NeMo2.0
            dummy_tensor = Tensor(0.0).cuda()
            return dummy_tensor

        # do the gather
        keys = list(losses_reduced_per_micro_batch[0].keys())
        assert sum(("avg" in keys, "loss_sum_and_microbatch_size" in keys)) == 1, (
            "Expected only either 'avg' or 'loss_sum_and_microbatch_size' in keys but got both"
        )
        key: Literal["avg", "loss_sum_and_microbatch_size"] = (
            "avg" if "avg" in keys else "loss_sum_and_microbatch_size"
        )

        loss_tensors_list: list[Tensor] = [loss_reduced[key] for loss_reduced in losses_reduced_per_micro_batch]
        # switch on the keys and allow other keys to pass through
        if key == "avg":
            return torch.concat(loss_tensors_list).mean()
        elif key == "loss_sum_and_microbatch_size":
            loss_sum_tensors_list = [
                loss_sum for loss_sum in losses_reduced_per_micro_batch if loss_tensors_list[NUM_TOKENS_IDX] > 0
            ]
            if len(loss_sum_tensors_list) == 0:
                # If we get no result, return zero.
                dummy_tensor = Tensor([0.0, 0.0]).cuda()
                return dummy_tensor
            else:
                # otherwise do a sum reduction.
                loss_sum = torch.vstack(loss_sum_tensors_list).sum(dim=0)
                return loss_sum
        else:
            raise ValueError(f"Unexpected: key must either be 'avg' or 'loss_sum_and_microbatch_size', not {key=}")


class BERTMLMLossWithReduction(MegatronLossReduction):  # noqa: D101
    def __init__(self, validation_step: bool = False, val_drop_last: bool = True) -> None:  # noqa: D107
        super().__init__()
        self.validation_step = validation_step
        self.val_drop_last = val_drop_last

        # NOTE: this handles an unknown scenario.
        self.LEGACY_VALIDATION = False

    def forward(
        self, batch: Dict[str, Tensor], forward_out: Dict[str, Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward impl.

        https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/megatron_parallel.py#L1733

        Note that Method signature is slightly different from NeMo as the NeMo signature is incorrect.
        """
        # neva returns (logits, loss_mask)
        if isinstance(forward_out, tuple):
            # NOTE(SKH): this comes from NeMo- when does this occur? Directly related to the incorrect method signature.
            forward_out, loss_mask = forward_out
            batch["loss_mask"] = loss_mask

        if "labels" not in batch:
            raise ValueError("Labels not provided in the batch. These are required for this loss computation.")

        # NOTE: token_logits is [sequence, batch] but labels and other fields, including the loss are [batch, sequence]
        unreduced_token_loss = unreduced_token_loss_fn(forward_out["token_logits"], batch["labels"])  # [b s]

        loss_sum, num_valid_tokens = masked_token_loss(unreduced_token_loss, batch["loss_mask"])

        if self.validation_step and not self.val_drop_last and loss_sum.isnan():
            assert num_valid_tokens == 0, "Got NaN loss with non-empty input"
            if batch["loss_mask"].count_nonzero() != 0:
                raise ValueError("Got NaN loss with non-empty input")
            loss_sum = torch.zeros_like(num_valid_tokens)

            if self.LEGACY_VALIDATION:
                # In previous implementations we had a custom return for this branch of the conditional, however the use
                #   for this is unclear.
                val_loss_for_microbatch = loss_sum.clone()
                # NOTE(SKH) - Requires a reduce to calculate, but now we do this exclusively in the reduce step. what triggers this?
                loss_sum_and_microbatch_size_all_gpu = 1 / 0
                # NOTE(SKH) have not implemented the loss sum and microbatch all gpu, as this is the reduce step.
                #             unclear how this will impact downstream code.
                return val_loss_for_microbatch, {"loss_sum_and_microbatch_size": loss_sum_and_microbatch_size_all_gpu}

        num_valid_tokens = num_valid_tokens.clone().detach().to(torch.int)
        loss_sum_and_ub_size = torch.cat([loss_sum.clone().detach().view(1), num_valid_tokens.view(1)])
        return loss_sum, num_valid_tokens, {"loss_sum_and_ub_size": loss_sum_and_ub_size}

    def reduce(self, losses_reduced_per_micro_batch) -> torch.Tensor:
        """Loss reduction impl.

        Taken from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L534-L552 .
        """
        if losses_reduced_per_micro_batch:
            if "avg" in losses_reduced_per_micro_batch[0]:
                # legacy behavior, average over the number of microbatches
                avg = [x["avg"] for x in losses_reduced_per_micro_batch]
                loss = torch.cat(avg).mean()
                return loss

            from megatron.core import parallel_state

            loss_sum_and_ub_size = [
                x["loss_sum_and_ub_size"] for x in losses_reduced_per_micro_batch if x["loss_sum_and_ub_size"][1] > 0
            ]
            loss = (
                torch.vstack(loss_sum_and_ub_size).sum(dim=0)
                if len(loss_sum_and_ub_size) > 0
                else torch.tensor([0.0, 0.0], device=torch.cuda.current_device())
            )
            torch.distributed.all_reduce(
                loss,
                group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )
            # average over the total number of tokens across the global batch.
            loss = loss[0] / loss[1]

            return loss

        return torch.tensor(0.0, device=torch.cuda.current_device())


def unreduced_token_loss_fn(logits: Tensor, labels: Tensor, cross_entropy_loss_fusion: bool = False) -> Tensor:
    """Computes the unreduced token loss given the logits and labels without regard to the loss mask.

    WARNING: This function does not apply a loss mask. Also, it does inplace operation on the inputs.

    Args:
        logits (Tensor): The predicted logits of shape [sequence_length, batch_size, num_classes].
        labels (Tensor): The true labels of shape [batch_size, sequence_length].
        cross_entropy_loss_fusion (bool): If True, use the fused kernel version of vocab parallel cross entropy. This
            should generally be preferred for speed as it packs more operations into a single kernel on the GPU. However
            some users have observed reduced training stability when using this method.

    Returns:
        Tensor: The unreduced token loss of shape [batch_size, sequence_length].
    """
    labels = labels.transpose(0, 1).contiguous()  # [b, s] -> [s, b]
    if cross_entropy_loss_fusion:
        loss = fused_vocab_parallel_cross_entropy(logits, labels)
    else:
        loss = tensor_parallel.vocab_parallel_cross_entropy(logits, labels)
    # [s b] => [b, s]
    loss = loss.transpose(0, 1).contiguous()
    return loss


def unreduced_sequence_loss_fn(self, logits: Tensor, labels: Tensor) -> Tensor:
    # TODO (@jstjohn): implement this function to handle the next sequence prediction task
    # TODO (@jstjohn): determine expected shapes of logits/labels in this case and add that to the docstring
    raise NotImplementedError("Sequence loss not implemented yet.")
