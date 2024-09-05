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
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Union

import torch
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.lightning.megatron_parallel import (
    MegatronLossReduction,
    masked_token_loss,
    masked_token_loss_context_parallel,
)
from torch import nn

from bionemo.llm.model.biobert.model import BioBertGenericConfig, MegatronBioBertModel
from bionemo.llm.model.loss import BERTMLMLossWithReduction, PerTokenLossDict, SameSizeLossDict
from bionemo.llm.utils import iomixin_utils as iom


"""This package demonstrates how you can take a pretrained geneformer module and fine-tune the classifier
token to output cell type predictions.
"""

__all__ = []


class SequenceLengthRMSEPlusBERTMLMLossWithReduction(BERTMLMLossWithReduction):
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
        """  # noqa: D205
        if "labels" not in batch:
            raise ValueError("Labels not provided in the batch. These are required for this loss computation.")

        unreduced_token_loss = self.unreduced_token_loss_fn(forward_out["token_logits"], batch["labels"])
        regression_output = forward_out["regression_output"]
        n_tokens = batch["attention_mask"].sum(dim=-1, keepdim=True).to(dtype=regression_output.dtype)
        assert len(n_tokens.shape) == 2
        assert n_tokens.shape[-1] == 1
        rmse_loss = torch.nn.functional.mse_loss(regression_output, n_tokens)

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
        loss_for_microbatch = loss_for_microbatch + rmse_loss  # add in the RMSE loss after reducing the logit loss
        # average the losses across the data parallel group, but also return the unreduced loss
        reduced_loss = average_losses_across_data_parallel_group([loss_for_microbatch])
        return loss_for_microbatch * cp_size, {"avg": reduced_loss}


class MegatronRegressionMLPHead(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        # FC layer over just the [CLS] token embedding
        # TODO use bias/activation fusion if requested
        self.linear_fc1 = nn.Linear(in_features=config.hidden_size, out_features=config.ffn_hidden_size)
        self.activation_function = config.activation_func
        self.linear_fc2 = nn.Linear(in_features=config.ffn_hidden_size, out_features=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.activation_function(self.linear_fc1(hidden_states)))


class MegatronBioBertFineTuneSeqLengthModel(MegatronBioBertModel):
    def __init__(self, config, *args, include_hiddens: bool = False, post_process: bool = True, **kwargs):
        super().__init__(config, *args, include_hiddens=True, post_process=post_process, **kwargs)
        self.include_hiddens_finetuning = (
            include_hiddens  # this include_hiddens is for the final output of fine-tuning
        )
        # If post_process is True that means that we are at the last megatron parallelism stage and we can
        #   apply the head.
        if post_process:
            # if we are doing post process (eg pipeline last stage) then we need to add the output layers
            self.regression_head = MegatronRegressionMLPHead(config)

    def forward(
        self,
        *args,
        **kwargs,
    ):
        output = super().forward(*args, **kwargs)
        # Stop early if we are not in post_process mode (for example if we are in the middle of model parallelism)
        if not self.post_process:
            return output  # we are not at the last pipeline stage so just return what the parent has
        # Double check that the output from the parent has everything we need to do prediction in this head.
        if ("hidden_states" not in output) or (not isinstance(output, dict)):
            raise ValueError(
                f"Expected to find 'hidden_states' in the output, and output to be dictionary-like, found {output},\n"
                "Make sure include_hiddens=True in the call to super().__init__"
            )
        # Get the hidden state from the parent output, and pull out the [CLS] token for this task
        hidden_states: torch.Tensor = output["hidden_states"][
            :, 0
        ]  # [b s h] => [b h], use [CLS] (first) token for reg
        # Predict our 1d regression target
        regression_output = self.regression_head(hidden_states)
        if not self.include_hiddens_finetuning:
            del output["hidden_states"]
        output["regression_output"] = regression_output
        return output


@dataclass
class FineTuneSeqLenBioBertConfig(
    BioBertGenericConfig[MegatronBioBertFineTuneSeqLengthModel], iom.IOMixinWithGettersSetters
):
    # When overriding fields in a dataclass _always_ declare types: https://github.com/python/cpython/issues/123269
    model_cls: Type[MegatronBioBertFineTuneSeqLengthModel] = MegatronBioBertFineTuneSeqLengthModel
    # typical case is fine-tune the base biobert that doesn't have this head. If you are instead loading a checkpoint
    # that has this new head and want to keep using these weights, please drop this next line or set to []
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=lambda: ["regression_head"])

    def get_loss_reduction_class(self) -> Type[MegatronLossReduction]:
        return SequenceLengthRMSEPlusBERTMLMLossWithReduction
