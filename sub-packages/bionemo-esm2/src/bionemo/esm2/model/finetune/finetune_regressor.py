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
from typing import Dict, List, Sequence, Tuple, Type

import numpy as np
import torch
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor
from torch.utils.data import Dataset

from bionemo.esm2.api import ESM2GenericConfig, ESM2Model
from bionemo.esm2.data import tokenizer
from bionemo.llm.data.types import BertSample
from bionemo.llm.model.biobert.model import BioBertOutput
from bionemo.llm.model.loss import BERTMLMLossWithReduction, PerTokenLossDict, SameSizeLossDict
from bionemo.llm.utils import iomixin_utils as iom


# This package demonstrates how you can take a pretrained ESM2 module and fine-tune the regressor
# to output sequence-level regression predictions.

__all__: Sequence[str] = (
    "RegressorLossReduction",
    "MegatronMLPHead",
    "ESM2FineTuneSeqModel",
    "ESM2FineTuneSeqConfig",
    "InMemorySingleValueDataset",
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
        targets = batch["labels"]  # [b, 1]
        regression_output = forward_out

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size == 1:
            loss = torch.nn.functional.mse_loss(regression_output, targets)
        else:  # TODO: support CP with masked_token_loss_context_parallel
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


class MegatronMLPHead(MegatronModule):
    """An MLP class for sequence-level regression."""

    def __init__(self, config: TransformerConfig):
        """Constructor."""
        super().__init__(config)

        layer_sizes = [config.hidden_size, 256, 1]
        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])]  # noqa: RUF007
        )
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=config.ft_dropout)

    def forward(self, hidden_states: Tensor) -> List[Tensor]:
        """Inference."""
        # [b, s, h]
        for layer in self.linear_layers[:-1]:
            hidden_states = self.dropout(self.act(layer(hidden_states)))

        output = self.linear_layers[-1](hidden_states)
        return output


class ESM2FineTuneSeqModel(ESM2Model):
    """ESM2 model that is suitable for fine-tuning on downstream tasks."""

    def __init__(self, config, *args, post_process: bool = True, return_embeddings: bool = False, **kwargs):
        """Constructs an instance of the ESM2 model suitable for fine-tuning."""
        super().__init__(config, *args, post_process=post_process, return_embeddings=True, **kwargs)

        # freeze encoder parameters
        if config.encoder_frozen:
            for _, param in self.named_parameters():
                param.requires_grad = False

        # If post_process is True that means that we are at the last megatron parallelism stage and we can
        #   apply the head.
        if post_process:
            # if we are doing post process (eg pipeline last stage) then we need to add the output layers
            self.regression_head = MegatronMLPHead(config)

    def forward(self, *args, **kwargs) -> BioBertOutput | Tensor:
        """Inference."""
        output = super().forward(*args, **kwargs)
        # Stop early if we are not in post_process mode (for example if we are in the middle of model parallelism)
        if not self.post_process:
            return output  # we are not at the last pipeline stage so just return what the parent has
        # Double check that the output from the parent has everything we need to do prediction in this head.
        if not isinstance(output, Tensor):
            raise ValueError(f"Expected the output to be the embedding Tensor, found {output}")
        # Get the hidden state from the parent output, and pull out the [CLS] token for this task
        embeddings: Tensor = output
        # Predict our 1d regression target
        regression_output = self.regression_head(embeddings)
        return regression_output


@dataclass
class ESM2FineTuneSeqConfig(
    ESM2GenericConfig[ESM2FineTuneSeqModel, RegressorLossReduction], iom.IOMixinWithGettersSetters
):
    """ExampleConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

    model_cls: Type[ESM2FineTuneSeqModel] = ESM2FineTuneSeqModel
    # typical case is fine-tune the base biobert that doesn't have this head. If you are instead loading a checkpoint
    # that has this new head and want to keep using these weights, please drop this next line or set to []
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=lambda: ["regression_head"])

    encoder_frozen: bool = True  # freeze encoder parameters
    ft_dropout: float = 0.25  # MLP layer dropout

    def get_loss_reduction_class(self) -> Type[RegressorLossReduction]:
        """Returns RegressorLossReduction class."""
        return RegressorLossReduction


class InMemorySingleValueDataset(Dataset):
    """An in-memory dataset that tokenizes strings into BertSample instances."""

    def __init__(
        self,
        data: Sequence[Tuple[str, float]],
        tokenizer: tokenizer.BioNeMoESMTokenizer = tokenizer.get_tokenizer(),
        seed: int = np.random.SeedSequence().entropy,  # type: ignore
    ):
        """Initializes a dataset for single-value regression fine-tuining.

        This is an in-memory dataset that does not apply masking to the sequence.

        Args:
            data (Sequence[Tuple[str, float]]): A sequence of tuples containing the sequence and target data.
            tokenizer (tokenizer.BioNeMoESMTokenizer, optional): The tokenizer to use. Defaults to tokenizer.get_tokenizer().
            seed: Random seed for reproducibility. This seed is mixed with the index of the sample to retrieve to ensure
                that __getitem__ is deterministic, but can be random across different runs. If None, a random seed is
                generated.
        """
        self.data = data
        self.seed = seed
        self._len = len(self.data)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """The size of the dataset."""
        return self._len

    def __getitem__(self, index: int) -> BertSample:
        """Obtains the BertSample at the given index."""
        sequence, target = self.data[index]
        tokenized_sequence = self._tokenize(sequence)
        # Overall mask for a token being masked in some capacity - either mask token, random token, or left as-is
        loss_mask = ~torch.isin(tokenized_sequence, Tensor(self.tokenizer.all_special_ids))

        return {
            "text": tokenized_sequence,
            "types": torch.zeros_like(tokenized_sequence, dtype=torch.int64),
            "attention_mask": torch.ones_like(tokenized_sequence, dtype=torch.int64),
            "labels": torch.tensor([target], dtype=torch.float),
            "loss_mask": loss_mask,
            "is_random": torch.zeros_like(tokenized_sequence, dtype=torch.int64),
        }

    def _tokenize(self, sequence: str) -> Tensor:
        """Tokenize a protein sequence.

        Args:
            sequence: The protein sequence.

        Returns:
            The tokenized sequence.
        """
        tensor = self.tokenizer.encode(sequence, add_special_tokens=True, return_tensors="pt")
        return tensor.flatten()  # type: ignore
