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


import os
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Type, TypedDict

import numpy as np
import torch
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from bionemo.esm2.api import ESM2GenericConfig, ESM2Model
from bionemo.esm2.data import tokenizer
from bionemo.esm2.model.finetune.dataset import InMemoryCSVDataset
from bionemo.llm.data.collate import MLM_LOSS_IGNORE_INDEX
from bionemo.llm.data.label2id_tokenizer import Label2IDTokenizer
from bionemo.llm.model.biobert.model import BioBertOutput
from bionemo.llm.model.loss import BERTMLMLossWithReduction, PerTokenLossDict, SameSizeLossDict
from bionemo.llm.utils import iomixin_utils as iom


"""This package demonstrates how you can take a pretrained ESM2 module and fine-tune the classifier
token to output secondary structure predictions.
"""

__all__: Sequence[str] = (
    "ClassifierLossReduction",
    "MegatronConvNetHead",
    "ESM2FineTuneTokenModel",
    "ESM2FineTuneTokenConfig",
    "InMemoryPerTokenValueDataset",
    "ClassifierInput",
    "Esm2FineTuneTokenOutput",
)


class ClassifierInput(TypedDict):
    """Used as input in the ClassifierLossReduction's forward method."""

    labels: Tensor
    loss_mask: Tensor


class Esm2FineTuneTokenOutput(BioBertOutput):
    """Inference output from ESM2FineTuneTokenModel."""

    classification_output: Tensor


class ClassifierLossReduction(BERTMLMLossWithReduction):
    """A class for calculating the cross entropy loss of classification output.

    This class used for calculating the loss, and for logging the reduced loss across micro batches.
    """

    def forward(
        self, batch: ClassifierInput, forward_out: Esm2FineTuneTokenOutput
    ) -> Tuple[Tensor, PerTokenLossDict | SameSizeLossDict]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside classification head.

        Returns:
            A tuple where the loss tensor will be used for backpropagation and the dict will be passed to
            the reduce method, which currently only works for logging.
        """
        targets = batch["labels"]  # [b, s]
        # [b, s, num_class] -> [b, num_class, s] to satisfy input dims for cross_entropy loss
        classification_output = forward_out["classification_output"].permute(0, 2, 1)
        loss_mask = batch["loss_mask"]  # [b, s]

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size == 1:
            losses = torch.nn.functional.cross_entropy(classification_output, targets, reduction="none")
            # losses may contain NaNs at masked locations. We use masked_select to filter out these NaNs
            masked_loss = torch.masked_select(losses, loss_mask)
            loss = masked_loss.sum() / loss_mask.sum()
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


class MegatronConvNetHead(MegatronModule):
    """A convolutional neural network class for residue-level classification."""

    def __init__(self, config: TransformerConfig):
        """Constructor."""
        super().__init__(config)

        self.finetune_model = torch.nn.Sequential(
            torch.nn.Conv2d(config.hidden_size, config.cnn_hidden_dim, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(config.cnn_dropout),
        )
        # class_heads (torch.nn.ModuleList): A list of convolutional layers, each corresponding to a different class head.
        # These are used for producing logits scores of varying sizes as specified in `output_sizes`.
        self.class_heads = torch.nn.Conv2d(32, config.cnn_num_classes, kernel_size=(7, 1), padding=(3, 0))

    def forward(self, hidden_states: Tensor) -> List[Tensor]:
        """Inference."""
        # [b, s, h] -> [b, h, s, 1]
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(dim=-1)
        hidden_states = self.finetune_model(hidden_states)  # [b, 32, s, 1]
        output = self.class_heads(hidden_states).squeeze(dim=-1).permute(0, 2, 1)  # [b, s, output_size]
        return output


class ESM2FineTuneTokenModel(ESM2Model):
    """An ESM2 model that is suitable for fine tuning."""

    def __init__(self, config, *args, include_hiddens: bool = False, post_process: bool = True, **kwargs):
        """Constructor."""
        super().__init__(config, *args, include_hiddens=True, post_process=post_process, **kwargs)

        # freeze encoder parameters
        if config.encoder_frozen:
            for _, param in self.named_parameters():
                param.requires_grad = False

        self.include_hiddens_finetuning = (
            include_hiddens  # this include_hiddens is for the final output of fine-tuning
        )
        # If post_process is True that means that we are at the last megatron parallelism stage and we can
        #   apply the head.
        if post_process:
            # if we are doing post process (eg pipeline last stage) then we need to add the output layers
            self.classification_head = MegatronConvNetHead(config)

    def forward(self, *args, **kwargs) -> Tensor | BioBertOutput | Esm2FineTuneTokenOutput:
        """Inference."""
        output: Tensor | BioBertOutput | Esm2FineTuneTokenOutput = super().forward(*args, **kwargs)
        # Stop early if we are not in post_process mode (for example if we are in the middle of model parallelism)
        if not self.post_process:
            return output  # we are not at the last pipeline stage so just return what the parent has
        # Double check that the output from the parent has everything we need to do prediction in this head.
        if not isinstance(output, dict) or "hidden_states" not in output:
            raise ValueError(
                f"Expected to find 'hidden_states' in the output, and output to be dictionary-like, found {output},\n"
                "Make sure include_hiddens=True in the call to super().__init__"
            )
        # Get the hidden state from the parent output, and pull out the [CLS] token for this task
        hidden_states: Tensor = output["hidden_states"]
        # Predict our 1d regression target
        classification_output = self.classification_head(hidden_states)
        if not self.include_hiddens_finetuning:
            del output["hidden_states"]
        output["classification_output"] = classification_output
        return output


@dataclass
class ESM2FineTuneTokenConfig(
    ESM2GenericConfig[ESM2FineTuneTokenModel, ClassifierLossReduction], iom.IOMixinWithGettersSetters
):
    """ExampleConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

    model_cls: Type[ESM2FineTuneTokenModel] = ESM2FineTuneTokenModel
    # typical case is fine-tune the base biobert that doesn't have this head. If you are instead loading a checkpoint
    # that has this new head and want to keep using these weights, please drop this next line or set to []
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=lambda: ["classification_head"])

    encoder_frozen: bool = True  # freeze encoder parameters
    cnn_num_classes: int = 3  # number of classes in each label
    cnn_dropout: float = 0.25
    cnn_hidden_dim: int = 32  # The number of output channels in the bottleneck layer of the convolution.

    def get_loss_reduction_class(self) -> Type[ClassifierLossReduction]:
        """The loss function type."""
        return ClassifierLossReduction


class InMemoryPerTokenValueDataset(InMemoryCSVDataset):
    """An in-memory dataset of labeled strings, which are tokenized on demand."""

    def __init__(
        self,
        data_path: str | os.PathLike,
        tokenizer: tokenizer.BioNeMoESMTokenizer = tokenizer.get_tokenizer(),
        seed: int = np.random.SeedSequence().entropy,  # type: ignore
    ):
        """Initializes a dataset for per-token classification fine-tuning.

        This is an in-memory dataset that does not apply masking to the sequence.

        Args:
            data_path (str | os.PathLike): A path to the CSV file containing sequences and labels (Optional)
            tokenizer (tokenizer.BioNeMoESMTokenizer, optional): The tokenizer to use. Defaults to tokenizer.get_tokenizer().
            seed: Random seed for reproducibility. This seed is mixed with the index of the sample to retrieve to ensure
                that __getitem__ is deterministic, but can be random across different runs. If None, a random seed is
                generated.
        """
        super().__init__(data_path=data_path, tokenizer=tokenizer, seed=seed)
        label_tokenizer = Label2IDTokenizer()
        self.label_tokenizer = label_tokenizer.build_vocab("CHE")
        self.label_cls_eos_id = MLM_LOSS_IGNORE_INDEX

    def transform_label(self, label: str) -> Tensor:
        """Transform the sequence label by tokenizing them.

        This method tokenizes the secondary structure token sequences.

        Args:
            label: secondary structure token sequences to be transformed

        Returns:
            tokenized label
        """
        label_ids = torch.tensor(self.label_tokenizer.text_to_ids(label))

        # # for multi-label classification with BCEWithLogitsLoss
        # tokenized_labels = torch.nn.functional.one_hot(label_ids, num_classes=self.label_tokenizer.vocab_size)
        # cls_eos = torch.full((1, self.label_tokenizer.vocab_size), self.label_cls_eos_id, dtype=tokenized_labels.dtype)

        # for multi-class (mutually exclusive) classification with CrossEntropyLoss
        tokenized_labels = label_ids
        cls_eos = torch.tensor([self.label_cls_eos_id], dtype=tokenized_labels.dtype)

        # add cls / eos label ids with padding value -100 to have the same shape as tokenized_sequence
        labels = torch.cat((cls_eos, tokenized_labels, cls_eos))
        return labels
