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
from typing import Dict, Iterable, List, Sequence, Tuple, Type, Union

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.lightning.megatron_parallel import MegatronLossReduction, ReductionT

from bionemo.esm2.api import ESM2Config, ESM2Model
from bionemo.llm.utils import iomixin_utils as iom


"""This package demonstrates how you can take a pretrained geneformer module and fine-tune the classifier
token to output cell type predictions.
"""

__all__ = []


class ClassifierLossReduction(MegatronLossReduction):
    """A class used for calculating the loss, and for logging the reduced loss across micro batches."""

    def forward(self, batch: Dict[str, torch.Tensor], forward_out: torch.Tensor) -> Tuple[torch.Tensor, ReductionT]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside LitAutoEncoder.

        Returns:
            A tuple containing [<loss_tensor>, ReductionT] where the loss tensor will be used for
                backpropagation and the ReductionT will be passed to the reduce method
                (which currently only works for logging.).
        """
        digits = batch["label"]
        digit_logits = forward_out["classification_output"]
        loss = torch.nn.functional.cross_entropy(digit_logits, digits)  # TODO: @farhadr what happens to loss_mask?
        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[ReductionT]) -> torch.Tensor:
        """Works across micro-batches. (data on single gpu).

        Note: This currently only works for logging and this loss will not be used for backpropagation.

        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """
        mse_losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return mse_losses.mean()


class MegatronConvNetHead(MegatronModule):
    """
    A convolutional neural network class for residue-level classification.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.finetune_model = torch.nn.Sequential(
            torch.nn.Conv2d(config.hidden_size, config.cnn_hidden_dim, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(config.cnn_dropout),
        )
        # class_heads (torch.nn.ModuleList): A list of convolutional layers, each corresponding to a different class head.
        # These are used for producing logits scores of varying sizes as specified in `output_sizes`.
        self.class_heads = torch.nn.ModuleList([])
        for head_size in config.cnn_output_sizes:
            self.class_heads.append(torch.nn.Conv2d(32, head_size, kernel_size=(7, 1), padding=(3, 0)))

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        # [b, s, h] -> [b, h, s, 1]
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(dim=-1)
        hidden_states = self.finetune_model(hidden_states)  # [b, 32, s, 1]
        outputs = []
        for head in self.class_heads:
            output = head(hidden_states)
            outputs.append(output.squeeze(dim=-1).permute(0, 2, 1))  # [b, s, output_size]
        return outputs


class ESM2FineTuneSeqLengthModel(ESM2Model):
    def __init__(self, config, *args, include_hiddens: bool = False, post_process: bool = True, **kwargs):
        super().__init__(config, *args, include_hiddens=True, post_process=post_process, **kwargs)
        self.include_hiddens_finetuning = (
            include_hiddens  # this include_hiddens is for the final output of fine-tuning
        )
        # If post_process is True that means that we are at the last megatron parallelism stage and we can
        #   apply the head.
        if post_process:
            # if we are doing post process (eg pipeline last stage) then we need to add the output layers
            self.classification_head = MegatronConvNetHead(config)

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
        classification_output = self.classification_head(hidden_states)
        if not self.include_hiddens_finetuning:
            del output["hidden_states"]
        output["classification_output"] = classification_output
        return output


@dataclass
class FineTuneSeqLenBioBertConfig(ESM2Config, iom.IOMixinWithGettersSetters):
    """ExampleConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

    model_cls: Type[ESM2FineTuneSeqLengthModel] = ESM2FineTuneSeqLengthModel

    # A list of integers where each integer represents the output size for each class head.
    cnn_output_sizes: List[int] = field(default_factory=lambda: [3])  # number of classes in each label
    cnn_dropout: float = 0.25
    cnn_hidden_dim: int = 32  # The number of output channels in the bottleneck layer of the convolution.

    def get_loss_reduction_class(self) -> Type[ClassifierLossReduction]:
        return ClassifierLossReduction


class Label2IDTokenizer(TokenizerSpec):
    """Initializes simple Char Tokenizer.
    Intended to be used for extracting class labels
    for classification models such as secondary
    structure prediction model, where each class is
    encoded with a character (ex. "C", "H", "E")

        Examples:
            >>> tokenizer = Label2IDTokenizer()
            >>> seqs = ['CHE', 'CCC', 'EHH']
            >>> tokenizer = tokenizer.build_vocab(s)
    """

    def __init__(self):
        self.vocab = {}
        self._update_index()

    def _update_index(self):
        """
        Updates the id_to_vocab index based on the current vocab
        """
        self.decode_vocab = {id_: token for token, id_ in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocab being used."""
        return len(self.vocab)

    def text_to_tokens(self, text: str) -> List[str]:
        return list(text)

    def tokens_to_text(self, tokens):
        return "".join(tokens)

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indexes/ids
        Args:
            tokens (List[str]):  Containing tokens
        Returns:
            (List[int]): Containing ID's for each token
        """
        ids = []
        for token in tokens:
            id_ = self.vocab.get(token)
            if id_ is None:
                raise ValueError(f"Do not recognize token: {token}")
            else:
                ids.append(id_)
        return ids

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert Ids to tokens
        Args:
            ids (List[int]): Containg ids for each token
        Returns:
            (List[str]): Containing tokens
        """
        tokens = []
        for id_ in ids:
            token = self.decode_vocab.get(id_)
            if token is None:
                raise ValueError(f"Do not recognize ID: {id_}")
            tokens.append(token)
        return tokens

    def text_to_ids(self, text: str) -> List[int]:
        """Converts text to ids
        Args:
            text (str): String containing text to convert
        Returns:
            (List[int]): Id's corresponding to the tokenization
            of the text
        """
        tokens = self.text_to_tokens(text)
        return self.tokens_to_ids(tokens)

    def ids_to_text(self, ids):
        tokens = self.ids_to_tokens(ids)
        return self.tokens_to_text(tokens)

    def build_vocab(self, strings: Union[str, Iterable[str]]):
        """Builds the vocabulary of the tokenizer from strings
        Args:
            strings: (Union[str, Iterable[str]]): Strings to
                build the vocabulary with. If a string is supplied,
                then the vocabulary is built from the single string.
                Otherwise, the vocabulary is progressively built
                from all of the strings in `strings`.
        """

        if isinstance(strings, str):
            strings = [strings]

        for string in strings:
            for token in string:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                    self.decode_vocab[self.vocab[token]] = token

        return self
