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


import functools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple, Type, Union

import pytorch_lightning as pl
import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.lightning.megatron_parallel import MegatronLossReduction, ReductionT
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset

from bionemo.core.data.resamplers import PRNGResampleDataset
from bionemo.esm2.api import ESM2GenericConfig, ESM2Model
from bionemo.esm2.data import tokenizer
from bionemo.llm.data import collate
from bionemo.llm.model.loss import BERTMLMLossWithReduction, PerTokenLossDict, SameSizeLossDict
from bionemo.llm.utils import iomixin_utils as iom
from bionemo.llm.utils.datamodule_utils import infer_num_samples


"""This package demonstrates how you can take a pretrained geneformer module and fine-tune the classifier
token to output cell type predictions.
"""

__all__ = []


class ClassifierLossReduction(BERTMLMLossWithReduction):
    """A class for calculating the cross entropy loss of classification output.

    This class used for calculating the loss, and for logging the reduced loss across micro batches.
    """

    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Union[PerTokenLossDict, SameSizeLossDict]]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside classification head.

        Returns:
            A tuple containing [<loss_tensor>, ReductionT] where the loss tensor will be used for
                backpropagation and the ReductionT will be passed to the reduce method
                (which currently only works for logging.).
        """
        targets = batch["labels"]
        classification_output = forward_out["classification_output"].permute(0, 2, 1)
        loss_mask = batch["loss_mask"]

        losses = torch.nn.functional.cross_entropy(classification_output, targets, reduction="none")
        masked_loss = losses * loss_mask
        loss = masked_loss.sum() / loss_mask.sum()

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
    """A convolutional neural network class for residue-level classification."""

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.finetune_model = torch.nn.Sequential(
            torch.nn.Conv2d(config.hidden_size, config.cnn_hidden_dim, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(config.cnn_dropout),
        )
        # class_heads (torch.nn.ModuleList): A list of convolutional layers, each corresponding to a different class head.
        # These are used for producing logits scores of varying sizes as specified in `output_sizes`.
        self.class_heads = torch.nn.Conv2d(32, config.cnn_num_classes, kernel_size=(7, 1), padding=(3, 0))

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:  # torch.Size([4, 1280])
        # [b, s, h] -> [b, h, s, 1]
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(dim=-1)
        hidden_states = self.finetune_model(hidden_states)  # [b, 32, s, 1]
        output = self.class_heads(hidden_states).squeeze(dim=-1).permute(0, 2, 1)  # [b, s, output_size]
        return output


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
        hidden_states: torch.Tensor = output["hidden_states"]
        # Predict our 1d regression target
        classification_output = self.classification_head(hidden_states)
        if not self.include_hiddens_finetuning:
            del output["hidden_states"]
        output["classification_output"] = classification_output
        return output


@dataclass
class ESM2FineTuneSeqLenBioBertConfig(ESM2GenericConfig[ESM2FineTuneSeqLengthModel], iom.IOMixinWithGettersSetters):
    """ExampleConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

    model_cls: Type[ESM2FineTuneSeqLengthModel] = ESM2FineTuneSeqLengthModel
    # typical case is fine-tune the base biobert that doesn't have this head. If you are instead loading a checkpoint
    # that has this new head and want to keep using these weights, please drop this next line or set to []
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=lambda: ["classification_head"])

    # A list of integers where each integer represents the output size for each class head.
    cnn_num_classes: int = 3  # number of classes in each label
    cnn_dropout: float = 0.25
    cnn_hidden_dim: int = 32  # The number of output channels in the bottleneck layer of the convolution.

    def get_loss_reduction_class(self) -> Type[MegatronLossReduction]:
        return ClassifierLossReduction


class Label2IDTokenizer(TokenizerSpec):
    """Char Tokenizer for sequence labels.

    Intended to be used for extracting class labels for classification models such as secondary structure prediction
    model, where each class is encoded with a character (ex. "C", "H", "E").

    Examples:
        >>> tokenizer = Label2IDTokenizer()
        >>> seqs = ['CHE', 'CCC', 'EHH']
        >>> tokenizer = tokenizer.build_vocab(s)
    """

    def __init__(self):
        self.vocab = {}
        self._update_index()

    def _update_index(self):
        """Updates the id_to_vocab index based on the current vocab."""
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
        """Convert tokens to indexes/ids.

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
        """Convert Ids to tokens.

        Args:
            ids (List[int]): Contains ids for each token
        Returns:
            (List[str]): Contains tokens
        """
        tokens = []
        for id_ in ids:
            token = self.decode_vocab.get(id_)
            if token is None:
                raise ValueError(f"Do not recognize ID: {id_}")
            tokens.append(token)
        return tokens

    def text_to_ids(self, text: str) -> List[int]:
        """Converts text to ids.

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
        """Builds the vocabulary of the tokenizer from strings.

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


class PerTokenValueDataset(Dataset):
    def __init__(self, tokenizer):
        self.data = [
            (
                "seq_0",
                "KIAIAVGDNLEMNLKLVLVKRESQSSISAEERVVAGRLSIVLVCSDAIIEQSKGFDLDGKAAVDDGGRLR",
                "CEHECHCCHECEHCHCCCHECEECCEEHECHHCEECCCCHCCCHCCCHHHHHCHCHCHCCHHECECCCHE",
            ),
            (
                "seq_1",
                "ISIYNAQEDGWSQGNVLKNGGSKAIVGYNNLAWRAFGVANLTAKYPKPVYIAGI",
                "HCCEHHHHCCHHHHCHCCCHHHHHCHHCHHCHCCCHHCCCHHCHHHCCCHHHHH",
            ),
            (
                "seq_2",
                "EVLEESLDFMYRVDWVEDALGNRRLVELSHGPLTFPDLWLRGVNLLSRKAFIEILLVPKKISRSMHNRKG",
                "ECHHHCHCHHCHCHHCHCHCCHCHCEHCCECCHCEHHCHCHCCHHCHHHCHCCCHCEECHCCHHEECHCH",
            ),
            (
                "seq_3",
                "SHKCGAVDEPREATAKDTVESCTSQVEADHCGCVGPAPPSSRDCSTQDKCSTSQTYSL",
                "CECCHECCCCCCCECCHCHCCCCEECCCHCCCCEHCCCCCHHCCCCCECCCCCCCECC",
            ),
            (
                "seq_4",
                "VKLWEGTTKNETQYGKPHYVIDATFYFDEGHDTTGLMTHANNKTHDTAAHATATVDVVTTE",
                "CHCECEHECEECCHCECCCECEECHHEHCECCEHCHCCEEEEEECHEECHEEHEHEEECCC",
            ),
            (
                "seq_5",
                "CGYEGGPIRSVRKVIRPSKEDMSGRVGVVIIHKSAFARAPDGIICTIKGFLYRWELVNLVYEHALPFLRH",
                "HCHHCHHCEHCHCHHCHHCHCEHCHHHCHCCCCCCEHEHHHHHEECCHCCHCHCHEHHCHHCCCEHCHCC",
            ),
        ]
        self._len = len(self.data)
        self.tokenizer = tokenizer
        label_tokenizer = Label2IDTokenizer()
        self.label_tokenizer = label_tokenizer.build_vocab("CHE")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        sequence = self.data[idx][1]
        tokenized_sequence = self._tokenize(sequence)
        # Overall mask for a token being masked in some capacity - either mask token, random token, or left as-is
        loss_mask = ~torch.isin(tokenized_sequence, torch.tensor(self.tokenizer.all_special_ids))
        labels = self._tokenize_labels(self.data[idx][2])

        return {
            "text": tokenized_sequence,
            "types": torch.zeros_like(tokenized_sequence, dtype=torch.int64),
            "attention_mask": torch.ones_like(tokenized_sequence, dtype=torch.int64),
            "labels": labels,
            "loss_mask": loss_mask,
            "is_random": torch.zeros_like(tokenized_sequence, dtype=torch.int64),
        }

    def _tokenize_labels(self, labels_sequence: str) -> torch.Tensor:
        label_ids = torch.tensor(self.label_tokenizer.text_to_ids(labels_sequence))

        # # for multi-label classification with BCEWithLogitsLoss
        # tokenized_labels = torch.nn.functional.one_hot(label_ids, num_classes=self.label_tokenizer.vocab_size)
        # cls_eos = torch.full((1, self.label_tokenizer.vocab_size), -1, dtype=tokenized_labels.dtype)

        # for multi-class (mutually exclusive) classification with CrossEntropyLoss
        tokenized_labels = label_ids
        cls_eos = torch.tensor([-1], dtype=tokenized_labels.dtype)

        # add cls / eos labels with padding value -1 to have the same shape as tokenized_sequence
        labels = torch.cat((cls_eos, tokenized_labels, cls_eos))
        return labels

    def _tokenize(self, sequence: str) -> torch.Tensor:
        """Tokenize a protein sequence.

        Args:
            sequence: The protein sequence.

        Returns:
            The tokenized sequence.
        """
        tensor = self.tokenizer.encode(sequence, add_special_tokens=True, return_tensors="pt")
        return tensor.flatten()  # type: ignore


class PerTokenValueDataModule(pl.LightningDataModule):
    def __init__(
        self,
        seed: int | None = 42,
        min_seq_length: int | None = None,
        max_seq_length: int = 1024,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        num_workers: int = 10,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        rampup_batch_size: list[int] | None = None,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        mask_random_prob: float = 0.1,
        tokenizer: tokenizer.BioNeMoAutoTokenizer = tokenizer.get_tokenizer(),
    ) -> None:
        super().__init__()
        self._seed = seed
        self._min_seq_length = min_seq_length
        self._max_seq_length = max_seq_length
        self._mask_prob = mask_prob
        self._mask_token_prob = mask_token_prob
        self._mask_random_prob = mask_random_prob
        self._tokenizer = tokenizer

        self._micro_batch_size = micro_batch_size
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._pin_memory = pin_memory

        self.data_sampler = MegatronDataSampler(
            seq_len=max_seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            dataloader_type="single",  # `MegatronPretrainingRandomSampler` from "cyclic" is failing.
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str) -> None:
        """Setup the ESMDataModule.

        Args:
            stage: Unused.

        Raises:
            RuntimeError: If the trainer is not attached, or if the trainer's max_steps is not set.
        """
        del stage  # Unused.

        if not hasattr(self, "trainer") or self.trainer is None:
            raise RuntimeError("Setup should be completed when trainer and config are attached.")

        if self.trainer.max_epochs is not None and self.trainer.max_epochs > 1:
            logging.warning(
                "Trainer is set to run for multiple epochs. This is not recommended due to the same shuffle being used "
                "in each. Instead set max_epochs to 1 and increase the number of max_steps."
            )

        max_train_steps = self.trainer.max_steps
        if max_train_steps <= 0:
            raise RuntimeError("Please specify trainer.max_steps")

        # Create training dataset
        _train_ds = PerTokenValueDataset(tokenizer=self._tokenizer)
        num_train_samples = int(max_train_steps * self.data_sampler.global_batch_size)

        self._train_ds = self._sample_and_shuffle_dataset(
            _train_ds, num_train_samples, "train"
        )  # shuffle manually without cyclic MegatronPretrainingRandomSampler

        # Create validation dataset
        _valid_ds = PerTokenValueDataset(tokenizer=self._tokenizer)
        num_val_samples = infer_num_samples(
            limit_batches=self.trainer.limit_val_batches,
            num_samples_in_dataset=len(_valid_ds),
            global_batch_size=self.data_sampler.global_batch_size,
            stage="val",
        )
        self._valid_ds = self._sample_and_shuffle_dataset(
            _valid_ds, num_val_samples, "val"
        )  # shuffle manually without cyclic MegatronPretrainingRandomSampler

        assert (
            hasattr(self, "trainer") and self.trainer is not None
        ), "Setup should be completed when trainer and config are attached."

    def _create_dataloader(self, dataset, **kwargs) -> torch.utils.data.DataLoader:
        assert self._tokenizer.pad_token_id is not None, "Tokenizer must have a pad token id."

        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            collate_fn=functools.partial(
                collate.bert_padding_collate_fn,
                padding_value=self._tokenizer.pad_token_id,
                min_length=self._min_seq_length,
                max_length=self._max_seq_length,
            ),
            **kwargs,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Returns the dataloader for training data."""
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Returns the dataloader for validation data."""
        return self._create_dataloader(self._valid_ds)

    def _sample_and_shuffle_dataset(self, dataset: Dataset, num_samples: int):
        """Sample the training dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from
            num_sample (int): number of samples to generate

        Returns:
            ResamplingMappedDataset: Resampled dataset

        """
        # This is where re-sampling occurs.
        return PRNGResampleDataset(
            dataset,
            num_samples=num_samples,
            seed=self._seed,
        )
