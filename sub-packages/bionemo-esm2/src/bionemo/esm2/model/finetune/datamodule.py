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

import pytorch_lightning as pl
import torch
import torch.utils.data
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from bionemo.core.data.multi_epoch_dataset import IdentityMultiEpochDatasetWrapper, MultiEpochDatasetResampler
from bionemo.esm2.data import tokenizer
from bionemo.esm2.model.finetune.finetune_regressor import InMemorySingleValueDataset
from bionemo.esm2.model.finetune.finetune_token_classifier import InMemoryPerTokenValueDataset
from bionemo.llm.data import collate
from bionemo.llm.utils.datamodule_utils import infer_num_samples


class ESM2FineTuneDataModule(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for fine-tuning ESM2 models.

    This DataModule is designed to handle the data preparation and loading for fine-tuning ESM2 models.
    It provides a flexible way to create and manage datasets, data loaders, and sampling strategies.
    """

    def __init__(
        self,
        train_dataset: InMemoryPerTokenValueDataset | InMemorySingleValueDataset,
        valid_dataset: InMemoryPerTokenValueDataset | InMemorySingleValueDataset,
        seed: int = 42,
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
        tokenizer: tokenizer.BioNeMoESMTokenizer = tokenizer.get_tokenizer(),
    ) -> None:
        """Initialize the ESM2FineTuneDataModule.

        Args:
            train_dataset: The training dataset.
            valid_dataset: The validation dataset.
            seed: The random seed to use for shuffling the datasets. Defaults to 42.
            min_seq_length: The minimum sequence length for the datasets. Defaults to None.
            max_seq_length: The maximum sequence length for the datasets. Defaults to 1024.
            micro_batch_size: The micro-batch size for the data loader. Defaults to 4.
            global_batch_size: The global batch size for the data loader. Defaults to 8.
            num_workers: The number of worker processes for the data loader. Defaults to 10.
            persistent_workers: Whether to persist the worker processes. Defaults to True.
            pin_memory: Whether to pin the data in memory. Defaults to True.
            rampup_batch_size: The batch size ramp-up schedule. Defaults to None.
            mask_prob: The probability of masking a token. Defaults to 0.15.
            mask_token_prob: The probability of replacing a masked token with a [MASK] token. Defaults to 0.8.
            mask_random_prob: The probability of replacing a masked token with a random token. Defaults to 0.1.
            tokenizer: The tokenizer to use for tokenization. Defaults to the BioNeMoESMTokenizer.

        Returns:
            None
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
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
        num_train_samples = int(max_train_steps * self.data_sampler.global_batch_size)
        self._train_ds = self._create_epoch_based_dataset(self.train_dataset, num_train_samples)

        # Create validation dataset
        num_val_samples = infer_num_samples(
            limit_batches=self.trainer.limit_val_batches,
            num_samples_in_dataset=len(self.valid_dataset),
            global_batch_size=self.data_sampler.global_batch_size,
            stage="val",
        )
        self._valid_ds = self._create_epoch_based_dataset(self.valid_dataset, num_val_samples)

        assert (
            hasattr(self, "trainer") and self.trainer is not None
        ), "Setup should be completed when trainer and config are attached."

    def _create_epoch_based_dataset(
        self,
        dataset: InMemoryPerTokenValueDataset | InMemorySingleValueDataset,
        total_samples: int,
    ):
        return MultiEpochDatasetResampler(
            IdentityMultiEpochDatasetWrapper(dataset), num_samples=total_samples, shuffle=True, seed=self._seed
        )

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

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Raises a not implemented error."""
        raise NotImplementedError("No test dataset provided for ESM2")
