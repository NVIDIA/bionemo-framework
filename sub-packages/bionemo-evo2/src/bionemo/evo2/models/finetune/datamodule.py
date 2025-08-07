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

"""DataModule for Evo2 fine-tuning."""

from typing import Optional

import torch
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from torch.utils.data import Dataset


class Evo2FineTuneDataModule(PreTrainingDataModule):
    """DataModule for Evo2 fine-tuning tasks."""

    def __init__(
        self,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        global_batch_size: int,
        micro_batch_size: int,
        tokenizer=None,
        min_seq_length: Optional[int] = None,
        max_seq_length: int = 8192,
        num_workers: int = 8,
        persistent_workers: bool = True,
        pin_memory: bool = True,
    ):
        """Initialize the data module.

        Args:
            train_dataset: Training dataset
            valid_dataset: Validation dataset
            global_batch_size: Global batch size across all GPUs
            micro_batch_size: Batch size per GPU
            tokenizer: Tokenizer to use
            min_seq_length: Minimum sequence length (optional)
            max_seq_length: Maximum sequence length
            num_workers: Number of data loading workers
            persistent_workers: Whether to keep workers alive between epochs
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.tokenizer = tokenizer
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length

        super().__init__(
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            seq_length=max_seq_length,
        )

    def train_dataloader(self):
        """Create training dataloader.

        Returns:
            DataLoader for training
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        """Create validation dataloader.

        Returns:
            DataLoader for validation
        """
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        """Collate function to pad sequences and create batch tensors.

        Args:
            batch: List of samples from the dataset

        Returns:
            Dictionary containing batched and padded tensors
        """
        # Find max length in batch
        max_len = min(max(len(item["tokens"]) for item in batch), self.max_seq_length)

        # Filter by min length if specified
        if self.min_seq_length is not None:
            batch = [item for item in batch if len(item["tokens"]) >= self.min_seq_length]
            if not batch:
                raise ValueError(f"No sequences in batch meet minimum length requirement of {self.min_seq_length}")

        # Pad sequences
        tokens_list = []
        labels_list = []
        labels_mask_list = []
        loss_mask_list = []

        for item in batch:
            tokens = item["tokens"][:max_len]
            seq_len = len(tokens)

            # Pad tokens
            if seq_len < max_len:
                padding = torch.zeros(max_len - seq_len, dtype=tokens.dtype)
                tokens = torch.cat([tokens, padding])
            tokens_list.append(tokens)

            # Create loss mask
            loss_mask = torch.zeros(max_len, dtype=torch.float)
            loss_mask[:seq_len] = 1.0
            loss_mask_list.append(loss_mask)

            # Handle labels
            if "labels" in item:
                labels = item["labels"]
                if len(labels.shape) == 0:  # Scalar label
                    labels_list.append(labels.unsqueeze(0))
                else:  # Token-level labels
                    if len(labels) < max_len:
                        padding = torch.zeros(max_len - len(labels), dtype=labels.dtype)
                        labels = torch.cat([labels, padding])
                    else:
                        labels = labels[:max_len]
                    labels_list.append(labels)

            # Handle label masks
            if "labels_mask" in item:
                mask = item["labels_mask"]
                if len(mask) < max_len:
                    padding = torch.zeros(max_len - len(mask), dtype=mask.dtype)
                    mask = torch.cat([mask, padding])
                else:
                    mask = mask[:max_len]
                labels_mask_list.append(mask)

        # Stack into batch tensors
        batch_dict = {
            "tokens": torch.stack(tokens_list),
            "loss_mask": torch.stack(loss_mask_list),
            "position_ids": torch.arange(max_len, dtype=torch.long).unsqueeze(0).expand(len(batch), -1),
        }

        if labels_list:
            # For sequence-level tasks, squeeze the extra dimension
            if labels_list[0].shape[-1] == 1:
                batch_dict["labels"] = torch.stack(labels_list).squeeze(-1)
            else:
                batch_dict["labels"] = torch.stack(labels_list)

        if labels_mask_list:
            batch_dict["labels_mask"] = torch.stack(labels_mask_list)

        return batch_dict
