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

"""Dataset classes for Evo2 fine-tuning."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from torch.utils.data import Dataset


class InMemoryNucleotideDataset(Dataset):
    """Base dataset class for nucleotide sequences."""

    def __init__(
        self,
        sequences: pd.Series,
        labels: Optional[pd.Series] = None,
        task_type: str = "regression",
        tokenizer=None,
        seed: int = np.random.SeedSequence().entropy,
    ):
        """Initialize the dataset.

        Args:
            sequences: Series of nucleotide sequences
            labels: Series of labels (optional)
            task_type: Type of task ("regression" or "classification")
            tokenizer: Tokenizer to use (defaults to Byte-Level)
            seed: Random seed for reproducibility
        """
        self.sequences = sequences
        self.labels = labels
        self.task_type = task_type
        self.tokenizer = tokenizer if tokenizer else get_nmt_tokenizer("Byte-Level")
        self.seed = seed
        self._generator = np.random.default_rng(seed=self.seed)

        # Simple label tokenizer for classification
        if task_type == "classification" and labels is not None:
            unique_labels = sorted(labels.unique())
            self.label_tokenizer = {label: idx for idx, label in enumerate(unique_labels)}
            self.label_tokenizer_vocab_size = len(unique_labels)
        else:
            self.label_tokenizer = None
            self.label_tokenizer_vocab_size = None

    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get an item from the dataset.

        Args:
            index: Index of the item to retrieve

        Returns:
            Dictionary containing tokenized sequence and optional labels
        """
        sequence = self.sequences.iloc[index]

        # Tokenize sequence
        if hasattr(self.tokenizer, "text_to_ids"):
            # NeMo tokenizer
            tokens = self.tokenizer.text_to_ids(sequence)
        else:
            # HuggingFace tokenizer
            tokens = self.tokenizer(sequence, return_tensors="pt")["input_ids"].squeeze()

        item = {"tokens": torch.tensor(tokens, dtype=torch.long)}

        if self.labels is not None:
            label = self.labels.iloc[index]
            item["labels"] = self.transform_label(label)

        return item

    def transform_label(self, label):
        """Transform label based on task type.

        Args:
            label: Raw label value

        Returns:
            Transformed label tensor
        """
        raise NotImplementedError("Subclasses must implement transform_label")

    @classmethod
    def from_csv(cls, data_path: Path, task_type: str, label_column: str = "labels", **kwargs):
        """Load dataset from CSV file.

        Args:
            data_path: Path to CSV file
            task_type: Type of task ("regression" or "classification")
            label_column: Name of the label column
            **kwargs: Additional arguments for the dataset

        Returns:
            Instance of the dataset
        """
        df = pd.read_csv(data_path)
        sequences = df["sequences"]
        labels = df[label_column] if label_column in df.columns else None
        return cls(sequences=sequences, labels=labels, task_type=task_type, **kwargs)


class InMemorySingleValueDataset(InMemoryNucleotideDataset):
    """Dataset for sequence-level regression/classification tasks."""

    def transform_label(self, label):
        """Transform label for sequence-level tasks.

        Args:
            label: Raw label value

        Returns:
            Tensor containing the transformed label
        """
        if self.task_type == "regression":
            return torch.tensor([label], dtype=torch.float)
        elif self.task_type == "classification":
            if isinstance(label, str):
                label_idx = self.label_tokenizer[label]
            else:
                label_idx = int(label)
            return torch.tensor([label_idx], dtype=torch.long)


class InMemoryPerTokenValueDataset(InMemoryNucleotideDataset):
    """Dataset for token-level classification tasks."""

    def __init__(self, sequences: pd.Series, labels: pd.Series, labels_mask: Optional[pd.Series] = None, **kwargs):
        """Initialize the per-token dataset.

        Args:
            sequences: Series of nucleotide sequences
            labels: Series of per-token labels
            labels_mask: Series of label masks (optional)
            **kwargs: Additional arguments for parent class
        """
        super().__init__(sequences, labels, **kwargs)
        self.labels_mask = labels_mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get an item from the dataset.

        Args:
            index: Index of the item to retrieve

        Returns:
            Dictionary containing tokens, labels, and optional mask
        """
        item = super().__getitem__(index)

        if self.labels_mask is not None:
            mask = self.labels_mask.iloc[index]
            # Parse mask string to tensor
            mask_values = [int(x) for x in mask.split()]
            item["labels_mask"] = torch.tensor(mask_values, dtype=torch.bool)

        return item

    def transform_label(self, label):
        """Transform per-token labels.

        Args:
            label: String or list of per-token labels

        Returns:
            Tensor containing per-token label indices
        """
        # Parse label string to tensor for per-token labels
        if isinstance(label, str):
            label_values = [self.label_tokenizer[x] if x in self.label_tokenizer else 0 for x in label.split()]
        else:
            label_values = label
        return torch.tensor(label_values, dtype=torch.long)

    @classmethod
    def from_csv(
        cls,
        data_path: Path,
        task_type: str,
        label_column: str = "labels",
        labels_mask_column: Optional[str] = None,
        **kwargs,
    ):
        """Load dataset from CSV file.

        Args:
            data_path: Path to CSV file
            task_type: Type of task
            label_column: Name of the label column
            labels_mask_column: Name of the mask column (optional)
            **kwargs: Additional arguments for the dataset

        Returns:
            Instance of the dataset
        """
        df = pd.read_csv(data_path)
        sequences = df["sequences"]
        labels = df[label_column] if label_column in df.columns else None
        labels_mask = df[labels_mask_column] if labels_mask_column and labels_mask_column in df.columns else None
        return cls(sequences=sequences, labels=labels, labels_mask=labels_mask, task_type=task_type, **kwargs)
