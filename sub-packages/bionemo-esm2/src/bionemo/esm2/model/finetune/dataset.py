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
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch import Tensor
from torch.utils.data import Dataset

from bionemo.esm2.data import tokenizer
from bionemo.llm.data.types import BertSample


class InMemoryCSVDataset(Dataset):
    """An in-memory dataset that tokenize strings into BertSample instances."""

    def __init__(
        self,
        data_path: str | os.PathLike,
        tokenizer: tokenizer.BioNeMoESMTokenizer = tokenizer.get_tokenizer(),
        seed: int = np.random.SeedSequence().entropy,  # type: ignore
    ):
        """Initializes a dataset for single-value regression fine-tuning.

        This is an in-memory dataset that does not apply masking to the sequence. But keeps track of <mask> in the
        dataset sequences provided.

        Args:
            data_path (str | os.PathLike): A path to the CSV file containing sequences and labels (Optional)
            tokenizer (tokenizer.BioNeMoESMTokenizer, optional): The tokenizer to use. Defaults to tokenizer.get_tokenizer().
            seed: Random seed for reproducibility. This seed is mixed with the index of the sample to retrieve to ensure
                that __getitem__ is deterministic, but can be random across different runs. If None, a random seed is
                generated.
        """
        self.sequences, self.labels = self.load_data(data_path)

        self.seed = seed
        self._len = len(self.sequences)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """The size of the dataset."""
        return self._len

    def __getitem__(self, index: int) -> BertSample:
        """Obtains the BertSample at the given index."""
        sequence = self.sequences[index]
        tokenized_sequence = self._tokenize(sequence)

        label = tokenized_sequence if len(self.labels) == 0 else self.transform_label(self.labels[index])
        # Overall mask for a token being masked in some capacity - either mask token, random token, or left as-is
        loss_mask = ~torch.isin(tokenized_sequence, Tensor(self.tokenizer.all_special_ids))

        return {
            "text": tokenized_sequence,
            "types": torch.zeros_like(tokenized_sequence, dtype=torch.int64),
            "attention_mask": torch.ones_like(tokenized_sequence, dtype=torch.int64),
            "labels": label,
            "loss_mask": loss_mask,
            "is_random": torch.zeros_like(tokenized_sequence, dtype=torch.int64),
        }

    def load_data(self, csv_path: str | os.PathLike) -> Tuple[Sequence, Sequence]:
        """Loads data from a CSV file, returning sequences and optionally labels.

        This method should be implemented by subclasses to process labels for their specific dataset.

        Args:
            csv_path (str | os.PathLike): The path to the CSV file containing the data.
            The file is expected to have at least one column named 'sequence'. A 'label' column is optional.

        Returns:
            Tuple[Sequence, Sequence]: A tuple where the first element is a list of sequences and the second element is
            a list of labels. If the 'label' column is not present, an empty list is returned for labels.
        """
        df = pd.read_csv(csv_path)
        sequences = df["sequences"].tolist()

        if "labels" in df.columns:
            labels = df["labels"].tolist()
        else:
            labels = []
        return sequences, labels

    def _tokenize(self, sequence: str) -> Tensor:
        """Tokenize a protein sequence.

        Args:
            sequence: The protein sequence.

        Returns:
            The tokenized sequence.
        """
        tensor = self.tokenizer.encode(sequence, add_special_tokens=True, return_tensors="pt")
        return tensor.flatten()  # type: ignore

    def transform_label(self, label):
        """Transform the label.

        This method should be implemented by subclass if label needs additional transformation.

        Args:
            label: label to be transformed

        Returns:
            transformed_label
        """
        return label
