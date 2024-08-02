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
import random
import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from bionemo.esm2.data import tokenizer
from bionemo.llm.data import masking
from bionemo.llm.data.types import BertSample


class ProteinSQLiteDataset(Dataset):
    """Dataset for protein sequences stored in a SQLite database."""

    def __init__(self, db_path: str | os.PathLike):
        """Initializes the dataset.

        Args:
            db_path: Path to the SQLite database.
        """
        self.conn = sqlite3.connect(str(db_path))
        self.cursor = self.conn.cursor()

    def __len__(self) -> int:
        """Returns the number of proteins in the dataset.

        Returns:
            Number of proteins in the dataset.
        """
        self.cursor.execute("SELECT COUNT(*) FROM protein")
        return int(self.cursor.fetchone()[0])

    def __getitem__(self, idx: str) -> str:
        """Returns the sequence of a protein at a given index.

        TODO: This method may want to support batched indexing for improved performance.

        Args:
            idx: An identifier for the protein sequence. For training data, these are UniRef90 IDs, while for validation
                data, they are UniRef50 IDs.

        Returns:
            The protein sequence as a string.
        """
        self.cursor.execute("SELECT sequence FROM protein WHERE id = ?", (idx,))
        return self.cursor.fetchone()[0]


class ESMMaskedResidueDataset(Dataset):
    """Dataset class for ESM pretraining that implements cluster sampling of UniRef50 and UniRef90 sequences.

    Megatron-LM expects the input datasets to be indexable, and for the output of the dataset for a given index to be
    deterministic. In cluster sampling, this can be tricky, since we need to perform weighted sampling over UniRef50
    clusters.

    Here, the getitem(i) returns a randomly sampled UniRef90 sequence from the i % len(dataset) UniRef50 cluster, with i
    controlling the random seed used for selecting the UniRef90 sequence and performing the masking.
    """

    def __init__(
        self,
        protein_dataset: ProteinSQLiteDataset,
        clusters: Sequence[Sequence[str]],
        total_samples: int,
        seed: int = np.random.SeedSequence().entropy,  # type: ignore
        max_seq_length: int = 1024,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        mask_random_prob: float = 0.1,
        tokenizer: tokenizer.HFTokenizer = tokenizer.get_tokenizer(),
    ) -> None:
        """Initializes the dataset.

        Args:
            protein_dataset: Dataset containing protein sequences, indexed by UniRef90 ids.
            clusters: UniRef90 ids for all training sequences, bucketed by UniRef50 cluster. Alternatively for
                validation, this can also just a list of UniRef50 ids, with each entry being a length-1 list with a
                single UniRef50 id.
            total_samples: Total number of samples to draw from the dataset.
            seed: Random seed for reproducibility. This seed is mixed with the index of the sample to retrieve to ensure
                that __getitem__ is deterministic, but can be random across different runs. If None, a random seed is
                generated.
            max_seq_length: Crop long sequences to a maximum of this length, including BOS and EOS tokens.
            mask_prob: The overall probability a token is included in the loss function. Defaults to 0.15.
            mask_token_prob: Proportion of masked tokens that get assigned the <MASK> id. Defaults to 0.8.
            mask_random_prob: Proportion of tokens that get assigned a random natural amino acid. Defaults to 0.1.
            tokenizer: The input ESM tokenizer. Defaults to the standard ESM tokenizer.
        """
        self.protein_dataset = protein_dataset
        self.clusters = clusters
        self.total_samples = total_samples
        self.seed = seed
        self.max_seq_length = max_seq_length

        if tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer does not have a mask token.")

        self.mask_config = masking.BertMaskConfig(
            mask_token=tokenizer.mask_token_id,
            random_tokens=range(4, 24),
            mask_prob=mask_prob,
            mask_token_prob=mask_token_prob,
            random_token_prob=mask_random_prob,
        )

        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Returns the total number of samples to be drawn.

        !!! note

            This is neither the actual number of clusters in the dataset nor the number of total sequences; since
            dataset[i] draws from the i % (num_clusters) cluster.

        """
        return self.total_samples

    def __getitem__(self, idx: int) -> BertSample:
        """Deterministically masks and returns a protein sequence from the dataset.

        This method samples from the i % len(dataset) cluster from the input clusters list. Random draws of the same
        cluster can be achieved by calling this method with i + len(dataset), i.e., wrapping around the dataset length.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A (possibly-truncated), masked protein sequence with CLS and EOS tokens and associated mask fields.
        """
        if idx not in range(len(self)):
            raise IndexError(f"Index {idx} out of range [0, {len(self)}).")

        # Initialize a random number generator with a seed that is a combination of the dataset seed and the index.
        rng = np.random.default_rng([self.seed, idx])
        cluster_idx = idx % len(self.clusters)
        if not len(self.clusters[cluster_idx]):
            raise ValueError(f"Cluster {cluster_idx} is empty.")

        sequence_id = rng.choice(self.clusters[cluster_idx])
        sequence = self.protein_dataset[sequence_id]

        # We crop the sequence to a maximum length of max_seq_length - 2 to account for the CLS and EOS tokens.
        cropped_sequence = _random_crop_string(sequence, self.max_seq_length - 2)

        # We don't want special tokens before we pass the input to the masking function; we add these in the collate_fn.
        tokenized_sequence = self._tokenize(cropped_sequence)

        torch_seed = rng.integers(low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max)
        masked_sequence, labels, loss_mask = masking.apply_bert_pretraining_mask(
            tokenized_sequence=tokenized_sequence,  # type: ignore
            random_seed=torch_seed,
            mask_config=self.mask_config,
        )

        masked_sequence, labels, loss_mask = masking.add_cls_and_eos_tokens(
            masked_sequence,
            labels,
            loss_mask,
            cls_token=self.tokenizer.cls_token_id,
            eos_token=self.tokenizer.eos_token_id,
        )

        return {
            "text": masked_sequence,
            "types": torch.zeros_like(masked_sequence, dtype=torch.int64),
            "attention_mask": torch.ones_like(masked_sequence, dtype=torch.int64),
            "labels": labels,
            "loss_mask": loss_mask,
            "is_random": torch.zeros_like(masked_sequence, dtype=torch.int64),
        }

    def _tokenize(self, sequence: str) -> torch.Tensor:
        """Tokenize a protein sequence.

        Args:
            sequence: The protein sequence.

        Returns:
            The tokenized sequence.
        """
        tensor = self.tokenizer.encode(sequence, add_special_tokens=False, return_tensors="pt")
        return tensor.flatten()  # type: ignore


def create_train_dataset(
    cluster_file: str | os.PathLike,
    db_path: str | os.PathLike,
    total_samples: int,
    seed: int,
    max_seq_length: int = 1024,
    mask_prob: float = 0.15,
    mask_token_prob: float = 0.8,
    mask_random_prob: float = 0.1,
    tokenizer: tokenizer.HFTokenizer = tokenizer.get_tokenizer(),
):
    """Creates a training dataset for ESM pretraining.

    Args:
        cluster_file: Path to the cluster file. The file should contain a "ur90_id" column, where each row contains a
            list of UniRef90 ids for a single UniRef50 cluster.
        db_path: Path to the SQLite database.
        total_samples: Total number of samples to draw from the dataset.
        seed: Random seed for reproducibility.
        max_seq_length: Crop long sequences to a maximum of this length, including BOS and EOS tokens.
        mask_prob: The overall probability a token is included in the loss function. Defaults to 0.15.
        mask_token_prob: Proportion of masked tokens that get assigned the <MASK> id. Defaults to 0.8.
        mask_random_prob: Proportion of tokens that get assigned a random natural amino acid. Defaults to 0.1.
        tokenizer: The input ESM tokenizer. Defaults to the standard ESM tokenizer.

    Returns:
        A dataset for ESM pretraining.

    Raises:
        ValueError: If the cluster file does not exist, the database file does not exist, or the cluster file does not
            contain a "ur90_id" column.
    """
    if not Path(cluster_file).exists():
        raise ValueError(f"Cluster file {cluster_file} not found.")

    if not Path(db_path).exists():
        raise ValueError(f"Database file {db_path} not found.")

    cluster_df = pd.read_parquet(cluster_file)
    if "ur90_id" not in cluster_df.columns:
        raise ValueError(f"Training cluster file must contain a 'ur90_id' column. Found columns {cluster_df.columns}.")

    protein_dataset = ProteinSQLiteDataset(db_path)
    return ESMMaskedResidueDataset(
        protein_dataset=protein_dataset,
        clusters=cluster_df["ur90_id"],
        total_samples=total_samples,
        seed=seed,
        max_seq_length=max_seq_length,
        mask_prob=mask_prob,
        mask_token_prob=mask_token_prob,
        mask_random_prob=mask_random_prob,
        tokenizer=tokenizer,
    )


def create_valid_dataset(
    cluster_file: str | os.PathLike,
    db_path: str | os.PathLike,
    total_samples: int,
    seed: int,
    max_seq_length: int = 1024,
    mask_prob: float = 0.15,
    mask_token_prob: float = 0.8,
    mask_random_prob: float = 0.1,
    tokenizer: tokenizer.HFTokenizer = tokenizer.get_tokenizer(),
):
    """Creates a validation dataset for ESM pretraining.

    Args:
        cluster_file: Path to the cluster file. The file should contain a single column named "ur50_id" with UniRef50
            IDs, with one UniRef50 ID per row.
        db_path: Path to the SQLite database.
        total_samples: Total number of samples to draw from the dataset.
        seed: Random seed for reproducibility.
        max_seq_length: Crop long sequences to a maximum of this length, including BOS and EOS tokens.
        mask_prob: The overall probability a token is included in the loss function. Defaults to 0.15.
        mask_token_prob: Proportion of masked tokens that get assigned the <MASK> id. Defaults to 0.8.
        mask_random_prob: Proportion of tokens that get assigned a random natural amino acid. Defaults to 0.1.
        tokenizer: The input ESM tokenizer. Defaults to the standard ESM tokenizer.

    Returns:
        A dataset for ESM pretraining.

    Raises:
        ValueError: If the cluster file does not exist, the database file does not exist, or the cluster file does not
            contain a "ur50_id" column.
    """
    if not Path(cluster_file).exists():
        raise ValueError(f"Cluster file {cluster_file} not found.")

    if not Path(db_path).exists():
        raise ValueError(f"Database file {db_path} not found.")

    protein_dataset = ProteinSQLiteDataset(db_path)

    cluster_df = pd.read_parquet(cluster_file)
    if "ur50_id" not in cluster_df.columns:
        raise ValueError(
            f"Validation cluster file must contain a 'ur50_id' column. Found columns {cluster_df.columns}."
        )

    # Create a single bucket for each UniRef50 cluster.
    clusters = cluster_df["ur50_id"].apply(lambda x: [x])
    return ESMMaskedResidueDataset(
        protein_dataset=protein_dataset,
        clusters=clusters,
        total_samples=total_samples,
        seed=seed,
        max_seq_length=max_seq_length,
        mask_prob=mask_prob,
        mask_token_prob=mask_token_prob,
        mask_random_prob=mask_random_prob,
        tokenizer=tokenizer,
    )


def _random_crop_string(s: str, crop_length: int):
    """Randomly crops a string to a maximum length."""
    if crop_length > len(s):
        return s

    start_index = random.randint(0, len(s) - crop_length)
    cropped_string = s[start_index : start_index + crop_length]
    return cropped_string
