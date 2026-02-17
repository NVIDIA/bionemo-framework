# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Hybrid HuggingFace windowed dataset with on-the-fly tokenization.

This module provides a memory-efficient approach to genomic sequence windowing:
1. Store only window mappings (sequence_idx, start_position) - ~4 GB for 238M windows
2. Load raw sequences from HF Arrow cache (memory-mapped, ~800 GB)
3. Tokenize on-the-fly during training (fast, ~50 μs per window)

This approach combines the benefits of:
- HuggingFace's efficient Arrow storage (fast random access)
- ShardedEden's mapping strategy (small footprint)
- DistributedSampler for true global shuffling

Usage:
    from hf_windowed_dataset import HFWindowedDataset, create_hf_windowed_dataloader

    # Create dataloader
    dataloader, sampler = create_hf_windowed_dataloader(
        distributed_config=dist_config,
        raw_dataset_path="/root/.cache/huggingface/datasets/metagenomes/...",
        mappings_path="/data/opengenome2/cache/metagenome_window_mappings.npy",
        tokenizer_name_or_path="./tokenizers/nucleotide_fast_tokenizer",
        micro_batch_size=8,
    )

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            # batch has input_ids, attention_mask, labels
            ...
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import datasets
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling

from distributed_config import DistributedConfig
from genomic_dataset import GenomicDataCollator


logger = logging.getLogger(__name__)


class HFWindowedDataset(Dataset):
    """PyTorch Dataset that combines HF Arrow storage with window mappings.

    This dataset provides efficient windowed access to genomic sequences:
    - Raw sequences stored in HF Arrow format (memory-mapped, ~800 GB)
    - Window mappings stored as numpy array (~4 GB for 238M windows)
    - Tokenization happens on-the-fly (~50 μs per window)

    The design mirrors ShardedEdenDataset but uses HuggingFace infrastructure
    instead of SQLite.
    """

    def __init__(
        self,
        raw_dataset: datasets.Dataset,
        mappings: np.ndarray,
        tokenizer: PreTrainedTokenizerBase,
        window_size: int = 8190,
        seq_length: int = 8192,
        text_column: str = "text",
        pad_to_seq_length: bool = True,
    ) -> None:
        """Initialize the HFWindowedDataset.

        Args:
            raw_dataset: HuggingFace Dataset with raw text sequences.
            mappings: Numpy array of shape (num_windows, 2) with (seq_idx, start_pos).
            tokenizer: HuggingFace tokenizer for on-the-fly tokenization.
            window_size: Size of text window to extract (default: 8190 chars).
            seq_length: Target sequence length after tokenization (default: 8192).
            text_column: Name of the column containing text (default: "text").
            pad_to_seq_length: Whether to pad sequences to seq_length (default: True).
        """
        self.raw_dataset = raw_dataset
        self.mappings = mappings
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.seq_length = seq_length
        self.text_column = text_column
        self.pad_to_seq_length = pad_to_seq_length

        # Token IDs for padding
        self._pad_id = tokenizer.pad_token_id
        if self._pad_id is None:
            self._pad_id = tokenizer.eos_token_id

        logger.info(
            f"HFWindowedDataset initialized: {len(self.mappings):,} windows, {len(self.raw_dataset):,} sequences"
        )

    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.mappings)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        """Get a single window.

        Args:
            idx: Window index.

        Returns:
            Dict with input_ids and attention_mask.
        """
        # 1. Look up mapping
        seq_idx, start_pos = self.mappings[idx]

        # 2. Fetch raw sequence from Arrow cache (memory-mapped, fast)
        raw_text = self.raw_dataset[int(seq_idx)][self.text_column]

        # 3. Slice the window
        window_text = raw_text[start_pos : start_pos + self.window_size]

        # 4. Tokenize on-the-fly
        encoding = self.tokenizer(
            window_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.seq_length,
            return_attention_mask=False,  # We'll create it ourselves for consistency
        )

        input_ids = encoding["input_ids"]
        original_len = len(input_ids)

        # 5. Pad to seq_length if requested
        if self.pad_to_seq_length:
            if original_len < self.seq_length:
                padding_length = self.seq_length - original_len
                input_ids = input_ids + [self._pad_id] * padding_length
                attention_mask = [1] * original_len + [0] * padding_length
            else:
                input_ids = input_ids[: self.seq_length]
                attention_mask = [1] * self.seq_length
        else:
            attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"HFWindowedDataset(windows={len(self.mappings):,}, "
            f"sequences={len(self.raw_dataset):,}, "
            f"window_size={self.window_size}, seq_length={self.seq_length})"
        )


def load_raw_dataset(
    path: str,
    split: str | None = None,
) -> datasets.Dataset:
    """Load raw dataset from path or HF cache.

    Args:
        path: Path to parquet directory or HF cache directory.
        split: Optional split name (for parquet loading).

    Returns:
        HuggingFace Dataset with raw sequences.
    """
    path_obj = Path(path)

    # Check if it's an HF cache directory (has dataset_info.json or state.json)
    if (path_obj / "dataset_info.json").exists() or (path_obj / "state.json").exists():
        logger.info(f"Loading from HF cache: {path}")
        dataset = datasets.load_from_disk(path)
        if isinstance(dataset, datasets.DatasetDict):
            if split and split in dataset:
                return dataset[split]
            else:
                # Return first available split
                first_split = next(iter(dataset.keys()))
                logger.warning(f"Using split '{first_split}' from DatasetDict")
                return dataset[first_split]
        return dataset  # type: ignore[return-value]

    # Otherwise treat as parquet directory
    logger.info(f"Loading from parquet: {path}")
    if split is None:
        split = "train"
    dataset = datasets.load_dataset(path, split=split, streaming=False)
    assert isinstance(dataset, datasets.Dataset)
    return dataset


def load_mappings(mappings_path: str) -> tuple[np.ndarray, dict]:
    """Load window mappings and metadata.

    Args:
        mappings_path: Path to the .npy mappings file.

    Returns:
        Tuple of (mappings array, metadata dict).
    """
    mappings_path = Path(mappings_path)

    # Load mappings
    mappings = np.load(mappings_path)
    logger.info(f"Loaded mappings: {mappings.shape[0]:,} windows, {mappings.nbytes / 1e9:.2f} GB")

    # Load metadata if available
    metadata_path = mappings_path.with_suffix(".json")
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return mappings, metadata


def create_hf_windowed_dataloader(
    distributed_config: DistributedConfig,
    raw_dataset_path: str,
    mappings_path: str,
    tokenizer_name_or_path: str,
    micro_batch_size: int = 8,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    window_size: int = 8190,
    seq_length: int = 8192,
    seed: int = 42,
    text_column: str = "text",
    uppercase_labels: bool = False,
    mask_degenerate_bases: bool = False,
    pad_sequences_to_be_divisible_by: int | None = None,
    split: str | None = None,
) -> tuple[DataLoader[Any], DistributedSampler[HFWindowedDataset]]:
    """Create a dataloader using the hybrid HF windowed dataset approach.

    This function creates a memory-efficient dataloader that:
    1. Uses pre-computed window mappings (~4 GB)
    2. Loads raw sequences from HF Arrow cache (~800 GB, memory-mapped)
    3. Tokenizes on-the-fly (~50 μs per window)
    4. Uses DistributedSampler for true global shuffling

    Args:
        distributed_config: Distributed training configuration.
        raw_dataset_path: Path to raw dataset (parquet or HF cache).
        mappings_path: Path to window mappings .npy file.
        tokenizer_name_or_path: Path to HuggingFace tokenizer.
        micro_batch_size: Batch size per GPU.
        num_workers: Number of DataLoader workers.
        prefetch_factor: Prefetch factor for DataLoader.
        window_size: Size of text window to extract (default: 8190).
        seq_length: Target sequence length after tokenization (default: 8192).
        seed: Random seed for DistributedSampler.
        text_column: Name of text column in dataset.
        uppercase_labels: Whether to uppercase labels (genomic masking).
        mask_degenerate_bases: Whether to mask non-ACGT bases.
        pad_sequences_to_be_divisible_by: Pad to multiple of this value.
        split: Dataset split (for parquet loading).

    Returns:
        Tuple of (DataLoader, DistributedSampler). Call sampler.set_epoch(epoch)
        at the start of each epoch.
    """
    # Load components
    raw_dataset = load_raw_dataset(raw_dataset_path, split=split)
    mappings, metadata = load_mappings(mappings_path)

    # Validate mappings match dataset
    if metadata:
        expected_sequences = metadata.get("num_sequences")
        if expected_sequences and expected_sequences != len(raw_dataset):
            logger.warning(
                f"Mappings were created for {expected_sequences:,} sequences, "
                f"but dataset has {len(raw_dataset):,}. Proceed with caution."
            )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        logger.warning(f"Tokenizer has no pad_token, using eos_token: {tokenizer.eos_token}")
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    dataset = HFWindowedDataset(
        raw_dataset=raw_dataset,
        mappings=mappings,
        tokenizer=tokenizer,
        window_size=window_size,
        seq_length=seq_length,
        text_column=text_column,
        pad_to_seq_length=True,
    )

    # Create sampler for distributed training
    sampler: DistributedSampler[HFWindowedDataset] = DistributedSampler(
        dataset,  # type: ignore[arg-type]
        num_replicas=distributed_config.world_size,
        rank=distributed_config.rank,
        shuffle=True,
        seed=seed,
    )

    # Create collator
    base_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=pad_sequences_to_be_divisible_by,
    )

    # Wrap with genomic collator if needed
    if uppercase_labels or mask_degenerate_bases:
        data_collator: Any = GenomicDataCollator(
            base_collator=base_collator,
            uppercase_labels=uppercase_labels,
            mask_degenerate_bases=mask_degenerate_bases,
        )
        logger.info(
            f"Using GenomicDataCollator (uppercase={uppercase_labels}, mask_degenerate={mask_degenerate_bases})"
        )
    else:
        data_collator = base_collator
        logger.info("Using standard DataCollatorForLanguageModeling")

    # Create dataloader
    dataloader: DataLoader[Any] = DataLoader(
        dataset,  # type: ignore[arg-type]
        sampler=sampler,
        batch_size=micro_batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    logger.info(
        f"Created HFWindowedDataloader: {len(dataset):,} windows, "
        f"batch={micro_batch_size}, workers={num_workers}, world={distributed_config.world_size}"
    )

    return dataloader, sampler
