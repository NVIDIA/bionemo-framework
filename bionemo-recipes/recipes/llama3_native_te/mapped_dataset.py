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

"""Mapped HuggingFace dataset with upfront windowing and caching.

This module provides a non-streaming approach to dataset processing that:
1. Performs all tokenization and windowing upfront
2. Caches the result to disk for reuse across training runs
3. Uses DistributedSampler for true global shuffling
4. Supports num_workers>1 for efficient data loading

This is an alternative to the streaming approach in dataset.py that provides
better shuffling characteristics similar to ShardedEdenDataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import datasets
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling

from distributed_config import DistributedConfig
from genomic_dataset import GenomicDataCollator


logger = logging.getLogger(__name__)


def create_windowed_mapped_dataset(
    tokenizer_name_or_path: str,
    load_dataset_kwargs: dict,
    max_seq_length: int = 8192,
    stride: int = 7992,
    text_column: str = "text",
    tokenize_batch_size: int = 100,  # Keep small! return_overflowing_tokens expands each sequence
    num_proc: int | None = None,
) -> tuple[datasets.Dataset, PreTrainedTokenizerBase]:
    """Create a windowed mapped dataset with all tokenization done upfront.

    This function loads a parquet dataset (non-streaming), applies tokenization with
    windowing to create overlapping windows, and returns a mapped Dataset that can
    be used with DistributedSampler for global shuffling.

    Args:
        tokenizer_name_or_path: Path to the HuggingFace tokenizer.
        load_dataset_kwargs: Keyword arguments for datasets.load_dataset().
            Must NOT include streaming=True.
        max_seq_length: Maximum sequence length for each window.
        stride: Stride for windowing (overlap = max_seq_length - stride).
        text_column: Name of the column containing text sequences.
        tokenize_batch_size: Batch size for tokenization map operation.
        num_proc: Number of processes for parallel tokenization. None for single-process.

    Returns:
        Tuple of (windowed_dataset, tokenizer).
    """
    logger.info(f"Loading dataset with kwargs: {load_dataset_kwargs}")

    # Force non-streaming for mapped dataset
    load_kwargs = {**load_dataset_kwargs}
    if load_kwargs.get("streaming", False):
        logger.warning("Ignoring streaming=True for mapped dataset, forcing streaming=False")
        load_kwargs["streaming"] = False

    raw_dataset = datasets.load_dataset(**load_kwargs)

    if isinstance(raw_dataset, datasets.DatasetDict):
        raise ValueError(
            f"Expected a single Dataset, got DatasetDict with keys: {list(raw_dataset.keys())}. "
            "Please specify a split in load_dataset_kwargs."
        )

    # Type assertion: we know it's a Dataset, not IterableDataset, since streaming=False
    if not isinstance(raw_dataset, datasets.Dataset):
        raise ValueError(f"Expected datasets.Dataset, got {type(raw_dataset)}")

    dataset: datasets.Dataset = raw_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        logger.warning(f"Tokenizer has no pad_token. Setting to eos_token: {tokenizer.eos_token}")
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_with_windowing(examples):
        """Tokenize with windowing (one-to-many expansion).

        Uses HuggingFace tokenizer's return_overflowing_tokens to create
        overlapping windows from each input sequence.
        """
        result = tokenizer(
            examples[text_column],
            max_length=max_seq_length,
            stride=stride,
            truncation=True,
            return_overflowing_tokens=True,
            add_special_tokens=True,
        )
        return result

    logger.info(f"Tokenizing dataset with max_seq_length={max_seq_length}, stride={stride}")
    original_len = len(dataset)
    logger.info(f"Original dataset size: {original_len} sequences")

    # Select only the text column before tokenization
    map_kwargs: dict[str, Any] = {
        "batched": True,
        "batch_size": tokenize_batch_size,
        "remove_columns": [text_column],
    }
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc

    tokenized_dataset = dataset.select_columns([text_column]).map(
        tokenize_with_windowing,
        **map_kwargs,
    )

    # Type assertion: map returns Dataset for non-streaming input
    assert isinstance(tokenized_dataset, datasets.Dataset)

    # Remove overflow_to_sample_mapping if present (not needed for training)
    if "overflow_to_sample_mapping" in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.remove_columns(["overflow_to_sample_mapping"])

    # Remove token_type_ids if present (not needed for causal LM)
    if "token_type_ids" in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.remove_columns(["token_type_ids"])

    tokenized_len = len(tokenized_dataset)
    logger.info(f"Created {tokenized_len} windows from {original_len} sequences")
    logger.info(f"Expansion ratio: {tokenized_len / original_len:.2f}x")

    return tokenized_dataset, tokenizer


def load_or_create_windowed_dataset(
    cache_dir: str,
    tokenizer_name_or_path: str,
    load_dataset_kwargs: dict,
    max_seq_length: int = 8192,
    stride: int = 7992,
    text_column: str = "text",
    tokenize_batch_size: int = 1000,
    num_proc: int | None = None,
    local_rank: int = 0,
    force_recreate: bool = False,
) -> tuple[datasets.Dataset, PreTrainedTokenizerBase]:
    """Load cached dataset or create it (with distributed synchronization).

    This function handles distributed training by ensuring only rank 0 creates
    the cache, while other ranks wait at a barrier.

    Args:
        cache_dir: Directory to store/load the cached dataset.
        tokenizer_name_or_path: Path to the HuggingFace tokenizer.
        load_dataset_kwargs: Keyword arguments for datasets.load_dataset().
        max_seq_length: Maximum sequence length for each window.
        stride: Stride for windowing.
        text_column: Name of the column containing text sequences.
        tokenize_batch_size: Batch size for tokenization.
        num_proc: Number of processes for parallel tokenization.
        local_rank: Local rank of the current process.
        force_recreate: If True, recreate the cache even if it exists.

    Returns:
        Tuple of (windowed_dataset, tokenizer).
    """
    cache_path = Path(cache_dir)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if cache exists (look for dataset_info.json which indicates a valid HF dataset)
    cache_exists = cache_path.exists() and (cache_path / "dataset_info.json").exists()

    if cache_exists and not force_recreate:
        if local_rank == 0:
            logger.info(f"Loading cached dataset from {cache_dir}")
        loaded_ds = datasets.load_from_disk(cache_dir)
        assert isinstance(loaded_ds, datasets.Dataset)
        logger.info(f"Loaded {len(loaded_ds)} windows from cache (rank {local_rank})")
        return loaded_ds, tokenizer

    # Only rank 0 creates the cache
    dataset: datasets.Dataset | None = None
    if local_rank == 0:
        logger.info(f"Creating windowed dataset (cache_dir={cache_dir})")
        dataset, tokenizer = create_windowed_mapped_dataset(
            tokenizer_name_or_path=tokenizer_name_or_path,
            load_dataset_kwargs=load_dataset_kwargs,
            max_seq_length=max_seq_length,
            stride=stride,
            text_column=text_column,
            tokenize_batch_size=tokenize_batch_size,
            num_proc=num_proc,
        )

        # Save to disk
        cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving dataset with {len(dataset)} windows to {cache_dir}")
        dataset.save_to_disk(cache_dir)
        logger.info("Dataset saved successfully")

    # Synchronize all ranks - other ranks wait for rank 0 to finish
    if dist.is_initialized():
        logger.info(f"Rank {local_rank} waiting at barrier...")
        dist.barrier()
        logger.info(f"Rank {local_rank} passed barrier")

    # All non-zero ranks load from cache after barrier
    if local_rank != 0:
        logger.info(f"Loading cached dataset from {cache_dir} (rank {local_rank})")
        loaded_ds = datasets.load_from_disk(cache_dir)
        assert isinstance(loaded_ds, datasets.Dataset)
        dataset = loaded_ds

    # At this point, dataset is guaranteed to be assigned:
    # - local_rank == 0: assigned in create_windowed_mapped_dataset
    # - local_rank != 0: assigned in load_from_disk above
    assert dataset is not None

    return dataset, tokenizer


def create_mapped_bshd_dataloader(
    distributed_config: DistributedConfig,
    tokenizer_name_or_path: str,
    cache_dir: str | None = None,
    load_dataset_kwargs: dict | None = None,
    micro_batch_size: int = 8,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    max_seq_length: int = 8192,
    stride: int = 7992,
    seed: int = 42,
    text_column: str = "text",
    uppercase_labels: bool = False,
    mask_degenerate_bases: bool = False,
    pad_sequences_to_be_divisible_by: int | None = None,
    tokenize_batch_size: int = 1000,
    num_proc: int | None = None,
    force_recreate: bool = False,
) -> tuple[DataLoader[Any], DistributedSampler[datasets.Dataset]]:
    """Create a BSHD dataloader from a cached mapped dataset.

    This dataloader provides true global shuffling via DistributedSampler,
    unlike streaming datasets which can only shuffle within a buffer.

    The dataset is either loaded from cache_dir or created from load_dataset_kwargs
    and cached for future use. In distributed settings, only rank 0 creates the
    cache while other ranks wait.

    Args:
        distributed_config: Distributed training configuration.
        tokenizer_name_or_path: Path to the HuggingFace tokenizer.
        cache_dir: Directory to store/load the cached dataset. If None, no caching.
        load_dataset_kwargs: Keyword arguments for datasets.load_dataset().
            Required if cache_dir doesn't exist.
        micro_batch_size: Batch size per GPU.
        num_workers: Number of DataLoader workers.
        prefetch_factor: Prefetch factor for DataLoader.
        max_seq_length: Maximum sequence length for each window.
        stride: Stride for windowing.
        seed: Random seed for DistributedSampler.
        text_column: Name of the column containing text sequences.
        uppercase_labels: Whether to uppercase labels (genomic masking).
        mask_degenerate_bases: Whether to mask non-ACGT bases (genomic masking).
        pad_sequences_to_be_divisible_by: Pad sequences to be divisible by this value.
        tokenize_batch_size: Batch size for tokenization.
        num_proc: Number of processes for parallel tokenization.
        force_recreate: If True, recreate the cache even if it exists.

    Returns:
        Tuple of (DataLoader, DistributedSampler). Call sampler.set_epoch(epoch)
        at the start of each epoch for varied shuffling.
    """
    if cache_dir is None and load_dataset_kwargs is None:
        raise ValueError("Either cache_dir or load_dataset_kwargs must be provided")

    dataset: datasets.Dataset
    tokenizer: PreTrainedTokenizerBase

    if cache_dir is not None:
        dataset, tokenizer = load_or_create_windowed_dataset(
            cache_dir=cache_dir,
            tokenizer_name_or_path=tokenizer_name_or_path,
            load_dataset_kwargs=load_dataset_kwargs or {},
            max_seq_length=max_seq_length,
            stride=stride,
            text_column=text_column,
            tokenize_batch_size=tokenize_batch_size,
            num_proc=num_proc,
            local_rank=distributed_config.local_rank,
            force_recreate=force_recreate,
        )
    else:
        # At this point, load_dataset_kwargs cannot be None (checked above)
        assert load_dataset_kwargs is not None
        dataset, tokenizer = create_windowed_mapped_dataset(
            tokenizer_name_or_path=tokenizer_name_or_path,
            load_dataset_kwargs=load_dataset_kwargs,
            max_seq_length=max_seq_length,
            stride=stride,
            text_column=text_column,
            tokenize_batch_size=tokenize_batch_size,
            num_proc=num_proc,
        )

    # Create DistributedSampler for true global shuffling
    # Note: HuggingFace Dataset is compatible with PyTorch's Dataset protocol
    sampler = DistributedSampler(
        dataset,  # type: ignore[arg-type]
        num_replicas=distributed_config.world_size,
        rank=distributed_config.rank,
        shuffle=True,
        seed=seed,
    )

    # Create collator
    base_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
        pad_to_multiple_of=pad_sequences_to_be_divisible_by,
    )

    # Wrap with genomic collator if masking options are enabled
    if uppercase_labels or mask_degenerate_bases:
        data_collator = GenomicDataCollator(
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

    dataloader = DataLoader(
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
        f"Created mapped BSHD dataloader: {len(dataset)} windows, "
        f"batch={micro_batch_size}, workers={num_workers}, world={distributed_config.world_size}"
    )

    return dataloader, sampler
