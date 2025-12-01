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

import logging

import datasets
import datasets.distributed
from torch.utils.data import DataLoader, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling, DataCollatorWithFlattening

from collator import TokenPackingDataset
from distributed_config import DistributedConfig
from genomic_dataset import GenomicDataCollator


logger = logging.getLogger(__name__)


def create_tokenized_dataset(
    distributed_config: DistributedConfig,
    tokenizer_path: str,
    load_dataset_kwargs: dict,
    max_seq_length: int = 8192,
    stride: int = 200,
    buffer_size: int = 500_000,
    use_lazy_tokenization: bool = True,
    sequence_column: str = "sequence",
):
    """Create a tokenized dataset with windowing.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_path: Path to the nucleotide tokenizer directory.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset`.
        max_seq_length: The maximum length of sequences (window size).
        stride: The stride for windowing (overlap = stride tokens).
        buffer_size: The buffer size for shuffle.
        use_lazy_tokenization: Whether to use datasets.set_transform for tokenization.
        sequence_column: Name of the column containing genomic sequences (default: "sequence").

    Returns:
        Tuple of (tokenized_dataset, tokenizer).
    """
    logger.info(f"Loading dataset with kwargs: {load_dataset_kwargs}")
    dataset = datasets.load_dataset(**load_dataset_kwargs)
    logger.info(f"Loaded dataset: {dataset}")

    if isinstance(dataset, datasets.IterableDataset):
        dataset = datasets.distributed.split_dataset_by_node(
            dataset,
            rank=distributed_config.rank,
            world_size=distributed_config.world_size,
        )
        dataset = dataset.shuffle(seed=42, buffer_size=buffer_size)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def tokenize_with_windowing(examples):
        """Tokenize nucleotide sequences with windowing (one-to-many mapping)."""
        # Tokenize with windowing using return_overflowing_tokens
        result = tokenizer(
            examples[sequence_column],
            max_length=max_seq_length,
            stride=stride,
            truncation=True,
            return_overflowing_tokens=True,
            add_special_tokens=True,
        )
        return result

    if isinstance(dataset, datasets.Dataset) and use_lazy_tokenization:
        # Using dataset.map on a non-streaming dataset will automatically perform and cache the transform
        tokenized_dataset = dataset.with_transform(tokenize_with_windowing)
    else:
        # WORKAROUND for OpenGenome2 inconsistent schema:
        # OpenGenome2 has inconsistent schemas across shards - some have 'record' column, some don't.
        # This causes dataset.column_names to be None for streaming IterableDataset.
        #
        # For IterableDataset with None column_names (OpenGenome2):
        #   - Must explicitly list columns to remove: [sequence_column, "record"]
        #   - IterableDataset.map() handles missing columns gracefully
        #
        # For regular Dataset (non-streaming, or streaming with consistent schema like ESM2):
        #   - Use dataset.column_names (which is available and accurate)
        #   - Dataset.map() raises error if column doesn't exist
        #
        # TODO: Remove this workaround once Arc Institute fixes OpenGenome2 schema consistency.
        # When all shards have the same columns, dataset.column_names will work for both cases.
        if isinstance(dataset, datasets.IterableDataset):
            # Streaming dataset: column_names may be None due to inconsistent schema
            columns_to_remove = [sequence_column, "record"]
        else:
            # Non-streaming dataset: use actual column names
            columns_to_remove = dataset.column_names

        logger.info(f"Applying dataset.map with columns to remove: {columns_to_remove}")

        tokenized_dataset = dataset.map(
            tokenize_with_windowing,
            batched=True,
            remove_columns=columns_to_remove,
        )

    return tokenized_dataset, tokenizer


def create_bshd_dataloader(
    distributed_config: DistributedConfig,
    tokenizer_path: str,
    load_dataset_kwargs: dict,
    micro_batch_size: int,
    num_workers: int = 1,
    max_seq_length: int = 8192,
    stride: int = 200,
    seed: int = 42,
    buffer_size: int = 500_000,
    use_lazy_tokenization: bool = True,
    use_stateful_dataloader: bool = False,
    sequence_column: str = "sequence",
    uppercase_labels: bool = False,
    mask_degenerate_bases: bool = True,
):
    """Create a BSHD dataloader for genomic sequences using CLM (causal language modeling).

    Args:
        distributed_config: The distributed configuration.
        tokenizer_path: Path to the nucleotide tokenizer directory.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset`.
        micro_batch_size: The batch size per device.
        num_workers: The number of workers to use for the dataloader.
        max_seq_length: The maximum length of sequences (window size).
        stride: The stride for windowing (overlap = stride tokens).
        seed: The seed to use for the distributed sampler and data collator.
        buffer_size: The buffer size for shuffle.
        use_lazy_tokenization: Whether to use datasets.set_transform for tokenization.
        use_stateful_dataloader: Whether to use the StatefulDataLoader to enable checkpointing the dataloader state.
        sequence_column: Name of the column containing genomic sequences (default: "sequence").
        uppercase_labels: Whether to uppercase labels (genomic masking). Default: False.
        mask_degenerate_bases: Whether to mask non-ACGT bases (genomic masking). Default: False.

    Returns:
        A tuple of (dataloader, dataset_or_sampler).
    """
    tokenized_dataset, tokenizer = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=max_seq_length,
        stride=stride,
        buffer_size=buffer_size,
        use_lazy_tokenization=use_lazy_tokenization,
        sequence_column=sequence_column,
    )

    if isinstance(tokenized_dataset, datasets.IterableDataset):
        sampler = None
    else:
        sampler = DistributedSampler(
            tokenized_dataset,
            rank=distributed_config.rank,
            num_replicas=distributed_config.world_size,
            seed=seed,
        )

    # Create base collator
    base_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
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
        # Use base collator directly for backward compatibility
        data_collator = base_collator
        logger.info("Using standard DataCollatorForLanguageModeling")

    # TODO(BIONEMO-3246) - remove the pin_memory=False once StatefulDataLoader supports pin_memory again.
    dataloader_class = StatefulDataLoader if use_stateful_dataloader else DataLoader
    train_dataloader = dataloader_class(
        tokenized_dataset,
        sampler=sampler,
        batch_size=micro_batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True if not use_stateful_dataloader else False,
        persistent_workers=num_workers > 0,
    )

    return train_dataloader, tokenized_dataset if sampler is None else sampler


def create_thd_dataloader(
    distributed_config: DistributedConfig,
    tokenizer_path: str,
    load_dataset_kwargs: dict,
    micro_batch_size: int | None = None,
    token_micro_batch_size: int | None = None,
    num_workers: int = 1,
    max_seq_length: int = 8192,
    stride: int = 200,
    buffer_size: int = 500_000,
    use_lazy_tokenization: bool = True,
    use_stateful_dataloader: bool = False,
    sequence_column: str = "sequence",
    uppercase_labels: bool = False,
    mask_degenerate_bases: bool = True,
):
    """Create a dataloader that packs up to the maximum number of tokens per batch.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_path: Path to the nucleotide tokenizer directory.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset`.
        micro_batch_size: The batch size per device.
        token_micro_batch_size: The maximum number of tokens per batch. If None, the micro_batch_size * max_seq_length
            will be used. Defaults to None.
        num_workers: The number of workers to use for the dataloader.
        max_seq_length: The maximum length of sequences (window size).
        stride: The stride for windowing (overlap = stride tokens).
        seed: The seed to use for the distributed sampler and data collator.
        buffer_size: The buffer size for shuffle.
        use_lazy_tokenization: Whether to use datasets.set_transform for tokenization.
        use_stateful_dataloader: Whether to use the StatefulDataLoader to enable checkpointing the dataloader state.
        sequence_column: Name of the column containing genomic sequences (default: "sequence").
        uppercase_labels: Whether to uppercase labels (genomic masking). Default: False.
        mask_degenerate_bases: Whether to mask degenerate bases (genomic masking). Default: True.

    Returns:
        A dataloader that can be used for training.
    """
    tokenized_dataset, tokenizer = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=max_seq_length,
        stride=stride,
        buffer_size=buffer_size,
        use_lazy_tokenization=use_lazy_tokenization,
        sequence_column=sequence_column,
    )

    assert isinstance(tokenized_dataset, datasets.IterableDataset), "THD token packing requires a streaming dataset."
    if token_micro_batch_size is None:
        assert micro_batch_size is not None, "Only one of micro_batch_size or token_micro_batch_size can be provided."
        token_micro_batch_size = micro_batch_size * max_seq_length
    else:
        assert micro_batch_size is None, "Only one of micro_batch_size or token_micro_batch_size can be provided."
        assert token_micro_batch_size >= max_seq_length, "token_micro_batch_size must be greater than max_seq_length."

    data_collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)

    if uppercase_labels or mask_degenerate_bases:
        # Wrap with genomic collator if masking options are enabled
        data_collator = GenomicDataCollator(
            base_collator=data_collator,
            uppercase_labels=uppercase_labels,
            mask_degenerate_bases=mask_degenerate_bases,
        )
        logger.info(
            f"Using GenomicDataCollator (uppercase={uppercase_labels}, mask_degenerate={mask_degenerate_bases})"
        )

    # TODO(BIONEMO-3246) - remove the pin_memory=False once StatefulDataLoader supports pin_memory again.
    dataloader_class = StatefulDataLoader if use_stateful_dataloader else DataLoader
    train_dataloader = dataloader_class(
        TokenPackingDataset(tokenized_dataset, max_tokens_per_batch=token_micro_batch_size),
        batch_size=None,  # The TokenPackingDataset will handle the batching.
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True if not use_stateful_dataloader else False,
        persistent_workers=num_workers > 0,
    )

    return train_dataloader, tokenized_dataset
