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
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from collator import (
    DataCollatorWithFlattening,
    TokenPackingDataset,
)
from distributed_config import DistributedConfig
from genomic_dataset import GenomicDataCollator


# Lazy import for tensor_dataset (optional, only needed for tensor dataset mode)
try:
    from tensor_dataset import create_tensor_dataloader
except ImportError:
    create_tensor_dataloader = None  # Not available, tensor dataset mode disabled


logger = logging.getLogger(__name__)


def create_dataloader_from_config(
    distributed_config: DistributedConfig,
    use_tensor_dataset: bool = False,
    tensor_dir: str | None = None,
    use_sequence_packing: bool = False,
    **kwargs,
):
    """Create a dataloader based on config, supporting both HF and pre-dumped tensor data.

    This is a convenience function that dispatches to the appropriate dataloader based on config.

    Args:
        distributed_config: Distributed training configuration
        use_tensor_dataset: If True, load from pre-dumped tensor files
        tensor_dir: Directory containing tensor files (required if use_tensor_dataset=True)
        use_sequence_packing: If True, use THD format with sequence packing
        **kwargs: Additional arguments passed to the underlying dataloader

    Returns:
        Tuple of (dataloader, dataset_or_sampler)
    """
    if use_tensor_dataset:
        if tensor_dir is None:
            raise ValueError("tensor_dir is required when use_tensor_dataset=True")
        logger.info(f"Using pre-dumped tensor dataset from {tensor_dir}")
        return create_tensor_dataloader(
            distributed_config=distributed_config,
            tensor_dir=tensor_dir,
            micro_batch_size=kwargs.get("micro_batch_size", 1),
            grad_acc_steps=kwargs.get("grad_acc_steps", 8),
            num_workers=kwargs.get("num_workers", 0),
            log_sequences=kwargs.get("log_sequences", False),
            log_dir=kwargs.get("sequence_log_dir"),
        )
    elif use_sequence_packing:
        return create_thd_dataloader(distributed_config=distributed_config, **kwargs)
    else:
        return create_bshd_dataloader(distributed_config=distributed_config, **kwargs)


def create_tokenized_dataset(
    distributed_config: DistributedConfig,
    tokenizer_name_or_path: str,
    load_dataset_kwargs: dict,
    max_seq_length: int = 8192,
    stride: int = 200,
    buffer_size: int = 5_000,
    text_column: str = "text",
    tokenize_batch_size: int = 100,
    shuffle: bool = True,
    skip_windowing: bool = False,
):
    """Create a tokenized dataset with windowing.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_name_or_path: Name or path to the nucleotide tokenizer directory.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset`.
        max_seq_length: The maximum length of sequences (window size).
        stride: The stride for windowing (overlap = stride tokens).
        buffer_size: The buffer size for shuffle.
        text_column: Name of the column containing genomic sequences (default: "text").
        tokenize_batch_size: The batch size for tokenization.
        shuffle: Whether to shuffle the data. Default: True.
        skip_windowing: If True, skip windowing (data is already windowed). Default: False.

    Returns:
        Tuple of (tokenized_dataset, tokenizer).
    """
    logger.info(f"Loading dataset with kwargs: {load_dataset_kwargs}")
    dataset = datasets.load_dataset(**load_dataset_kwargs)

    if isinstance(dataset, datasets.IterableDataset):
        # Hugging Face's `split_dataset_by_node` is quite sensitive to the total number of shards -- if the number of
        # shards is not perfectly divisible by the world size, it defaults to loading the same shards on all nodes and
        # using strided sampling to avoid loading the same data on all nodes. This can be quite inefficient with large
        # numbers of shards and workers, so we use `dataset.shard` instead.
        if distributed_config.world_size > dataset.num_shards:
            logger.info(f"Sharding dataset with {dataset.num_shards} shards with split_dataset_by_node")
            dataset = datasets.distributed.split_dataset_by_node(
                dataset, rank=distributed_config.rank, world_size=distributed_config.world_size
            )
        else:
            logger.info(f"Sharding dataset with {dataset.num_shards} shards with dataset.shard")
            dataset = dataset.shard(num_shards=distributed_config.world_size, index=distributed_config.rank)

        if shuffle:
            dataset = dataset.shuffle(seed=42, buffer_size=buffer_size)
        else:
            logger.info("Shuffle disabled - preserving data order from source")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    if skip_windowing:
        # Data is already windowed - just tokenize without overflow handling
        logger.info("Windowing disabled - data is assumed to be pre-windowed")

        def tokenize_simple(examples):
            """Tokenize pre-windowed sequences (one-to-one mapping)."""
            result = tokenizer(
                examples[text_column],
                max_length=max_seq_length,
                truncation=True,
                padding=False,
                add_special_tokens=True,
            )
            return result

        tokenized_dataset = dataset.select_columns(text_column).map(
            tokenize_simple,
            batched=True,
            batch_size=tokenize_batch_size,
            remove_columns=[text_column],
        )
    else:
        # Standard windowing with return_overflowing_tokens

        def tokenize_with_windowing(examples):
            """Tokenize nucleotide sequences with windowing (one-to-many mapping)."""
            # Tokenize with windowing using return_overflowing_tokens
            result = tokenizer(
                examples[text_column],
                max_length=max_seq_length,
                stride=stride,
                truncation=True,
                return_overflowing_tokens=True,
                add_special_tokens=True,
            )
            return result

        tokenized_dataset = dataset.select_columns(text_column).map(
            tokenize_with_windowing,
            batched=True,
            batch_size=tokenize_batch_size,
            remove_columns=[text_column],
        )

    # Even in THD mode, we use a base MLM collator that requires a padding token to be set.
    if tokenizer.pad_token is None:
        logger.warning(f"Tokenizer does not have a padding token. Setting it to the EOS token: {tokenizer.eos_token}")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenized_dataset, tokenizer


def create_pretokenized_dataset(
    distributed_config: DistributedConfig,
    tokenizer_name_or_path: str,
    load_dataset_kwargs: dict,
    shuffle: bool = False,
    seed: int = 42,
):
    """Load a pre-tokenized dataset (e.g., from dump_sharded_eden_as_parquet.py).

    The dataset should have an 'input_ids' column containing tokenized sequences.
    This function loads the data without any tokenization or windowing.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_name_or_path: Path to tokenizer (for collator configuration).
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset`.
        shuffle: Whether to shuffle. Default False to preserve dump order.
        seed: Random seed for shuffling.

    Returns:
        Tuple of (dataset, tokenizer).
    """
    logger.info(f"Loading pre-tokenized dataset with kwargs: {load_dataset_kwargs}")
    dataset = datasets.load_dataset(**load_dataset_kwargs)

    # Load tokenizer for collator
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For map-style datasets, we don't shard here - let DistributedSampler handle it
    # For iterable datasets, shard by node
    if isinstance(dataset, datasets.IterableDataset):
        if distributed_config.world_size > 1:
            dataset = datasets.distributed.split_dataset_by_node(
                dataset, rank=distributed_config.rank, world_size=distributed_config.world_size
            )
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        else:
            logger.info("Pre-tokenized dataset: shuffle disabled - preserving dump order")
    else:
        logger.info(f"Pre-tokenized dataset: {len(dataset)} samples (map-style)")
        if not shuffle:
            logger.info("Pre-tokenized dataset: shuffle disabled - DistributedSampler will preserve order")

    return dataset, tokenizer


class IndexTrackingDataset(torch.utils.data.Dataset):
    """Wrapper that tracks which indices are accessed and logs them.

    This is used to verify that batches of GBS samples are kept together
    across optimizer steps when using DistributedSampler.
    """

    def __init__(self, dataset, rank: int, log_dir: str | None = None):
        """Initialize the tracking wrapper.

        Args:
            dataset: The underlying dataset to wrap.
            rank: The current rank (for logging).
            log_dir: Directory to write logs. If None, no logging.
        """
        self.dataset = dataset
        self.rank = rank
        self.log_dir = log_dir
        self.access_count = 0
        self.log_file = None  # Don't open file in __init__ - defer to avoid worker issues
        self._file_initialized = False

    def _ensure_log_file(self):
        """Lazily initialize log file (called on first access, works with DataLoader workers)."""
        if self.log_file is None and self.log_dir:
            import os

            os.makedirs(self.log_dir, exist_ok=True)
            # Include worker info in filename to avoid conflicts
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else "main"
            log_path = os.path.join(self.log_dir, f"sample_indices_rank{self.rank}_worker{worker_id}.csv")
            self.log_file = open(log_path, "w")
            self.log_file.write("access_order,dataset_index,first_5_tokens\n")
            self._file_initialized = True

    def __len__(self):
        """Return length of underlying dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get item and log the access."""
        item = self.dataset[idx]

        # Lazily initialize log file on first access
        if self.log_dir and not self._file_initialized:
            self._ensure_log_file()

        if self.log_file:
            # Log the index and first few tokens
            input_ids = item["input_ids"]
            if hasattr(input_ids, "tolist"):
                first_tokens = input_ids[:5].tolist()
            else:
                first_tokens = input_ids[:5]
            self.log_file.write(f"{self.access_count},{idx},{first_tokens}\n")

            # Flush every access to ensure logs are captured even if training stops early
            self.log_file.flush()

        self.access_count += 1
        return item

    def close(self):
        """Close the log file."""
        if self.log_file:
            self.log_file.close()


def create_bshd_dataloader(
    distributed_config: DistributedConfig,
    tokenizer_name_or_path: str,
    load_dataset_kwargs: dict,
    micro_batch_size: int,
    num_workers: int = 1,
    prefetch_factor: int = 4,
    max_seq_length: int = 8192,
    stride: int = 200,
    seed: int = 42,
    buffer_size: int = 500_000,
    use_stateful_dataloader: bool = False,
    text_column: str = "text",
    uppercase_labels: bool = False,
    mask_degenerate_bases: bool = False,
    pad_sequences_to_be_divisible_by: int | None = None,
    shuffle: bool = True,
    skip_windowing: bool = False,
    skip_tokenization: bool = False,
    log_sample_indices: bool = False,
    sample_log_dir: str | None = None,
):
    """Create a BSHD dataloader for llama3 pre-training.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_name_or_path: Name or path to the nucleotide tokenizer directory.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset`.
        micro_batch_size: The batch size per device.
        num_workers: The number of workers to use for the dataloader.
        prefetch_factor: The prefetch factor to use for the dataloader.
        max_seq_length: The maximum length of sequences (window size).
        stride: The stride for windowing (overlap = stride tokens).
        seed: The seed to use for the distributed sampler and data collator.
        buffer_size: The buffer size for shuffle.
        use_stateful_dataloader: Whether to use the StatefulDataLoader to enable checkpointing the dataloader state.
        text_column: Name of the column containing text sequences (default: "text").
        uppercase_labels: Whether to uppercase labels (genomic masking). Default: False.
        mask_degenerate_bases: Whether to mask non-ACGT bases (genomic masking). Default: False.
        pad_sequences_to_be_divisible_by: The number to pad sequences to be divisible by, required for FP8 training.
            Default: None.
        shuffle: Whether to shuffle the data. Set to False to preserve exact data ordering (e.g., when using
            pre-ordered data from John's ShardedEdenDataset). Default: True.
        skip_windowing: If True, data is already windowed to max_seq_length and windowing is skipped.
            Use this when loading pre-windowed data. Default: False.
        skip_tokenization: If True, data is already tokenized (has 'input_ids' column).
            Use this when loading pre-tokenized parquet files from dump_sharded_eden_as_parquet.py.
            Default: False.
        log_sample_indices: If True, log which dataset indices are accessed by each rank.
            Useful for verifying batch composition. Default: False.
        sample_log_dir: Directory to write sample index logs. Required if log_sample_indices=True.

    Returns:
        A tuple of (dataloader, dataset_or_sampler).
    """
    if skip_tokenization:
        # Load pre-tokenized data directly
        tokenized_dataset, tokenizer = create_pretokenized_dataset(
            distributed_config=distributed_config,
            tokenizer_name_or_path=tokenizer_name_or_path,
            load_dataset_kwargs=load_dataset_kwargs,
            shuffle=shuffle,
            seed=seed,
        )
    else:
        tokenized_dataset, tokenizer = create_tokenized_dataset(
            distributed_config=distributed_config,
            tokenizer_name_or_path=tokenizer_name_or_path,
            load_dataset_kwargs=load_dataset_kwargs,
            max_seq_length=max_seq_length,
            stride=stride,
            buffer_size=buffer_size,
            text_column=text_column,
            tokenize_batch_size=micro_batch_size * prefetch_factor,
            shuffle=shuffle,
            skip_windowing=skip_windowing,
        )

    # Wrap with index tracking if logging is enabled (only for map-style datasets)
    if log_sample_indices and not isinstance(tokenized_dataset, datasets.IterableDataset):
        if sample_log_dir is None:
            raise ValueError("sample_log_dir must be provided when log_sample_indices=True")
        logger.info(f"Wrapping dataset with IndexTrackingDataset, logging to {sample_log_dir}")
        tokenized_dataset = IndexTrackingDataset(
            tokenized_dataset, rank=distributed_config.rank, log_dir=sample_log_dir
        )

    if isinstance(tokenized_dataset, datasets.IterableDataset):
        sampler = None
    else:
        sampler = DistributedSampler(
            tokenized_dataset,
            rank=distributed_config.rank,
            num_replicas=distributed_config.world_size,
            seed=seed,
            shuffle=shuffle,  # Pass shuffle flag to sampler
        )
        if not shuffle:
            logger.info("DistributedSampler shuffle disabled - preserving data order")

    # Create base collator
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
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return train_dataloader, tokenized_dataset if sampler is None else sampler


def create_thd_dataloader(
    distributed_config: DistributedConfig,
    tokenizer_name_or_path: str,
    load_dataset_kwargs: dict,
    micro_batch_size: int | None = None,
    token_micro_batch_size: int | None = None,
    num_workers: int = 1,
    prefetch_factor: int = 4,
    max_seq_length: int = 8192,
    stride: int = 200,
    buffer_size: int = 500_000,
    use_stateful_dataloader: bool = False,
    text_column: str = "text",
    uppercase_labels: bool = False,
    mask_degenerate_bases: bool = False,
    split_samples_in_token_packing: bool = True,
    pad_sequences_to_be_divisible_by: int | None = None,
):
    """Create a dataloader that packs up to the maximum number of tokens per batch.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_name_or_path: Name or path to the nucleotide tokenizer directory.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset`.
        micro_batch_size: The batch size per device.
        token_micro_batch_size: The maximum number of tokens per batch. If None, the micro_batch_size * max_seq_length
            will be used. Defaults to None.
        num_workers: The number of workers to use for the dataloader.
        prefetch_factor: The prefetch factor to use for the dataloader.
        max_seq_length: The maximum length of sequences (window size).
        stride: The stride for windowing (overlap = stride tokens).
        seed: The seed to use for the distributed sampler and data collator.
        buffer_size: The buffer size for shuffle.
        use_stateful_dataloader: Whether to use the StatefulDataLoader to enable checkpointing the dataloader state.
        text_column: Name of the column containing genomic sequences (default: "text").
        uppercase_labels: Whether to uppercase labels (genomic masking). Default: False.
        mask_degenerate_bases: Whether to mask degenerate bases (genomic masking). Default: False.
        split_samples_in_token_packing: Whether to split samples to form batches with exactly token_micro_batch_size
            tokens. Default: True.
        pad_sequences_to_be_divisible_by: If provided, sequences will be padded to be divisible by this value.
            This is useful for context parallelism. Defaults to None.

    Returns:
        A tuple of (dataloader, dataset_or_sampler).
    """
    tokenized_dataset, tokenizer = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_name_or_path,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=max_seq_length,
        stride=stride,
        buffer_size=buffer_size,
        text_column=text_column,
    )

    assert isinstance(tokenized_dataset, datasets.IterableDataset), "THD token packing requires a streaming dataset."
    if token_micro_batch_size is None:
        assert micro_batch_size is not None, "Only one of micro_batch_size or token_micro_batch_size can be provided."
        token_micro_batch_size = micro_batch_size * max_seq_length
    else:
        assert micro_batch_size is None, "Only one of micro_batch_size or token_micro_batch_size can be provided."
        assert token_micro_batch_size >= max_seq_length, "token_micro_batch_size must be greater than max_seq_length."

    # Create base MLM collator and wrap with flattening collator
    data_collator = DataCollatorWithFlattening(
        collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        pad_sequences_to_be_divisible_by=pad_sequences_to_be_divisible_by,
    )

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
        TokenPackingDataset(
            tokenized_dataset,
            max_tokens_per_batch=token_micro_batch_size,
            split_samples=split_samples_in_token_packing,
        ),
        batch_size=None,  # The TokenPackingDataset will handle the batching.
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True if not use_stateful_dataloader else False,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return train_dataloader, tokenized_dataset
