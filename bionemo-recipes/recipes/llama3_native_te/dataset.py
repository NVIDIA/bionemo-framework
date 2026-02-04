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
    use_sharded_eden: bool = False,
    sharded_eden_kwargs: dict | None = None,
    **kwargs,
):
    """Create a dataloader based on config, supporting multiple data sources.

    This is a convenience function that dispatches to the appropriate dataloader based on config.

    Args:
        distributed_config: Distributed training configuration
        use_tensor_dataset: If True, load from pre-dumped tensor files
        tensor_dir: Directory containing tensor files (required if use_tensor_dataset=True)
        use_sequence_packing: If True, use THD format with sequence packing
        use_sharded_eden: If True, use ShardedEdenDataset directly from SQLite
        sharded_eden_kwargs: Arguments for create_sharded_eden_dataloader (required if use_sharded_eden=True)
        **kwargs: Additional arguments passed to the underlying dataloader

    Returns:
        Tuple of (dataloader, dataset_or_sampler)
    """
    if use_sharded_eden:
        if sharded_eden_kwargs is None:
            raise ValueError("sharded_eden_kwargs is required when use_sharded_eden=True")
        logger.info("Using ShardedEdenDataset directly from SQLite")
        return create_sharded_eden_dataloader(
            distributed_config=distributed_config,
            **sharded_eden_kwargs,
        )
    elif use_tensor_dataset:
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
    debug_window_order: bool = False,
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
        debug_window_order: If True, log window ordering info to verify if windows from
            same sequence are yielded consecutively. Default: False.

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
        # Track window order for debugging if enabled
        _debug_window_counter = [0]  # Mutable counter for window tracking
        _debug_seq_counter = [0]  # Track global sequence IDs

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

            # Debug logging: track which sequence each window came from
            if debug_window_order and distributed_config.rank == 0:
                sample_mapping = result.get("overflow_to_sample_mapping", list(range(len(result["input_ids"]))))
                batch_start = _debug_window_counter[0]
                seq_start = _debug_seq_counter[0]

                # Log first 200 windows to see the pattern
                if batch_start < 200:
                    for i, (local_seq_id, input_ids) in enumerate(zip(sample_mapping, result["input_ids"])):
                        global_idx = batch_start + i
                        global_seq_id = seq_start + local_seq_id
                        first_tokens = input_ids[:6]
                        if global_idx < 200:
                            logger.info(
                                f"[WINDOW_DEBUG] idx={global_idx:4d} seq_id={global_seq_id:4d} "
                                f"local_seq={local_seq_id:2d} first_tokens={first_tokens}"
                            )

                _debug_window_counter[0] += len(result["input_ids"])
                _debug_seq_counter[0] += len(set(sample_mapping))

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
    debug_window_order: bool = False,
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
        debug_window_order: If True, log window ordering to verify if windows from the same
            sequence are yielded consecutively (rank 0 only). Default: False.

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
            debug_window_order=debug_window_order,
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


# =============================================================================
# ShardedEdenDataset Integration (Direct SQLite access without Megatron)
# =============================================================================


class HuggingFaceTokenizerAdapter:
    """Adapter that wraps a HuggingFace tokenizer with the interface expected by ShardedEdenDataset.

    ShardedEdenDataset expects a tokenizer with:
    - bos_id: int
    - eos_id: int
    - _sep_id: int
    - pad_id: int
    - text_to_ids(text: str) -> List[int]
    """

    def __init__(self, hf_tokenizer):
        """Initialize the adapter with a HuggingFace tokenizer.

        Args:
            hf_tokenizer: A HuggingFace tokenizer (e.g., from AutoTokenizer)
        """
        self.hf_tokenizer = hf_tokenizer

        # Map HuggingFace token IDs to Megatron-style attributes
        self.bos_id = hf_tokenizer.bos_token_id
        self.eos_id = hf_tokenizer.eos_token_id
        self.pad_id = hf_tokenizer.pad_token_id

        # Handle sep token - use eos if sep not defined
        if hf_tokenizer.sep_token_id is not None:
            self._sep_id = hf_tokenizer.sep_token_id
        else:
            # Fallback: use eos_id or a special token
            self._sep_id = self.eos_id

        # Ensure pad_id is set (some tokenizers don't have it)
        if self.pad_id is None:
            self.pad_id = self.eos_id

        # Ensure bos_id is set
        if self.bos_id is None:
            # For tokenizers without BOS, use 0 or another sensible default
            self.bos_id = 0

        logger.info(
            f"HuggingFaceTokenizerAdapter initialized: bos_id={self.bos_id}, eos_id={self.eos_id}, "
            f"sep_id={self._sep_id}, pad_id={self.pad_id}"
        )

    def text_to_ids(self, text: str) -> list[int]:
        """Convert text to token IDs without adding special tokens."""
        return self.hf_tokenizer.encode(text, add_special_tokens=False)


class ShardedEdenDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper that converts ShardedEdenDataset output to HuggingFace format.

    ShardedEdenDataset returns: {"tokens": tensor, "labels": tensor, "loss_mask": tensor, ...}
    HuggingFace expects: {"input_ids": tensor, "attention_mask": tensor, ...}

    This wrapper does the conversion and creates proper attention_mask for padding.
    """

    def __init__(self, sharded_eden_dataset, pad_id: int, log_indices: bool = False, log_dir: str | None = None):
        """Initialize the wrapper.

        Args:
            sharded_eden_dataset: The underlying ShardedEdenDataset
            pad_id: The padding token ID (used to create attention_mask)
            log_indices: Whether to log accessed indices
            log_dir: Directory for logging
        """
        self.dataset = sharded_eden_dataset
        self.pad_id = pad_id
        self.log_indices = log_indices
        self.log_dir = log_dir

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get an item and convert to HuggingFace format."""
        sample = self.dataset[idx]
        tokens = sample["tokens"]
        labels = sample["labels"].clone()  # Clone to avoid modifying original
        loss_mask = sample.get("loss_mask")

        # CRITICAL: HuggingFace models ignore loss where labels == -100
        # ShardedEdenDataset sets labels = pad_id for padding, but we need -100
        # Use loss_mask to identify positions that should be ignored in loss
        if loss_mask is not None:
            # Where loss_mask == 0, set labels to -100 (HF's ignore index)
            labels[loss_mask == 0] = -100

        # Create attention_mask in HuggingFace EXTENDED format:
        # - 0.0 = attend to this token
        # - Large negative value = don't attend (masked/padding)
        #
        # The model (modeling_llama_te.py line 349) converts 2D mask to 4D via:
        #   attention_mask[:, None, None, :] < -1
        # So we need: padding tokens = -10000 (< -1 = True = masked)
        #             real tokens = 0 (< -1 = False = not masked)
        padding_mask = tokens == self.pad_id  # True for padding tokens
        attention_mask = padding_mask.float() * -10000.0  # -10000 for padding, 0 for real tokens

        # Convert from ShardedEdenDataset format to HuggingFace format
        return {
            "input_ids": tokens,  # Rename tokens -> input_ids
            "labels": labels,  # Now with -100 for ignored positions
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,  # Keep for debugging/logging
            "position_ids": sample.get("position_ids"),
        }


def create_sharded_eden_dataloader(
    distributed_config: DistributedConfig,
    tokenizer_name_or_path: str,
    sequence_db_dir: str,
    window_db_path: str,
    micro_batch_size: int = 1,
    seq_length: int = 8192,
    stride: int = 7992,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 42,
    rc_aug: bool = False,
    log_windows: bool = False,
    log_dir: str | None = None,
):
    """Create a dataloader using ShardedEdenDataset directly (no Megatron dependencies).

    This approach:
    1. Uses ShardedEdenDataset to read directly from SQLite databases
    2. Wraps HuggingFace tokenizer with Megatron-compatible interface
    3. Uses standard PyTorch DistributedSampler with shuffle=True
    4. Returns data in HuggingFace format (input_ids, labels, etc.)

    Args:
        distributed_config: Distributed training configuration
        tokenizer_name_or_path: Path to HuggingFace tokenizer or model name
        sequence_db_dir: Directory containing per-sample SQLite databases
        window_db_path: Path to the window mappings database (train split)
        micro_batch_size: Batch size per GPU
        seq_length: Sequence length (must match database)
        stride: Stride for windowing (must match database)
        num_workers: DataLoader workers
        shuffle: Whether to shuffle data (recommended: True)
        seed: Random seed for shuffling
        rc_aug: Whether to enable reverse complement augmentation
        log_windows: Whether to log window access patterns
        log_dir: Directory for window access logs

    Returns:
        Tuple of (dataloader, sampler)
    """
    # Import standalone version (no NeMo/Megatron dependencies)
    from sharded_eden_standalone import ShardedEdenDatasetStandalone

    # Load HuggingFace tokenizer
    logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    # Create adapter for ShardedEdenDataset
    tokenizer_adapter = HuggingFaceTokenizerAdapter(hf_tokenizer)

    # Create ShardedEdenDataset (standalone version, no NeMo/Megatron)
    logger.info("Creating ShardedEdenDatasetStandalone (no NeMo/Megatron dependencies):")
    logger.info(f"  sequence_db_dir: {sequence_db_dir}")
    logger.info(f"  window_db_path: {window_db_path}")
    logger.info(f"  seq_length: {seq_length}, stride: {stride}")
    logger.info(f"  shuffle: {shuffle}, rc_aug: {rc_aug}")

    eden_dataset = ShardedEdenDatasetStandalone(
        tokenizer=tokenizer_adapter,
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_db_path,
        seq_length=seq_length,
        stride=stride,
        create_attention_mask=False,
        rc_aug=rc_aug,
        skip_stats=True,
        log_windows=log_windows,
        log_dir=log_dir,
    )

    logger.info(f"Dataset has {len(eden_dataset)} windows")

    # Wrap with HuggingFace-compatible output format (pass pad_id for attention_mask creation)
    wrapped_dataset = ShardedEdenDatasetWrapper(eden_dataset, pad_id=tokenizer_adapter.pad_id)

    # Create DistributedSampler
    sampler = DistributedSampler(
        wrapped_dataset,
        num_replicas=distributed_config.world_size,
        rank=distributed_config.rank,
        shuffle=shuffle,
        seed=seed,
    )

    # Create DataLoader
    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=eden_dataset.collate_fn,  # Use the dataset's collate function
    )

    logger.info(f"Created ShardedEden DataLoader with {len(dataloader)} batches per epoch")
    logger.info(f"  Samples per GPU: {len(wrapped_dataset) // distributed_config.world_size}")

    return dataloader, sampler
