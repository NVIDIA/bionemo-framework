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
import random
from collections.abc import Iterator
from typing import Any

import datasets
import datasets.distributed
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from collator import (
    DataCollatorWithFlattening,
    TokenPackingDataset,
)
from distributed_config import DistributedConfig
from genomic_dataset import GenomicDataCollator


logger = logging.getLogger(__name__)


class InterleavedShuffleDataset(IterableDataset):
    """Shuffle by interleaving data from multiple stream positions.

    This provides pseudo-global shuffling for streaming datasets without requiring
    a massive buffer. Instead of filling one buffer sequentially, we maintain
    multiple smaller buffers that pull from different NON-OVERLAPPING regions of
    the dataset, then sample across all buffers.

    For a dataset with 1M windows (10 chunks x 100k items each, 5k buffer per chunk):
    - Chunk 0 reads items [0, 100k), buffers 5k at a time
    - Chunk 1 reads items [100k, 200k), buffers 5k at a time
    - ...
    - Chunk 9 reads items [900k, 1M), buffers 5k at a time
    - Samples are drawn uniformly across all chunk buffers, mixing items from all regions.
    """

    def __init__(
        self,
        dataset: datasets.IterableDataset,
        num_interleave_chunks: int = 10,
        chunk_buffer_size: int = 5000,
        skip_per_chunk: int | None = None,
        seed: int = 42,
    ):
        """Initialize the interleaved shuffle dataset.

        Args:
            dataset: The underlying HuggingFace IterableDataset.
            num_interleave_chunks: Number of non-overlapping regions to interleave from.
            chunk_buffer_size: Buffer size per chunk (total memory = num_chunks x chunk_buffer_size items).
            skip_per_chunk: Number of items per chunk region. Each chunk reads exactly this many items
                from its region. If None, defaults to 50000 (total coverage = num_chunks x 50k).
            seed: Random seed for shuffling. Incremented each epoch for different ordering.
        """
        self.dataset = dataset
        self.num_interleave_chunks = num_interleave_chunks
        self.chunk_buffer_size = chunk_buffer_size
        self.skip_per_chunk = skip_per_chunk if skip_per_chunk is not None else 50000
        self.seed = seed
        self._epoch = 0

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate with interleaved shuffling.

        Each chunk reads a non-overlapping region of the dataset using skip/take.
        Items are sampled uniformly across all chunk buffers for pseudo-global mixing.
        """
        # Use different seed each epoch for varied ordering
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        # Create iterators for non-overlapping regions using skip + take
        iterators: list[Iterator] = []
        exhausted: list[bool] = []
        for i in range(self.num_interleave_chunks):
            skip_amount = i * self.skip_per_chunk
            # Each chunk reads ONLY skip_per_chunk items from its region (no overlap)
            chunk_iter = iter(self.dataset.skip(skip_amount).take(self.skip_per_chunk))
            iterators.append(chunk_iter)
            exhausted.append(False)

        total_coverage = self.num_interleave_chunks * self.skip_per_chunk
        logger.info(
            f"InterleavedShuffleDataset: {self.num_interleave_chunks} chunks x "
            f"{self.skip_per_chunk} items each = {total_coverage} total coverage, "
            f"{self.chunk_buffer_size} buffer per chunk, epoch={self._epoch - 1}"
        )

        # Maintain a buffer for each chunk
        buffers: list[list] = [[] for _ in range(self.num_interleave_chunks)]

        # Initial fill: partially fill each buffer
        for i in range(self.num_interleave_chunks):
            self._fill_buffer(i, iterators, buffers, exhausted)

        while True:
            # Find non-empty buffers
            non_empty = [i for i in range(self.num_interleave_chunks) if buffers[i]]
            if not non_empty:
                break

            # Sample from a random non-empty buffer
            chunk_idx = rng.choice(non_empty)
            item_idx = rng.randrange(len(buffers[chunk_idx]))
            yield buffers[chunk_idx].pop(item_idx)

            # Refill this buffer if it's running low and iterator not exhausted
            if not exhausted[chunk_idx] and len(buffers[chunk_idx]) < self.chunk_buffer_size // 2:
                self._fill_buffer(chunk_idx, iterators, buffers, exhausted)

    def _fill_buffer(
        self,
        chunk_idx: int,
        iterators: list[Iterator],
        buffers: list[list],
        exhausted: list[bool],
    ) -> None:
        """Fill a chunk's buffer up to chunk_buffer_size.

        Args:
            chunk_idx: Index of the chunk to fill.
            iterators: List of iterators for each chunk.
            buffers: List of buffers for each chunk.
            exhausted: List tracking whether each iterator is exhausted.
        """
        while len(buffers[chunk_idx]) < self.chunk_buffer_size:
            try:
                item = next(iterators[chunk_idx])
                buffers[chunk_idx].append(item)
            except StopIteration:
                exhausted[chunk_idx] = True
                break


def create_tokenized_dataset(
    distributed_config: DistributedConfig,
    tokenizer_name_or_path: str,
    load_dataset_kwargs: dict,
    max_seq_length: int = 8192,
    stride: int = 200,
    buffer_size: int = 50_000,
    text_column: str = "text",
    tokenize_batch_size: int = 100,
    shuffle_sequences: bool = True,
    shuffle_windows: bool = False,
    interleaved_shuffle: bool = False,
    interleave_chunks: int = 10,
    interleave_skip: int = 50_000,
):
    """Create a tokenized dataset with windowing.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_name_or_path: Name or path to the nucleotide tokenizer directory.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset`.
        max_seq_length: The maximum length of sequences (window size).
        stride: The stride for windowing (overlap = stride tokens).
        buffer_size: The buffer size for shuffle operations.
        text_column: Name of the column containing genomic sequences (default: "text").
        tokenize_batch_size: The batch size for tokenization.
        shuffle_sequences: Whether to shuffle raw sequences before windowing. This randomizes
            the order of sequences but windows from the same sequence remain consecutive.
            Fast because sequences are larger items. Default: True.
        shuffle_windows: Whether to shuffle windows after tokenization. This interleaves windows
            from different sequences for better batch diversity, but requires filling a buffer
            and can be slower. Default: False.
        interleaved_shuffle: Whether to use interleaved shuffling for windows. This provides
            pseudo-global shuffling by pulling from multiple stream positions simultaneously.
            More memory-efficient than large buffer_size. Overrides shuffle_windows. Default: False.
        interleave_chunks: Number of stream positions to interleave from. Default: 10.
        interleave_skip: Number of windows to skip between chunk starting positions. Default: 50000.
            With 10 chunks and 50k skip, covers first 500k windows of the dataset.

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

        # Pre-windowing shuffle: randomizes sequence order (fast, since sequences are larger items)
        if shuffle_sequences:
            logger.info(f"Shuffling sequences before windowing with buffer_size={buffer_size}")
            dataset = dataset.shuffle(seed=42, buffer_size=buffer_size)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

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

    # Apply window shuffling if enabled
    if isinstance(tokenized_dataset, datasets.IterableDataset):
        if interleaved_shuffle:
            # Interleaved shuffle: pull from multiple stream positions for pseudo-global coverage
            # This is more memory-efficient than a massive buffer
            logger.info(
                f"Using interleaved window shuffle: {interleave_chunks} chunks, "
                f"{buffer_size // interleave_chunks} buffer per chunk, {interleave_skip} skip between chunks"
            )
            tokenized_dataset = InterleavedShuffleDataset(
                tokenized_dataset,
                num_interleave_chunks=interleave_chunks,
                chunk_buffer_size=buffer_size // interleave_chunks,
                skip_per_chunk=interleave_skip,
                seed=42,
            )
        elif shuffle_windows:
            # Standard buffer shuffle (limited to buffer_size items mixing)
            logger.info(f"Shuffling windows with buffer_size={buffer_size}")
            tokenized_dataset = tokenized_dataset.shuffle(seed=42, buffer_size=buffer_size)

    # Even in THD mode, we use a base MLM collator that requires a padding token to be set.
    if tokenizer.pad_token is None:
        logger.warning(f"Tokenizer does not have a padding token. Setting it to the EOS token: {tokenizer.eos_token}")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenized_dataset, tokenizer


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
    buffer_size: int = 50_000,
    use_stateful_dataloader: bool = False,
    text_column: str = "text",
    uppercase_labels: bool = False,
    mask_degenerate_bases: bool = False,
    pad_sequences_to_be_divisible_by: int | None = None,
    shuffle_sequences: bool = True,
    shuffle_windows: bool = False,
    interleaved_shuffle: bool = False,
    interleave_chunks: int = 10,
    interleave_skip: int = 50_000,
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
        buffer_size: The buffer size for shuffle operations.
        use_stateful_dataloader: Whether to use the StatefulDataLoader to enable checkpointing the dataloader state.
        text_column: Name of the column containing text sequences (default: "text").
        uppercase_labels: Whether to uppercase labels (genomic masking). Default: False.
        mask_degenerate_bases: Whether to mask non-ACGT bases (genomic masking). Default: False.
        pad_sequences_to_be_divisible_by: The number to pad sequences to be divisible by, required for FP8 training.
            Default: None.
        shuffle_sequences: Whether to shuffle raw sequences before windowing. Randomizes sequence order
            but windows from the same sequence remain consecutive. Fast. Default: True.
        shuffle_windows: Whether to shuffle windows after tokenization. Interleaves windows from different
            sequences for better batch diversity, but can be slower. Default: False.
        interleaved_shuffle: Whether to use interleaved shuffling for pseudo-global window coverage.
            More memory-efficient than large buffer_size. Overrides shuffle_windows. Default: False.
        interleave_chunks: Number of stream positions to interleave from. Default: 10.
        interleave_skip: Windows to skip between chunk starting positions. Default: 50000.

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
        tokenize_batch_size=micro_batch_size * prefetch_factor,
        shuffle_sequences=shuffle_sequences,
        shuffle_windows=shuffle_windows,
        interleaved_shuffle=interleaved_shuffle,
        interleave_chunks=interleave_chunks,
        interleave_skip=interleave_skip,
    )

    if isinstance(tokenized_dataset, (datasets.IterableDataset, InterleavedShuffleDataset)):
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
    buffer_size: int = 50_000,
    use_stateful_dataloader: bool = False,
    text_column: str = "text",
    uppercase_labels: bool = False,
    mask_degenerate_bases: bool = False,
    split_samples_in_token_packing: bool = True,
    pad_sequences_to_be_divisible_by: int | None = None,
    shuffle_sequences: bool = True,
    shuffle_windows: bool = False,
    interleaved_shuffle: bool = False,
    interleave_chunks: int = 10,
    interleave_skip: int = 50_000,
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
        buffer_size: The buffer size for shuffle operations.
        use_stateful_dataloader: Whether to use the StatefulDataLoader to enable checkpointing the dataloader state.
        text_column: Name of the column containing genomic sequences (default: "text").
        uppercase_labels: Whether to uppercase labels (genomic masking). Default: False.
        mask_degenerate_bases: Whether to mask degenerate bases (genomic masking). Default: False.
        split_samples_in_token_packing: Whether to split samples to form batches with exactly token_micro_batch_size
            tokens. Default: True.
        pad_sequences_to_be_divisible_by: If provided, sequences will be padded to be divisible by this value.
            This is useful for context parallelism. Defaults to None.
        shuffle_sequences: Whether to shuffle raw sequences before windowing. Randomizes sequence order
            but windows from the same sequence remain consecutive. Fast. Default: True.
        shuffle_windows: Whether to shuffle windows after tokenization. Interleaves windows from different
            sequences for better batch diversity, but can be slower. Default: False.
        interleaved_shuffle: Whether to use interleaved shuffling for pseudo-global window coverage.
            More memory-efficient than large buffer_size. Overrides shuffle_windows. Default: False.
        interleave_chunks: Number of stream positions to interleave from. Default: 10.
        interleave_skip: Windows to skip between chunk starting positions. Default: 50000.

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
        shuffle_sequences=shuffle_sequences,
        shuffle_windows=shuffle_windows,
        interleaved_shuffle=interleaved_shuffle,
        interleave_chunks=interleave_chunks,
        interleave_skip=interleave_skip,
    )

    assert isinstance(tokenized_dataset, (datasets.IterableDataset, InterleavedShuffleDataset)), (
        "THD token packing requires a streaming dataset."
    )
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
