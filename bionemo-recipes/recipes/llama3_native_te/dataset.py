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
from distributed_config import DistributedConfig
from torch.utils.data import DataLoader, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling


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
        # Pass examples[sequence_column] directly to tokenizer
        # With batched=True + batch_size=1 + return_overflowing_tokens,
        # the tokenizer returns windows that dataset.map() will unpack
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
        logger.info("Using with_transform for lazy tokenization")
        tokenized_dataset = dataset.with_transform(tokenize_with_windowing)
    else:
        # For streaming datasets with huge genomic sequences, process one sequence at a time (batch_size=1)
        # The tokenizer's return_overflowing_tokens will automatically expand windows into separate items
        # IMPORTANT: Must explicitly remove the sequence column - dataset.column_names is None for IterableDataset!
        logger.info(f"Applying dataset.map with windowing (max_seq_length={max_seq_length}, stride={stride})")
        logger.info(f"  Removing column: [{sequence_column}]")
        tokenized_dataset = dataset.map(
            tokenize_with_windowing,
            batched=True,
            batch_size=1,  # Process one huge sequence at a time to avoid OOM
            remove_columns=[sequence_column],  # Explicitly remove the text column
        )
        logger.info("Dataset tokenization mapping complete")

    return tokenized_dataset, tokenizer


def create_bshd_dataloader(
    distributed_config: DistributedConfig,
    tokenizer_path: str,
    load_dataset_kwargs: dict,
    micro_batch_size: int,
    num_workers: int = 0,
    max_seq_length: int = 8192,
    stride: int = 200,
    seed: int = 42,
    buffer_size: int = 500_000,
    use_lazy_tokenization: bool = True,
    use_stateful_dataloader: bool = False,
    sequence_column: str = "sequence",
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

    # Use DataCollatorForLanguageModeling with mlm=False for CLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling (no masking)
    )

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
