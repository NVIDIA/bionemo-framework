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
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from collator import MLMDataCollatorWithFlattening, TokenPackingDataset
from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)


def create_tokenized_dataset(
    distributed_config: DistributedConfig,
    tokenizer_name: str,
    load_dataset_kwargs: dict,
    max_seq_length: int = 1024,
    buffer_size: int = 10_000,
    use_lazy_tokenization: bool = True,
):
    """Create a tokenized dataset."""
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

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        """Tokenize the protein sequences."""
        return tokenizer(
            examples["sequence"],
            truncation=True,
            max_length=max_seq_length,
        )

    if isinstance(dataset, datasets.Dataset) and use_lazy_tokenization:
        # Using dataset.map on a non-streaming dataset will automatically perform and cache the transform, which can
        # trigger an expensive tokenization.
        tokenized_dataset = dataset.with_transform(tokenize_function)

    else:
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

    return tokenized_dataset, tokenizer


def create_bshd_dataloader(
    distributed_config: DistributedConfig,
    tokenizer_name: str,
    load_dataset_kwargs: dict,
    micro_batch_size: int,
    num_workers: int,
    max_seq_length: int = 1024,
    seed: int = 42,
    buffer_size: int = 10_000,
    use_lazy_tokenization: bool = True,
    mlm_probability: float = 0.15,
):
    """Create a dataloader for the dataset.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_name: The name of the tokenizer to pull from the HuggingFace Hub.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset` for the train dataset.
        micro_batch_size: The batch size (number of sequences) per device.
        num_workers: The number of workers to use for the dataloader.
        max_seq_length: The maximum length of the protein sequences.
        seed: The seed to use for the distributed sampler and data collator.
        buffer_size: The buffer size to use for the distributed sampler.
        use_lazy_tokenization: Whether to use datasets.set_transform for tokenization if the dataset is a
            non-streaming datasets.Dataset. Defaults to True.
        mlm_probability: The probability of masking tokens for MLM (default 0.15). Set to 0 for no masking.
        **kwargs: Unused, here to enable kwargs to match the signature of create_thd_dataloader.

    Returns:
        A dataloader that can be used for training.
    """
    tokenized_dataset, tokenizer = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=max_seq_length,
        buffer_size=buffer_size,
        use_lazy_tokenization=use_lazy_tokenization,
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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=max_seq_length,
        seed=seed,
    )

    train_dataloader = StatefulDataLoader(
        tokenized_dataset,
        sampler=sampler,
        batch_size=micro_batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_dataloader, tokenized_dataset if sampler is None else sampler


def create_thd_dataloader(
    distributed_config: DistributedConfig,
    tokenizer_name: str,
    load_dataset_kwargs: dict,
    micro_batch_size: int | None = None,
    token_micro_batch_size: int | None = None,
    num_workers: int = 1,
    max_seq_length: int = 1024,
    seed: int = 42,
    buffer_size: int = 10_000,
    mlm_probability: float = 0.15,
):
    """Create a dataloader that packs up to the maximum number of tokens per batch.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_name: The name of the tokenizer to pull from the HuggingFace Hub.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset` for the train dataset.
        micro_batch_size: The batch size (number of sequences) per device. This will set the token_micro_batch_size to
            micro_batch_size * max_seq_length. Defaults to None.
        token_micro_batch_size: The maximum number of tokens per batch. If None, the micro_batch_size * max_seq_length
            will be used. Defaults to None.
        num_workers: The number of workers to use for the dataloader. For iterable datasets, this should be 1.
        max_seq_length: The maximum length of the protein sequences.
        seed: The seed to use for the distributed sampler and data collator.
        buffer_size: The buffer size to use for the distributed sampler.
        mlm_probability: The probability of masking tokens for MLM (default 0.15). Set to 0 for no masking.
        **kwargs: Unused, here to enable kwargs to match the signature of create_bshd_dataloader.

    Returns:
        A dataloader that can be used for training.
    """
    tokenized_dataset, tokenizer = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=max_seq_length,
        buffer_size=buffer_size,
    )

    assert isinstance(tokenized_dataset, datasets.IterableDataset), "THD token packing requires a streaming dataset."
    if token_micro_batch_size is None:
        assert micro_batch_size is not None, "Only one of micro_batch_size or token_micro_batch_size can be provided."
        token_micro_batch_size = micro_batch_size * max_seq_length
    else:
        assert micro_batch_size is None, "Only one of micro_batch_size or token_micro_batch_size can be provided."
        assert token_micro_batch_size >= max_seq_length, "token_micro_batch_size must be greater than max_seq_length."

    # For THD, we pad out to the maximum number of tokens per batch for consistent array shapes.
    data_collator = MLMDataCollatorWithFlattening(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=token_micro_batch_size,
        seed=seed,
    )

    train_dataloader = StatefulDataLoader(
        TokenPackingDataset(tokenized_dataset, max_tokens_per_batch=token_micro_batch_size),
        batch_size=None,  # The TokenPackingDataset will handle the batching.
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_dataloader, tokenized_dataset
