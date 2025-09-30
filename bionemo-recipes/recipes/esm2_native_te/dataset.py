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
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from collator import MLMDataCollatorWithFlattening
from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)


def infinite_dataloader(dataloader, dataset_or_sampler):
    """Create an infinite iterator that automatically restarts at the end of each epoch.

    Args:
        dataloader: The DataLoader to loop through.
        dataset_or_sampler: The dataset or sampler to set epochs for.
    """
    epoch = 0
    while True:
        dataset_or_sampler.set_epoch(epoch)  # Update epoch for proper shuffling
        for batch in dataloader:
            yield batch
        epoch += 1  # Increment epoch counter after completing one full pass


def create_dataloader(
    distributed_config: DistributedConfig,
    tokenizer_name: str,
    load_dataset_kwargs: dict,
    micro_batch_size: int,
    num_workers: int,
    max_seq_length: int = 1024,
    seed: int = 42,
    use_sequence_packing: bool = False,
    sequence_packing_pad_to_multiple_of: int | None = None,
    buffer_size: int = 10_000,
    use_lazy_tokenization: bool = True,
):
    """Create a dataloader for the dataset.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_name: The name of the tokenizer to pull from the HuggingFace Hub.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset` for the train dataset.
        micro_batch_size: The batch size per device.
        num_workers: The number of workers to use for the dataloader.
        max_seq_length: The maximum length of the protein sequences.
        seed: The seed to use for the distributed sampler and data collator.
        use_sequence_packing: Whether to use sequence packing.
        sequence_packing_pad_to_multiple_of: The padding to use for the sequence packing collator, for fp8 support.
        buffer_size: The buffer size to use for the distributed sampler.
        use_lazy_tokenization: Whether to use datasets.set_transform for tokenization if the dataset is a
            non-streaming datasets.Dataset. Defaults to True.

    Returns:
        A dataloader that just infinitely loops over the dataset.
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
        sampler = None
    else:
        sampler = DistributedSampler(
            dataset,
            rank=distributed_config.rank,
            num_replicas=distributed_config.world_size,
            seed=seed,
        )

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

    if use_sequence_packing:
        data_collator = MLMDataCollatorWithFlattening(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            pad_to_multiple_of=sequence_packing_pad_to_multiple_of,
            seed=seed,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            pad_to_multiple_of=max_seq_length,
            seed=seed,
        )

    train_dataloader = DataLoader(
        tokenized_dataset,
        sampler=sampler,
        batch_size=micro_batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # Create the infinite iterator
    train_iterator = infinite_dataloader(train_dataloader, dataset if sampler is None else sampler)

    return train_iterator
