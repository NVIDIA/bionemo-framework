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
import os
import warnings
from collections import defaultdict
from pathlib import Path

import datasets
import datasets.distributed
import torch
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from collator import MLMDataCollatorWithFlattening
from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)


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
    mlm_probability: float = 0.15,
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
        mlm_probability: The probability of masking tokens for MLM (default 0.15). Set to 0 for no masking.

    Returns:
        A dataloader that can be used for training.
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
            mlm_probability=mlm_probability,
            pad_to_multiple_of=sequence_packing_pad_to_multiple_of,
            seed=seed,
        )
    else:
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

    return train_dataloader, dataset if sampler is None else sampler


def save_dataloader(
    dataloader: StatefulDataLoader,
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
):
    """Save the dataloader state to a file.

    Here we save the dataloader state based on global rank.

    Args:
        dataloader: The dataloader to save the state of.
        ckpt_path: The path to save the dataloader state to.
        dist_config: The distributed configuration.
    """
    num_workers = dataloader.num_workers
    ckpt_path = Path(ckpt_path)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    dataloader_path = ckpt_path / f"dataloader_rank_{dist_config.rank}_num_workers_{num_workers}.pt"

    dataloader_state = dataloader.state_dict()
    torch.save(dataloader_state, dataloader_path)
    logger.info(f"Saved DDP dataloader state to {dataloader_path}")


def load_dataloader(
    dataloader: StatefulDataLoader,
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
    num_workers: int,
):
    """Load the dataloader state from a file.

    Here we load the dataloader state based on global rank.

    Args:
        dataloader: The dataloader to load the state of.
        ckpt_path: The path to load the dataloader state from.
        dist_config: The distributed configuration.
        num_workers: The number of workers to load the dataloader state for.
    """
    # First we check to see if the checkpoint folder exists.
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        logger.info("No checkpoint folder found, starting from scratch")
        return

    # Then we check to see if there are any dataloader files in the checkpoint path.
    dataloader_files = [f for f in ckpt_path.iterdir() if f.name.startswith("dataloader_rank_")]

    if not dataloader_files:
        logger.info("No Dataloader checkpoint found, starting from scratch")
        return

    # Parse the files by rank and step.
    dataloader_files_by_rank = defaultdict(list)
    for file in dataloader_files:
        rank = int(file.stem.split("_")[2])
        dataloader_files_by_rank[rank].append(file)

    if dist_config.rank not in dataloader_files_by_rank:
        raise ValueError(
            f"No dataloader checkpoint found for rank {dist_config.rank}, available ranks: {list(dataloader_files_by_rank.keys())}"
        )

    # Check to see if the num_workers matches the num_workers in the checkpoint.
    if dataloader.num_workers != num_workers:
        warnings.warn(
            f"No dataloader checkpoint found for num_workers {num_workers}, saved num_workers from ckpt: {dataloader.num_workers}",
            UserWarning,
            stacklevel=2,
        )
        return

    dataloader_path = ckpt_path / f"dataloader_rank_{dist_config.rank}_num_workers_{dataloader.num_workers}.pt"

    dataloader_state = torch.load(dataloader_path)
    dataloader.load_state_dict(dataloader_state)
    logger.info(f"Loaded DDP dataloader state from {dataloader_path}")
    return dataloader
