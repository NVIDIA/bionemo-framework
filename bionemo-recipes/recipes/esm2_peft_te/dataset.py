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

import datasets
import datasets.distributed
from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from distributed_config import DistributedConfig


def create_dataloader(
    distributed_config: DistributedConfig,
    perform_validation: bool,
    tokenizer_name: str,
    micro_batch_size: int,
    val_micro_batch_size: int,
    num_workers: int,
    max_seq_length: int,
    stride: int,
    validation_samples: int,
    seed: int,
    ss3_classification: bool,
    load_dataset_kwargs: dict,
) -> tuple[DataLoader, DataLoader | None, IterableDataset | DistributedSampler]:
    """Create a dataloader for the secondary structure dataset."""
    train_dataset = load_dataset(**load_dataset_kwargs)

    print(
        f"Loading dataset: path: '{load_dataset_kwargs['path']}' | data_files: '{load_dataset_kwargs['data_files']}'."
    )

    # Split into train/val BEFORE any distributed operations
    if perform_validation:
        val_dataset = train_dataset.take(validation_samples)
        train_dataset = train_dataset.skip(validation_samples)

    if isinstance(train_dataset, IterableDataset):
        train_dataset = datasets.distributed.split_dataset_by_node(
            train_dataset,
            rank=distributed_config.rank,
            world_size=distributed_config.world_size,
        )
        train_dataset = train_dataset.shuffle(seed=seed, buffer_size=10_000)

        val_dataset = datasets.distributed.split_dataset_by_node(
            val_dataset,
            rank=distributed_config.rank,
            world_size=distributed_config.world_size,
        )

    ss8_token_map = {"H": 0, "I": 1, "G": 2, "E": 3, "B": 4, "S": 5, "T": 6, "~": 7}  # '~' denotes coil / unstructured
    ss3_token_map = {"H": 0, "I": 0, "G": 0, "E": 1, "B": 1, "S": 2, "T": 2, "~": 2}  # '~' denotes coil / unstructured

    if ss3_classification:
        ss_token_map = ss3_token_map
    else:
        ss_token_map = ss8_token_map

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenize_args = {
        "max_length": max_seq_length,
        "truncation": True,
        "stride": stride,
        "return_overflowing_tokens": True,
        "return_offsets_mapping": True,
    }

    def tokenize(example):
        """Tokenize both the input protein sequence and the secondary structure labels."""
        result = tokenizer(example["Sequence"], **tokenize_args)

        # While we can use the rust-based tokenizer for the protein sequence, we manually encode the secondary structure
        # labels. Our goal is to return a list of integer labels with the same shape as the input_ids.
        labels = []
        for batch_idx in range(len(result["input_ids"])):
            sequence_labels = []

            # This array maps the possibly-chunked result["input_ids"] to the original sequence. Because of
            # `return_overflowing_tokens`, each input sequence may be split into multiple input rows.
            offsets = result["offset_mapping"][batch_idx]

            # This gets the original secondary structure sequence for the current chunk.
            ss_sequence = example["Secondary_structure"][result["overflow_to_sample_mapping"][batch_idx]]

            for offset_start, offset_end in offsets:
                if offset_start == offset_end:
                    sequence_labels.append(-100)  # Start and end of the sequence tokens can be ignored.
                elif offset_end == offset_start + 1:  # All tokens are single-character.
                    ss_char = ss_sequence[offset_start]
                    ss_label_value = ss_token_map[ss_char]  # Encode the secondary structure character
                    sequence_labels.append(ss_label_value)
                else:
                    raise ValueError(f"Invalid offset: {offset_start} {offset_end}")

            labels.append(sequence_labels)

        return {"input_ids": result["input_ids"], "labels": labels}

    train_tokenized_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=[col for col in train_dataset.features if col not in ["input_ids", "labels"]],
    )

    if isinstance(train_tokenized_dataset, IterableDataset):
        train_sampler = None
    else:
        train_sampler = DistributedSampler(
            train_tokenized_dataset,
            rank=distributed_config.rank,
            num_replicas=distributed_config.world_size,
            seed=seed,
        )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding="max_length", max_length=1024)
    train_dataloader = DataLoader(
        train_tokenized_dataset,
        sampler=train_sampler,
        batch_size=micro_batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    if perform_validation:
        val_tokenized_dataset = val_dataset.map(
            tokenize,
            batched=True,
            remove_columns=[col for col in val_dataset.features if col not in ["input_ids", "labels"]],
        )

        if isinstance(train_tokenized_dataset, IterableDataset):
            val_sampler = None
        else:
            val_sampler = DistributedSampler(
                val_tokenized_dataset,
                rank=distributed_config.rank,
                num_replicas=distributed_config.world_size,
                seed=seed,
            )

        collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding="max_length", max_length=1024)
        val_dataloader = DataLoader(
            val_tokenized_dataset,
            sampler=val_sampler,
            batch_size=val_micro_batch_size,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader, train_tokenized_dataset if train_sampler is None else train_sampler
