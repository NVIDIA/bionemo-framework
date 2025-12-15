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
    DataCollatorWithFlattening,
)

from collator import TokenPackingDataset
from distributed_config import DistributedConfig
from utils import SS3_LABEL2ID, SS8_LABEL2ID


def create_dataloader(
    distributed_config: DistributedConfig,
    use_sequence_packing: bool,
    tokenizer_name: str,
    micro_batch_size: int,
    val_micro_batch_size: int,
    num_workers: int,
    max_seq_length: int,
    stride: int,
    seed: int,
    ss3_classification: bool,
    load_dataset_kwargs: dict,
) -> tuple[DataLoader, DataLoader | None, IterableDataset | DistributedSampler]:
    """Create a dataloader for the secondary structure dataset."""
    dataset_or_dataset_dict = load_dataset(**load_dataset_kwargs)

    if isinstance(dataset_or_dataset_dict, dict):
        train_dataset = dataset_or_dataset_dict.get("train")
        assert train_dataset, "'train' split must be specified."
        val_dataset = dataset_or_dataset_dict.get("validation")
    else:
        train_dataset = dataset_or_dataset_dict
        val_dataset = None

    print(
        f"Loading dataset: path: '{load_dataset_kwargs['path']}' | data_files: '{load_dataset_kwargs['data_files']}'."
    )

    perform_validation = val_dataset is not None

    if isinstance(train_dataset, IterableDataset):
        train_dataset = datasets.distributed.split_dataset_by_node(
            train_dataset,
            rank=distributed_config.rank,
            world_size=distributed_config.world_size,
        )
        train_dataset = train_dataset.shuffle(seed=seed, buffer_size=10_000)

        if perform_validation:
            val_dataset = datasets.distributed.split_dataset_by_node(
                val_dataset,
                rank=distributed_config.rank,
                world_size=distributed_config.world_size,
            )

    if ss3_classification:
        ss_token_map = SS3_LABEL2ID
    else:
        ss_token_map = SS8_LABEL2ID

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

    if use_sequence_packing:
        assert isinstance(train_tokenized_dataset, datasets.IterableDataset), (
            "THD token packing requires a streaming dataset."
        )
        collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)
        train_tokenized_dataset = TokenPackingDataset(
            train_tokenized_dataset, max_tokens_per_batch=micro_batch_size * max_seq_length
        )
        batch_size = None  # The TokenPackingDataset will handle the batching.
    else:
        collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer, padding="max_length", max_length=max_seq_length
        )
        batch_size = micro_batch_size

    train_dataloader = DataLoader(
        train_tokenized_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
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

        if isinstance(val_tokenized_dataset, IterableDataset):
            val_sampler = None
        else:
            val_sampler = DistributedSampler(
                val_tokenized_dataset,
                rank=distributed_config.rank,
                num_replicas=distributed_config.world_size,
                seed=seed,
            )

        if use_sequence_packing:
            assert isinstance(val_tokenized_dataset, datasets.IterableDataset), (
                "THD token packing requires a streaming dataset."
            )
            collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)
            val_tokenized_dataset = TokenPackingDataset(
                val_tokenized_dataset, max_tokens_per_batch=micro_batch_size * max_seq_length
            )
            val_batch_size = None  # The TokenPackingDataset will handle the batching.
        else:
            collator = DataCollatorForTokenClassification(
                tokenizer=tokenizer, padding="max_length", max_length=max_seq_length
            )
            val_batch_size = val_micro_batch_size

        val_dataloader = DataLoader(
            val_tokenized_dataset,
            sampler=val_sampler,
            batch_size=val_batch_size,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader, train_tokenized_dataset if train_sampler is None else train_sampler
