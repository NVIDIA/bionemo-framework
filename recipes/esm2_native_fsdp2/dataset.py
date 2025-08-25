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

from pathlib import Path

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling


# Create the dataset -- here, we just use a simple parquet file with some raw protein sequences
# stored in the repo itself to avoid external dependencies.
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")


def infinite_dataloader(dataloader, sampler):
    """Create an infinite iterator that automatically restarts at the end of each epoch."""
    epoch = 0
    while True:
        sampler.set_epoch(epoch)  # Update epoch for proper shuffling
        for batch in dataloader:
            yield batch
        epoch += 1  # Increment epoch counter after completing one full pass


def create_dataloader(data_dir, batch_size, max_length=1024):
    """Create a dataloader for the dataset.

    Args:
        data_dir: The directory containing the dataset.
        batch_size: The batch size.
        max_length: The maximum length of the protein sequences.

    Returns:
        A dataloader that just infinitely loops over the dataset.
        The number of batches in the dataloader.
    """
    # We copy this parquet file to the container to avoid external dependencies, modify if you're
    # using a local dataset. If you're reading this and scaling up the dataset to a larger size,
    # look into `set_transform` and other streaming options from the `datasets` library.
    data_path = Path(data_dir) / "train.parquet"
    dataset = load_dataset("parquet", data_files=data_path.as_posix(), split="train")

    def tokenize_function(examples):
        """Tokenize the protein sequences."""
        return tokenizer(
            examples["sequence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=max_length,
    )

    # Create dataloader with distributed sampler
    train_sampler = DistributedSampler(tokenized_dataset)
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # Create the infinite iterator
    train_iterator = infinite_dataloader(train_dataloader, train_sampler)

    return train_iterator, len(train_dataloader)
