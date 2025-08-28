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
from typing import Literal

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling


logger = logging.getLogger(__name__)


def create_datasets_and_collator(
    pretained_model: str | os.PathLike,
    max_length: int,
    data_size: Literal["full", "sanity", "parquet"],
) -> tuple[Dataset, Dataset, DataCollatorForLanguageModeling]:
    """Create the datasets and the data collator.

    Args:
        pretained_model: The path or tag of the pre-trained model to load the tokenizer from.
        max_length: The maximum length of the sequences.
        data_size: The size of the dataset to load. If "full", use and pre-process the full UR100P
            CSV dataset. This takes a long time without a cached dataset. If "small", use and
            pre-process the parquet version of the dataset, which is much faster than "full". If
            "sanity", truncates the evaluation dataset to 100 examples.

    Returns:
        A tuple containing the train dataset, the eval dataset, and the data collator.
    """
    tokenizer = AutoTokenizer.from_pretrained(pretained_model)

    def tokenize(examples):
        """Tokenize the examples."""
        return tokenizer(
            examples["sequence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    if data_size == "full":
        logger.info("Loading full dataset")
        train_dataset = load_dataset("chandar-lab/UR100P", split="train")
        eval_dataset = load_dataset("chandar-lab/UR100P", data_dir="UniProt", split="test")

    elif data_size == "sanity":
        logger.info("Loading sanity dataset")
        train_dataset = load_dataset("parquet", data_files="train.parquet", split="train")
        eval_dataset = load_dataset("parquet", data_files="train.parquet", split="train").select(range(10))

    elif data_size == "parquet":
        logger.info("Loading parquet branch dataset")
        train_dataset = load_dataset(
            "chandar-lab/UR100P",
            split="train",
            revision="refs/convert/parquet",
        )

        eval_dataset = load_dataset(
            "chandar-lab/UR100P",
            split="test",
            revision="refs/convert/parquet",
        )

    else:
        raise ValueError(f"Invalid data size: {data_size}")

    train_dataset = train_dataset.shuffle(seed=42)

    for dataset in [train_dataset, eval_dataset]:
        dataset.set_transform(tokenize, output_all_columns=True)
        dataset.remove_columns(["sequence", "name"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=max_length,
        seed=42,
    )

    return train_dataset, eval_dataset, data_collator
