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

from dataclasses import dataclass

import torch

from dataset import create_dataloader


@dataclass
class MockDistributedConfig:
    rank: int
    local_rank: int
    world_size: int


def test_iterable_dataloader_yields_different_values_per_rank():
    """Test that the iterable dataloader yields different values per rank."""

    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
    }

    rank1_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=2,
    )

    rank1_dataloader = create_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    )

    rank1_duplicate_dataloader = create_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    )

    rank2_config = MockDistributedConfig(
        rank=1,
        local_rank=1,
        world_size=2,
    )

    rank2_dataloader = create_dataloader(
        distributed_config=rank2_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    )

    rank1_batch = next(rank1_dataloader)
    rank1_duplicate_batch = next(rank1_duplicate_dataloader)
    rank2_batch = next(rank2_dataloader)

    for key, value in rank1_batch.items():
        assert (value != rank2_batch[key]).any()
        torch.testing.assert_close(value, rank1_duplicate_batch[key])


def test_map_dataset_dataloader_yields_different_values_per_rank():
    """Test that the map-style dataloader yields different values per rank."""

    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        # The only difference here is that this dataset doesn't set streaming to True
    }

    rank1_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=2,
    )

    rank1_dataloader = create_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    )

    rank1_duplicate_dataloader = create_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    )

    rank2_config = MockDistributedConfig(
        rank=1,
        local_rank=1,
        world_size=2,
    )

    rank2_dataloader = create_dataloader(
        distributed_config=rank2_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    )

    rank1_batch = next(rank1_dataloader)
    rank1_duplicate_batch = next(rank1_duplicate_dataloader)
    rank2_batch = next(rank2_dataloader)

    for key, value in rank1_batch.items():
        assert (value != rank2_batch[key]).any()
        torch.testing.assert_close(value, rank1_duplicate_batch[key])
