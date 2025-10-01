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

import pytest
import torch

from dataset import create_dataloader


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)

@dataclass
class MockDistributedConfig:
    rank: int
    local_rank: int
    world_size: int

def test_stateful_dataloader_works_with_iterator():
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
    }

    dist_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=1,
    )

    # First, collect reference batches from a fresh dataloader
    reference_dataloader_info = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        use_stateful_dataloader=True,
        mlm_probability=0,
    )

    # Collect 10 batches in total. Save the state of the sixth batch at iteration 5.
    reference_batches = []
    for i, batch in enumerate(reference_dataloader_info.iterator):
        reference_batches.append(batch["input_ids"])
        if i == 5:
            # save the state of the fifth batch
            dataloader_state = reference_dataloader_info.dataloader.state_dict()
        if i == 9:  # Collect 10 batches total
            break

    # Now test checkpoint/restore
    new_dataloader_info = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        use_stateful_dataloader=True,
        mlm_probability=0,
    )

    new_dataloader = new_dataloader_info.dataloader
    new_dataloader.load_state_dict(dataloader_state)

    # Note: Maybe the transform is non deterministic? Like the lazy loading map function.
    # Get three batches of data
    loaded_batches = []
    for i, batch in enumerate(new_dataloader):
        loaded_batches.append(batch["input_ids"])
        if i == 2:
            break

    assert len(reference_batches) == 10
    assert len(loaded_batches) == 3

    assert torch.equal(loaded_batches[0], reference_batches[6])
    assert torch.equal(loaded_batches[1], reference_batches[7])
    assert torch.equal(loaded_batches[2], reference_batches[8])


def test_stateful_dataloader():
    """Test that the stateful dataloader works with streaming = False.
    First we create a fresh dataloader and collect 10 batches, specified by 0th first index [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Save the state of the dataloader after the sixth batch (at iteration 5).
    Then we create another dataloader called loaded_dataloader and collect 3 batches which should be [6, 7, 8].
    then we compare the first 3 batches of the loaded_dataloader to batches 6, 7, 8 of the reference_batches.
    """

    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
    }

    dist_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=1,
    )

    # First, collect reference batches from a fresh dataloader
    reference_dataloader_info = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        use_stateful_dataloader=True,
        mlm_probability=0,
    )

    # Collect 10 batches in total. Save the state of the sixth batch at iteration 5.
    reference_batches = []
    for i, batch in enumerate(reference_dataloader_info.dataloader):
        reference_batches.append(batch["input_ids"])
        if i == 5:
            # save the state of the fifth batch
            dataloader_state = reference_dataloader_info.dataloader.state_dict()
        if i == 9:  # Collect 10 batches total
            break

    # Now test checkpoint/restore
    new_dataloader_info = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        use_stateful_dataloader=True,
        mlm_probability=0,
    )

    new_dataloader = new_dataloader_info.dataloader
    new_dataloader.load_state_dict(dataloader_state)

    # Note: Maybe the transform is non deterministic? Like the lazy loading map function.
    # Get three batches of data
    loaded_batches = []
    for i, batch in enumerate(new_dataloader):
        loaded_batches.append(batch["input_ids"])
        if i == 2:
            break

    assert len(reference_batches) == 10
    assert len(loaded_batches) == 3

    assert torch.equal(loaded_batches[0], reference_batches[6])
    assert torch.equal(loaded_batches[1], reference_batches[7])
    assert torch.equal(loaded_batches[2], reference_batches[8])


def test_stateful_dataloader_with_multiple_workers():
    """Test that the stateful dataloader works with multiple GPUs."""
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
    }

    dist_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=1,
    )

    # First, collect reference batches from a fresh dataloader
    reference_dataloader_info = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=4,
        use_stateful_dataloader=True,
        mlm_probability=0,
    )

    # Collect 10 batches in total. Save the state of the sixth batch at iteration 5.
    reference_batches = []
    for i, batch in enumerate(reference_dataloader_info.dataloader):
        reference_batches.append(batch["input_ids"])
        if i == 5:
            # save the state of the fifth batch
            dataloader_state = reference_dataloader_info.dataloader.state_dict()
        if i == 9:  # Collect 10 batches total
            break

    # Now test checkpoint/restore
    new_dataloader_info = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=4,
        use_stateful_dataloader=True,
        mlm_probability=0,
    )

    new_dataloader = new_dataloader_info.dataloader
    new_dataloader.load_state_dict(dataloader_state)

    # Note: Maybe the transform is non deterministic? Like the lazy loading map function.
    # Get three batches of data
    loaded_batches = []
    for i, batch in enumerate(new_dataloader):
        loaded_batches.append(batch["input_ids"])
        if i == 2:
            break

    assert len(reference_batches) == 10
    assert len(loaded_batches) == 3

    assert torch.equal(loaded_batches[0], reference_batches[6])
    assert torch.equal(loaded_batches[1], reference_batches[7])
    assert torch.equal(loaded_batches[2], reference_batches[8])



@requires_multi_gpu
def test_stateful_dataloader_with_multiple_gpus():
    pass



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
    ).iterator

    rank1_duplicate_dataloader = create_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    ).iterator

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
    ).iterator

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
    ).iterator

    rank1_duplicate_dataloader = create_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    ).iterator

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
    ).iterator

    rank1_batch = next(rank1_dataloader)
    rank1_duplicate_batch = next(rank1_duplicate_dataloader)
    rank2_batch = next(rank2_dataloader)

    for key, value in rank1_batch.items():
        assert (value != rank2_batch[key]).any()
        torch.testing.assert_close(value, rank1_duplicate_batch[key])


def test_lazy_tokenization_returns_batch():
    """Test that the lazy tokenization works."""

    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": False,
    }

    config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=2,
    )

    dataloader = create_dataloader(
        distributed_config=config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        use_lazy_tokenization=True,
    ).iterator

    batch = next(dataloader)
    assert batch is not None
