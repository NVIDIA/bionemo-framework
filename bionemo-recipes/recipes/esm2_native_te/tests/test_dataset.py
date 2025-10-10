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

import os
import shutil
from dataclasses import dataclass

import pytest
import torch

from dataset import create_dataloader, load_dataloader, save_dataloader


@dataclass
class MockDistributedConfig:
    rank: int
    local_rank: int
    world_size: int


def test_stateful_dataloader_load_fails_if_num_workers_mismatch(tmp_path):
    dataloader_path = tmp_path / "dl_test_num_workers_mismatch"
    shutil.rmtree(dataloader_path, ignore_errors=True)
    os.makedirs(dataloader_path, exist_ok=True)
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
    }

    rank0_dist_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=1,
    )

    reference_dataloader, _ = create_dataloader(
        distributed_config=rank0_dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )

    save_dataloader(
        dataloader=reference_dataloader,
        ckpt_path=dataloader_path,
        dist_config=rank0_dist_config,
    )

    # Now try to load the dataloader state for rank 1.
    with pytest.warns(
        UserWarning, match=r"No dataloader checkpoint found for num_workers 2, saved num_workers from ckpt: 1"
    ):
        load_dataloader(
            dataloader=reference_dataloader,
            ckpt_path=dataloader_path,
            num_workers=2,
            dist_config=rank0_dist_config,
        )


def test_stateful_dataloader_load_fails_if_rank_mismatch(tmp_path):
    """In this test we are going to save the dataloader state using only a rank0
    distributed config. Then we will try to load the dataloader state with a rank1 and a rank2 dist config and we shuold hit an error.
    """
    dataloader_path = tmp_path / "dl_test_ranks_mismatch"
    shutil.rmtree(dataloader_path, ignore_errors=True)
    os.makedirs(dataloader_path, exist_ok=True)
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
    }

    rank0_dist_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=2,
    )
    rank1_dist_config = MockDistributedConfig(
        rank=1,
        local_rank=1,
        world_size=2,
    )

    reference_dataloader, _ = create_dataloader(
        distributed_config=rank0_dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )
    # Save dataloader state for rank 0
    for i, _ in enumerate(reference_dataloader):
        if i == 5:
            dataloader_path = dataloader_path / f"step_{i}"
            os.makedirs(dataloader_path, exist_ok=True)
            save_dataloader(
                dataloader=reference_dataloader,
                ckpt_path=dataloader_path,
                dist_config=rank0_dist_config,
            )
            break

    # Now try to load the dataloader state for rank 1.
    with pytest.raises(ValueError, match=r"No dataloader checkpoint found for rank 1, available ranks: \[0\]"):
        load_dataloader(
            dataloader=reference_dataloader,
            ckpt_path=dataloader_path,
            num_workers=1,
            dist_config=rank1_dist_config,
        )


def test_load_dataset_state_from_latest_checkpoint(tmp_path):
    dataloader_path = tmp_path / "dl_test"
    os.makedirs(dataloader_path, exist_ok=True)
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

    reference_dataloader, _ = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )

    for i, _ in enumerate(reference_dataloader):
        if i in [1, 5, 9]:
            dataloader_path = dataloader_path / f"step_{i}"
            os.makedirs(dataloader_path, exist_ok=True)
            save_dataloader(
                dataloader=reference_dataloader,
                ckpt_path=dataloader_path,
                dist_config=dist_config,
            )
        if i == 9:
            break
    new_dataloader, _ = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )

    new_dataloader = load_dataloader(
        dataloader=new_dataloader,
        ckpt_path=dataloader_path,
        dist_config=dist_config,
        num_workers=1,
    )
    assert new_dataloader.state_dict()["_snapshot"]["_snapshot_step"] == 10
    shutil.rmtree(dataloader_path)


def test_map_style_stateful_dataloader_resumption_multi_process(tmp_path):  # noqa: C901
    dataloader_path = tmp_path / "dl_test"
    os.makedirs(dataloader_path, exist_ok=True)
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": False,
    }

    rank0_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=2,
    )
    rank1_config = MockDistributedConfig(
        rank=1,
        local_rank=1,
        world_size=2,
    )

    # Based on local rank.
    # Create dataloader for process 0
    rank0_dataloader, _ = create_dataloader(
        distributed_config=rank0_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )

    rank1_dataloader, _ = create_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )

    dataloader_path_step_4 = dataloader_path / "step_4"
    dataloader_path_step_5 = dataloader_path / "step_5"
    os.makedirs(dataloader_path_step_4, exist_ok=True)
    os.makedirs(dataloader_path_step_5, exist_ok=True)

    # Run 10 batches, save state at step 5
    reference_rank0_batches = []
    for i, batch in enumerate(rank0_dataloader):
        reference_rank0_batches.append(batch["input_ids"])
        if i == 5:
            save_dataloader(
                dataloader=rank0_dataloader,
                ckpt_path=dataloader_path_step_5,
                dist_config=rank0_config,
            )
        if i == 9:
            break

    # Run 10 batches, save state at step 4
    reference_rank1_batches = []

    for i, batch in enumerate(rank1_dataloader):
        reference_rank1_batches.append(batch["input_ids"])
        if i == 4:
            save_dataloader(
                dataloader=rank1_dataloader,
                ckpt_path=dataloader_path_step_4,
                dist_config=rank1_config,
            )
        if i == 9:
            break

    # Load rank0 dataloader state at step 5
    rank0_dataloader_info_reloaded, _ = create_dataloader(
        distributed_config=rank0_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )

    rank0_dataloader_reloaded = load_dataloader(
        dataloader=rank0_dataloader_info_reloaded,
        ckpt_path=dataloader_path_step_5,
        num_workers=1,
        dist_config=rank0_config,
    )

    # Load rank1 dataloader state at step 4
    rank1_dataloader_info_reloaded, _ = create_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )

    rank1_dataloader_reloaded = load_dataloader(
        dataloader=rank1_dataloader_info_reloaded,
        ckpt_path=dataloader_path_step_4,
        num_workers=1,
        dist_config=rank1_config,
    )

    # Run 3 more steps on loaded_rank0_dataloader and save the batches
    loaded_rank0_batches = []
    for i, batch in enumerate(rank0_dataloader_reloaded):
        loaded_rank0_batches.append(batch["input_ids"])
        if i == 2:  # Collect 3 batches (indices 0-2) to match with reference batches 7-9
            break
    # Run 4 more steps on loaded_rank1_dataloader and save the batches
    loaded_rank1_batches = []
    for i, batch in enumerate(rank1_dataloader_reloaded):
        loaded_rank1_batches.append(batch["input_ids"])
        if i == 3:  # Collect 4 batches (indices 0-3) to match with reference batches 6-9
            break

    assert torch.equal(loaded_rank0_batches[0], reference_rank0_batches[6])
    assert torch.equal(loaded_rank0_batches[1], reference_rank0_batches[7])
    assert torch.equal(loaded_rank0_batches[2], reference_rank0_batches[8])

    assert torch.equal(loaded_rank1_batches[0], reference_rank1_batches[5])
    assert torch.equal(loaded_rank1_batches[1], reference_rank1_batches[6])
    assert torch.equal(loaded_rank1_batches[2], reference_rank1_batches[7])
    assert torch.equal(loaded_rank1_batches[3], reference_rank1_batches[8])

    shutil.rmtree(dataloader_path)


def test_iterable_stateful_dataloader_resumption_multi_process(tmp_path):  # noqa: C901
    dataloader_path = tmp_path / "dl_test"
    os.makedirs(dataloader_path, exist_ok=True)
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
    }

    rank0_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=2,
    )
    rank1_config = MockDistributedConfig(
        rank=1,
        local_rank=1,
        world_size=2,
    )

    # Based on local rank.
    # Create dataloader for process 0
    rank0_dataloader_info, _ = create_dataloader(
        distributed_config=rank0_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )

    rank1_dataloader_info, _ = create_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )
    dataloader_path_step_4 = dataloader_path / "step_4"
    dataloader_path_step_5 = dataloader_path / "step_5"
    os.makedirs(dataloader_path_step_4, exist_ok=True)
    os.makedirs(dataloader_path_step_5, exist_ok=True)

    # Run 10 batches, save state at step 5
    reference_rank0_batches = []
    for i, batch in enumerate(rank0_dataloader_info):
        reference_rank0_batches.append(batch["input_ids"])
        if i == 5:
            save_dataloader(
                dataloader=rank0_dataloader_info,
                ckpt_path=dataloader_path_step_5,
                dist_config=rank0_config,
            )
        if i == 9:
            break

    # Run 10 batches, save state at step 4
    reference_rank1_batches = []

    for i, batch in enumerate(rank1_dataloader_info):
        reference_rank1_batches.append(batch["input_ids"])
        if i == 4:
            save_dataloader(
                dataloader=rank1_dataloader_info,
                ckpt_path=dataloader_path_step_4,
                dist_config=rank1_config,
            )
        if i == 9:
            break

    # Load rank0 dataloader state at step 5
    rank0_dataloader_reloaded, _ = create_dataloader(
        distributed_config=rank0_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )

    rank0_dataloader_reloaded = load_dataloader(
        dataloader=rank0_dataloader_reloaded,
        ckpt_path=dataloader_path_step_5,
        num_workers=1,
        dist_config=rank0_config,
    )

    # Load rank1 dataloader state at step 4
    rank1_dataloader_reloaded, _ = create_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )

    rank1_dataloader_reloaded = load_dataloader(
        dataloader=rank1_dataloader_reloaded,
        ckpt_path=dataloader_path_step_4,
        num_workers=1,
        dist_config=rank1_config,
    )

    # Run 3 more steps on loaded_rank0_dataloader and save the batches
    loaded_rank0_batches = []
    for i, batch in enumerate(rank0_dataloader_reloaded):
        loaded_rank0_batches.append(batch["input_ids"])
        if i == 2:  # Collect 3 batches (indices 0-2) to match with reference batches 7-9
            break
    # Run 4 more steps on loaded_rank1_dataloader and save the batches
    loaded_rank1_batches = []
    for i, batch in enumerate(rank1_dataloader_reloaded):
        loaded_rank1_batches.append(batch["input_ids"])
        if i == 3:  # Collect 4 batches (indices 0-3) to match with reference batches 6-9
            break

    assert torch.equal(loaded_rank0_batches[0], reference_rank0_batches[6])
    assert torch.equal(loaded_rank0_batches[1], reference_rank0_batches[7])
    assert torch.equal(loaded_rank0_batches[2], reference_rank0_batches[8])

    assert torch.equal(loaded_rank1_batches[0], reference_rank1_batches[5])
    assert torch.equal(loaded_rank1_batches[1], reference_rank1_batches[6])
    assert torch.equal(loaded_rank1_batches[2], reference_rank1_batches[7])
    assert torch.equal(loaded_rank1_batches[3], reference_rank1_batches[8])

    shutil.rmtree(dataloader_path)


def test_stateful_dataloader_works_save_dataloader_and_load_dataloader_single_process(tmp_path):
    # Test uses rank 0.
    dataloader_path = tmp_path / "dl_test"
    os.makedirs(dataloader_path, exist_ok=True)
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
    reference_dataloader_info, _ = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )

    # Collect 10 batches in total. Save the state of the sixth batch at iteration 5.
    reference_batches = []
    for i, batch in enumerate(reference_dataloader_info):
        reference_batches.append(batch["input_ids"])
        if i == 5:
            # save the state of the fifth batch
            save_dataloader(
                dataloader=reference_dataloader_info,
                ckpt_path=dataloader_path,
                dist_config=dist_config,
            )
        if i == 9:  # Collect 10 batches total
            break

    # Now test checkpoint/restore
    new_dataloader_info, _ = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )

    new_dataloader = load_dataloader(
        dataloader=new_dataloader_info,
        ckpt_path=dataloader_path,
        num_workers=1,
        dist_config=dist_config,
    )

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

    shutil.rmtree(dataloader_path)


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
    reference_dataloader_info, _ = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )

    # Collect 10 batches in total. Save the state of the sixth batch at iteration 5.
    reference_batches = []
    for i, batch in enumerate(reference_dataloader_info):
        reference_batches.append(batch["input_ids"])
        if i == 5:
            # save the state of the fifth batch
            dataloader_state = reference_dataloader_info.state_dict()
        if i == 9:  # Collect 10 batches total
            break

    # Now test checkpoint/restore
    new_dataloader_info, _ = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
    )

    new_dataloader = new_dataloader_info
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


def test_stateful_dataloader_with_multiple_workers(tmp_path):
    """Test that the stateful dataloader works with multiple GPUs."""
    dataloader_path = tmp_path / "dl_test_multi_workers"
    shutil.rmtree(dataloader_path, ignore_errors=True)
    os.makedirs(dataloader_path, exist_ok=True)
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
    reference_dataloader, _ = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=4,
        mlm_probability=0,
    )

    # Collect 10 batches in total. Save the state of the sixth batch at iteration 5.
    reference_batches = []
    for i, batch in enumerate(reference_dataloader):
        reference_batches.append(batch["input_ids"])
        if i == 5:
            # save the state of the fifth batch
            dataloader_path = dataloader_path / f"step_{i}"
            os.makedirs(dataloader_path, exist_ok=True)
            save_dataloader(
                dataloader=reference_dataloader,
                ckpt_path=dataloader_path,
                dist_config=dist_config,
            )
        if i == 9:  # Collect 10 batches total
            break

    # Now test checkpoint/restore
    new_dataloader, _ = create_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=4,
        mlm_probability=0,
    )

    load_dataloader(
        dataloader=new_dataloader,
        ckpt_path=dataloader_path,
        num_workers=4,
        dist_config=dist_config,
    )

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


def test_iterable_dataloader_yields_different_values_per_rank():
    """Test that the iterable dataloader yields different values per rank."""
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
        # The only difference here is that this dataset doesn't set streaming to True
    }

    rank1_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=2,
    )

    rank1_dataloader, _ = create_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    )

    rank1_duplicate_dataloader, _ = create_dataloader(
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

    rank2_dataloader, _ = create_dataloader(
        distributed_config=rank2_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    )

    rank1_batch = next(iter(rank1_dataloader))
    rank1_duplicate_batch = next(iter(rank1_duplicate_dataloader))
    rank2_batch = next(iter(rank2_dataloader))

    for key, value in rank1_batch.items():
        assert rank1_batch[key] is not None
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

    rank1_dataloader, _ = create_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    )

    rank1_duplicate_dataloader, _ = create_dataloader(
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

    rank2_dataloader, _ = create_dataloader(
        distributed_config=rank2_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    )

    rank1_batch = next(iter(rank1_dataloader))
    rank1_duplicate_batch = next(iter(rank1_duplicate_dataloader))
    rank2_batch = next(iter(rank2_dataloader))

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

    dataloader, _ = create_dataloader(
        distributed_config=config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        use_lazy_tokenization=True,
    )

    batch = next(iter(dataloader))
    assert batch is not None
