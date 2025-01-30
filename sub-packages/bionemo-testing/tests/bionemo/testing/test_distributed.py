# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing.spawn
from megatron.core import parallel_state

from bionemo.testing.distributed import dist_environment


MAX_WORLD_SIZE = 4
AVAILABLE_WORLD_SIZE = torch.cuda.device_count()
WORLD_SIZES = range(min(MAX_WORLD_SIZE, torch.cuda.device_count()))


# TODO @sichu improve documentation
#  arguments will be recognized as fixture if decorating the tested function directly into test function,
#  so we have to create a test function for each tested function.


def all_reduce_sum(rank: int, world_size: int):
    """Private test function for torch.distributed mean reduce."""
    with dist_environment(rank=rank, world_size=world_size):
        tensor = torch.tensor([rank + 1]).cuda(rank)
        dist.all_reduce(tensor)
        assert tensor.item() == world_size * (world_size + 1) / 2


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_all_reduce_sum(world_size: int):
    """Multiprocessing test of _test_all_reduce_sum."""
    torch.multiprocessing.spawn(
        fn=all_reduce_sum,
        args=(world_size,),
        nprocs=world_size,
    )


def data_parallel_group(rank: int, world_size: int):
    """Private test function for dp parallel state."""
    with dist_environment(rank=rank, world_size=world_size):
        assert parallel_state.get_data_parallel_rank() == rank
        assert parallel_state.get_data_parallel_world_size() == world_size
        assert parallel_state.get_data_parallel_src_rank() == 0


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_data_parallel_group(world_size: int):
    """Multiprocessing test of _test_data_parallel_group."""
    torch.multiprocessing.spawn(
        fn=data_parallel_group,
        args=(world_size,),
        nprocs=world_size,
    )


def tensor_model_parallel_group(rank: int, world_size: int):
    """Private test function for tp parallel state."""
    with dist_environment(rank=rank, world_size=world_size, tensor_model_parallel_size=world_size):
        assert parallel_state.get_tensor_model_parallel_rank() == rank
        assert parallel_state.get_tensor_model_parallel_world_size() == world_size
        assert parallel_state.get_tensor_model_parallel_src_rank() == 0


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_tensor_model_parallel_group(world_size: int):
    """Multiprocessing test of _test_tensor_model_parallel_group."""
    torch.multiprocessing.spawn(
        fn=tensor_model_parallel_group,
        args=(world_size,),
        nprocs=world_size,
    )


def pipeline_model_parallel_group(rank: int, world_size: int):
    """Private test function for pp parallel state."""
    with dist_environment(rank=rank, world_size=world_size, pipeline_model_parallel_size=world_size):
        assert parallel_state.get_pipeline_model_parallel_rank() == rank
        assert parallel_state.get_pipeline_model_parallel_world_size() == world_size
        if rank == 0:
            assert parallel_state.is_pipeline_first_stage()
        if rank == world_size:
            assert parallel_state.is_pipeline_last_stage()


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_pipeline_model_parallel_group(world_size: int):
    """Multiprocessing test of _test_pipeline_model_parallel_group."""
    torch.multiprocessing.spawn(
        fn=pipeline_model_parallel_group,
        args=(world_size,),
        nprocs=world_size,
    )
