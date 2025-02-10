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


import os
from contextlib import contextmanager
from typing import List, Optional

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing.spawn
from pytest import MonkeyPatch
from torch._C._distributed_c10d import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh

from bionemo.moco.distributions.prior.discrete.mask import DiscreteMaskedPrior
from bionemo.moco.distributions.time.uniform import UniformTimeDistribution
from bionemo.moco.interpolants.continuous_time.discrete.mdlm import MDLM
from bionemo.moco.schedules.noise.continuous_noise_transforms import LogLinearExpNoiseTransform


@pytest.fixture
def mdlm():
    time_distribution = UniformTimeDistribution(discrete_time=False)
    prior = DiscreteMaskedPrior(num_classes=20)
    noise_schedule = LogLinearExpNoiseTransform()
    mdlm = MDLM(time_distribution, prior, noise_schedule)
    return mdlm


DEVICE_MESH: Optional[DeviceMesh] = None


def mdlm_parallel_interpolate(
    rank: int,
    mdlm,
    world_size: int = 1,
    device_type: str = "cuda",
):
    with parallel_context(rank=rank, world_size=world_size):  # , backend="nccl", device_type=device_type):
        data_gpu = torch.randint(0, 16, (5, 10)).to("cuda")
        t_gpu = mdlm.sample_time(5, device=data_gpu.device)
        result = mdlm.interpolate(data_gpu, t_gpu)
        print(t_gpu, torch.distributed.get_rank())
        assert result.shape == (5, 10)


@pytest.mark.parametrize("world_size", [1, 2])
def test_mdlm_parallel_interpolate(
    mdlm,
    world_size,
    device_type: str = "cuda",
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Check if world_size number of devices are visible
    visible_devices = torch.cuda.device_count() if device_type == "cuda" else 1  # assume 1 for non-CUDA (e.g., CPU)
    if world_size > visible_devices:
        pytest.skip(f"Insufficient devices: {world_size} devices requested, but only {visible_devices} are visible")

    torch.multiprocessing.spawn(
        fn=mdlm_parallel_interpolate,
        args=(
            mdlm,
            world_size,
            device_type,
        ),
        nprocs=world_size,
    )


def initialize_parallel_states(
    context_parallel_size: int = 1,
) -> None:
    # Get world size and rank. Ensure some consistencies.
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    assert world_size % context_parallel_size == 0, "world size not divisible by context_parallel_size"

    global DEVICE_MESH
    data_parallel_size = world_size // context_parallel_size
    DEVICE_MESH = dist.device_mesh.init_device_mesh(
        "cuda",
        (context_parallel_size, data_parallel_size),
        mesh_dim_names=("cp", "dp"),
    )


def clean_up_parallel_states() -> None:
    global DEVICE_MESH
    DEVICE_MESH = None


def get_data_parallel_group() -> ProcessGroup:
    global DEVICE_MESH
    if DEVICE_MESH is None:
        raise ValueError("device mesh is not set.")
    return DEVICE_MESH.get_group("dp")


def get_context_parallel_group() -> ProcessGroup:
    global DEVICE_MESH
    if DEVICE_MESH is None:
        raise ValueError("device mesh is not set.")
    return DEVICE_MESH.get_group("cp")


def get_data_parallel_ranks() -> List[int]:
    dp_group = get_data_parallel_group()
    return dist.get_process_group_ranks(dp_group)


def get_context_parallel_ranks() -> List[int]:
    cp_group = get_context_parallel_group()
    return dist.get_process_group_ranks(cp_group)


def get_data_parallel_src_rank() -> int:
    return get_data_parallel_ranks()[0]


def get_context_parallel_src_rank() -> int:
    return get_context_parallel_ranks()[0]


def get_context_parallel_group_rank(global_rank: int) -> int:
    cp_group = get_context_parallel_group()
    return dist.get_group_rank(cp_group, global_rank)


def shift_context_parallel_rank(global_rank: int, offset: int) -> int:
    cp_group = get_context_parallel_group()
    group_rank = dist.get_group_rank(cp_group, global_rank)
    group_rank = (group_rank + offset) % cp_group.size()  # mod to respect ring topology
    return dist.get_global_rank(cp_group, group_rank)


def get_context_parallel_prev_rank(global_rank: int) -> int:
    return shift_context_parallel_rank(global_rank, -1)


def get_context_parallel_next_rank(global_rank: int) -> int:
    return shift_context_parallel_rank(global_rank, +1)


DEFAULT_MASTER_ADDR = "localhost"
DEFAULT_MASTER_PORT = "29500"


def clean_up_distributed():
    """Clean up dist and torch cuda cache."""
    if dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()


@contextmanager
def parallel_context(
    rank: int = 0,
    world_size: int = 1,
    context_parallel_size: int = 1,
):
    """Context manager for torch distributed testing."""
    with MonkeyPatch.context() as context:
        clean_up_distributed()

        # distributed and parallel state set up
        if not os.environ.get("MASTER_ADDR", None):
            context.setenv("MASTER_ADDR", DEFAULT_MASTER_ADDR)
        if not os.environ.get("MASTER_PORT", None):
            context.setenv("MASTER_PORT", DEFAULT_MASTER_PORT)
        context.setenv("RANK", str(rank))

        dist.init_process_group(backend="nccl", world_size=world_size)
        initialize_parallel_states(context_parallel_size=context_parallel_size)

        yield

        clean_up_parallel_states()
        clean_up_distributed()
