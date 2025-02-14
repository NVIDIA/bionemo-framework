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
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing.spawn
from pytest import MonkeyPatch
from torch.distributed import ProcessGroup


DEFAULT_MASTER_ADDR = "localhost"
DEFAULT_MASTER_PORT = "29500"


@contextmanager
def parallel_context(
    rank: int = 0,
    world_size: int = 1,
):
    """Context manager for torch distributed testing.

    Sets up and cleans up the distributed environment, including the device mesh.

    Args:
        rank (int): The rank of the process. Defaults to 0.
        world_size (int): The world size of the distributed environment. Defaults to 1.

    Yields:
        None
    """
    with MonkeyPatch.context() as context:
        clean_up_distributed()

        # distributed and parallel state set up
        if not os.environ.get("MASTER_ADDR", None):
            context.setenv("MASTER_ADDR", DEFAULT_MASTER_ADDR)
        if not os.environ.get("MASTER_PORT", None):
            context.setenv("MASTER_PORT", DEFAULT_MASTER_PORT)
        context.setenv("RANK", str(rank))

        dist.init_process_group(backend="nccl", world_size=world_size)

        yield

        clean_up_parallel_states()
        clean_up_distributed()


def clean_up_parallel_states() -> None:
    """Cleans up the parallel states after distributed testing.

    Resets the global DEVICE_MESH variable to None.
    """
    global DEVICE_MESH
    DEVICE_MESH = None


def get_data_parallel_group() -> ProcessGroup:
    """Retrieves the data parallel process group.

    Args:
        None

    Returns:
        ProcessGroup: The data parallel process group.

    Raises:
        ValueError: If the device mesh is not set.
    """
    global DEVICE_MESH
    if DEVICE_MESH is None:
        raise ValueError("device mesh is not set.")
    return DEVICE_MESH.get_group("dp")


def get_data_parallel_ranks() -> List[int]:
    """Retrieves the ranks of the data parallel group.

    Args:
        None

    Returns:
        List[int]: A list of global ranks in the data parallel group.
    """
    dp_group = get_data_parallel_group()
    return dist.get_process_group_ranks(dp_group)


def get_data_parallel_src_rank() -> int:
    """Retrieves the source rank of the data parallel group.

    Args:
        None

    Returns:
        int: The global rank of the first process in the data parallel group.
    """
    return get_data_parallel_ranks()[0]


def clean_up_distributed() -> None:
    """Cleans up the distributed environment.

    Destroys the process group and empties the CUDA cache.

    Args:
        None

    Returns:
        None
    """
    if dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()
