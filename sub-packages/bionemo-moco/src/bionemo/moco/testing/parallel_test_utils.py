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
import socket
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.multiprocessing.spawn
from pytest import MonkeyPatch


DEFAULT_MASTER_ADDR = "localhost"
DEFAULT_MASTER_PORT = "29500"


def find_free_network_port(address: str = "localhost") -> int:
    """Finds a free port for the specified address. Defaults to localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((address, 0))
    addr_port = s.getsockname()
    s.close()
    if addr_port is None:
        # Could not find any free port.
        return None, None
    return addr_port


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
            network_address, free_network_port = find_free_network_port(address=DEFAULT_MASTER_ADDR)
            context.setenv("MASTER_PORT", free_network_port if free_network_port is not None else DEFAULT_MASTER_PORT)
        context.setenv("RANK", str(rank))

        dist.init_process_group(backend="nccl", world_size=world_size)

        yield

        clean_up_distributed()


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
