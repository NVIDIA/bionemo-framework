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

import torch
import torch.distributed as dist
import torch.multiprocessing.spawn
from megatron.core import parallel_state
from pytest import MonkeyPatch


DEFAULT_MASTER_ADDR = "localhost"
DEFAULT_MASTER_PORT = "29500"
DEFAULT_NCCL_TIMEOUT = "30"  # in second


def clean_up_states():
    """Clean up parallel states, torch.distributed and torch cuda cache."""
    parallel_state.destroy_model_parallel()  # destroy parallel state before distributed
    if dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()


@contextmanager
def dist_environment(
    rank: int,
    world_size: int = -1,
    **initialize_model_parallel_kwargs,
):
    """Context manager for torch distributed testing."""
    with MonkeyPatch.context() as context:
        clean_up_states()

        if not os.environ.get("MASTER_ADDR", None):
            context.setenv("MASTER_ADDR", DEFAULT_MASTER_ADDR)
        if not os.environ.get("MASTER_PORT", None):
            context.setenv("MASTER_PORT", DEFAULT_MASTER_PORT)
        if not os.environ.get("NCCL_TIMEOUT", None):
            context.setenv("NCCL_TIMEOUT", DEFAULT_NCCL_TIMEOUT)
        context.setenv("RANK", str(rank))

        dist.init_process_group(backend="nccl", world_size=world_size)
        parallel_state.initialize_model_parallel(**initialize_model_parallel_kwargs)

        yield

        clean_up_states()
