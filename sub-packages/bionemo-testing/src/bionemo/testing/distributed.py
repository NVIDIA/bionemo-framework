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

from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.multiprocessing.spawn
from megatron.core import parallel_state
from pytest import MonkeyPatch


MASTER_ADDR = "localhost"
MASTER_PORT = "29500"
NCCL_TIMEOUT = "30"  # in second


@contextmanager
def dist_environment(
    rank: int,
    world_size: int = -1,
    **initialize_model_parallel_kwargs,
):
    """Context manager for torch distributed testing."""
    with MonkeyPatch.context() as context:
        # clean up
        torch.cuda.empty_cache()
        parallel_state.destroy_model_parallel()

        # init
        context.setenv("MASTER_ADDR", MASTER_ADDR)
        context.setenv("MASTER_PORT", MASTER_PORT)
        context.setenv("NCCL_TIMEOUT", NCCL_TIMEOUT)
        context.setenv("RANK", str(rank))
        dist.init_process_group(backend="nccl", world_size=world_size)
        parallel_state.initialize_model_parallel(**initialize_model_parallel_kwargs)

        yield

        # clean up
        torch.cuda.empty_cache()
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
