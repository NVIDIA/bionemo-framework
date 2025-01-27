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
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing.spawn
from megatron.core import parallel_state
from megatron.core.tensor_parallel import random as tp_random
from pytest import MonkeyPatch


DEFAULT_MASTER_ADDR = "localhost"
DEFAULT_MASTER_PORT = "29500"
DEFAULT_NCCL_TIMEOUT = "30"  # in second


def clean_up_distributed_and_parallel_states():
    """Clean up parallel states, torch.distributed and torch cuda cache."""
    parallel_state.destroy_model_parallel()  # destroy parallel state before distributed
    if dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()


@contextmanager
def dist_environment(
    rank: int = 0,
    world_size: int = 1,
    seed: Optional[int] = 42,
    **initialize_model_parallel_kwargs,
):
    """Context manager for torch distributed testing."""
    with MonkeyPatch.context() as context:
        clean_up_distributed_and_parallel_states()

        # distributed and parallel state set up
        if not os.environ.get("MASTER_ADDR", None):
            context.setenv("MASTER_ADDR", DEFAULT_MASTER_ADDR)
        if not os.environ.get("MASTER_PORT", None):
            context.setenv("MASTER_PORT", DEFAULT_MASTER_PORT)
        if not os.environ.get("NCCL_TIMEOUT", None):
            context.setenv("NCCL_TIMEOUT", DEFAULT_NCCL_TIMEOUT)
        context.setenv("RANK", str(rank))

        dist.init_process_group(backend="nccl", world_size=world_size)
        parallel_state.initialize_model_parallel(**initialize_model_parallel_kwargs)

        # tensor parallel random seed set up
        # do not call torch.cuda.manual_seed after so!
        initial_states = None
        if tp_random.get_cuda_rng_tracker().is_initialized():
            initial_states = tp_random.get_cuda_rng_tracker().get_states()
        if seed is not None:
            tp_random.model_parallel_cuda_manual_seed(seed)

        yield

        # restore/unset tensor parallel random seed
        if initial_states is not None:
            tp_random.get_cuda_rng_tracker().set_states(initial_states)
        else:
            # Reset to the unset state
            tp_random.get_cuda_rng_tracker().reset()

        clean_up_distributed_and_parallel_states()
