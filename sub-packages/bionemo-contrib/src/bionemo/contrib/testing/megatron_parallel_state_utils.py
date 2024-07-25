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
from typing import Optional

import megatron.core.num_microbatches_calculator
import pytorch_lightning as pl
import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel import random as tp_random
from nemo.utils import logging


"""
This package contains utilities for managing the state of distributed model parallelism in Megatron and Apex.

In general you should just use the context manager `distributed_model_parallel_state` to manage the state of
your test. This context manager will handle the setup and teardown of the distributed model parallel state for you.

Example usage:
```python

from bionemo.contrib.testing import megatron_parallel_state_utils

def my_test():
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        # your test code that requires megatron/apex parallel state to be set up here
```

"""


def _reset_microbatch_calculator():
    """Resets _GLOBAL_NUM_MICROBATCHES_CALCULATOR in megatron which is used in NeMo to initilised model parallel in
    nemo.collections.nlp.modules.common.megatron.megatron_init.initialize_model_parallel_for_nemo
    """
    megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None


def _dummy() -> None:
    return


def _teardown_apex_megatron_cuda():
    """Cleans GPU allocation and model and data parallel settings after usage of a model:
    - sets the global variables related to model and data parallelism to None in Apex and Megatron:.
    - releases all unoccupied cached GPU memory currently held by the caching CUDA allocator, see torch.cuda.empty_cache
    """
    torch.cuda.empty_cache()
    _reset_microbatch_calculator()
    parallel_state.destroy_model_parallel()


def _initialize_distributed_parallel_state(
    local_rank: int = 0,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_model_parallel_split_rank: int = 0,
    interactive: bool = False,
) -> None:
    # initialize pytorch DDP
    # if not interactive and not torch.distributed.is_initialized():
    if not torch.distributed.is_initialized():
        logging.info("pytorch DDP is not initialized. Initializing with pytorch-lightening...")
        trainer = pl.Trainer(devices=1, strategy="ddp" if not interactive else "auto", num_nodes=1)

        if trainer.strategy.launcher is not None:
            trainer.strategy.launcher.launch(_dummy, trainer=trainer)
        trainer.strategy.setup_environment()

    if not interactive and parallel_state.is_unitialized():
        logging.info("Megatron DDP is not initialized. Initializing...")
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
        )


@contextmanager
def clean_parallel_state_context():
    """Puts you into a clean parallel state, and again tears it down at the end"""
    try:
        _teardown_apex_megatron_cuda()
        yield
    finally:
        _teardown_apex_megatron_cuda()


@contextmanager
def distributed_model_parallel_state(seed: Optional[int] = 42):
    """Context manager for handling creating and cleaning up distributed model parallel state for tests.
    Use like:
    with distributed_model_parallel_state():
        # your test code here
    # After the block your state is cleaned up.
    """
    try:
        _teardown_apex_megatron_cuda()
        _initialize_distributed_parallel_state()
        # Our goal is to set required state on entry, and then restore current state on exit for the RNGs.
        #  there are two possibilities that are handled below:
        # 1. If the RNG state is not initialized, we need to set it up and then
        #     unset it on exit to restore the current state. We track that this is the case when `initial_states` is `None`.
        # 2. If the RNG state is initialized, we need to track this state and reset it on exit to be what it was on entry.
        #    We track that this is the case when `initial_states` is not `None`.
        if tp_random.get_cuda_rng_tracker().is_initialized():
            initial_states = tp_random.get_cuda_rng_tracker().get_states()
        else:
            initial_states = None
        if seed is not None:
            # Set the seed if provided, this case is valid whether or not the RNG had state previously.
            #  on exit the RNG state will be restored to what it was on entry.
            tp_random.model_parallel_cuda_manual_seed(seed)
        else:
            # This is the case where the RNG state is not initialized and no seed was provided.
            #  We need to raise an error in this case, as we cannot restore the RNG state on exit and we need a seed
            #  to initialize the RNG state to. This only happens if the user overrides the default seed and sets it
            #  to None, and additionally if the RNG state was not initialized externally, as there is a default seed of 42.
            if initial_states is None:
                raise ValueError(
                    "You must provide a seed if the initial parallel state is unset. "
                    "Either provide a seed or leave the default seed (rather setting to None) "
                    "or initialize the RNG state externally."
                )
        yield
    finally:
        if initial_states is not None:
            tp_random.get_cuda_rng_tracker().set_states(initial_states)
        else:
            # Reset to the unset state
            tp_random.get_cuda_rng_tracker().reset()
        _teardown_apex_megatron_cuda()


__all__ = [
    "clean_parallel_state_context",
    "distributed_model_parallel_state",
]
