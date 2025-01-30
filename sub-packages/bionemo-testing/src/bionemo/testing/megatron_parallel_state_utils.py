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

"""This package contains utilities for managing the state of distributed model parallelism in Megatron and Apex.

In general you should just use the context manager `distributed_model_parallel_state` to manage the state of
your test. This context manager will handle the setup and teardown of the distributed model parallel state for you.

Example usage:
```python

from bionemo.testing import megatron_parallel_state_utils

def my_test():
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        # your test code that requires megatron/apex parallel state to be set up here
```

"""

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence
from unittest import mock
from unittest.mock import MagicMock

import lightning.pytorch as pl
import megatron.core.num_microbatches_calculator
import torch
import torch.distributed
from megatron.core import parallel_state
from megatron.core.tensor_parallel import random as tp_random
from nemo import lightning as nl
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategyNotebook
from nemo.utils import logging
from torch.testing._internal.distributed.fake_pg import FakeStore

from bionemo.core.utils.dtypes import PrecisionTypes


__all__: Sequence[str] = (
    "clean_parallel_state_context",
    "distributed_model_parallel_state",
    "mock_distributed_parallel_state",
)


def _reset_microbatch_calculator():
    """Resets _GLOBAL_NUM_MICROBATCHES_CALCULATOR in megatron which is used in NeMo to initilised model parallel in
    nemo.collections.nlp.modules.common.megatron.megatron_init.initialize_model_parallel_for_nemo
    """  # noqa: D205, D415
    megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None


def _dummy() -> None:
    return


def _teardown_apex_megatron_cuda():
    """Cleans GPU allocation and model and data parallel settings after usage of a model:
    - sets the global variables related to model and data parallelism to None in Apex and Megatron:.
    - releases all unoccupied cached GPU memory currently held by the caching CUDA allocator, see torch.cuda.empty_cache
    """  # noqa: D205, D415
    torch.cuda.empty_cache()
    _reset_microbatch_calculator()
    parallel_state.destroy_model_parallel()


def _initialize_distributed_parallel_state(
    devices: int = 1,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_model_parallel_split_rank: int = 0,
    context_parallel_size: int = 1,
    interactive: bool = False,
    precision: PrecisionTypes = "fp32",
) -> pl.Trainer | None:
    trainer = None
    # initialize pytorch DDP
    # if not interactive and not torch.distributed.is_initialized():
    if not torch.distributed.is_initialized():
        logging.info("pytorch DDP is not initialized. Initializing with pytorch-lightning...")
        trainer = pl.Trainer(
            devices=devices,
            strategy="ddp" if not interactive else NLPDDPStrategyNotebook(),
            num_nodes=1,
            # plugins=nl.MegatronMixedPrecision(
            #     precision=precision,
            #     params_dtype=get_autocast_dtype(precision),
            #     pipeline_dtype=get_autocast_dtype(precision),
            #     autocast_enabled=False,
            # ),
        )

        if trainer.strategy.launcher is not None:
            trainer.strategy.launcher.launch(_dummy, trainer=trainer)
        trainer.strategy.setup_environment()

    if parallel_state.is_unitialized():
        logging.info("Megatron DDP is not initialized. Initializing...")
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
            context_parallel_size=context_parallel_size,
        )

    return trainer


@contextmanager
def clean_parallel_state_context() -> Iterator[None]:
    """Puts you into a clean parallel state, and again tears it down at the end."""
    try:
        _teardown_apex_megatron_cuda()
        yield
    except Exception as e:
        # TODO (@skothenhill) verify this is a problem and that this is a solution. Had issues with keyboard interrupts being ignored inside context manager.
        raise Exception from e
    finally:
        _teardown_apex_megatron_cuda()


@contextmanager
def distributed_model_parallel_state(
    seed: Optional[int] = 42,
    devices: int = 1,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_model_parallel_split_rank: int = 0,
    context_parallel_size: int = 1,
    interactive: bool = False,
    precision: PrecisionTypes = "fp32",
) -> Iterator[None]:
    """Context manager for handling creating and cleaning up distributed model parallel state for tests.
    Use like:
    with distributed_model_parallel_state():
        # your test code here
    # After the block your state is cleaned up.
    """  # noqa: D205
    initial_states: Optional[Any] = None
    trainer: pl.Trainer | None = None

    try:
        _teardown_apex_megatron_cuda()
        trainer = _initialize_distributed_parallel_state(
            devices=devices,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
            context_parallel_size=context_parallel_size,
            interactive=interactive,
            precision=precision,
        )
        # Our goal is to set required state on entry, and then restore current state on exit for the RNGs.
        #  there are two possibilities that are handled below:
        # 1. If the RNG state is not initialized, we need to set it up and then
        #     unset it on exit to restore the current state. We track that this is the case when `initial_states` is `None`.
        # 2. If the RNG state is initialized, we need to track this state and reset it on exit to be what it was on entry.
        #    We track that this is the case when `initial_states` is not `None`.
        if tp_random.get_cuda_rng_tracker().is_initialized():
            initial_states = tp_random.get_cuda_rng_tracker().get_states()
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
        if trainer is not None:
            nl.teardown(trainer)


@contextmanager
def mock_distributed_parallel_state(
    world_size: int = 8,
    rank: int = 0,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    seed: int | None = 42,
):
    """A context manager that facilitates easy mocking of torch.distributed for an arbitrary GPU in a simulated cluster.

    Key functions that are mocked:
        * `torch.distributed.new_group` when `backend="gloo"` which doesn't support a `backend="fake"`
        * `torch.distributed.destroy_process_group` when `backend="gloo"` since new "gloo" groups are not actually made
        * `torch._C._cuda_setDevice` which changes the current device behind the scenes. We assign devices round-robin
            to support `world_size > torch.cuda.device_count()`.

    Outside of this mocking, a fake cluster is initialized using `backend="fake"` in `torch.distributed`. This sets up
        enough global state and environment for megatron to think that it is initializing a larger cluster with some
        settings where the current context has some user defined rank. You can then test the megatron state on a
        hypothetical rank in some large world size.

    Args:
        world_size: The world size (cluster size). Defaults to 8.
        rank: the GPU number globally in the cluster. Defaults to 0.
        tensor_model_parallel_size: tensor model parallel setting for megatron. Defaults to 1.
        pipeline_model_parallel_size: pipeline model parallel setting for megatron. Defaults to 1.
        virtual_pipeline_model_parallel_size: virtual pipeline model parallel size for megatron. Defaults to None.
        context_parallel_size: context parallel size. Defaults to 1.
        expert_model_parallel_size: expert model parallel size. Defaults to 1.
        seed: seed for RNG state. Defaults to 42.
    """
    # First set up mocks for torch.distributed state/info
    ori_device_count = torch.cuda.device_count()
    # Conditionally mock torch.distributed.new_group based on backend argument
    ori_dist_new_group = torch.distributed.new_group

    def mock_new_group(*args, **kwargs):
        if kwargs.get("backend") == "gloo":
            # Return a specific mock if backend is 'gloo'
            return MagicMock(name="gloo_group")
        else:
            # Return another mock or a different behavior for other backends
            return ori_dist_new_group(*args, **kwargs)

    ori_destroy_pg = torch.distributed.destroy_process_group

    def mock_destroy_gloo_group(pg=None):
        if isinstance(pg, MagicMock):
            return None
        ori_destroy_pg(pg)

    # The next mock is required to "set the device" to one that is greater than the number of actual GPUs
    #  the consequence of this mock is that the device is always dev 0
    ori_set_device = torch._C._cuda_setDevice

    def mock_set_device(device):
        if ori_device_count > 0:
            ori_set_device(device % ori_device_count)  # wrap around the request

    with (
        mock.patch("torch.distributed.new_group", side_effect=mock_new_group),
        mock.patch("torch.distributed.destroy_process_group", side_effect=mock_destroy_gloo_group),
        mock.patch("torch._C._cuda_setDevice", side_effect=mock_set_device),
    ):
        # Next set up state etc
        state_util = _MockMegatronParallelStateSingleton()  # static singleton class
        state_util.world_size = world_size
        state_util.rank = rank
        initial_states: Optional[Any] = None
        try:
            state_util.set_world_size(world_size=world_size, rank=rank)
            state_util.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
                context_parallel_size=context_parallel_size,
                expert_model_parallel_size=expert_model_parallel_size,
            )
            # Our goal is to set required state on entry, and then restore current state on exit for the RNGs.
            #  there are two possibilities that are handled below:
            # 1. If the RNG state is not initialized, we need to set it up and then
            #     unset it on exit to restore the current state. We track that this is the case when `initial_states` is `None`.
            # 2. If the RNG state is initialized, we need to track this state and reset it on exit to be what it was on entry.
            #    We track that this is the case when `initial_states` is not `None`.
            if tp_random.get_cuda_rng_tracker().is_initialized():
                initial_states = tp_random.get_cuda_rng_tracker().get_states()
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
            state_util.destroy_model_parallel()


class _MockMegatronParallelStateSingleton:
    _instance = None

    def __init__(
        self,
        world_size=torch.cuda.device_count(),
        rank=int(os.getenv("LOCAL_RANK", 0)),
        inited=False,
        store=FakeStore(),
    ):
        """A singleton to deal with global megatron state for simulating a fake cluster.

        Args:
            world_size: the cluster size. Defaults to torch.cuda.device_count().
            rank: rank of this node. Defaults to int(os.getenv("LOCAL_RANK", 0)).
            inited: if this global cluster has been initiated. Defaults to False.
            store: the FakeStore for process groups. Defaults to FakeStore().
        """
        self.world_size = world_size
        self.rank = rank
        self.inited = inited
        # Fake store idea: see https://github.com/pytorch/pytorch/blob/main/test/distributed/test_fake_pg.py
        self.store = store

    def __new__(cls):
        # Makes this a singleton
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize_distributed(self):
        torch.cuda.set_device(self.rank % self.world_size)
        # Fake store idea: see https://github.com/pytorch/pytorch/blob/main/test/distributed/test_fake_pg.py
        torch.distributed.init_process_group(
            backend="fake",
            world_size=self.world_size,
            rank=self.rank,
            store=self.store,
        )
        self.inited = True

    def set_world_size(self, world_size=None, rank=None):
        self.world_size = torch.cuda.device_count() if world_size is None else world_size
        if torch.distributed.is_initialized() and self.world_size != torch.distributed.get_world_size():
            torch.distributed.destroy_process_group()

        if rank is None:
            self.rank = int(os.environ.get("LOCAL_RANK", 0))
            if self.rank >= self.world_size:
                self.rank = -1
        else:
            self.rank = rank

    def destroy_model_parallel(self):
        if not self.inited:
            return
        # torch.distributed.barrier()
        parallel_state.destroy_model_parallel()
        self.inited = False
        torch.distributed.destroy_process_group()

    def initialize_model_parallel(
        self,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        **kwargs,
    ):
        parallel_state.destroy_model_parallel()
        self.initialize_distributed()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            **kwargs,
        )
        self.inited = True
