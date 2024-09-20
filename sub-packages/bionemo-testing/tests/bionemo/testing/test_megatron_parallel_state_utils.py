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


import torch
import torch.distributed as dist
from megatron.core import parallel_state
from nemo import lightning as nl

from bionemo.testing import megatron_parallel_state_utils


def test_load_megatron_strategy():
    # This will clean up most of the megatron global state that can get created
    with megatron_parallel_state_utils.distributed_model_parallel_state(43):
        strategy = nl.MegatronStrategy(tensor_model_parallel_size=1)
        assert strategy.tensor_model_parallel_size == 1


def test_construct_nemo_lightning_trainer():
    # This will clean up most of the megatron global state that can get created
    with megatron_parallel_state_utils.distributed_model_parallel_state(43):
        trainer = nl.Trainer(
            devices=1,
            max_steps=5,
            accelerator="gpu",
            strategy=nl.MegatronStrategy(tensor_model_parallel_size=1),
        )
        assert trainer.max_steps == 5


def test_rank0_first_pipeline():
    with megatron_parallel_state_utils.mock_distributed_parallel_state(
        world_size=8, rank=0, pipeline_model_parallel_size=8
    ):
        assert parallel_state.is_pipeline_first_stage()
        assert not parallel_state.is_pipeline_last_stage()


def test_rank4_mid_pipeline():
    with megatron_parallel_state_utils.mock_distributed_parallel_state(
        world_size=8, rank=4, pipeline_model_parallel_size=8
    ):
        assert not parallel_state.is_pipeline_first_stage()
        assert not parallel_state.is_pipeline_last_stage()


def test_rank7_last_pipeline():
    with megatron_parallel_state_utils.mock_distributed_parallel_state(
        world_size=8, rank=7, pipeline_model_parallel_size=8
    ):
        assert not parallel_state.is_pipeline_first_stage()
        assert parallel_state.is_pipeline_last_stage()


def test_get_pp_group():
    with megatron_parallel_state_utils.mock_distributed_parallel_state(world_size=2, pipeline_model_parallel_size=2):
        assert parallel_state.get_pipeline_model_parallel_group() is not None


def test_get_tp_group():
    with megatron_parallel_state_utils.mock_distributed_parallel_state(world_size=2, tensor_model_parallel_size=2):
        assert parallel_state.get_tensor_model_parallel_group() is not None


def test_get_cp_group():
    with megatron_parallel_state_utils.mock_distributed_parallel_state(world_size=2, context_parallel_size=2):
        assert parallel_state.get_context_parallel_group() is not None


def test_all_reduce():
    # Adapted from https://github.com/pytorch/pytorch/blob/main/test/distributed/test_fake_pg.py
    with megatron_parallel_state_utils.mock_distributed_parallel_state(world_size=2, rank=1):
        output = torch.ones(3, 3).cuda() * dist.get_rank()
        dist.all_reduce(output)
        assert tuple(output.shape) == (3, 3)


def test_allgather():
    # Adapted from https://github.com/pytorch/pytorch/blob/main/test/distributed/test_fake_pg.py
    with megatron_parallel_state_utils.mock_distributed_parallel_state(world_size=2, rank=1):
        input_tensor = torch.ones(3, 3) * dist.get_rank()
        output_tensors = [torch.empty_like(input_tensor) for _ in range(2)]
        dist.all_gather(output_tensors, input_tensor)
        for _, out_tensor in enumerate(output_tensors):
            assert tuple(out_tensor.shape) == (3, 3)


def test_reduce_scatter():
    # Adapted from https://github.com/pytorch/pytorch/blob/main/test/distributed/test_fake_pg.py
    with megatron_parallel_state_utils.mock_distributed_parallel_state(world_size=2, rank=1):
        to_reduce_scatter = [torch.ones(3, 3) * rank for rank in range(2)]
        output_tensor = torch.empty(3, 3)

        dist.reduce_scatter(output_tensor, to_reduce_scatter)
        assert tuple(output_tensor.shape) == (3, 3)
