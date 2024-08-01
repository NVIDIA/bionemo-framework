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
