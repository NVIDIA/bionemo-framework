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

import pytest
from nemo import lightning as nl

from bionemo.llm import lightning as bnptl
from bionemo.testing import lightning_basic as lb
from bionemo.testing import megatron_parallel_state_utils


def test_mixin_strategy_contract_get_loss_reduction():
    with megatron_parallel_state_utils.clean_parallel_state_context():
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            ddp="megatron",
            find_unused_parameters=True,
            enable_nemo_ckpt_io=False,
        )
        strategy.connect(bnptl.LightningPassthroughPredictionMixin())
        mixin = bnptl.LightningPassthroughPredictionMixin()
        strategy_reduction_function = strategy._get_loss_reduction("predict")
        assert isinstance(strategy_reduction_function(mixin), bnptl.PassthroughLossReduction)


@pytest.mark.needs_gpu
def test_train_mnist_litautoencoder_with_megatron_strategy_single_gpu():
    with megatron_parallel_state_utils.clean_parallel_state_context():
        model = lb.LitAutoEncoder(config=lb.ExampleConfig())
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            ddp="megatron",
            find_unused_parameters=True,
            enable_nemo_ckpt_io=False,
        )
        trainer = nl.Trainer(accelerator="gpu", devices=1, strategy=strategy, max_steps=10, num_nodes=1)
        data_module = lb.MNISTDataModule()
        trainer.fit(model, data_module)
