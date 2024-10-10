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


"""
This file tests the lora-related utilities for ESM2 finetuning.
"""

import copy

import pytest
import torch
from nemo.collections.nlp.parts.peft_config import LoraPEFTConfig
from omegaconf.omegaconf import open_dict

from bionemo.model.protein.downstream import FineTuneProteinModel
from bionemo.model.protein.esm1nv.esm1nv_model import ESM2nvLoRAModel
from bionemo.model.utils import setup_trainer
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import Deterministic, distributed_model_parallel_state, teardown_apex_megatron_cuda


@pytest.fixture(scope="function")
def cfg(config_path_for_tests):
    cfg = load_model_config(config_name="esm2nv_lora_finetune_test", config_path=config_path_for_tests)
    return cfg


@pytest.fixture(scope="function")
def model(cfg) -> FineTuneProteinModel:
    with Deterministic(), distributed_model_parallel_state():
        with open_dict(cfg):
            cfg.model.encoder_cfg = cfg
        trainer = setup_trainer(cfg)
        model = FineTuneProteinModel(cfg.model, trainer)
        yield model
        teardown_apex_megatron_cuda()


@pytest.mark.needs_80gb_memory_gpu
def test_lora_layers_added(cfg, model):
    contains_lora = any("lora_kqv_adapter" in name for name, _ in model.named_modules())
    assert contains_lora is True


@pytest.mark.needs_80gb_memory_gpu
def test_lora_layers_trainable(cfg, model):
    for param, _ in model.named_parameters():
        if "lora_kqv_adapter" in param:
            assert _.requires_grad is True


@pytest.mark.needs_80gb_memory_gpu
def test_finetuneproteinmodel_last_lora_adapter_layer_linear_in_trained(cfg):
    cfg.trainer.max_steps = 5
    with Deterministic(), distributed_model_parallel_state():
        with open_dict(cfg):
            cfg.model.encoder_cfg = cfg
        trainer = setup_trainer(cfg)

        model = FineTuneProteinModel(cfg.model, trainer)
        lora_before = copy.deepcopy(model.encoder_model.model.get_peft_state_dict())
        trainer.fit(model)
        lora_after = copy.deepcopy(model.encoder_model.model.get_peft_state_dict())

        lora_before_weights = lora_before[
            "model.language_model.encoder.layers.32.self_attention.adapter_layer.lora_kqv_adapter.linear_in.weight"
        ]
        lora_after_weights = lora_after[
            "model.language_model.encoder.layers.32.self_attention.adapter_layer.lora_kqv_adapter.linear_in.weight"
        ]

        diff = torch.max(torch.abs(lora_before_weights.cuda() - lora_after_weights.cuda())).detach().cpu().numpy()
        assert diff > 0.0


@pytest.mark.needs_80gb_memory_gpu
def test_finetuneproteinmodel_last_lora_adapter_layer_linear_out_trained(cfg):
    cfg.trainer.max_steps = 5
    with Deterministic(), distributed_model_parallel_state():
        with open_dict(cfg):
            cfg.model.encoder_cfg = cfg
        trainer = setup_trainer(cfg)

        model = FineTuneProteinModel(cfg.model, trainer)
        lora_before = copy.deepcopy(model.encoder_model.model.get_peft_state_dict())
        trainer.fit(model)
        lora_after = copy.deepcopy(model.encoder_model.model.get_peft_state_dict())

        lora_before_weights = lora_before[
            "model.language_model.encoder.layers.32.self_attention.adapter_layer.lora_kqv_adapter.linear_out.weight"
        ]
        lora_after_weights = lora_after[
            "model.language_model.encoder.layers.32.self_attention.adapter_layer.lora_kqv_adapter.linear_out.weight"
        ]

        diff = torch.max(torch.abs(lora_before_weights.cuda() - lora_after_weights.cuda())).detach().cpu().numpy()
        assert diff > 0.0


@pytest.mark.needs_80gb_memory_gpu
def test_finetuneproteinmodel_first_lora_adapter_layer_linear_out_trained(cfg):
    cfg.trainer.max_steps = 5
    with Deterministic(), distributed_model_parallel_state():
        with open_dict(cfg):
            cfg.model.encoder_cfg = cfg
        trainer = setup_trainer(cfg)

        model = FineTuneProteinModel(cfg.model, trainer)
        lora_before = copy.deepcopy(model.encoder_model.model.get_peft_state_dict())
        trainer.fit(model)
        lora_after = copy.deepcopy(model.encoder_model.model.get_peft_state_dict())

        lora_before_weights = lora_before[
            "model.language_model.encoder.layers.0.self_attention.adapter_layer.lora_kqv_adapter.linear_out.weight"
        ]
        lora_after_weights = lora_after[
            "model.language_model.encoder.layers.0.self_attention.adapter_layer.lora_kqv_adapter.linear_out.weight"
        ]

        diff = torch.max(torch.abs(lora_before_weights.cuda() - lora_after_weights.cuda())).detach().cpu().numpy()
        assert diff > 0.0


@pytest.mark.needs_80gb_memory_gpu
def test_finetuneproteinmodel_first_lora_adapter_layer_linear_in_trained(cfg):
    cfg.trainer.max_steps = 5
    with Deterministic(), distributed_model_parallel_state():
        with open_dict(cfg):
            cfg.model.encoder_cfg = cfg
        trainer = setup_trainer(cfg)

        model = FineTuneProteinModel(cfg.model, trainer)
        lora_before = copy.deepcopy(model.encoder_model.model.get_peft_state_dict())
        trainer.fit(model)
        lora_after = copy.deepcopy(model.encoder_model.model.get_peft_state_dict())

        lora_before_weights = lora_before[
            "model.language_model.encoder.layers.0.self_attention.adapter_layer.lora_kqv_adapter.linear_in.weight"
        ]
        lora_after_weights = lora_after[
            "model.language_model.encoder.layers.0.self_attention.adapter_layer.lora_kqv_adapter.linear_in.weight"
        ]

        diff = torch.max(torch.abs(lora_before_weights.cuda() - lora_after_weights.cuda())).detach().cpu().numpy()
        assert diff > 0.0


@pytest.fixture(scope="function")
def model_lora(cfg) -> ESM2nvLoRAModel:
    with Deterministic(), distributed_model_parallel_state():
        with open_dict(cfg):
            cfg.model.encoder_cfg = cfg
        trainer = setup_trainer(cfg)
        model = ESM2nvLoRAModel(cfg.model, trainer)
        yield model


def test_nlpadaptermodelmixin_add_adapters(cfg):
    with Deterministic(), distributed_model_parallel_state():
        with open_dict(cfg):
            cfg.model.encoder_cfg = cfg
        trainer = setup_trainer(cfg)
        model_lora = ESM2nvLoRAModel(cfg.model, trainer)

        peft_cfg = LoraPEFTConfig(cfg.model)
        model_lora.add_adapter(peft_cfg)
        contains_lora = any("lora_kqv_adapter" in name for name, _ in model_lora.named_modules())
        assert contains_lora is True


def test_nlpadaptermodelmixin_state_dict_func_with_peft_true(cfg):
    with Deterministic(), distributed_model_parallel_state():
        with open_dict(cfg):
            cfg.model.encoder_cfg = cfg
        trainer = setup_trainer(cfg)
        model_lora = ESM2nvLoRAModel(cfg.model, trainer)

        peft_cfg = LoraPEFTConfig(cfg.model)
        model_lora.add_adapter(peft_cfg)

        peft_layers1 = model_lora.get_peft_state_dict().keys()  # only retrieves peft layers
        peft_layers2 = model_lora.state_dict().keys()  # should only retrieve peft layers if flags are set correctly

        assert peft_layers1 == peft_layers2


def test_nlpadaptermodelmixin_state_dict_func_with_peft_false(cfg):
    with Deterministic(), distributed_model_parallel_state():
        with open_dict(cfg):
            cfg.model.encoder_cfg = cfg
        trainer = setup_trainer(cfg)
        model_lora = ESM2nvLoRAModel(cfg.model, trainer)

        peft_cfg = LoraPEFTConfig(cfg.model)
        model_lora.add_adapter(peft_cfg)

        peft_layers1 = model_lora.get_peft_state_dict().keys()  # only retrieves peft layers

        model_lora.use_peft = False
        all_keys = model_lora.state_dict().keys()  # should return all layers of model

        assert peft_layers1 != all_keys
