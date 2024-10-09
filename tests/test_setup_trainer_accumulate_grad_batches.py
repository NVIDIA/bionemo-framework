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
import pathlib

import pytest
from omegaconf import DictConfig, open_dict

from bionemo.model.protein.downstream import FineTuneProteinModel
from bionemo.model.protein.esm1nv import esm1nv_model
from bionemo.model.utils import (
    setup_trainer,
)
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import (
    distributed_model_parallel_state,
)


os.environ["BIONEMO_HOME"] = os.environ.get("BIONEMO_HOME", "/workspace/bionemo")
BIONEMO_HOME = os.environ["BIONEMO_HOME"]

THIS_FILE_DIR = pathlib.Path(os.path.abspath(__file__))
PROJ_BASE_DIR = THIS_FILE_DIR.parent
CONFIG_PATH = "../examples/tests/conf"
PREPEND_CONFIG_DIR = PROJ_BASE_DIR / "examples" / "protein" / "esm2nv" / "conf"


@pytest.fixture(scope="function")
def esm2_cfg(config_path_for_tests) -> DictConfig:
    return load_model_config(config_name="esm2nv_data_test", config_path=config_path_for_tests)


@pytest.fixture(scope="function")
def esm2_downstream_flip_sec_str_cfg(config_path_for_tests) -> DictConfig:
    return load_model_config(config_name="esm2nv_downstream_flip_sec_str", config_path=config_path_for_tests)


@pytest.mark.needs_fork
def test_setup_trainer_reset_accumulate_grad_batches(esm2_cfg):
    esm2_cfg.model.micro_batch_size = 1
    esm2_cfg.trainer.num_nodes = 1
    esm2_cfg.trainer.devices = 1
    esm2_cfg.trainer.accumulate_grad_batches = 3

    assert "global_batch_size" not in esm2_cfg.model
    _ = setup_trainer(esm2_cfg)  # reset_accumulate_grad_batches should be True by default
    assert esm2_cfg.model.global_batch_size == 3
    assert esm2_cfg.trainer.accumulate_grad_batches == 1


@pytest.mark.needs_fork
def test_setup_trainer_not_reset_accumulate_grad_batches(esm2_cfg):
    esm2_cfg.model.micro_batch_size = 1
    esm2_cfg.trainer.num_nodes = 1
    esm2_cfg.trainer.devices = 1
    esm2_cfg.trainer.accumulate_grad_batches = 3

    assert "global_batch_size" not in esm2_cfg.model
    _ = setup_trainer(
        esm2_cfg, reset_accumulate_grad_batches=False
    )  # reset_accumulate_grad_batches should be True by default
    assert esm2_cfg.model.global_batch_size == 3
    assert esm2_cfg.trainer.accumulate_grad_batches == 3


@pytest.mark.needs_fork
def test_init_esm2_pretraining_without_accumulate_grad_batches(esm2_cfg):
    esm2_cfg.trainer.accumulate_grad_batches = 1
    trainer = setup_trainer(esm2_cfg)

    with distributed_model_parallel_state():
        _ = esm1nv_model.ESM2nvModel(esm2_cfg.model, trainer)  # model should fail at instantiation otherwise


@pytest.mark.needs_fork
def test_init_esm2_pretraining_with_accumulate_grad_batches(esm2_cfg):
    esm2_cfg.trainer.accumulate_grad_batches = 2
    trainer = setup_trainer(esm2_cfg)

    with distributed_model_parallel_state():
        _ = esm1nv_model.ESM2nvModel(esm2_cfg.model, trainer)  # model should fail at instantiation otherwise


@pytest.mark.needs_fork
def test_esm2_finetuning_without_accumulate_grad_batches(esm2_downstream_flip_sec_str_cfg):
    esm2_downstream_flip_sec_str_cfg.trainer.accumulate_grad_batches = 1
    trainer = setup_trainer(
        esm2_downstream_flip_sec_str_cfg, reset_accumulate_grad_batches=False
    )  # retain accumulate_grad_batches value for EncoderFinetuning class

    # reconfigure for encoder finetuning init
    with open_dict(esm2_downstream_flip_sec_str_cfg):
        esm2_downstream_flip_sec_str_cfg.model.encoder_cfg = esm2_downstream_flip_sec_str_cfg

    with distributed_model_parallel_state():
        _ = FineTuneProteinModel(esm2_downstream_flip_sec_str_cfg.model, trainer)


@pytest.mark.needs_fork
def test_esm2_finetuning_with_accumulate_grad_batches(esm2_downstream_flip_sec_str_cfg):
    esm2_downstream_flip_sec_str_cfg.trainer.accumulate_grad_batches = 2
    trainer = setup_trainer(
        esm2_downstream_flip_sec_str_cfg, reset_accumulate_grad_batches=False
    )  # retain accumulate_grad_batches value for EncoderFinetuning class

    # reconfigure for encoder finetuning init
    with open_dict(esm2_downstream_flip_sec_str_cfg):
        esm2_downstream_flip_sec_str_cfg.model.encoder_cfg = esm2_downstream_flip_sec_str_cfg

    with distributed_model_parallel_state():
        _ = FineTuneProteinModel(esm2_downstream_flip_sec_str_cfg.model, trainer)
