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
from pathlib import Path

import pytest
from pytorch_lightning import LightningModule

from bionemo.callbacks import setup_dwnstr_task_validation_callbacks
from bionemo.model.molecule.megamolbart import MegaMolBARTModel
from bionemo.model.protein.downstream import FineTuneProteinModel
from bionemo.model.protein.esm1nv import ESM1nvModel
from bionemo.model.protein.prott5nv import ProtT5nvModel
from bionemo.model.utils import initialize_distributed_parallel_state, setup_trainer
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import teardown_apex_megatron_cuda


CONFIG_NAME = [
    "esm1nv_test_seqlen256",
    "esm1nv_test_seqlen1024",
    "prott5nv_test_seqlen256",
    "prott5nv_test_seqlen1024",
    "megamolbart_test_seqlen1024",
    "megamolbart_test_seqlen256",
]
MODEL_CLASS = [ESM1nvModel, ESM1nvModel, ProtT5nvModel, ProtT5nvModel, MegaMolBARTModel, MegaMolBARTModel]

MODEL_PARAMETERS = [
    43415936,
    44005760,
    198773888,
    199363712,
    45320192,
    44926976,
]


@pytest.fixture(scope="module")
def bionemo_home() -> Path:
    try:
        x = os.environ["BIONEMO_HOME"]
    except KeyError:
        raise ValueError("Need to set BIONEMO_HOME in order to run unit tests! See docs for instructions.")
    else:
        yield Path(x).absolute()


@pytest.fixture(scope="module")
def config_path_for_tests(bionemo_home) -> str:
    yield str(bionemo_home / "examples" / "tests" / "conf")


@pytest.mark.needs_gpu
@pytest.mark.parametrize(
    "config_name, model_class, model_parameters",
    list(zip(CONFIG_NAME, MODEL_CLASS, MODEL_PARAMETERS)),
)
def test_model_size(config_name: str, model_class: LightningModule, model_parameters: int, config_path_for_tests):
    """Check that number of model weights are correct"""
    cfg = load_model_config(config_name=config_name, config_path=config_path_for_tests)
    initialize_distributed_parallel_state()
    callbacks = setup_dwnstr_task_validation_callbacks(cfg)
    trainer = setup_trainer(cfg, callbacks=callbacks)
    if model_class == FineTuneProteinModel:
        model = model_class(cfg, trainer)
    else:
        model = model_class(cfg.model, trainer)
    teardown_apex_megatron_cuda()
    assert model.num_weights == model_parameters
