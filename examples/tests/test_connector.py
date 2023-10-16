# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib

import pytest
from hydra import compose, initialize

from bionemo.model.molecule.megamolbart import MegaMolBARTModel
from bionemo.model.protein.downstream import FineTuneProteinModel
from bionemo.model.protein.esm1nv import ESM1nvModel
from bionemo.model.protein.prott5nv import ProtT5nvModel
from bionemo.model.utils import setup_trainer
from bionemo.utils.callbacks.callback_utils import setup_callbacks
from bionemo.utils.tests import BioNemoSearchPathConfig, register_searchpath_config_plugin, update_relative_config_dir


PREPEND_CONFIG_DIR = [
    '../protein/esm1nv/conf',
    '../protein/esm1nv/conf',
    '../protein/prott5nv/conf',
    '../protein/prott5nv/conf',
    '../molecule/megamolbart/conf',
    '../molecule/megamolbart/conf',
]
CONFIG_NAME = [
    'esm1nv_test_seqlen256',
    'esm1nv_test_seqlen1024',
    'prott5nv_test_seqlen256',
    'prott5nv_test_seqlen1024',
    'megamolbart_test_seqlen1024',
    'megamolbart_test_seqlen256',
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

os.environ['PROJECT_MOUNT'] = os.environ.get('PROJECT_MOUNT', '/workspace/bionemo')
THIS_FILE_DIR = pathlib.Path(os.path.abspath(__file__)).parent


def get_cfg(prepend_config_path, config_name, config_path='conf'):
    prepend_config_path = pathlib.Path(prepend_config_path)

    class TestSearchPathConfig(BioNemoSearchPathConfig):
        def __init__(self) -> None:
            super().__init__()
            self.prepend_config_dir = update_relative_config_dir(prepend_config_path, THIS_FILE_DIR)

    register_searchpath_config_plugin(TestSearchPathConfig)
    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name)

    return cfg


@pytest.mark.needs_gpu
@pytest.mark.parametrize(
    'prepend_config_path, config_name, model_class, model_parameters',
    list(zip(PREPEND_CONFIG_DIR, CONFIG_NAME, MODEL_CLASS, MODEL_PARAMETERS)),
)
def test_model_size(prepend_config_path, config_name, model_class, model_parameters):
    '''Check that number of model weights are correct'''

    cfg = get_cfg(prepend_config_path, config_name)
    callbacks = setup_callbacks(cfg)

    trainer = setup_trainer(cfg, callbacks=callbacks)
    if model_class == FineTuneProteinModel:
        model = model_class(cfg, trainer)
    else:
        model = model_class(cfg.model, trainer)
    assert model.num_weights == model_parameters
