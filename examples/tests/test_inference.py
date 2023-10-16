# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

import logging
import os
import pathlib
import pickle

import pytest
from hydra import compose, initialize

import examples.infer as infer
from bionemo.utils.tests import (
    BioNemoSearchPathConfig,
    clean_directory,
    register_searchpath_config_plugin,
    update_relative_config_dir,
)


# logger
logging.getLogger('nemo_logger').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

PREPEND_CONFIG_DIR = ['../conf/', '../conf/', '../conf/']

CONFIG_NAME = ['prott5nv_infer', 'esm1nv_infer', 'megamolbart_infer']

HIDDEN_SIZE = [768, 768, 512]

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
@pytest.mark.needs_checkpoint
@pytest.mark.parametrize(
    'prepend_config_path, config_name, hidden_size', list(zip(PREPEND_CONFIG_DIR, CONFIG_NAME, HIDDEN_SIZE))
)
def test_model_inference(prepend_config_path, config_name, hidden_size):
    '''Check that number of model weights are correct'''

    cfg = get_cfg(prepend_config_path, config_name)
    clean_directory(cfg.exp_manager.exp_dir)
    infer.main(cfg)

    embeddings = pickle.load(open(cfg.model.data.output_fname, "rb"))
    dict_keys = cfg.model.downstream_task.outputs + ["sequence", "id"]
    for emb in embeddings:
        for key in dict_keys:
            if key not in emb.keys():
                assert False, f'Missing key {key} in embeddings file {cfg.model.data.output_fname}'
        if "hiddens" in dict_keys:
            assert emb["hiddens"].shape == (len(emb["sequence"]), hidden_size)
        if "embeddings" in dict_keys:
            assert emb["embeddings"].shape == (hidden_size,)
