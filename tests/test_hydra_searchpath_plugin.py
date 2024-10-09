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


import hydra
from hydra.core.global_hydra import GlobalHydra


def test_nemo_config_searchpath_plugin():
    """
    Tests if NeMoConfigSearchPathConfig has been initialized correctly
    """
    with hydra.initialize():
        config_loader = GlobalHydra.instance().config_loader()
        search_paths_obj = config_loader.get_search_path()
    search_paths = [s.path for s in search_paths_obj.config_search_path]
    nemo_search_path_str = "file:///workspace/nemo/examples/nlp/language_modeling/conf"
    assert nemo_search_path_str in search_paths


def test_load_config_from_nemo():
    """
    Tests if config from NeMo directory /workspace/nemo/examples/nlp/language_modeling/conf can be loaded
    """
    nemo_config = "megatron_model_base_config"
    with hydra.initialize():
        cfg = hydra.compose(config_name=nemo_config)
    assert cfg is not None
    assert "hidden_size" in cfg
