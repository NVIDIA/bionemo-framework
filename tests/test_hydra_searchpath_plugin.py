# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
