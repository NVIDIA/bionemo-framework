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
from logging import Logger
from typing import Optional

from hydra import compose, initialize_config_dir
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.global_hydra import GlobalHydra
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig, OmegaConf

from bionemo.utils.tests import register_searchpath_config_plugin


def load_model_config(
    config_name: str, config_path: str, *, prepend_config_path: str = None, logger: Optional[Logger] = None
) -> DictConfig:
    """
    Gets config using hydra compose method. Registers hydra search path if prepend_config_path provided
    More details: https://hydra.cc/docs/advanced/search_path/
    Args:
        config_name: a name of the config file
        config_path: an absolute config path
        prepend_config_path: an absolute path to prepend to hydra search path
        logger: a logger to print config to the console

    Returns: hydra config as dict

    """

    if not os.path.isabs(config_path):
        raise ValueError("This methods requires config_path to be an absolute path")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    if prepend_config_path is not None:
        if not os.path.isabs(prepend_config_path):
            raise ValueError("This methods requires config_path to be an absolute path")

        class C(SearchPathPlugin):
            def __init__(self) -> None:
                super().__init__()

            def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
                search_path.prepend(provider="bionemo-searchpath-plugin", path=f"file://{prepend_config_path}")

        register_searchpath_config_plugin(C)

    with initialize_config_dir(version_base=None, config_dir=str(config_path)):
        cfg = compose(config_name=config_name.replace(".yaml", ""))
    if logger is not None:
        logger.info("\n\n************** Experiment configuration ***********")
        logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    return cfg
