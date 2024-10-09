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

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


BIONEMO_PREPEND_SEARCH_PATH: str = os.getenv("BIONEMO_PREPEND_SEARCH_PATH", None)
BIONEMO_APPEND_SEARCH_PATH: str = os.getenv("BIONEMO_APPEND_SEARCH_PATH", None)


class DefaultConfigSearchPathConfig(SearchPathPlugin):
    """
    Hydra SearchPathPlugin that appends or prepends list of path that Hydra searches in order to find
    non-primary configs. It adds more flexibility and customisation to the order of the searchpaths than
    the native hydra solutions.

    When a config is requested, the first matching config in the search path is selected, hence a user needs to be
    careful to name configs uniquely especially regarding file names in /workspace/nemo/examples/nlp/language_modeling/conf

    See hydra.searchpath details: https://hydra.cc/docs/advanced/search_path/

    *** USAGE ****
    APPEND_SEARCH_PATH=<PATH1>,<PATH_2>,..,<PATH_M> PREPEND_SEARCH_PATH=<PATH1>,<PATH_2>,..,<PATH_M> python ....

    with native support from hydra:
        python examples/molecule/megamolbart/pretrain.py --config-path $BIONEMO_HOME/examples/tests/conf
        --config-name megamolbart_test hydra.searchpath=[file://$BIONEMO_HOME/examples/molecule/megamolbart/conf]

    with this new solution
        BIONEMO_APPEND_SEARCH_PATH=$BIONEMO_HOME/examples/molecule/megamolbart/conf
        python examples/molecule/megamolbart/pretrain.py --config-path $BIONEMO_HOME/examples/tests/conf --config-name megamolbart_test

        BIONEMO_PREPEND_SEARCH_PATH=$BIONEMO_HOME/examples/molecule/megamolbart/conf
        python examples/molecule/megamolbart/pretrain.py --config-path $BIONEMO_HOME/examples/tests/conf --config-name megamolbart_test


    """

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        if BIONEMO_PREPEND_SEARCH_PATH is not None:
            new_search_paths: list = BIONEMO_PREPEND_SEARCH_PATH.split(",")
            for i, new_path in enumerate(new_search_paths):
                search_path.prepend(provider=f"prepend--searchpath-plugin-{i}", path=f"file://{new_path}")

        if BIONEMO_APPEND_SEARCH_PATH is not None:
            new_search_paths: list = BIONEMO_APPEND_SEARCH_PATH.split(",")
            for i, new_path in enumerate(new_search_paths):
                search_path.append(provider=f"append--searchpath-plugin-{i}", path=f"file://{new_path}")
