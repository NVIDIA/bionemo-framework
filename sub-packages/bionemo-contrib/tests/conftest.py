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


# conftest.py
import pathlib
import os
import pytest


_BNMO_ROOT = pathlib.Path(os.path.abspath(os.path.dirname(__file__))).parent.parent.parent
@pytest.fixture(scope="session")
def bionemo2_root_path():
    # TODO: os.env("BIONEMO_HOME") when its setup.
    return _BNMO_ROOT


@pytest.fixture(scope="session")
def single_cell_test_data_processed(bionemo2_root_path):
    return bionemo2_root_path / "test_data/cellxgene_2023-12-15_small/processed_data/test"


@pytest.fixture(scope="session")
def single_cell_test_data_input(bionemo2_root_path):
    return bionemo2_root_path / "test_data/cellxgene_2023-12-15_small/input_data/test"


@pytest.fixture(scope="session")
def geneformer_nemo1_checkpoint_path(bionemo2_root_path):
    return bionemo2_root_path / "models/singlecell/geneformer/geneformer-10M-240530.nemo"


@pytest.fixture(scope="session")
def single_cell_processed_data_input(bionemo2_root_path):
    return bionemo2_root_path / "test_data/cellxgene_2023-12-15_small/processed_data"
