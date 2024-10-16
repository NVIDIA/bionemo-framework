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

from bionemo.testing.data.load import load


@pytest.fixture
def test_processed_directory() -> Path:
    """Gets the path to the directory with test processed data.

    Returns:
        A Path object that is the directory with test data.
    """
    bionemo2_root = Path("/workspace/bionemo2")
    return  bionemo2_root / "data/cellxgene_2023-12-15_small/processed_data/test"

    # return load("scdl/sample") / "scdl_data"

@pytest.fixture
def sc_test_data_directory() -> Path:
    """Gets the path to the directory with the Single Cell test data

    Returns:
        A Path object that is the directory with test data.
    """
    bionemo2_root = Path("/workspace/bionemo2")
    return  bionemo2_root / "sc_data"

@pytest.fixture
def sc_test_h5ad() -> Path:
    """Gets the path to the file with the input Single Cell test h5ad data

    Returns:
        A Path object that is the directory with test data.
    """
    bionemo2_root = Path("/workspace/bionemo2")
    base_h5ad_path = bionemo2_root / "data/cellxgene_2023-12-15_small/scdl_h5ad_test/"
    def _test_h5ad_file_path(file_name) -> Path: 
        return base_h5ad_path / file_name 
    return _test_h5ad_file_path

@pytest.fixture
def test_input_directory() -> Path:
    """Gets the path to the directory with test processed data.

    Returns:
        A Path object that is the directory with test data.
    """
    bionemo2_root = Path("/workspace/bionemo2")
    base_input_path =  bionemo2_root / "data/cellxgene_2023-12-15_small/input_data/test"
    def _test_input_directory(metadata_path) -> Path:
        
        path = base_input_path / metadata_path.replace("data/cellxgene_2023-12-15_small/input_data/test/", "")
        print("Path", path)
        return path

    return  _test_input_directory

@pytest.fixture
def test_directory() -> Path:
    """Gets the path to the single cell directory with test data.

    Returns:
        A Path object that is the directory with test data.
    """
    return load("scdl/sample") / "scdl_data"