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


from pathlib import Path

import pytest

from bionemo.core.data.load import load as core_load
from bionemo.testing.data.load import load


@pytest.fixture
def test_directory() -> Path:
    """Gets the path to the original synthetic single cell directory with test data (no feature ids).

    Returns:
        A Path object that is the directory with specified test data.
    """
    return load("scdl/sample") / "scdl_data"


@pytest.fixture
def test_directory_feat_ids() -> Path:
    """Gets the path to the directory with the synthetic single cell data (with the feature ids appended).

    Returns:
        A Path object that is the directory with specified test data.
    """
    return load("scdl/sample_scdl_feature_ids") / "scdl_data_with_feature_ids"


@pytest.fixture
def cellx_small_directory() -> Path:
    """Gets the path to the directory with with cellx small dataset in Single Cell Memmap format.

    Returns:
        A Path object that is the directory with the specified test data.
    """
    return load("single_cell/testdata-20241203") / "cellxgene_2023-12-15_small_processed_scdl"


@pytest.fixture(scope="session")
def data_path() -> Path:
    """Gets the path to the directory with cellx small dataset in Single Cell Memmap format.

    This is a session-scoped fixture to avoid redundant loading across multiple tests.
    This is the most commonly used test data across the test suite.

    Returns:
        A Path object that is the directory with the specified test data.
    """
    return core_load("single_cell/testdata-20241203") / "cellxgene_2023-12-15_small_processed_scdl"


@pytest.fixture(scope="session")
def geneformer_10m_checkpoint() -> Path:
    """Gets the path to the Geneformer 10M checkpoint version 2.0.

    This is a session-scoped fixture to avoid redundant loading across multiple tests.

    Returns:
        A Path object that is the checkpoint path.
    """
    return core_load("geneformer/10M_240530:2.0")


@pytest.fixture(scope="session")
def geneformer_nemo1_checkpoint() -> Path:
    """Gets the path to the Geneformer NeMo 1.0 QA checkpoint.

    This is a session-scoped fixture to avoid redundant loading.

    Returns:
        A Path object that is the checkpoint path.
    """
    return core_load("geneformer/qa")


@pytest.fixture(scope="session")
def geneformer_nemo1_release_checkpoint() -> Path:
    """Gets the path to the Geneformer NeMo 1.0 release checkpoint.

    This is a session-scoped fixture to avoid redundant loading.

    Returns:
        A Path object that is the checkpoint path.
    """
    return core_load("geneformer/10M_240530:1.0")


@pytest.fixture(scope="session")
def celltype_bench_data() -> Path:
    """Gets the path to the celltype benchmark golden values directory.

    This is a session-scoped fixture to avoid redundant loading.

    Returns:
        A Path object that is the directory with celltype benchmark data.
    """
    return core_load("single_cell/celltype-bench-golden-vals") / "hs-celltype-bench-subset7500"


@pytest.fixture(scope="session")
def nemo1_per_layer_outputs() -> Path:
    """Gets the path to NeMo 1.0 per-layer outputs.

    This is a session-scoped fixture to avoid redundant loading.

    Returns:
        A Path object that is the directory with per-layer outputs.
    """
    return core_load("single_cell/nemo1-geneformer-per-layer-outputs")


@pytest.fixture(scope="session")
def nemo1_expected_values() -> Path:
    """Gets the path to NeMo 1.0 expected values.

    This is a session-scoped fixture to avoid redundant loading.

    Returns:
        A Path object that is the directory with expected values.
    """
    return core_load("single_cell/nemo1-geneformer-golden-vals")
