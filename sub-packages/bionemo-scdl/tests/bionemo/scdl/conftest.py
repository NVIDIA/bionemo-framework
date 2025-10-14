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


import shutil
from pathlib import Path

import numpy as np
import pytest

from bionemo.scdl.data.load import load
from bionemo.scdl.index.row_feature_index import VariableFeatureIndex


@pytest.fixture
def test_directory() -> Path:
    """Gets the path to the directory with test data.

    Returns:
        A Path object that is the directory with test data.
    """
    return load("scdl/sample_scdl_feature_ids") / "scdl_data_with_feature_ids"


@pytest.fixture
def test_neighbor_directory() -> Path:
    """Gets the path to the directory with neighbor test data.

    Returns:
        A Path object that is the directory with neighbor test data.
    """
    return load("scdl/sample_scdl_neighbor")


@pytest.fixture
def create_cellx_val_data(tmpdir) -> Path:
    """Gets the path to the directory with test data.

    Returns:
        A Path object that is the directory with test data.
    """
    cellx_input_val_path = (
        load("scdl/testdata-20240506") / "cellxgene_2023-12-15_small" / "input_data" / "val" / "assay__10x_3_v2/"
    )
    file1 = (
        cellx_input_val_path
        / "sex__female/development_stage__74-year-old_human_stage/self_reported_ethnicity__Asian/tissue_general__lung/dataset_id__f64e1be1-de15-4d27-8da4-82225cd4c035/sidx_40575621_2_0.h5ad"
    )
    file2 = (
        cellx_input_val_path
        / "sex__male/development_stage__82-year-old_human_stage/self_reported_ethnicity__European/tissue_general__lung/dataset_id__f64e1be1-de15-4d27-8da4-82225cd4c035/sidx_40596188_1_0.h5ad"
    )
    collated_dir = tmpdir / "collated_val"
    collated_dir.mkdir()
    shutil.copy(file1, collated_dir)
    shutil.copy(file2, collated_dir)
    return collated_dir


# ==== Fixtures for VariableFeatureIndex ======
@pytest.fixture
def create_first_VariableFeatureIndex() -> VariableFeatureIndex:
    """
    Instantiate a FeatureIndex.
    Returns:
        A FeatureIndex with known values.
    """
    index = VariableFeatureIndex()
    seed_features = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    num_rows = 12
    label = None
    index.append_features(num_rows, seed_features, label)
    return index, seed_features, num_rows, label


@pytest.fixture
def create_second_VariableFeatureIndex() -> VariableFeatureIndex:
    """
    Instantiate another FeatureIndex.
    Returns:
        A FeatureIndex with known values.
    """

    index = VariableFeatureIndex()
    seed_features = {
        "feature_name": np.array(["FF", "GG", "HH", "II", "ZZ"]),
        "gene_name": np.array(["RET", "NTRK", "PPARG", "TSHR", "EGFR"]),
        "spare": np.array([None, None, None, None, None]),
    }
    num_rows = 8
    label = "MY_DATAFRAME"
    index.append_features(num_rows, seed_features, label)
    return index, seed_features, num_rows, label


@pytest.fixture
def create_empty_VariableFeatureIndex() -> VariableFeatureIndex:
    """
    Instantiate an empty FeatureIndex.
    Returns:
        A FeatureIndex with no features.
    """
    num_empty_rows = 10
    index_with_empty_features = VariableFeatureIndex()
    index_with_empty_features.append_features(num_empty_rows, {})
    return index_with_empty_features, num_empty_rows
