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

import pytest

from bionemo.scdl.data.load import load
from bionemo.scdl.index.feature_index import ObservedFeatureIndex, VariableFeatureIndex


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


# ===== Variables that are used to create sample FeatureIndex Objects ======
@pytest.fixture
def seed_features_first():
    """
    Shared seed features used by both Variable and Observed first-index fixtures.
    """
    return {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}


@pytest.fixture
def seed_features_second():
    """
    Seed features for the second VariableFeatureIndex fixture.
    """
    return {
        "feature_name": np.array(["FF", "GG", "HH", "II", "ZZ"]),
        "gene_name": np.array(["RET", "NTRK", "PPARG", "TSHR", "EGFR"]),
        "spare": np.array([None, None, None, None, None]),
    }


# ==== Fixtures for VariableFeatureIndex ======
@pytest.fixture
def create_first_VariableFeatureIndex(seed_features_first) -> VariableFeatureIndex:
    """
    Instantiate a FeatureIndex.

    Returns:
        A FeatureIndex with known values.
    """
    index = VariableFeatureIndex()
    index.seed_features = seed_features_first
    index.append_features(12, index.seed_features)
    return index


@pytest.fixture
def create_second_VariableFeatureIndex(seed_features_second) -> VariableFeatureIndex:
    """
    Instantiate another FeatureIndex.

    Returns:
        A FeatureIndex with known values.
    """
    index2 = VariableFeatureIndex()
    index2.seed_features = seed_features_second
    index2.append_features(8, index2.seed_features, "MY_DATAFRAME")
    return index2


# ==== Fixtures for ObservedFeatureIndex ======
@pytest.fixture
def create_first_ObservedFeatureIndex(seed_features_first) -> ObservedFeatureIndex:
    """
    Instantiate a FeatureIndex.

    Returns:
        A FeatureIndex with known values.
    """
    index = ObservedFeatureIndex()
    index.seed_features = seed_features_first
    index.append_features(3, index.seed_features)
    return index


@pytest.fixture
def create_second_ObservedFeatureIndex(seed_features_first) -> ObservedFeatureIndex:
    """
    Instantiate another FeatureIndex.

    Returns:
        A FeatureIndex with known values.
    """
    index2 = ObservedFeatureIndex()
    index2.seed_features = seed_features_first
    index2.append_features(5, index2.seed_features, "MY_DATAFRAME")
    return index2
