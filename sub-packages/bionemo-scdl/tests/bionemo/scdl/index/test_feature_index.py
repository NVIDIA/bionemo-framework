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


import re

import numpy as np
import pandas as pd
import pytest

from bionemo.scdl.index.feature_index import FeatureIndex, are_dicts_equal


def test_equal_dicts():
    dict1 = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
    dict2 = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
    assert are_dicts_equal(dict1, dict2) is True


def test_unequal_values():
    dict1 = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
    dict3 = {"a": np.array([1, 2, 3]), "b": np.array([7, 8, 9])}

    assert are_dicts_equal(dict1, dict3) is False


def test_unequal_keys():
    dict1 = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
    dict4 = {"a": np.array([1, 2, 3]), "c": np.array([4, 5, 6])}
    assert are_dicts_equal(dict1, dict4) is False


def test_different_lengths():
    dict1 = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
    smaller_dict = {"a": np.array([1, 2, 3])}
    assert are_dicts_equal(dict1, smaller_dict) is False


@pytest.fixture
def create_first_FeatureIndex() -> FeatureIndex:
    """
    Instantiate a FeatureIndex.

    Returns:
        A FeatureIndex with known values.
    """
    one_feats = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    index = FeatureIndex()
    index.append_features(12, one_feats)
    return index


@pytest.fixture
def create_same_features_first_FeatureIndex() -> FeatureIndex:
    """
    Instantiate a FeatureIndex.

    Returns:
        A FeatureIndex with known values.
    """
    one_feats = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    index = FeatureIndex()
    index.append_features(6, one_feats)
    return index


@pytest.fixture
def create_second_FeatureIndex() -> FeatureIndex:
    """
    Instantiate another FeatureIndex.

    Returns:
        A FeatureIndex with known values.
    """
    two_feats = {
        "feature_name": np.array(["FF", "GG", "HH", "II", "ZZ"]),
        "gene_name": np.array(["RET", "NTRK", "PPARG", "TSHR", "EGFR"]),
        "spare": np.array([None, None, None, None, None]),
    }

    index2 = FeatureIndex()
    index2.append_features(8, two_feats, "MY_DATAFRAME")
    return index2


def test_dataframe_results_in_error():
    two_feats = pd.DataFrame(
        {
            "feature_name": ["FF", "GG", "HH", "II", "ZZ"],
            "gene_name": ["RET", "NTRK", "PPARG", "TSHR", "EGFR"],
            "spare": [None, None, None, None, None],
        }
    )
    index = FeatureIndex()
    with pytest.raises(TypeError) as error_info:
        index.append_features(8, two_feats, "MY_DATAFRAME")
        assert "Expected a dictionary, but received a Pandas DataFrame." in str(error_info.value)


def test_feature_index_internals_on_empty_index():
    index = FeatureIndex()
    assert len(index) == 0
    assert index.number_of_rows() == 0


def test_feature_index_internals_on_single_index(create_first_FeatureIndex):
    assert len(create_first_FeatureIndex) == 1
    assert [3] == create_first_FeatureIndex.column_dims()
    assert create_first_FeatureIndex.number_of_rows() == 12

    vals = create_first_FeatureIndex.number_of_values()
    assert vals == [12 * 3]
    assert len(vals) == 1


def test_feature_index_internals_on_append_empty_features(create_first_FeatureIndex):
    index = FeatureIndex()
    index.append_features(10, {})
    create_first_FeatureIndex.concat(index)
    assert len(create_first_FeatureIndex) == 2
    assert [3, 0] == create_first_FeatureIndex.column_dims()
    assert create_first_FeatureIndex.number_of_rows() == 22

    vals = create_first_FeatureIndex.number_of_values()
    assert vals == [12 * 3, 0]
    assert len(vals) == 2


def test_feature_index_internals_on_append_different_features(create_first_FeatureIndex, create_second_FeatureIndex):
    one_feats = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    two_feats = {
        "feature_name": np.array(["FF", "GG", "HH", "II", "ZZ"]),
        "gene_name": np.array(["RET", "NTRK", "PPARG", "TSHR", "EGFR"]),
        "spare": np.array([None, None, None, None, None]),
    }
    create_first_FeatureIndex.concat(create_second_FeatureIndex)
    assert len(create_first_FeatureIndex) == 2
    assert create_first_FeatureIndex.number_vars_at_row(1) == 3
    assert create_first_FeatureIndex.number_vars_at_row(13) == 5
    assert create_first_FeatureIndex.number_vars_at_row(19) == 5
    assert create_first_FeatureIndex.number_vars_at_row(2) == 3
    assert sum(create_first_FeatureIndex.number_of_values()) == (12 * 3) + (8 * 5)
    assert create_first_FeatureIndex.number_of_values()[1] == (8 * 5)
    assert create_first_FeatureIndex.number_of_rows() == 20
    feats, label = create_first_FeatureIndex.lookup(row=3, select_features=None)
    assert np.all(feats[0] == one_feats["feature_name"])
    assert np.all(feats[1] == one_feats["feature_int"])
    assert label is None
    feats, label = create_first_FeatureIndex.lookup(row=15, select_features=None)
    assert np.all(feats[0] == two_feats["feature_name"])
    assert np.all(feats[1] == two_feats["gene_name"])
    assert np.all(feats[2] == two_feats["spare"])
    assert label == "MY_DATAFRAME"


def test_feature_index_internals_on_append_same_features(create_first_FeatureIndex):
    one_feats = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    create_first_FeatureIndex.concat(create_first_FeatureIndex)
    assert len(create_first_FeatureIndex) == 1
    assert create_first_FeatureIndex.number_vars_at_row(1) == 3
    assert create_first_FeatureIndex.number_vars_at_row(13) == 3
    assert create_first_FeatureIndex.number_vars_at_row(19) == 3
    assert create_first_FeatureIndex.number_vars_at_row(2) == 3
    assert sum(create_first_FeatureIndex.number_of_values()) == 2 * (12 * 3)
    assert create_first_FeatureIndex.number_of_values()[0] == 2 * (12 * 3)
    assert create_first_FeatureIndex.number_of_rows() == 24
    feats, label = create_first_FeatureIndex.lookup(row=3, select_features=None)
    assert np.all(feats[0] == one_feats["feature_name"])
    assert np.all(feats[1] == one_feats["feature_int"])
    assert label is None
    feats, label = create_first_FeatureIndex.lookup(row=15, select_features=None)
    assert np.all(feats[0] == one_feats["feature_name"])
    assert np.all(feats[1] == one_feats["feature_int"])
    assert label is None


def test_concat_length(
    create_first_FeatureIndex,
    create_second_FeatureIndex,
):
    create_first_FeatureIndex.concat(create_second_FeatureIndex)
    assert len(create_first_FeatureIndex) == 2


def test_concat_number_variables_at_each_row(
    create_first_FeatureIndex,
    create_second_FeatureIndex,
):
    create_first_FeatureIndex.concat(create_second_FeatureIndex)
    assert create_first_FeatureIndex.number_vars_at_row(1) == 3
    assert create_first_FeatureIndex.number_vars_at_row(13) == 5
    assert create_first_FeatureIndex.number_vars_at_row(19) == 5
    assert create_first_FeatureIndex.number_vars_at_row(2) == 3


def test_concat_number_values(
    create_first_FeatureIndex,
    create_second_FeatureIndex,
):
    create_first_FeatureIndex.concat(create_second_FeatureIndex)

    assert sum(create_first_FeatureIndex.number_of_values()) == (12 * 3) + (8 * 5)
    assert create_first_FeatureIndex.number_of_values()[1] == (8 * 5)
    assert create_first_FeatureIndex.number_of_rows() == 20


def test_concat_lookup_results(
    create_first_FeatureIndex,
    create_second_FeatureIndex,
):
    one_feats = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    two_feats = {
        "feature_name": np.array(["FF", "GG", "HH", "II", "ZZ"]),
        "gene_name": np.array(["RET", "NTRK", "PPARG", "TSHR", "EGFR"]),
        "spare": np.array([None, None, None, None, None]),
    }
    create_first_FeatureIndex.concat(create_second_FeatureIndex)
    feats, label = create_first_FeatureIndex.lookup(row=3, select_features=None)
    assert np.all(feats[0] == one_feats["feature_name"])
    assert np.all(feats[1] == one_feats["feature_int"])
    assert label is None
    feats, label = create_first_FeatureIndex.lookup(row=15, select_features=None)
    assert np.all(feats[0] == two_feats["feature_name"])
    assert np.all(feats[1] == two_feats["gene_name"])
    assert np.all(feats[2] == two_feats["spare"])
    assert label == "MY_DATAFRAME"


def test_feature_lookup_empty():
    index = FeatureIndex()
    with pytest.raises(IndexError, match=r"There are no features to lookup"):
        index.lookup(row=1)


def test_feature_lookup_negative(create_first_FeatureIndex):
    with pytest.raises(IndexError, match=r"Row index -1 is not valid. It must be non-negative."):
        create_first_FeatureIndex.lookup(row=-1)


def test_feature_lookup_too_large(create_first_FeatureIndex):
    with pytest.raises(
        IndexError, match=re.escape("Row index 12544 is larger than number of rows in FeatureIndex (12).")
    ):
        create_first_FeatureIndex.lookup(row=12544)


def test_save_reload_row_feature_index_identical(tmp_path, create_first_FeatureIndex, create_second_FeatureIndex):
    create_first_FeatureIndex.concat(create_second_FeatureIndex)
    create_first_FeatureIndex.save(tmp_path / "features")
    index_reload = FeatureIndex.load(tmp_path / "features")
    assert len(create_first_FeatureIndex) == len(index_reload)
    assert create_first_FeatureIndex.column_dims() == index_reload.column_dims()
    assert create_first_FeatureIndex.number_of_rows() == index_reload.number_of_rows()
    assert create_first_FeatureIndex.version() == index_reload.version()

    assert create_first_FeatureIndex.number_of_values() == index_reload.number_of_values()

    for row in range(create_first_FeatureIndex.number_of_rows()):
        features_one, labels_one = create_first_FeatureIndex.lookup(row=row, select_features=None)
        features_reload, labels_reload = index_reload.lookup(row=row, select_features=None)
        assert labels_one == labels_reload
        assert np.all(np.array(features_one, dtype=object) == np.array(features_reload))
