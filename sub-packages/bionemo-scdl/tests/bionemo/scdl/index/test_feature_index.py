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

from bionemo.scdl.index.feature_index import FeatureIndex, ObservedFeatureIndex, VariableFeatureIndex, are_dicts_equal


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
def create_first_VariableFeatureIndex() -> VariableFeatureIndex:
    """
    Instantiate a FeatureIndex.

    Returns:
        A FeatureIndex with known values.
    """
    one_feats = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    index = VariableFeatureIndex()
    index.append_features(12, one_feats)
    return index


@pytest.fixture
def create_same_features_first_VariableFeatureIndex() -> VariableFeatureIndex:
    """
    Instantiate a FeatureIndex.

    Returns:
        A FeatureIndex with known values.
    """
    one_feats = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    index = VariableFeatureIndex()
    index.append_features(6, one_feats)
    return index


@pytest.fixture
def create_second_VariableFeatureIndex() -> VariableFeatureIndex:
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

    index2 = VariableFeatureIndex()
    index2.append_features(8, two_feats, "MY_DATAFRAME")
    return index2


@pytest.fixture
def create_first_ObservedFeatureIndex() -> ObservedFeatureIndex:
    """
    Instantiate a FeatureIndex.

    Returns:
        A FeatureIndex with known values.
    """
    one_feats = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    index = ObservedFeatureIndex()
    index.append_features(3, one_feats)
    return index


@pytest.fixture
def create_second_ObservedFeatureIndex() -> ObservedFeatureIndex:
    """
    Instantiate another FeatureIndex.

    Returns:
        A FeatureIndex with known values.
    """
    two_feats = {
        "f_name": np.array(["FDDF", "GG", "HdfH", "II", "ZZ"]),
        "f_int": np.array([1, 2, 3, 4, 5]),
        "f_spare": np.array([None, None, None, None, None]),
    }

    index2 = ObservedFeatureIndex()
    index2.append_features(5, two_feats, "MY_DATAFRAME")
    return index2


def test_dataframe_results_in_error():
    two_feats = pd.DataFrame(
        {
            "feature_name": ["FF", "GG", "HH", "II", "ZZ"],
            "gene_name": ["RET", "NTRK", "PPARG", "TSHR", "EGFR"],
            "spare": [None, None, None, None, None],
        }
    )
    index = VariableFeatureIndex()
    with pytest.raises(TypeError) as error_info:
        index.append_features(8, two_feats, "MY_DATAFRAME")
        assert "expects a dict of arrays" in str(error_info.value)

def test_feature_index_internals_on_empty_index():
    index = FeatureIndex()
    assert len(index) == 0
    assert index.number_of_rows() == 0


def test_feature_index_internals_on_single_index(create_first_ObservedFeatureIndex, create_first_VariableFeatureIndex):
    assert len(create_first_VariableFeatureIndex) == 1
    assert [3] == create_first_VariableFeatureIndex.column_dims()
    assert create_first_VariableFeatureIndex.number_of_rows() == 12

    vals = create_first_VariableFeatureIndex.number_of_values()
    assert vals == [12 * 3]

    assert len(create_first_ObservedFeatureIndex) == 1
    assert [2] == create_first_ObservedFeatureIndex.column_dims()
    assert create_first_ObservedFeatureIndex.number_of_rows() == 3

    vals = create_first_ObservedFeatureIndex.number_of_values()
    assert vals == [3 * 2]


def test_feature_index_internals_on_append_empty_features(
    create_first_VariableFeatureIndex, create_first_ObservedFeatureIndex
):
    index = VariableFeatureIndex()
    index.append_features(10, {})
    create_first_VariableFeatureIndex.concat(index)
    assert len(create_first_VariableFeatureIndex) == 2
    assert [3, 0] == create_first_VariableFeatureIndex.column_dims()
    assert create_first_VariableFeatureIndex.number_of_rows() == 22

    vals = create_first_VariableFeatureIndex.number_of_values()
    assert vals == [12 * 3, 0]
    assert len(vals) == 2


def test_obs_feature_index_internals_on_append_incompatible_features_observed(create_first_ObservedFeatureIndex):
    index = ObservedFeatureIndex()
    with pytest.raises(ValueError, match=r"Number of observations 10 does not match the number of .obs entries 0"):
        index.append_features(10, {})

    with pytest.raises(ValueError, match=r"Number of observations 10 does not match the number of .obs entries 3"):
        index.append_features(10, {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])})


def test_feature_index_internals_on_append_different_features(
    create_first_VariableFeatureIndex, create_second_VariableFeatureIndex
):
    one_feats = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    two_feats = {
        "feature_name": np.array(["FF", "GG", "HH", "II", "ZZ"]),
        "gene_name": np.array(["RET", "NTRK", "PPARG", "TSHR", "EGFR"]),
        "spare": np.array([None, None, None, None, None]),
    }
    create_first_VariableFeatureIndex.concat(create_second_VariableFeatureIndex)
    assert len(create_first_VariableFeatureIndex) == 2
    assert create_first_VariableFeatureIndex.number_vars_at_row(1) == 3
    assert create_first_VariableFeatureIndex.number_vars_at_row(13) == 5
    assert create_first_VariableFeatureIndex.number_vars_at_row(19) == 5
    assert create_first_VariableFeatureIndex.number_vars_at_row(2) == 3
    assert sum(create_first_VariableFeatureIndex.number_of_values()) == (12 * 3) + (8 * 5)
    assert create_first_VariableFeatureIndex.number_of_values()[1] == (8 * 5)
    assert create_first_VariableFeatureIndex.number_of_rows() == 20
    feats, label = create_first_VariableFeatureIndex.lookup(row=3, select_features=None)
    assert np.all(feats[0] == one_feats["feature_name"])
    assert np.all(feats[1] == one_feats["feature_int"])
    assert label is None
    feats, label = create_first_VariableFeatureIndex.lookup(row=15, select_features=None)
    assert np.all(feats[0] == two_feats["feature_name"])
    assert np.all(feats[1] == two_feats["gene_name"])
    assert np.all(feats[2] == two_feats["spare"])
    assert label == "MY_DATAFRAME"


def test_feature_index_internals_on_append_different_features_observed(
    create_first_ObservedFeatureIndex, create_second_ObservedFeatureIndex
):
    one_feats = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    two_feats = {
        "feature_name": np.array(["FDDF", "GG", "HdfH", "II", "ZZ"]),
        "feature_int": np.array([1, 2, 3, 4, 5]),
        "f_spare": np.array([None, None, None, None, None]),
    }
    create_first_ObservedFeatureIndex.concat(create_second_ObservedFeatureIndex)
    assert len(create_first_ObservedFeatureIndex) == 2
    for row in range(3):
        assert create_first_ObservedFeatureIndex.number_vars_at_row(row) == 2
    for row in range(3, 8):
        assert create_first_ObservedFeatureIndex.number_vars_at_row(row) == 3
    vals = create_first_ObservedFeatureIndex.number_of_values()
    assert vals[0] == 3 * 2
    assert vals[1] == 3 * 5
    assert create_first_ObservedFeatureIndex.number_of_rows() == 8
    for j in range(3):
        feats, label = create_first_ObservedFeatureIndex.lookup(row=j, select_features=None)
        assert feats == [one_feats["feature_name"][j], one_feats["feature_int"][j]]
        assert label is None
    for j in range(5):
        feats, label = create_first_ObservedFeatureIndex.lookup(row=3 + j, select_features=None)
        assert feats == [two_feats["feature_name"][j], two_feats["feature_int"][j], two_feats["f_spare"][j]]
        assert label == "MY_DATAFRAME"


def test_feature_index_internals_on_append_same_features(create_first_VariableFeatureIndex):
    one_feats = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    create_first_VariableFeatureIndex.concat(create_first_VariableFeatureIndex)
    assert len(create_first_VariableFeatureIndex) == 1
    assert create_first_VariableFeatureIndex.number_vars_at_row(1) == 3
    assert create_first_VariableFeatureIndex.number_vars_at_row(13) == 3
    assert create_first_VariableFeatureIndex.number_vars_at_row(19) == 3
    assert create_first_VariableFeatureIndex.number_vars_at_row(2) == 3
    assert sum(create_first_VariableFeatureIndex.number_of_values()) == 2 * (12 * 3)
    assert create_first_VariableFeatureIndex.number_of_values()[0] == 2 * (12 * 3)
    assert create_first_VariableFeatureIndex.number_of_rows() == 24
    feats, label = create_first_VariableFeatureIndex.lookup(row=3, select_features=None)
    assert np.all(feats[0] == one_feats["feature_name"])
    assert np.all(feats[1] == one_feats["feature_int"])
    assert label is None
    feats, label = create_first_VariableFeatureIndex.lookup(row=15, select_features=None)
    assert np.all(feats[0] == one_feats["feature_name"])
    assert np.all(feats[1] == one_feats["feature_int"])
    assert label is None


def test_feature_index_internals_on_append_same_features_observed(create_first_ObservedFeatureIndex):
    one_feats = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    create_first_ObservedFeatureIndex.concat(create_first_ObservedFeatureIndex)
    assert len(create_first_ObservedFeatureIndex) == 1
    for j in range(6):
        assert create_first_ObservedFeatureIndex.number_vars_at_row(j) == 2
    assert sum(create_first_ObservedFeatureIndex.number_of_values()) == 6 * 2
    assert create_first_ObservedFeatureIndex.number_of_rows() == 6
    for j in range(6):
        feats, label = create_first_ObservedFeatureIndex.lookup(row=j, select_features=None)
        assert np.all(feats == [one_feats["feature_name"][j % 3], one_feats["feature_int"][j % 3]])
        assert label is None
    with pytest.raises(IndexError, match=re.escape("Row index 6 is larger than number of rows in FeatureIndex (6).")):
        create_first_ObservedFeatureIndex.lookup(row=6)


def test_concat_length(
    create_first_VariableFeatureIndex,
    create_second_VariableFeatureIndex,
):
    create_first_VariableFeatureIndex.concat(create_second_VariableFeatureIndex)
    assert len(create_first_VariableFeatureIndex) == 2


def test_concat_number_variables_at_each_row(
    create_first_VariableFeatureIndex,
    create_second_VariableFeatureIndex,
):
    create_first_VariableFeatureIndex.concat(create_second_VariableFeatureIndex)
    assert create_first_VariableFeatureIndex.number_vars_at_row(1) == 3
    assert create_first_VariableFeatureIndex.number_vars_at_row(13) == 5
    assert create_first_VariableFeatureIndex.number_vars_at_row(19) == 5
    assert create_first_VariableFeatureIndex.number_vars_at_row(2) == 3


def test_concat_number_values(
    create_first_VariableFeatureIndex,
    create_second_VariableFeatureIndex,
):
    create_first_VariableFeatureIndex.concat(create_second_VariableFeatureIndex)

    assert sum(create_first_VariableFeatureIndex.number_of_values()) == (12 * 3) + (8 * 5)
    assert create_first_VariableFeatureIndex.number_of_values()[1] == (8 * 5)
    assert create_first_VariableFeatureIndex.number_of_rows() == 20


def test_concat_lookup_results(
    create_first_VariableFeatureIndex,
    create_second_VariableFeatureIndex,
):
    one_feats = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    two_feats = {
        "feature_name": np.array(["FF", "GG", "HH", "II", "ZZ"]),
        "gene_name": np.array(["RET", "NTRK", "PPARG", "TSHR", "EGFR"]),
        "spare": np.array([None, None, None, None, None]),
    }
    create_first_VariableFeatureIndex.concat(create_second_VariableFeatureIndex)
    feats, label = create_first_VariableFeatureIndex.lookup(row=3, select_features=None)
    assert np.all(feats[0] == one_feats["feature_name"])
    assert np.all(feats[1] == one_feats["feature_int"])
    assert label is None
    feats, label = create_first_VariableFeatureIndex.lookup(row=15, select_features=None)
    assert np.all(feats[0] == two_feats["feature_name"])
    assert np.all(feats[1] == two_feats["gene_name"])
    assert np.all(feats[2] == two_feats["spare"])
    assert label == "MY_DATAFRAME"


def test_VariableFeatureIndex_lookup_empty():
    index = VariableFeatureIndex()
    with pytest.raises(IndexError, match=r"There are no features to lookup"):
        index.lookup(row=1)


def test_ObservedFeatureIndex_lookup_empty():
    index = ObservedFeatureIndex()
    with pytest.raises(IndexError, match=r"There are no features to lookup"):
        index.lookup(row=1)


def test_VariableFeatureIndex_lookup_negative(create_first_VariableFeatureIndex):
    with pytest.raises(IndexError, match=r"Row index -1 is not valid. It must be non-negative."):
        create_first_VariableFeatureIndex.lookup(row=-1)


def test_ObservedFeatureIndex_lookup_negative(create_first_ObservedFeatureIndex):
    with pytest.raises(IndexError, match=r"Row index -1 is not valid. It must be non-negative."):
        create_first_ObservedFeatureIndex.lookup(row=-1)


def test_VariableFeatureIndex_lookup_too_large(create_first_VariableFeatureIndex):
    with pytest.raises(
        IndexError, match=re.escape("Row index 12544 is larger than number of rows in FeatureIndex (12).")
    ):
        create_first_VariableFeatureIndex.lookup(row=12544)


def test_save_reload_VariableFeatureIndex_identical(
    tmp_path, create_first_VariableFeatureIndex, create_second_VariableFeatureIndex
):
    create_first_VariableFeatureIndex.concat(create_second_VariableFeatureIndex)
    create_first_VariableFeatureIndex.save(tmp_path / "features")
    index_reload = VariableFeatureIndex.load(tmp_path / "features")
    assert len(create_first_VariableFeatureIndex) == len(index_reload)
    assert create_first_VariableFeatureIndex.column_dims() == index_reload.column_dims()
    assert create_first_VariableFeatureIndex.number_of_rows() == index_reload.number_of_rows()
    assert create_first_VariableFeatureIndex.version() == index_reload.version()

    assert create_first_VariableFeatureIndex.number_of_values() == index_reload.number_of_values()

    for row in range(create_first_VariableFeatureIndex.number_of_rows()):
        features_one, labels_one = create_first_VariableFeatureIndex.lookup(row=row, select_features=None)
        features_reload, labels_reload = index_reload.lookup(row=row, select_features=None)
        assert labels_one == labels_reload
        assert np.all(np.array(features_one, dtype=object) == np.array(features_reload))


def test_save_reload_ObservedFeatureIndex_identical(
    tmp_path, create_first_ObservedFeatureIndex, create_second_ObservedFeatureIndex
):
    create_first_ObservedFeatureIndex.concat(create_second_ObservedFeatureIndex)
    create_first_ObservedFeatureIndex.save(tmp_path / "features")
    index_reload = ObservedFeatureIndex.load(tmp_path / "features")
    assert len(create_first_ObservedFeatureIndex) == len(index_reload)
    assert create_first_ObservedFeatureIndex.column_dims() == index_reload.column_dims()
    assert create_first_ObservedFeatureIndex.number_of_rows() == index_reload.number_of_rows()
    assert create_first_ObservedFeatureIndex.version() == index_reload.version()

    assert create_first_ObservedFeatureIndex.number_of_values() == index_reload.number_of_values()

    for row in range(create_first_ObservedFeatureIndex.number_of_rows()):
        features_one, labels_one = create_first_ObservedFeatureIndex.lookup(row=row, select_features=None)
        features_reload, labels_reload = index_reload.lookup(row=row, select_features=None)
        assert labels_one == labels_reload
        assert np.all(np.array(features_one) == np.array(features_reload))


def test_observed_getitem_slice_contiguous_across_blocks():
    df1_feats = {
        "feature_name": np.array(["FF", "GG", "HH"]),
        "feature_int": np.array([1, 2, 3]),
    }
    df2_feats = {
        "f_name": np.array(["A", "B", "C", "D", "E"]),
        "f_int": np.array([10, 11, 12, 13, 14]),
        "f_spare": np.array([None, None, None, None, None]),
    }
    obs = ObservedFeatureIndex()
    obs.append_features(3, df1_feats)
    obs.append_features(5, df2_feats, label="blk2")

    out, labels = obs[1:7]
    assert labels == [None, "blk2"]
    assert isinstance(out, list)
    assert len(out) == 2
    pd.testing.assert_frame_equal(out[0], pd.DataFrame(df1_feats).iloc[1:])
    pd.testing.assert_frame_equal(out[1], pd.DataFrame(df2_feats).iloc[:4])


def test_observed_getitem_slice_with_step_and_order_preserved():
    df1_feats = {
        "x": np.array([0, 1, 2, 3]),
    }
    df2_feats = {
        "y": np.array([10, 11, 12]),
    }
    obs = ObservedFeatureIndex()
    obs.append_features(4, df1_feats)
    obs.append_features(3, df2_feats, label="b2")

    out, labels = obs[0:7:2]
    assert labels == [None, "b2"]
    assert isinstance(out, list)
    assert len(out) == 2
    (pd.testing.assert_frame_equal(out[0].reset_index(drop=True), pd.DataFrame({"x": [0, 2]})),)
    pd.testing.assert_frame_equal(out[1].reset_index(drop=True), pd.DataFrame({"y": [10, 12]}))
