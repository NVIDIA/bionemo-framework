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
    index = VariableFeatureIndex()
    index.seed_features = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    index.append_features(12, index.seed_features)
    return index


@pytest.fixture
def create_same_features_first_VariableFeatureIndex() -> VariableFeatureIndex:
    """
    Instantiate a FeatureIndex.

    Returns:
        A FeatureIndex with known values.
    """
    index = VariableFeatureIndex()
    index.seed_features = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    index.append_features(6, index.seed_features)
    return index


@pytest.fixture
def create_second_VariableFeatureIndex() -> VariableFeatureIndex:
    """
    Instantiate another FeatureIndex.

    Returns:
        A FeatureIndex with known values.
    """
    index2 = VariableFeatureIndex()
    index2.seed_features = {
        "feature_name": np.array(["FF", "GG", "HH", "II", "ZZ"]),
        "gene_name": np.array(["RET", "NTRK", "PPARG", "TSHR", "EGFR"]),
        "spare": np.array([None, None, None, None, None]),
    }
    index2.append_features(8, index2.seed_features, "MY_DATAFRAME")
    return index2


@pytest.fixture
def create_first_ObservedFeatureIndex() -> ObservedFeatureIndex:
    """
    Instantiate a FeatureIndex.

    Returns:
        A FeatureIndex with known values.
    """
    index = ObservedFeatureIndex()
    index.seed_features = {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])}
    index.append_features(3, index.seed_features)
    return index


@pytest.fixture
def create_second_ObservedFeatureIndex() -> ObservedFeatureIndex:
    """
    Instantiate another FeatureIndex.

    Returns:
        A FeatureIndex with known values.
    """
    index2 = ObservedFeatureIndex()
    index2.seed_features = {
        "f_name": np.array(["FDDF", "GG", "HdfH", "II", "ZZ"]),
        "f_int": np.array([1, 2, 3, 4, 5]),
        "f_spare": np.array([None, None, None, None, None]),
    }
    index2.append_features(5, index2.seed_features, "MY_DATAFRAME")
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
    with pytest.raises(TypeError, match="VariableFeatureIndex.append_features expects a dict of arrays"):
        index.append_features(8, two_feats, "MY_DATAFRAME")


def test_feature_index_internals_on_empty_index():
    index = FeatureIndex()
    assert len(index) == 0
    assert index.number_of_rows() == 0


def test_feature_index_internals_on_single_index_both_variable_and_observed(
    create_first_ObservedFeatureIndex, create_first_VariableFeatureIndex
):
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


def testObeservedFetureIndex_on_append_incompatible_features_observed(create_first_ObservedFeatureIndex):
    index = ObservedFeatureIndex()
    with pytest.raises(ValueError, match=r"Provided empty features but n_obs > 0"):
        index.append_features(10, {})

    with pytest.raises(ValueError, match=r"Number of observations 10 does not match feature array length 3"):
        index.append_features(10, {"feature_name": np.array(["FF", "GG", "HH"]), "feature_int": np.array([1, 2, 3])})


def testVariableFeatureIndex_on_append_different_features__concat_and_basic_counts(
    create_first_VariableFeatureIndex, create_second_VariableFeatureIndex
):
    create_first_VariableFeatureIndex.concat(create_second_VariableFeatureIndex)
    assert len(create_first_VariableFeatureIndex) == 2


def testVariableFeatureIndex_on_append_different_features__number_vars(
    create_first_VariableFeatureIndex, create_second_VariableFeatureIndex
):
    create_first_VariableFeatureIndex.concat(create_second_VariableFeatureIndex)
    assert create_first_VariableFeatureIndex.number_vars_at_row(1) == 3
    assert create_first_VariableFeatureIndex.number_vars_at_row(13) == 5
    assert create_first_VariableFeatureIndex.number_vars_at_row(19) == 5
    assert create_first_VariableFeatureIndex.number_vars_at_row(2) == 3


def testVariableFeatureIndex_on_append_different_features__number_of_values(
    create_first_VariableFeatureIndex, create_second_VariableFeatureIndex
):
    create_first_VariableFeatureIndex.concat(create_second_VariableFeatureIndex)
    assert sum(create_first_VariableFeatureIndex.number_of_values()) == (12 * 3) + (8 * 5)
    assert create_first_VariableFeatureIndex.number_of_values()[1] == (8 * 5)
    assert create_first_VariableFeatureIndex.number_of_rows() == 20


def testVariableFeatureIndex_on_append_different_features__lookup_first(
    create_first_VariableFeatureIndex, create_second_VariableFeatureIndex
):
    one_feats = create_first_VariableFeatureIndex.seed_features
    create_first_VariableFeatureIndex.concat(create_second_VariableFeatureIndex)
    feats, label = create_first_VariableFeatureIndex.lookup(row=3, select_features=None)
    assert np.all(feats[0] == one_feats["feature_name"])
    assert np.all(feats[1] == one_feats["feature_int"])
    assert label is None


def testVariableFeatureIndex_on_append_different_features__lookup_second(
    create_first_VariableFeatureIndex, create_second_VariableFeatureIndex
):
    two_feats = create_second_VariableFeatureIndex.seed_features
    create_first_VariableFeatureIndex.concat(create_second_VariableFeatureIndex)
    feats, label = create_first_VariableFeatureIndex.lookup(row=15, select_features=None)
    assert np.all(feats[0] == two_feats["feature_name"])
    assert np.all(feats[1] == two_feats["gene_name"])
    assert np.all(feats[2] == two_feats["spare"])
    assert label == "MY_DATAFRAME"


def testObeservedFetureIndex_on_append_different_features_observed(
    create_first_ObservedFeatureIndex, create_second_ObservedFeatureIndex
):
    one_feats = create_first_ObservedFeatureIndex.seed_features
    two_feats = create_second_ObservedFeatureIndex.seed_features
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


def testVariableFeatureIndex_on_append_features_twice_length(create_first_VariableFeatureIndex):
    create_first_VariableFeatureIndex.concat(create_first_VariableFeatureIndex)
    assert len(create_first_VariableFeatureIndex) == 1


def testVariableFeatureIndex_on_append_features_twice_number_vars(create_first_VariableFeatureIndex):
    create_first_VariableFeatureIndex.concat(create_first_VariableFeatureIndex)
    assert create_first_VariableFeatureIndex.number_vars_at_row(1) == 3
    assert create_first_VariableFeatureIndex.number_vars_at_row(13) == 3
    assert create_first_VariableFeatureIndex.number_vars_at_row(19) == 3
    assert create_first_VariableFeatureIndex.number_vars_at_row(2) == 3


def testVariableFeatureIndex_on_append_features_twice_number_of_values(create_first_VariableFeatureIndex):
    create_first_VariableFeatureIndex.concat(create_first_VariableFeatureIndex)
    assert sum(create_first_VariableFeatureIndex.number_of_values()) == 2 * (12 * 3)
    assert create_first_VariableFeatureIndex.number_of_values()[0] == 2 * (12 * 3)
    assert create_first_VariableFeatureIndex.number_of_rows() == 24


def testVariableFeatureIndex_on_append_features_twice_lookup(create_first_VariableFeatureIndex):
    seed = create_first_VariableFeatureIndex.seed_features
    create_first_VariableFeatureIndex.concat(create_first_VariableFeatureIndex)
    feats, label = create_first_VariableFeatureIndex.lookup(row=3, select_features=None)
    assert np.all(feats[0] == seed["feature_name"])
    assert np.all(feats[1] == seed["feature_int"])
    assert label is None
    feats, label = create_first_VariableFeatureIndex.lookup(row=15, select_features=None)
    assert np.all(feats[0] == seed["feature_name"])
    assert np.all(feats[1] == seed["feature_int"])
    assert label is None


def testObeservedFetureIndex_on_append_features_twice_length(create_first_ObservedFeatureIndex):
    create_first_ObservedFeatureIndex.concat(create_first_ObservedFeatureIndex)
    assert len(create_first_ObservedFeatureIndex) == 1


def testObeservedFetureIndex_on_append_features_twice_number_vars(create_first_ObservedFeatureIndex):
    create_first_ObservedFeatureIndex.concat(create_first_ObservedFeatureIndex)
    for j in range(6):
        assert create_first_ObservedFeatureIndex.number_vars_at_row(j) == 2


def testObeservedFetureIndex_on_append_features_twice_number_of_values(create_first_ObservedFeatureIndex):
    create_first_ObservedFeatureIndex.concat(create_first_ObservedFeatureIndex)
    assert sum(create_first_ObservedFeatureIndex.number_of_values()) == 6 * 2


def testObeservedFetureIndex_on_append_features_twice_number_of_rows(create_first_ObservedFeatureIndex):
    create_first_ObservedFeatureIndex.concat(create_first_ObservedFeatureIndex)
    assert create_first_ObservedFeatureIndex.number_of_rows() == 6


def testObeservedFetureIndex_on_append_features_twice_lookup(create_first_ObservedFeatureIndex):
    one_feats = create_first_ObservedFeatureIndex.seed_features
    create_first_ObservedFeatureIndex.concat(create_first_ObservedFeatureIndex)
    for j in range(6):
        feats, label = create_first_ObservedFeatureIndex.lookup(row=j, select_features=None)
        assert np.all(feats == [one_feats["feature_name"][j % 3], one_feats["feature_int"][j % 3]])
        assert label is None


def testObeservedFetureIndex_on_append_features_twice_lookup_out_of_bounds(create_first_ObservedFeatureIndex):
    create_first_ObservedFeatureIndex.concat(create_first_ObservedFeatureIndex)
    with pytest.raises(IndexError, match=re.escape("Row index 6 is larger than number of rows in FeatureIndex (6).")):
        create_first_ObservedFeatureIndex.lookup(row=6)


@pytest.mark.parametrize("index_cls", [VariableFeatureIndex, ObservedFeatureIndex])
def test_feature_index_lookup_empty(index_cls):
    index = index_cls()
    with pytest.raises(IndexError, match=r"There are no features to lookup"):
        index.lookup(row=1)


@pytest.mark.parametrize(
    "fixture_name",
    [
        "create_first_VariableFeatureIndex",
        "create_first_ObservedFeatureIndex",
    ],
)
def test_feature_index_lookup_negative(request, fixture_name):
    index = request.getfixturevalue(fixture_name)
    with pytest.raises(IndexError, match=r"Row index -1 is not valid. It must be non-negative."):
        index.lookup(row=-1)


@pytest.mark.parametrize(
    "fixture_name",
    [
        "create_first_VariableFeatureIndex",
        "create_first_ObservedFeatureIndex",
    ],
)
def test_feature_index_lookup_too_large(request, fixture_name):
    index = request.getfixturevalue(fixture_name)
    with pytest.raises(
        IndexError, match=re.escape("Row index 12544 is larger than number of rows in FeatureIndex (12).")
    ):
        index.lookup(row=12544)


@pytest.mark.parametrize(
    "first_fixture,second_fixture,index_cls",
    [
        ("create_first_VariableFeatureIndex", "create_second_VariableFeatureIndex", VariableFeatureIndex),
        ("create_first_ObservedFeatureIndex", "create_second_ObservedFeatureIndex", ObservedFeatureIndex),
    ],
)
def test_save_reload_FeatureIndex_identical(tmp_path, request, first_fixture, second_fixture, index_cls):
    index_one = request.getfixturevalue(first_fixture)
    index_two = request.getfixturevalue(second_fixture)
    index_one.concat(index_two)
    index_one.save(tmp_path / "features")
    index_reload = index_cls.load(tmp_path / "features")

    assert len(index_one) == len(index_reload)
    assert index_one.column_dims() == index_reload.column_dims()
    assert index_one.number_of_rows() == index_reload.number_of_rows()
    assert index_one.version() == index_reload.version()

    assert index_one.number_of_values() == index_reload.number_of_values()

    for row in range(index_one.number_of_rows()):
        features_one, labels_one = index_one.lookup(row=row, select_features=None)
        features_reload, labels_reload = index_reload.lookup(row=row, select_features=None)
        assert labels_one == labels_reload
        assert np.all(np.array(features_one, dtype=object) == np.array(features_reload, dtype=object))


def testObeservedFetureIndex_getitem_slice_contiguous_across_blocks():
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
    assert set(out[0].keys()) == set(df1_feats.keys())
    expected = [
        {
            "feature_name": df1_feats["feature_name"][1:],
            "feature_int": df1_feats["feature_int"][1:],
        },
        {
            "f_name": df2_feats["f_name"][:4],
            "f_int": df2_feats["f_int"][:4],
            "f_spare": df2_feats["f_spare"][:4],
        },
    ]
    for actual, exp in zip(out, expected):
        assert set(actual.keys()) == set(exp.keys())
        for k in exp:
            assert np.array_equal(actual[k], exp[k])


def testObeservedFetureIndex_getitem_slice_with_step_and_order_preserved():
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
    expected = [
        {"x": df1_feats["x"][::2]},
        {"y": df2_feats["y"][::2]},
    ]
    for actual, exp in zip(out, expected):
        assert set(actual.keys()) == set(exp.keys())
        for k in exp:
            assert np.array_equal(actual[k], exp[k])
