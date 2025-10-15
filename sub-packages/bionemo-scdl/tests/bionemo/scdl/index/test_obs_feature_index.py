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

import numpy as np
import pandas as pd
import pytest

from bionemo.scdl.index.row_feature_index import ObservedFeatureIndex


def test_appending_dataframe_results_in_error():
    two_feats = pd.DataFrame({"feature_name": ["FF", "GG"], "gene_name": ["RET", "NTRK"]})
    index = ObservedFeatureIndex()
    with pytest.raises(TypeError) as error_info:
        index.append_features(two_feats, "MY_DATAFRAME")
        assert "Expected a dictionary, but received a Pandas DataFrame." in str(error_info.value)


def test_append_features_mismatched_lengths_raises():
    index = ObservedFeatureIndex()
    bad_features = {
        "feature_name": np.array(["A", "B", "C"]),
        "feature_int": np.array([1, 2]),
    }
    with pytest.raises(ValueError, match="All feature arrays must have the same length"):
        index.append_features(bad_features, label="BAD")


def test_ObservedFeatureIndex_behaviour_on_empty_index(assert_index_state):
    index = ObservedFeatureIndex()
    assert_index_state(index, length=0, rows=0, col_widths=[], values=[0])
    with pytest.raises(IndexError, match=r"There are no features to lookup"):
        index.lookup(row=0)


def test_feature_lookup_negative(make_feat_dictionary):
    idx = ObservedFeatureIndex()
    num_rows = 3
    num_cols = 2
    idx.append_features(make_feat_dictionary(num_cols, num_rows), None)
    with pytest.raises(IndexError, match=r"Row index -1 is not valid. It must be non-negative."):
        idx.lookup(row=-1)


def test_feature_lookup_too_large(make_feat_dictionary):
    idx = ObservedFeatureIndex()
    num_rows = 2
    num_cols = 3
    idx.append_features(make_feat_dictionary(num_cols, num_rows), None)
    with pytest.raises(IndexError):
        idx.lookup(row=num_rows)


def test_select_feature_index_behavior(
    make_feat_dictionary,
):
    """lookup(select_features=...) returns requested columns in order; errors on missing names."""
    index = ObservedFeatureIndex()
    num_rows = 5
    num_cols = 3
    expected_label = None
    seed_features = make_feat_dictionary(num_cols, num_rows, key_prefix="c")
    index.append_features(seed_features, expected_label)

    feats, label = index.lookup(0, select_features=[])
    assert feats == []
    assert label == expected_label

    feats, label = index.lookup(0, select_features=list(seed_features.keys()))
    assert label == expected_label
    selected = {k: seed_features[k] for k in seed_features.keys()}

    assert np.array_equal(np.stack(feats), [l[0] for l in list(selected.values())])

    with pytest.raises(
        ValueError, match="Provided feature column does_not_exist in select_features not present in dataset."
    ):
        index.lookup(0, select_features=["does_not_exist"])  # missing feature name should raise


def test_concat_non_empty_with_empty_index_structure(make_feat_dictionary, assert_index_state):
    base = ObservedFeatureIndex()
    num_rows, num_cols = 3, 2
    base.append_features(make_feat_dictionary(num_cols, num_rows), None)
    empty = ObservedFeatureIndex()
    empty.append_features(make_feat_dictionary(0, 0), label="empty")
    base.concat(empty)
    assert_index_state(base, rows=num_rows, col_widths=[num_cols], values=[num_rows * num_cols], length=1)


def test_concat_two_indices_structure(make_feat_dictionary, assert_index_state):
    idx = ObservedFeatureIndex()
    num_rows_a, num_cols_a = 2, 2
    idx.append_features(make_feat_dictionary(num_cols_a, num_rows_a), label="A")
    other = ObservedFeatureIndex()
    num_rows_b, num_cols_b = 3, 3
    other.append_features(make_feat_dictionary(num_cols_b, num_rows_b), label="B")
    idx.concat(other)
    assert_index_state(idx, rows=num_rows_a + num_rows_b, col_widths=[num_cols_a, num_cols_b], length=2)


def test_concat_same_feature_index_twice_structure(make_feat_dictionary, assert_index_state):
    """
    Test that concatenating the same VariableFeatureIndex twice does not increase the number of index types,
    and doubles the number of rows, keeping feature column counts correct.
    """
    first_index = ObservedFeatureIndex()
    num_rows, num_cols = 4, 3
    seed_features = make_feat_dictionary(num_cols, num_rows)

    first_index.append_features(seed_features, None)
    first_index.concat(first_index)
    # Should still be a single feature type, not two
    assert_index_state(
        first_index, length=1, rows=2 * num_rows, col_widths=[num_cols], values=[2 * (num_rows * num_cols)]
    )


def test_concat_multiblock_source_adds_rows_correctly(make_feat_dictionary, assert_index_state):
    source = ObservedFeatureIndex()
    num_rows_a, num_cols_a = 3, 3

    feats_a = make_feat_dictionary(num_cols_a, num_rows_a)
    num_rows_b, num_cols_b = 4, 2
    feats_b = make_feat_dictionary(num_cols_b, num_rows_b)
    source.append_features(feats_a, label="A")
    source.append_features(feats_b, label="B")

    target = ObservedFeatureIndex()
    target.concat(source)
    assert_index_state(
        target,
        rows=num_rows_a + num_rows_b,
        col_widths=[num_cols_a, num_cols_b],
        values=[num_rows_a * num_cols_a, num_rows_b * num_cols_b],
        length=2,
    )


def test_concat_multiblock_number_vars_at_rows_correct_values(make_feat_dictionary):
    """Lookup returns correct features for each row, and number_vars_at_row reflects the block-specific feature col_widths across boundaries."""
    idx = ObservedFeatureIndex()
    num_rows_a = 3
    num_cols_a = 3
    feats_a = make_feat_dictionary(num_cols_a, num_rows_a)
    num_rows_b = 2
    num_cols_b = 2
    feats_b = make_feat_dictionary(num_cols_b, num_rows_b)
    idx.append_features(feats_a, label="A")
    idx.append_features(feats_b, label="B")
    # Rows in first block
    for r in range(0, num_rows_a):
        assert idx.number_vars_at_row(r) == num_cols_a
        feats, label = idx.lookup(row=r, select_features=None)
        assert np.array_equal(np.stack(feats), np.stack([l[r] for l in list(feats_a.values())]))
        assert label == "A"
    # Rows in second block
    for r in range(num_rows_a, num_rows_a + num_rows_b):
        assert idx.number_vars_at_row(r) == num_cols_b
        feats, label = idx.lookup(row=r, select_features=None)
        assert np.array_equal(np.stack(feats), np.stack([l[r - num_rows_a] for l in list(feats_b.values())]))
        assert label == "B"


def test_save_reload_row_ObservedFeatureIndex_same_feature_indices(tmp_path, make_feat_dictionary, assert_index_state):
    first_index = ObservedFeatureIndex()
    num_rows, num_cols = 3, 3
    first_index.append_features(make_feat_dictionary(num_cols, num_rows), None)
    first_index.concat(first_index)
    first_index.save(tmp_path / "features")
    index_reload = ObservedFeatureIndex.load(tmp_path / "features")
    assert_index_state(
        index_reload,
        length=len(first_index),
        col_widths=first_index.column_dims(),
        rows=first_index.number_of_rows(),
        values=first_index.number_of_values(),
    )
    assert first_index.version() == index_reload.version()

    for row in range(first_index.number_of_rows()):
        features_one, labels_one = first_index.lookup(row=row, select_features=None)
        features_reload, labels_reload = index_reload.lookup(row=row, select_features=None)
        assert labels_one == labels_reload
        assert np.all(np.array(features_one, dtype=object) == np.array(features_reload))


def test_ObservedFeatureIndex_getitem_int_returns_row_values_and_label(make_feat_dictionary):
    idx = ObservedFeatureIndex()
    n_obs = 3
    features = make_feat_dictionary(2, n_obs, key_prefix="feat_")
    idx.append_features(features, label="LBL")

    vals0, label0 = idx[0]
    assert label0 == "LBL"
    # The keys
    assert vals0 == [features["feat_0"][0], features["feat_1"][0]]

    vals_last, label_last = idx[-1]
    assert label_last == "LBL"
    assert vals_last == [features["feat_0"][-1], features["feat_1"][-1]]


def testObeservedFetureIndex_getitem_contiguous_slice(make_feat_dictionary):
    # First block: 3 rows, 2 columns (let's use width=3, num_cols=2)
    df1_feats = make_feat_dictionary(2, 3, key_prefix="feature_")
    # Second block: 5 rows, 3 columns (width=5, num_cols=3, prefix f_)
    df2_feats = make_feat_dictionary(3, 5, key_prefix="f_")
    obs = ObservedFeatureIndex()
    obs.append_features(df1_feats)
    obs.append_features(df2_feats, label="blk2")

    out, labels = obs[1:7]  # This should start at 1 in block 1, and continue into block 2

    assert labels == [None, "blk2"]
    assert isinstance(out, list)
    assert len(out) == 2
    assert set(out[0].keys()) == set(df1_feats.keys())
    # Prepare expected slices
    expected = [
        {k: v[1:] for k, v in df1_feats.items()},
        {k: v[:4] for k, v in df2_feats.items()},
    ]
    for actual, exp in zip(out, expected):
        assert set(actual.keys()) == set(exp.keys())
        for k in exp:
            assert np.array_equal(actual[k], exp[k])


def testObeservedFetureIndex_getitem_slice_with_step_and_order_preserved(make_feat_dictionary):
    # First block: 4 rows, 1 column, prefix x
    df1_feats = make_feat_dictionary(1, 4, key_prefix="x")
    # Second block: 3 rows, 1 column, prefix y
    df2_feats = make_feat_dictionary(1, 3, key_prefix="y")
    obs = ObservedFeatureIndex()
    obs.append_features(df1_feats)
    obs.append_features(df2_feats, label="b2")

    out, labels = obs[0:7:2]  # Step of 2

    assert labels == [None, "b2"]
    assert isinstance(out, list)
    assert len(out) == 2
    expected = [
        {k: v[::2] for k, v in df1_feats.items()},
        {k: v[::2] for k, v in df2_feats.items()},
    ]
    for actual, exp in zip(out, expected):
        assert set(actual.keys()) == set(exp.keys())
        for k in exp:
            assert np.array_equal(actual[k], exp[k])
