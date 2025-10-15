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

from bionemo.scdl.index.row_feature_index import VariableFeatureIndex


#  Testing VariableFeatureIndex functionality
def test_appending_dataframe_results_in_error():
    two_feats = pd.DataFrame({"feature_name": ["FF", "GG"], "gene_name": ["RET", "NTRK"]})
    index = VariableFeatureIndex()
    with pytest.raises(TypeError) as error_info:
        index.append_features(8, two_feats, "MY_DATAFRAME")
        assert "Expected a dictionary, but received a Pandas DataFrame." in str(error_info.value)


def test_append_features_mismatched_lengths_raises():
    index = VariableFeatureIndex()
    bad_features = {
        "feature_name": np.array(["A", "B", "C"]),
        "feature_int": np.array([1, 2]),
    }
    with pytest.raises(ValueError, match="All feature arrays must have the same length"):
        index.append_features(5, bad_features, label="BAD")


def test_VariableFeatureIndex_behaviour_on_empty_index(assert_index_state):
    index = VariableFeatureIndex()
    assert_index_state(index, length=0, rows=0, col_widths=[], values=[0])
    with pytest.raises(IndexError, match=r"There are no features to lookup"):
        index.lookup(row=0)


def test_feature_lookup_negative(make_feat_dictionary):
    idx = VariableFeatureIndex()
    num_rows = 2
    cols = 3
    col_widths = 2
    idx.append_features(num_rows, make_feat_dictionary(cols, col_widths), None)
    with pytest.raises(IndexError, match=r"Row index -1 is not valid. It must be non-negative."):
        idx.lookup(row=-1)


def test_feature_lookup_too_large(make_feat_dictionary):
    idx = VariableFeatureIndex()
    num_rows = 2
    cols = 3
    col_widths = 2
    idx.append_features(num_rows, make_feat_dictionary(cols, col_widths), None)
    with pytest.raises(IndexError):
        idx.lookup(row=num_rows)


def test_select_features_behavior(make_feat_dictionary):
    """lookup(select_features=...) returns requested columns in order; errors on missing names."""
    index = VariableFeatureIndex()
    num_rows = 5
    cols = 3
    col_widths = 2
    seed_features = make_feat_dictionary(cols, col_widths, key_prefix="c")
    index.append_features(num_rows, seed_features, expected_label := None)

    feats, label = index.lookup(0, select_features=[])
    assert feats == []
    assert label == expected_label

    feats, label = index.lookup(0, select_features=list(seed_features.keys()))
    assert label == expected_label
    selected = {k: seed_features[k] for k in seed_features.keys()}
    assert np.array_equal(np.stack(feats), np.stack(list(selected.values())))

    with pytest.raises(
        ValueError, match="Provided feature column does_not_exist in select_features not present in dataset."
    ):
        index.lookup(0, select_features=["does_not_exist"])  # missing feature name should raise


def test_concat_non_empty_with_empty_index_structure(make_feat_dictionary, assert_index_state):
    base = VariableFeatureIndex()
    num_rows, cols, col_widths = 3, 3, 2
    base.append_features(num_rows, make_feat_dictionary(cols, col_widths), None)
    empty = VariableFeatureIndex()
    empty_num_rows = 4
    empty.append_features(empty_num_rows, make_feat_dictionary(0, 0), label="empty")
    base.concat(empty)
    assert_index_state(
        base, rows=empty_num_rows + num_rows, col_widths=[col_widths, 0], values=[num_rows * col_widths, 0], length=2
    )


def test_concat_two_blocks_structure(make_feat_dictionary, assert_index_state):
    idx = VariableFeatureIndex()
    num_rows_a, cols_a, col_widths_a = 2, 2, 2
    idx.append_features(num_rows_a, make_feat_dictionary(cols_a, col_widths_a), label="A")
    other = VariableFeatureIndex()
    num_rows_b, cols_b, col_widths_b = 3, 3, 3
    other.append_features(num_rows_b, make_feat_dictionary(cols_b, col_widths_b), label="B")
    idx.concat(other)
    assert_index_state(idx, rows=num_rows_a + num_rows_b, col_widths=[cols_a, cols_b], length=2)


def test_concat_same_feature_index_twice_structure(make_feat_dictionary, assert_index_state):
    """
    Test that concatenating the same VariableFeatureIndex twice does not increase the number of index types,
    and doubles the number of rows, keeping feature column counts correct.
    """
    first_index = VariableFeatureIndex()
    num_rows, cols, col_widths = 4, 3, 2
    seed_features = make_feat_dictionary(cols, col_widths)

    first_index.append_features(num_rows, seed_features, None)
    first_index.concat(first_index)
    # Should still be a single feature type, not two
    assert_index_state(
        first_index, length=1, rows=2 * num_rows, col_widths=[col_widths], values=[2 * (num_rows * col_widths)]
    )


def test_concat_multiblock_source_adds_rows_correctly(make_feat_dictionary, assert_index_state):
    source = VariableFeatureIndex()
    num_rows_a = 3
    cols_a = 3
    col_widths_a = 3
    feats_a = make_feat_dictionary(cols_a, col_widths_a)
    num_rows_b = 4
    cols_b = 2
    col_widths_b = 2
    feats_b = make_feat_dictionary(cols_b, col_widths_b)
    source.append_features(num_rows_a, feats_a, label="A")
    source.append_features(num_rows_b, feats_b, label="B")

    target = VariableFeatureIndex()
    target.concat(source)
    assert_index_state(
        target,
        rows=7,
        col_widths=[col_widths_a, col_widths_b],
        values=[num_rows_a * col_widths_a, num_rows_b * col_widths_b],
        length=2,
    )


def test_concat_multiblock_number_vars_at_rows_correct_values(make_feat_dictionary):
    """Lookup returns correct features for each row, and number_vars_at_row reflects the block-specific feature col_widths across boundaries."""
    idx = VariableFeatureIndex()
    num_rows_a = 3
    cols_a = 3
    col_widths_a = 3
    feats_a = make_feat_dictionary(cols_a, col_widths_a)
    num_rows_b = 2
    cols_b = 2
    col_widths_b = 2
    feats_b = make_feat_dictionary(cols_b, col_widths_b)
    idx.append_features(num_rows_a, feats_a, label="A")
    idx.append_features(num_rows_b, feats_b, label="B")

    # Rows in first block
    for r in range(0, num_rows_a):
        assert idx.number_vars_at_row(r) == col_widths_a
        feats, label = idx.lookup(row=r, select_features=None)
        assert np.array_equal(np.stack(feats), np.stack(list(feats_a.values())))
        assert label == "A"
    # Rows in second block
    for r in range(num_rows_a, num_rows_a + num_rows_b):
        assert idx.number_vars_at_row(r) == col_widths_b
        feats, label = idx.lookup(row=r, select_features=None)
        assert np.array_equal(np.stack(feats), np.stack(list(feats_b.values())))
        assert label == "B"


def test_save_reload_row_VariableFeatureIndex_same_feature_indices(tmp_path, make_feat_dictionary, assert_index_state):
    first_index = VariableFeatureIndex()
    num_rows, cols, col_widths = 3, 3, 2
    first_index.append_features(num_rows, make_feat_dictionary(cols, col_widths), None)
    first_index.concat(first_index)
    first_index.save(tmp_path / "features")
    index_reload = VariableFeatureIndex.load(tmp_path / "features")
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
