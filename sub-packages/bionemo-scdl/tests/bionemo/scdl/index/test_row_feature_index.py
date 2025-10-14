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

from bionemo.scdl.index.row_feature_index import VariableFeatureIndex, are_dicts_equal


# Testing dictionary equality function
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


def test_appending_dataframe_results_in_error():
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
        assert "Expected a dictionary, but received a Pandas DataFrame." in str(error_info.value)


def test_VariableFeatureIndex_internals_on_empty_index():
    index = VariableFeatureIndex()
    assert len(index) == 0
    assert index.number_of_rows() == 0
    with pytest.raises(IndexError, match=r"There are no features to lookup"):
        index.lookup(row=0)


def test_feature_lookup_negative(create_first_VariableFeatureIndex):
    first_index, _, _, _ = create_first_VariableFeatureIndex
    with pytest.raises(IndexError, match=r"Row index -1 is not valid. It must be non-negative."):
        first_index.lookup(row=-1)


def test_feature_lookup_too_large(create_first_VariableFeatureIndex):
    first_index, _, num_rows, _ = create_first_VariableFeatureIndex
    lookup_row = num_rows + 1
    with pytest.raises(
        IndexError,
        match=re.escape(
            f"Row index {lookup_row} is larger than number of rows in FeatureIndex ({first_index.number_of_rows()})."
        ),
    ):
        first_index.lookup(row=lookup_row)


def test_select_features_behavior(create_second_VariableFeatureIndex):
    index, seed_features, _, expected_label = create_second_VariableFeatureIndex

    feats, label = index.lookup(0, select_features=[])
    assert feats == []
    assert label == expected_label

    feats, label = index.lookup(0, select_features=seed_features.keys())
    assert label == expected_label
    selected = {k: seed_features[k] for k in seed_features.keys()}
    assert np.all(feats == np.stack(list(selected.values())))

    with pytest.raises(
        ValueError, match="Provided feature column does_not_exist in select_features not present in dataset."
    ):
        index.lookup(0, select_features=["does_not_exist"])  # missing feature name should raise


def test_concat_empty_index_correct_length(create_first_VariableFeatureIndex, create_empty_VariableFeatureIndex):
    """
    After concatenating an index with empty features, index should have two sets of features.
    """
    first_index, _, _, _ = create_first_VariableFeatureIndex
    empty_index, _ = create_empty_VariableFeatureIndex
    first_index.concat(empty_index)
    assert len(first_index) == 2


def test_concat_empty_index_column_dims(create_first_VariableFeatureIndex, create_empty_VariableFeatureIndex):
    """
    The column_dims after concatenating empty features should be [original_cols, 0].
    """
    first_index, _, _, _ = create_first_VariableFeatureIndex
    original_num_cols = first_index.column_dims()[0]
    empty_index, _ = create_empty_VariableFeatureIndex
    first_index.concat(empty_index)
    assert first_index.column_dims() == [original_num_cols, 0]


def test_concat_empty_index_row_count(create_first_VariableFeatureIndex, create_empty_VariableFeatureIndex):
    """
    number_of_rows() should be updated by the number of empty rows appended.
    """
    first_index, _, num_rows, _ = create_first_VariableFeatureIndex
    empty_index, num_empty_rows = create_empty_VariableFeatureIndex
    first_index.concat(empty_index)
    assert first_index.number_of_rows() == num_rows + num_empty_rows


def test_concat_empty_index_value_counts(create_first_VariableFeatureIndex, create_empty_VariableFeatureIndex):
    """
    number_of_values() after concatenation of empty should match expected [original*original_cols, 0].
    """
    first_index, _, num_rows, _ = create_first_VariableFeatureIndex
    original_num_cols = first_index.column_dims()[0]
    empty_index, _ = create_empty_VariableFeatureIndex
    first_index.concat(empty_index)
    vals = first_index.number_of_values()
    assert vals == [num_rows * original_num_cols, 0]


def test_concat_different_feature_indices_structure(
    create_first_VariableFeatureIndex, create_second_VariableFeatureIndex
):
    """
    Test that concatenating two different VariableFeatureIndices results in a new index with the correct structure.
    """
    # Get the actual feature values directly from the test indices
    first_index, _, first_num_rows, _ = create_first_VariableFeatureIndex
    second_index, _, second_num_rows, _ = create_second_VariableFeatureIndex
    first_columns = first_index.column_dims()[0]
    second_columns = second_index.column_dims()[0]
    first_index.concat(second_index)
    assert len(first_index) == 2

    # Column dimension unchanged
    assert first_index.column_dims() == [first_columns, second_columns]
    # Row count doubled
    assert first_index.number_of_rows() == first_num_rows + second_num_rows
    # Number of values doubled
    assert first_index.number_of_values() == [(first_num_rows * first_columns), (second_num_rows * second_columns)]


def test_concat_different_feature_indices_number_vars(
    create_first_VariableFeatureIndex, create_second_VariableFeatureIndex
):
    """
    After concatenation, every row in index should still return the same number of variables.
    """
    first_index, _, first_num_rows, _ = create_first_VariableFeatureIndex
    second_index, _, second_num_rows, _ = create_second_VariableFeatureIndex
    original_num_cols = first_index.column_dims()[0]
    second_num_cols = second_index.column_dims()[0]
    first_index.concat(second_index)
    for row_index in range(first_num_rows):
        assert first_index.number_vars_at_row(row_index) == original_num_cols
    for row_index in range(first_num_rows, first_index.number_of_rows()):
        assert first_index.number_vars_at_row(row_index) == second_num_cols


def test_concat_different_feature_indices_correct_feature_values(
    create_first_VariableFeatureIndex, create_second_VariableFeatureIndex
):
    """
    Features and labels should be identical for every row before and after the concatenation.
    """
    first_index, first_seed_features, first_num_rows, label = create_first_VariableFeatureIndex
    second_index, second_seed_features, _, second_label = create_second_VariableFeatureIndex
    first_index.concat(second_index)

    # Check first set of rows (before concat)
    for row_index in range(len(first_index)):
        feats, label = first_index.lookup(row=row_index, select_features=None)
        assert np.all(feats == np.stack(list(first_seed_features.values())))
        assert label == label

    # Check second set of rows (after concat)
    for row_index in range(first_num_rows, first_index.number_of_rows()):
        feats, label = first_index.lookup(row=row_index, select_features=None)
        assert np.all(feats == np.stack(list(second_seed_features.values())))
        assert label == second_label


def test_concat_same_feature_index_twice_structure(create_first_VariableFeatureIndex):
    """
    Test that concatenating the same VariableFeatureIndex twice does not increase the number of index types,
    and doubles the number of rows, keeping feature column counts correct.
    """
    first_index, _, num_rows, _ = create_first_VariableFeatureIndex
    original_num_cols = first_index.column_dims()[0]
    original_num_rows = num_rows
    first_index.concat(first_index)
    # Should still be a single feature type, not two
    assert len(first_index) == 1
    # Column dimension unchanged
    assert first_index.column_dims() == [original_num_cols]
    # Row count doubled
    assert first_index.number_of_rows() == 2 * original_num_rows
    # Number of values doubled
    assert first_index.number_of_values() == [2 * (original_num_rows * original_num_cols)]


def test_save_reload_row_VariableFeatureIndex_same_feature_indices(
    tmp_path, create_first_VariableFeatureIndex, create_second_VariableFeatureIndex
):
    first_index, _, _, _ = create_first_VariableFeatureIndex
    second_index, _, _, _ = create_second_VariableFeatureIndex
    first_index.concat(second_index)
    first_index.save(tmp_path / "features")
    index_reload = VariableFeatureIndex.load(tmp_path / "features")
    assert len(first_index) == len(index_reload)
    assert first_index.column_dims() == index_reload.column_dims()
    assert first_index.number_of_rows() == index_reload.number_of_rows()
    assert first_index.version() == index_reload.version()

    assert first_index.number_of_values() == index_reload.number_of_values()

    for row in range(first_index.number_of_rows()):
        features_one, labels_one = first_index.lookup(row=row, select_features=None)
        features_reload, labels_reload = index_reload.lookup(row=row, select_features=None)
        assert labels_one == labels_reload
        assert np.all(np.array(features_one, dtype=object) == np.array(features_reload))


def test_concat_multiblock_source_adds_rows_correctly():
    source = VariableFeatureIndex()
    feats_a = {"x": np.array([1, 2, 3])}
    feats_b = {"x": np.array([10, 20])}
    source.append_features(3, feats_a, label="A")
    source.append_features(4, feats_b, label="B")

    target = VariableFeatureIndex()
    target.concat(source)

    assert target.number_of_rows() == 7
    assert len(target) == 2
    assert target.column_dims() == [len(feats_a["x"]), len(feats_b["x"])]
