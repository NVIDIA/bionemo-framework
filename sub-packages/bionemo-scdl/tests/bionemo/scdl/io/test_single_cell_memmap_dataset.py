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

from typing import Tuple

import numpy as np
import os
import pytest

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset


first_array_values = [1, 2, 3, 4, 5]
second_array_values = [10, 9, 8, 7, 6, 5, 4, 3]


@pytest.fixture
def generate_dataset(tmp_path, test_directory) -> SingleCellMemMapDataset:
    """
    Create a SingleCellMemMapDataset, save and reload it

    Args:
        tmp_path: temporary directory fixture
    Returns:
        A SingleCellMemMapDataset
    """
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad")
    ds.save()
    del ds
    reloaded = SingleCellMemMapDataset(tmp_path / "scy")
    return reloaded


@pytest.fixture
def create_and_fill_mmap_arrays(tmp_path) -> Tuple[np.memmap, np.memmap]:
    """
    Instantiate and fill two np.memmap arrays.

    Args:
        tmp_path: temporary directory fixture
    Returns:
        Two instantiated np.memmap arrays.
    """
    arr1 = np.memmap(tmp_path / "x.npy", dtype="uint32", shape=(len(first_array_values),), mode="w+")
    arr1[:] = np.array(first_array_values, dtype="uint32")

    arr2 = np.memmap(tmp_path / "y.npy", dtype="uint32", shape=(len(second_array_values),), mode="w+")
    arr2[:] = np.array(second_array_values, dtype="uint32")
    return arr1, arr2


@pytest.fixture
def compare_fn():
    def _compare(dns: SingleCellMemMapDataset, dt: SingleCellMemMapDataset) -> bool:
        """
        Returns whether two SingleCellMemMapDatasets are equal

        Args:
            dns: SingleCellMemMapDataset
            dnt: SingleCellMemMapDataset
        Returns:
            True if these datasets are equal.
        """

        assert dns.number_of_rows() == dt.number_of_rows()
        assert dns.number_of_values() == dt.number_of_values()
        assert dns.number_nonzero_values() == dt.number_nonzero_values()
        assert dns.number_of_variables() == dt.number_of_variables()
        assert dns.number_of_rows() == dt.number_of_rows()
        for row_idx in range(len(dns)):
            assert (dns[row_idx][0] == dt[row_idx][0]).all()
            assert (dns[row_idx][1] == dt[row_idx][1]).all()

    return _compare


def test_empty_dataset_save_and_reload(tmp_path):
    ds = SingleCellMemMapDataset(data_path=tmp_path / "scy", num_rows=2, num_elements=10)
    ds.save()
    del ds
    reloaded = SingleCellMemMapDataset(tmp_path / "scy")
    assert reloaded.number_of_rows() == 0
    assert reloaded.number_of_variables() == [0]
    assert reloaded.number_of_values() == 0
    assert len(reloaded) == 0
    assert len(reloaded[1][0]) == 0


def test_wrong_arguments_for_dataset(tmp_path):
    with pytest.raises(
        ValueError, match=r"An np.memmap path, an h5ad path, or the number of elements and rows is required"
    ):
        SingleCellMemMapDataset(data_path=tmp_path / "scy")


def test_load_h5ad(tmp_path, test_directory):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad")
    assert ds.number_of_rows() == 8
    assert ds.number_of_variables() == [10]
    assert len(ds) == 8
    assert ds.number_of_values() == 80
    assert ds.number_nonzero_values() == 5
    np.isclose(ds.sparsity(), 0.9375, rtol=1e-6)
    assert len(ds) == 8


def test_h5ad_no_file(tmp_path):
    ds = SingleCellMemMapDataset(data_path=tmp_path / "scy", num_rows=2, num_elements=10)
    with pytest.raises(FileNotFoundError, match=rf"Error: could not find h5ad path {tmp_path}/a"):
        ds.load_h5ad(anndata_path=tmp_path / "a")


def test_SingleCellMemMapDataset_constructor(generate_dataset):
    assert generate_dataset.number_of_rows() == 8
    assert generate_dataset.number_of_variables() == [10]
    assert generate_dataset.number_of_values() == 80
    assert generate_dataset.number_nonzero_values() == 5
    assert np.isclose(generate_dataset.sparsity(), 0.9375, rtol=1e-6)
    assert len(generate_dataset) == 8

    assert generate_dataset.shape() == (8, [10])


def test_SingleCellMemMapDataset_get_row(generate_dataset):
    assert len(generate_dataset[0][0]) == 1
    vals, cols = generate_dataset[0]
    assert vals[0] == 6.0
    assert cols[0] == 2
    assert len(generate_dataset[1][1]) == 0
    assert len(generate_dataset[1][0]) == 0
    vals, cols = generate_dataset[2]
    assert vals[0] == 19.0
    assert cols[0] == 2
    vals, cols = generate_dataset[7]
    assert vals[0] == 1.0
    assert cols[0] == 8


def test_SingleCellMemMapDataset_get_row_colum(generate_dataset):
    assert generate_dataset.get_row_column(0, 0, impute_missing_zeros=True) == 0.0
    assert generate_dataset.get_row_column(0, 0, impute_missing_zeros=False) is None
    assert generate_dataset.get_row_column(0, 2) == 6.0
    assert generate_dataset.get_row_column(6, 3) == 16.0
    assert generate_dataset.get_row_column(3, 2) == 12.0


def test_SingleCellMemMapDataset_get_row_padded(generate_dataset):
    padded_row, feats = generate_dataset.get_row_padded(0, return_features=True, feature_vars=["feature_name"])
    assert len(padded_row) == 10
    assert padded_row[2] == 6.0
    assert len(feats[0]) == 10
    assert generate_dataset.get_row_padded(0)[0][0] == 0.0
    assert generate_dataset.data[0] == 6.0
    assert generate_dataset.data[1] == 19.0
    assert len(generate_dataset.get_row_padded(2)[0]) == 10


def test_concat_SingleCellMemMapDatasets_same(tmp_path, test_directory):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad")
    dt = SingleCellMemMapDataset(tmp_path / "sct", h5ad_path=test_directory / "adata_sample0.h5ad")
    dt.concat(ds)

    assert dt.number_of_rows() == 2 * ds.number_of_rows()
    assert dt.number_of_values() == 2 * ds.number_of_values()
    assert dt.number_nonzero_values() == 2 * ds.number_nonzero_values()


def test_concat_SingleCellMemMapDatasets_empty(tmp_path, test_directory):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad")
    exp_rows = np.array(ds.row_index)
    exp_cols = np.array(ds.col_index)
    exp_data = np.array(ds.data)

    ds.concat([])
    assert (np.array(ds.row_index) == exp_rows).all()
    assert (np.array(ds.col_index) == exp_cols).all()
    assert (np.array(ds.data) == exp_data).all()


@pytest.mark.parametrize("extend_copy_size", [1, 10 * 1_024 * 1_024])
def test_concat_SingleCellMemMapDatasets_underlying_memmaps(tmp_path, test_directory, extend_copy_size):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad")
    dt = SingleCellMemMapDataset(tmp_path / "sct", h5ad_path=test_directory / "adata_sample1.h5ad")
    exp_rows = np.append(dt.row_index, ds.row_index[1:] + len(dt.col_index))
    exp_cols = np.append(dt.col_index, ds.col_index)
    exp_data = np.append(dt.data, ds.data)

    dt.concat(ds, extend_copy_size)
    assert (np.array(dt.row_index) == exp_rows).all()
    assert (np.array(dt.col_index) == exp_cols).all()
    assert (np.array(dt.data) == exp_data).all()


def test_concat_SingleCellMemMapDatasets_diff(tmp_path, test_directory):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad")
    dt = SingleCellMemMapDataset(tmp_path / "sct", h5ad_path=test_directory / "adata_sample1.h5ad")

    exp_number_of_rows = ds.number_of_rows() + dt.number_of_rows()
    exp_n_val = ds.number_of_values() + dt.number_of_values()
    exp_nnz = ds.number_nonzero_values() + dt.number_nonzero_values()
    dt.concat(ds)
    assert dt.number_of_rows() == exp_number_of_rows
    assert dt.number_of_values() == exp_n_val
    assert dt.number_nonzero_values() == exp_nnz


def test_concat_SingleCellMemMapDatasets_multi(tmp_path, compare_fn, test_directory):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad")
    dt = SingleCellMemMapDataset(tmp_path / "sct", h5ad_path=test_directory / "adata_sample1.h5ad")
    dx = SingleCellMemMapDataset(tmp_path / "sccx", h5ad_path=test_directory / "adata_sample2.h5ad")
    exp_n_obs = ds.number_of_rows() + dt.number_of_rows() + dx.number_of_rows()
    dt.concat(ds)
    dt.concat(dx)
    assert dt.number_of_rows() == exp_n_obs
    dns = SingleCellMemMapDataset(tmp_path / "scdns", h5ad_path=test_directory / "adata_sample1.h5ad")
    dns.concat([ds, dx])
    compare_fn(dns, dt)


def test_lazy_load_SingleCellMemMapDatasets_one_dataset(tmp_path, compare_fn, test_directory):
    ds_regular = SingleCellMemMapDataset(tmp_path / "sc1", h5ad_path=test_directory / "adata_sample1.h5ad")
    ds_lazy = SingleCellMemMapDataset(
        tmp_path / "sc2",
        h5ad_path=test_directory / "adata_sample1.h5ad",
        paginated_load_cutoff=0,
        load_block_row_size=2,
    )
    compare_fn(ds_regular, ds_lazy)


def test_lazy_load_SingleCellMemMapDatasets_another_dataset(tmp_path, compare_fn, test_directory):
    ds_regular = SingleCellMemMapDataset(tmp_path / "sc1", h5ad_path=test_directory / "adata_sample0.h5ad")
    ds_lazy = SingleCellMemMapDataset(
        tmp_path / "sc2",
        h5ad_path=test_directory / "adata_sample0.h5ad",
        paginated_load_cutoff=0,
        load_block_row_size=3,
    )
    compare_fn(ds_regular, ds_lazy)


# Test creating a dataset with neighbor support
def test_create_dataset_with_neighbor_support(tmp_path):
    # Create a simple dataset with neighbor support
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scnn",
        num_rows=5,
        num_elements=10,
        load_neighbors=True,
        neighbor_key='next_cell_ids',
        neighbor_sampling_strategy='random',
        fallback_to_identity=True
    )

    # Verify neighbor configuration
    assert ds.load_neighbors is True
    assert ds.neighbor_key == 'next_cell_ids'
    assert ds.neighbor_sampling_strategy == 'random'
    assert ds.fallback_to_identity is True
    assert ds._has_neighbors is False  # No neighbors loaded yet

def test_empty_dataset_save_and_reload_with_neighbors(tmp_path):
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scnn",
        num_rows=2,
        num_elements=10,
        load_neighbors=True,
        neighbor_key='next_cell_ids',
        neighbor_sampling_strategy='random',
        fallback_to_identity=True
    )
    ds.save()
    del ds
    reloaded = SingleCellMemMapDataset(
        tmp_path / "scnn",
        load_neighbors=True,
        neighbor_key='next_cell_ids',
        neighbor_sampling_strategy='random',
        fallback_to_identity=True
    )
    assert reloaded.number_of_rows() == 0
    assert reloaded.number_of_variables() == [0]
    assert reloaded.number_of_values() == 0
    assert len(reloaded) == 0
    assert len(reloaded[1][0]) == 0
    # Test neighbor configuration is preserved
    assert reloaded.load_neighbors is True
    assert reloaded.neighbor_key == 'next_cell_ids'
    assert reloaded.neighbor_sampling_strategy == 'random'
    assert reloaded.fallback_to_identity is True
    assert reloaded._has_neighbors is False  # No neighbors loaded for empty dataset

def test_neighbor_matrix_extraction(tmp_path, test_neighbor_directory):
    # Use the NGC sample neighbor dataset
    sample_h5ad_path = test_neighbor_directory / "adata_sample0_neighbors.h5ad"

    # Create dataset with neighbors using the NGC sample file
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scnn",
        h5ad_path=sample_h5ad_path,
        load_neighbors=True,
        neighbor_key='next_cell_ids',
        neighbor_sampling_strategy='random',
        fallback_to_identity=True
    )

    # Test that neighbor data was extracted
    assert ds._has_neighbors is True
    assert ds._neighbor_indptr is not None
    assert ds._neighbor_indices is not None
    assert ds._neighbor_data is not None

    # Test basic properties of the neighbor data
    assert ds.number_of_rows() == 8
    assert len(ds._neighbor_indices) == 29  # 29 nonzero entries
    assert len(ds._neighbor_indptr) == 9    # 8 cells + 1 (CSR format)
    assert len(ds._neighbor_data) == 29     # 29 nonzero values

    # Test that the neighbor matrix structure is valid (CSR format)
    # indptr should be monotonically increasing
    assert all(ds._neighbor_indptr[i] <= ds._neighbor_indptr[i+1] for i in range(len(ds._neighbor_indptr)-1))

    # All indices should be valid cell indices (0 to 7)
    assert all(0 <= idx < 8 for idx in ds._neighbor_indices)

    # All data values should be positive (pseudotime values)
    assert all(val > 0 for val in ds._neighbor_data)

def test_sample_neighbor_index(tmp_path, monkeypatch, test_neighbor_directory):
    """Test neighbor index sampling using real sample data."""

    # Path to the NGC sample neighbor data
    sample_neighbor_file = test_neighbor_directory / "adata_sample0_neighbors.h5ad"

    # Create dataset with real neighbor data
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scn",
        h5ad_path=sample_neighbor_file,
        load_neighbors=True,
        neighbor_key='next_cell_ids',
        neighbor_sampling_strategy='random',
        fallback_to_identity=True
    )

    # Mock numpy's random choice to make sampling deterministic
    def mock_choice(arr, p=None):
        # Always return the first element for predictable testing
        return arr[0]

    monkeypatch.setattr(np.random, 'choice', mock_choice)

    # Test sampling for cells that have neighbors
    for cell_idx in range(ds.number_of_rows()):
        start_idx = ds._neighbor_indptr[cell_idx]
        end_idx = ds._neighbor_indptr[cell_idx + 1]

        if start_idx < end_idx:  # Cell has neighbors
            # Get the expected neighbor (first one due to our mock)
            expected_neighbor = ds._neighbor_indices[start_idx]
            sampled_neighbor = ds.sample_neighbor_index(cell_idx)
            assert sampled_neighbor == expected_neighbor, f"Cell {cell_idx} should sample neighbor {expected_neighbor}, got {sampled_neighbor}"

    # Test fallback behavior for cell 0 which has no neighbors
    cell_idx = 0
    sampled_neighbor = ds.sample_neighbor_index(cell_idx)
    assert sampled_neighbor == cell_idx, f"Cell {cell_idx} with no neighbors should return itself, got {sampled_neighbor}"

    # Test that sampling respects the probability distribution when using weighted sampling
    # Reset to use actual random sampling (remove mock)
    monkeypatch.undo()

    # Sample multiple times from a cell with neighbors to ensure randomness works
    cell_with_neighbors = None
    for cell_idx in range(ds.number_of_rows()):
        start_idx = ds._neighbor_indptr[cell_idx]
        end_idx = ds._neighbor_indptr[cell_idx + 1]
        if end_idx - start_idx > 1:  # Cell has multiple neighbors
            cell_with_neighbors = cell_idx
            break

    if cell_with_neighbors is not None:
        # Sample multiple times and ensure we get valid neighbors
        samples = []
        for _ in range(10):
            neighbor = ds.sample_neighbor_index(cell_with_neighbors)
            samples.append(neighbor)
            # Verify the sampled neighbor is valid
            start_idx = ds._neighbor_indptr[cell_with_neighbors]
            end_idx = ds._neighbor_indptr[cell_with_neighbors + 1]
            valid_neighbors = ds._neighbor_indices[start_idx:end_idx]
            assert neighbor in valid_neighbors, f"Sampled neighbor {neighbor} not in valid neighbors {valid_neighbors}"

def test_get_row_with_neighbor(tmp_path, monkeypatch, test_neighbor_directory):
    """Test get_row_with_neighbor using real sample data."""

    # Path to the NGC sample neighbor data
    sample_neighbor_file = test_neighbor_directory / "adata_sample0_neighbors.h5ad"

    # Create dataset with real neighbor data
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scnn",
        h5ad_path=sample_neighbor_file,
        load_neighbors=True,
        neighbor_key='next_cell_ids',
        neighbor_sampling_strategy='random',
        fallback_to_identity=True
    )

    # Verify neighbors are loaded
    assert ds._has_neighbors is True

    # Mock sample_neighbor_index to return predictable neighbors for testing
    def mock_sample_neighbor(idx):
        if idx == 0:
            return 2  # Cell 0's neighbor is cell 2 (both have data)
        elif idx == 2:
            return 0  # Cell 2's neighbor is cell 0 (both have data)
        else:
            return idx  # Fallback to self for other cells
        
    # Use monkeypatch to mock the method properly
    monkeypatch.setattr(ds, 'sample_neighbor_index', mock_sample_neighbor)

    # Test get_row_with_neighbor with neighbor (cell 0 -> cell 2)
    result = ds.get_row_with_neighbor(0, include_neighbor=True)

    # Validate structure and content
    assert isinstance(result, dict)
    assert set(result.keys()) == {'current_cell', 'next_cell', 'current_cell_index', 'next_cell_index', 'features'}
    assert result['current_cell_index'] == 0
    assert result['next_cell_index'] == 2

    # Test cell data structure (should be tuples of (values, indices))
    current_values, current_cols = result['current_cell']
    next_values, next_cols = result['next_cell']

    # Verify that we get actual data from the real dataset
    assert isinstance(current_values, np.ndarray)
    assert isinstance(current_cols, np.ndarray)
    assert isinstance(next_values, np.ndarray)
    assert isinstance(next_cols, np.ndarray)

    # Verify that the data is non-empty (cells should have some gene expression)
    assert len(current_values) > 0, "Current cell should have some gene expression data"
    assert len(next_values) > 0, "Next cell should have some gene expression data"
    assert len(current_values) == len(current_cols), "Values and columns should have same length"
    assert len(next_values) == len(next_cols), "Values and columns should have same length"

    # Verify the actual values match what we expect from existing tests
    assert current_values[0] == 6.0, f"Expected cell 0 to have value 6.0, got {current_values[0]}"
    assert current_cols[0] == 2, f"Expected cell 0 to have column 2, got {current_cols[0]}"
    assert next_values[0] == 19.0, f"Expected cell 2 to have value 19.0, got {next_values[0]}"
    assert next_cols[0] == 2, f"Expected cell 2 to have column 2, got {next_cols[0]}"

    # Test get_row_with_neighbor without neighbor (should return just the current cell)
    result_no_neighbor = ds.get_row_with_neighbor(0, include_neighbor=False)
    assert isinstance(result_no_neighbor, tuple)
    assert len(result_no_neighbor) == 2

    # The result should be the same as the current_cell from the with-neighbor case
    values_only, features_only = result_no_neighbor
    current_values_only, current_cols_only = values_only
    assert np.array_equal(current_values_only, current_values)
    assert np.array_equal(current_cols_only, current_cols)

    # Test with cell 1 which has no gene expression data (should handle gracefully)
    result_empty = ds.get_row_with_neighbor(1, include_neighbor=True)
    assert result_empty['current_cell_index'] == 1
    assert result_empty['next_cell_index'] == 1  # Should fallback to itself

def test_get_row_padded_with_neighbor(tmp_path, monkeypatch, test_neighbor_directory):
    """Test get_row_padded_with_neighbor using real sample data."""

    # Path to the NGC sample neighbor data
    sample_neighbor_file = test_neighbor_directory / "adata_sample0_neighbors.h5ad"

    # Create dataset with real neighbor data
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scnn",
        h5ad_path=sample_neighbor_file,
        load_neighbors=True,
        neighbor_key='next_cell_ids',
        neighbor_sampling_strategy='random',
        fallback_to_identity=True
    )

    # Verify neighbors are loaded
    assert ds._has_neighbors is True

    # Mock sample_neighbor_index to return predictable neighbors for testing
    def mock_sample_neighbor(cell_index):
        if cell_index == 0:
            return 2  # Cell 0's neighbor is cell 2 (both have data)
        elif cell_index == 2:
            return 0  # Cell 2's neighbor is cell 0 (both have data)
        else:
            return cell_index  # Fallback to self for other cells

    # Use monkeypatch to mock the method properly
    monkeypatch.setattr(ds, 'sample_neighbor_index', mock_sample_neighbor)

    # Test get_row_padded_with_neighbor with neighbor (cell 0 -> cell 2)
    result = ds.get_row_padded_with_neighbor(0, include_neighbor=True)

    # Validate structure and content
    assert isinstance(result, dict)
    assert set(result.keys()) == {'current_cell', 'next_cell', 'current_cell_index', 'next_cell_index', 'features'}
    assert result['current_cell_index'] == 0
    assert result['next_cell_index'] == 2

    # Test padded data (should be dense arrays with zeros for missing values)
    current_padded = result['current_cell']
    next_padded = result['next_cell']

    # Verify that we get dense numpy arrays
    assert isinstance(current_padded, np.ndarray)
    assert isinstance(next_padded, np.ndarray)

    # Both should have the same length (number of features/genes)
    assert len(current_padded) == len(next_padded)
    assert len(current_padded) == 10  # We know our sample data has 10 features

    # Verify the actual values match what we expect from existing tests
    # Cell 0 has value 6.0 at column 2, so current_padded[2] should be 6.0
    assert current_padded[2] == 6.0, f"Expected cell 0 to have value 6.0 at index 2, got {current_padded[2]}"
    # Cell 2 has value 19.0 at column 2, so next_padded[2] should be 19.0
    assert next_padded[2] == 19.0, f"Expected cell 2 to have value 19.0 at index 2, got {next_padded[2]}"

    # All other positions should be 0.0 (since data is sparse)
    for i in range(10):
        if i != 2:  # Skip the non-zero position
            assert current_padded[i] == 0.0, f"Expected cell 0 to have 0.0 at index {i}, got {current_padded[i]}"
            assert next_padded[i] == 0.0, f"Expected cell 2 to have 0.0 at index {i}, got {next_padded[i]}"

    # Test get_row_padded_with_neighbor without neighbor (should return just the current cell)
    result_no_neighbor = ds.get_row_padded_with_neighbor(0, include_neighbor=False)
    assert isinstance(result_no_neighbor, tuple)
    assert len(result_no_neighbor) == 2

    # The result should be the same as the current_cell from the with-neighbor case
    padded_only, features_only = result_no_neighbor
    assert np.array_equal(padded_only, current_padded)

def test_get_neighbor_stats(tmp_path, test_neighbor_directory):

    # Path to the NGC sample neighbor data
    sample_neighbor_file = test_neighbor_directory / "adata_sample0_neighbors.h5ad"

    # Create dataset with real neighbor data
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scn",
        h5ad_path=sample_neighbor_file,
        load_neighbors=True,
        neighbor_key='next_cell_ids',
        neighbor_sampling_strategy='random',
        fallback_to_identity=True
    )

    # Verify neighbors are loaded
    assert ds._has_neighbors is True

    # Get and check stats using real neighbor data
    stats = ds.get_neighbor_stats()

    # Validate the structure of the stats dictionary
    expected_keys = {
        "has_neighbors", 
        "total_connections", 
        "min_neighbors_per_cell", 
        "max_neighbors_per_cell", 
        "avg_neighbors_per_cell", 
        "cells_with_no_neighbors"
    }
    assert set(stats.keys()) == expected_keys

    # Test basic properties with real data
    assert stats["has_neighbors"] is True
    assert isinstance(stats["total_connections"], int)
    assert isinstance(stats["min_neighbors_per_cell"], int)
    assert isinstance(stats["max_neighbors_per_cell"], int)
    assert isinstance(stats["avg_neighbors_per_cell"], float)
    assert isinstance(stats["cells_with_no_neighbors"], int)

    # Validate logical constraints
    assert stats["total_connections"] >= 0
    assert stats["min_neighbors_per_cell"] >= 0
    assert stats["max_neighbors_per_cell"] >= stats["min_neighbors_per_cell"]
    assert stats["cells_with_no_neighbors"] >= 0
    assert stats["cells_with_no_neighbors"] <= ds.number_of_rows()
    assert stats["avg_neighbors_per_cell"] >= 0

    # Based on our known real data properties (from previous tests)
    # We know our sample has 8 cells and 29 total connections
    assert ds.number_of_rows() == 8
    assert stats["total_connections"] == 29

    # Calculate expected average: 29 connections / 8 cells = 3.625
    expected_avg = 29.0 / 8.0
    assert abs(stats["avg_neighbors_per_cell"] - expected_avg) < 1e-6

    # Test that the maximum is reasonable (shouldn't exceed total cells - 1)
    assert stats["max_neighbors_per_cell"] <= 7  # Can't have more neighbors than other cells

    # Verify that cells with no neighbors count makes sense
    # (should be <= total number of cells)
    assert 0 <= stats["cells_with_no_neighbors"] <= 8

    # Test individual cell neighbor counts to validate stats
    neighbor_counts = []
    for cell_idx in range(ds.number_of_rows()):
        neighbors = ds.get_neighbor_indices_for_cell(cell_idx)
        neighbor_counts.append(len(neighbors))

    # Validate that computed stats match individual cell data
    assert min(neighbor_counts) == stats["min_neighbors_per_cell"]
    assert max(neighbor_counts) == stats["max_neighbors_per_cell"]
    assert sum(neighbor_counts) == stats["total_connections"]
    assert neighbor_counts.count(0) == stats["cells_with_no_neighbors"]

    # Test case with neighbors disabled (create a new dataset without neighbors)
    ds_no_neighbors = SingleCellMemMapDataset(
        data_path=tmp_path / "scn_no_neighbors",
        h5ad_path=sample_neighbor_file,
        load_neighbors=False,  # Disable neighbor loading
        neighbor_key='next_cell_ids',
        neighbor_sampling_strategy='random',
        fallback_to_identity=True
    )

    # Verify no neighbors were loaded
    assert ds_no_neighbors._has_neighbors is False

    # Get stats for dataset without neighbors
    stats_no_neighbors = ds_no_neighbors.get_neighbor_stats()
    assert stats_no_neighbors == {"has_neighbors": False}