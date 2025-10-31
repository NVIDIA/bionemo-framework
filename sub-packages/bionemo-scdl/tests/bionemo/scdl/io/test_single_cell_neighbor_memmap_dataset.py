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
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad", use_X_not_raw=True)
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


# Test creating a dataset with neighbor support
def test_create_dataset_with_neighbor_support(tmp_path):
    # Create a simple dataset with neighbor support
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scnn",
        num_rows=5,
        num_elements=10,
        load_neighbors=True,
        neighbor_key="next_cell_ids",
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
        use_X_not_raw=True,
    )

    # Verify neighbor configuration
    assert ds.load_neighbors is True
    assert ds.neighbor_key == "next_cell_ids"
    assert ds.neighbor_sampling_strategy == "random"
    assert ds.fallback_to_identity is True
    assert ds._has_neighbors is False  # No neighbors loaded yet


def test_empty_dataset_save_and_reload_with_neighbors(tmp_path):
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scnn",
        num_rows=2,
        num_elements=10,
        load_neighbors=True,
        neighbor_key="next_cell_ids",
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
        use_X_not_raw=True,
    )
    ds.save()
    del ds
    reloaded = SingleCellMemMapDataset(
        tmp_path / "scnn",
        load_neighbors=True,
        neighbor_key="next_cell_ids",
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
    )
    assert reloaded.number_of_rows() == 0
    assert reloaded.number_of_variables() == [0]
    assert reloaded.number_of_values() == 0
    assert len(reloaded) == 0
    assert len(reloaded[1][0]) == 0
    # Test neighbor configuration is preserved
    assert reloaded.load_neighbors is True
    assert reloaded.neighbor_key == "next_cell_ids"
    assert reloaded.neighbor_sampling_strategy == "random"
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
        neighbor_key="next_cell_ids",
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
        use_X_not_raw=True,
    )

    # Test that neighbor data was extracted
    assert ds._has_neighbors is True
    assert ds._neighbor_indptr is not None
    assert ds._neighbor_indices is not None
    assert ds._neighbor_data is not None

    # Test basic properties of the neighbor data
    assert ds.number_of_rows() == 8
    assert len(ds._neighbor_indices) == 29  # 29 nonzero entries
    assert len(ds._neighbor_indptr) == 9  # 8 cells + 1 (CSR format)
    assert len(ds._neighbor_data) == 29  # 29 nonzero values

    # Test that the neighbor matrix structure is valid (CSR format)
    # indptr should be monotonically increasing
    assert all(ds._neighbor_indptr[i] <= ds._neighbor_indptr[i + 1] for i in range(len(ds._neighbor_indptr) - 1))

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
        neighbor_key="next_cell_ids",
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
        use_X_not_raw=True,
    )

    # Mock numpy's random choice to make sampling deterministic
    def mock_choice(arr, p=None):
        # Always return the first element for predictable testing
        return arr[0]

    monkeypatch.setattr(np.random, "choice", mock_choice)

    # Test sampling for cells that have neighbors
    for cell_idx in range(ds.number_of_rows()):
        start_idx = ds._neighbor_indptr[cell_idx]
        end_idx = ds._neighbor_indptr[cell_idx + 1]

        if start_idx < end_idx:  # Cell has neighbors
            # Get the expected neighbor (first one due to our mock)
            expected_neighbor = ds._neighbor_indices[start_idx]
            sampled_neighbor = ds.sample_neighbor_index(cell_idx)
            assert sampled_neighbor == expected_neighbor, (
                f"Cell {cell_idx} should sample neighbor {expected_neighbor}, got {sampled_neighbor}"
            )

    # Test fallback behavior for cell 0 which has no neighbors
    cell_idx = 0
    sampled_neighbor = ds.sample_neighbor_index(cell_idx)
    assert sampled_neighbor == cell_idx, (
        f"Cell {cell_idx} with no neighbors should return itself, got {sampled_neighbor}"
    )

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
        neighbor_key="next_cell_ids",
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
        use_X_not_raw=True,
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
    monkeypatch.setattr(ds, "sample_neighbor_index", mock_sample_neighbor)

    # Test get_row_with_neighbor
    result = ds.get_row_with_neighbor(0)

    # Validate structure and content
    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "current_cell",
        "next_cell",
        "current_cell_index",
        "next_cell_index",
        "var_features",
        "obs_features",
    }
    assert result["current_cell_index"] == 0
    assert result["next_cell_index"] == 2

    # Test cell data structure (should be tuples of (values, indices))
    current_values, current_cols = result["current_cell"]
    next_values, next_cols = result["next_cell"]

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

    # Test that calling the function on a dataset without neighbors raises ValueError
    ds_no_neighbors = SingleCellMemMapDataset(
        data_path=tmp_path / "scnn_no_neighbors",
        h5ad_path=sample_neighbor_file,
        load_neighbors=False,  # No neighbors
        neighbor_key="next_cell_ids",
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
        use_X_not_raw=True,
    )

    # Should raise ValueError when trying to use neighbor functions without neighbors
    try:
        ds_no_neighbors.get_row_with_neighbor(0)
        assert False, "Should have raised ValueError for dataset without neighbors"
    except ValueError as e:
        assert "Cannot include neighbor data" in str(e)

    # Test with cell 1 which has no gene expression data (should handle gracefully)
    result_empty = ds.get_row_with_neighbor(1)
    assert result_empty["current_cell_index"] == 1
    assert result_empty["next_cell_index"] == 1  # Should fallback to itself


def test_get_row_padded_with_neighbor(tmp_path, monkeypatch, test_neighbor_directory):
    """Test get_row_padded_with_neighbor using real sample data."""

    # Path to the NGC sample neighbor data
    sample_neighbor_file = test_neighbor_directory / "adata_sample0_neighbors.h5ad"

    # Create dataset with real neighbor data
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scnn",
        h5ad_path=sample_neighbor_file,
        load_neighbors=True,
        neighbor_key="next_cell_ids",
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
        use_X_not_raw=True,
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
    monkeypatch.setattr(ds, "sample_neighbor_index", mock_sample_neighbor)

    # Test get_row_padded_with_neighbor (always returns neighbor data in simplified API)
    result = ds.get_row_padded_with_neighbor(0)

    # Validate structure and content
    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "current_cell",
        "next_cell",
        "current_cell_index",
        "next_cell_index",
        "var_features",
        "obs_features",
    }
    assert result["current_cell_index"] == 0
    assert result["next_cell_index"] == 2

    # Test padded data (should be dense arrays with zeros for missing values)
    current_padded = result["current_cell"]
    next_padded = result["next_cell"]

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

    # Test that calling the function on a dataset without neighbors raises ValueError
    ds_no_neighbors = SingleCellMemMapDataset(
        data_path=tmp_path / "scnn_no_neighbors",
        h5ad_path=sample_neighbor_file,
        load_neighbors=False,  # No neighbors
        neighbor_key="next_cell_ids",
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
        use_X_not_raw=True,
    )

    # Should raise ValueError when trying to use neighbor functions without neighbors
    try:
        ds_no_neighbors.get_row_padded_with_neighbor(0)
        assert False, "Should have raised ValueError for dataset without neighbors"
    except ValueError as e:
        assert "Cannot include neighbor data" in str(e)


def test_get_neighbor_stats(tmp_path, test_neighbor_directory):
    # Path to the NGC sample neighbor data
    sample_neighbor_file = test_neighbor_directory / "adata_sample0_neighbors.h5ad"

    # Create dataset with real neighbor data
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scn",
        h5ad_path=sample_neighbor_file,
        load_neighbors=True,
        neighbor_key="next_cell_ids",
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
        use_X_not_raw=True,
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
        "cells_with_no_neighbors",
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
        neighbor_key="next_cell_ids",
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
        use_X_not_raw=True,
    )

    # Verify no neighbors were loaded
    assert ds_no_neighbors._has_neighbors is False

    # Get stats for dataset without neighbors
    stats_no_neighbors = ds_no_neighbors.get_neighbor_stats()
    assert stats_no_neighbors == {"has_neighbors": False}


def test_paginated_neighbor_data_extraction(tmp_path, test_neighbor_directory):
    """Test paginated neighbor data extraction using forced paginated loading."""

    # Path to the NGC sample neighbor data
    sample_neighbor_file = test_neighbor_directory / "adata_sample0_neighbors.h5ad"

    # Create dataset with paginated loading forced (by setting cutoff to 0)
    ds_paginated = SingleCellMemMapDataset(
        data_path=tmp_path / "scn_paginated",
        h5ad_path=sample_neighbor_file,
        load_neighbors=True,
        neighbor_key="next_cell_ids",
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
        paginated_load_cutoff=0,  # Force paginated loading for any file size
        load_block_row_size=3,  # Use small block size to test chunking
        use_X_not_raw=True,
    )

    # Create dataset with regular loading for comparison
    ds_regular = SingleCellMemMapDataset(
        data_path=tmp_path / "scn_regular",
        h5ad_path=sample_neighbor_file,
        load_neighbors=True,
        neighbor_key="next_cell_ids",
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
        paginated_load_cutoff=999999,  # Ensure regular loading
        use_X_not_raw=True,
    )

    # Verify both datasets loaded neighbors successfully
    assert ds_paginated._has_neighbors is True
    assert ds_regular._has_neighbors is True

    # Verify that neighbor data structures are identical between paginated and regular loading
    assert ds_paginated.number_of_rows() == ds_regular.number_of_rows()
    assert len(ds_paginated._neighbor_indptr) == len(ds_regular._neighbor_indptr)
    assert len(ds_paginated._neighbor_indices) == len(ds_regular._neighbor_indices)
    assert len(ds_paginated._neighbor_data) == len(ds_regular._neighbor_data)

    # Verify that the actual neighbor data is identical
    assert np.array_equal(ds_paginated._neighbor_indptr, ds_regular._neighbor_indptr)
    assert np.array_equal(ds_paginated._neighbor_indices, ds_regular._neighbor_indices)
    assert np.array_equal(ds_paginated._neighbor_data, ds_regular._neighbor_data)

    # Test that neighbor functionality works identically
    for cell_idx in range(ds_paginated.number_of_rows()):
        paginated_neighbors = ds_paginated.get_neighbor_indices_for_cell(cell_idx)
        regular_neighbors = ds_regular.get_neighbor_indices_for_cell(cell_idx)
        assert np.array_equal(paginated_neighbors, regular_neighbors)

        paginated_weights = ds_paginated.get_neighbor_weights_for_cell(cell_idx)
        regular_weights = ds_regular.get_neighbor_weights_for_cell(cell_idx)
        assert np.array_equal(paginated_weights, regular_weights)

    # Test that neighbor stats are identical
    paginated_stats = ds_paginated.get_neighbor_stats()
    regular_stats = ds_regular.get_neighbor_stats()
    assert paginated_stats == regular_stats

    # Verify the expected structure from our known test data
    assert ds_paginated.number_of_rows() == 8
    assert paginated_stats["total_connections"] == 29
    assert paginated_stats["has_neighbors"] is True


def test_get_neighbor_weights_for_cell(tmp_path, test_neighbor_directory):
    """Test get_neighbor_weights_for_cell method for coverage."""

    # Path to the NGC sample neighbor data
    sample_neighbor_file = test_neighbor_directory / "adata_sample0_neighbors.h5ad"

    # Create dataset with neighbors
    ds_with_neighbors = SingleCellMemMapDataset(
        data_path=tmp_path / "scn_with_neighbors",
        h5ad_path=sample_neighbor_file,
        load_neighbors=True,
        neighbor_key="next_cell_ids",
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
        use_X_not_raw=True,
    )

    # Test normal operation - get weights for a cell that has neighbors
    weights = ds_with_neighbors.get_neighbor_weights_for_cell(2)  # Cell 2 has neighbors
    assert isinstance(weights, np.ndarray)
    assert len(weights) > 0  # Should have neighbor weights

    # Test cell with no neighbors (cell 0 and 1 have no neighbors based on indptr)
    weights_empty = ds_with_neighbors.get_neighbor_weights_for_cell(0)
    assert isinstance(weights_empty, np.ndarray)
    assert len(weights_empty) == 0  # Should be empty

    # Test IndexError for out of bounds cell index
    with pytest.raises(IndexError, match="Cell index .* out of bounds"):
        ds_with_neighbors.get_neighbor_weights_for_cell(999)

    with pytest.raises(IndexError, match="Cell index .* out of bounds"):
        ds_with_neighbors.get_neighbor_weights_for_cell(-1)

    # Create dataset without neighbors to test error conditions
    ds_without_neighbors = SingleCellMemMapDataset(
        data_path=tmp_path / "scn_without_neighbors",
        h5ad_path=sample_neighbor_file,
        load_neighbors=False,  # No neighbors requested
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
        use_X_not_raw=True,
    )

    # Test with load_neighbors=False - should return empty array
    weights_no_neighbors = ds_without_neighbors.get_neighbor_weights_for_cell(0)
    assert isinstance(weights_no_neighbors, np.ndarray)
    assert len(weights_no_neighbors) == 0

    # Create dataset that requests neighbors but has no neighbor data to test ValueError
    ds_neighbors_requested = SingleCellMemMapDataset(
        data_path=tmp_path / "scn_neighbors_requested",
        h5ad_path=sample_neighbor_file,
        load_neighbors=True,
        neighbor_key="nonexistent_key",  # This key doesn't exist, so no neighbors will be loaded
        neighbor_sampling_strategy="random",
        fallback_to_identity=True,
        use_X_not_raw=True,
    )

    # Test ValueError when neighbors were requested but not available
    with pytest.raises(ValueError, match="Neighbor functionality was enabled but no neighbor data is available"):
        ds_neighbors_requested.get_neighbor_weights_for_cell(0)
