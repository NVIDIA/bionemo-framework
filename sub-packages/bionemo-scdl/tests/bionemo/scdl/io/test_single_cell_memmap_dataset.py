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

# NOTE: add neighbor test dataset
# Test creating a dataset with neighbor support
def test_create_dataset_with_neighbor_support(tmp_path):
    # Create a simple dataset with neighbor support
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scnn", 
        num_rows=5, 
        num_elements=10,
        load_neighbors=True,
        neighbor_key='test_neighbors',
        neighbor_sampling_strategy='random',
        fallback_to_identity=True
    )
    
    # Verify neighbor configuration
    assert ds.load_neighbors is True
    assert ds.neighbor_key == 'test_neighbors'
    assert ds.neighbor_sampling_strategy == 'random'
    assert ds.fallback_to_identity is True
    assert ds._has_neighbors is False  # No neighbors loaded yet

def test_neighbor_matrix_extraction(tmp_path, monkeypatch):
    # Create test AnnData with neighbor information
    from scipy.sparse import csr_matrix
    
    # Mock the AnnData object with neighbors
    class MockAnnData:
        def __init__(self):
            self.obsp = {
                'next_cell_ids': csr_matrix(([1.0, 2.0, 3.0], ([0, 1, 2], [1, 2, 0])), shape=(3, 3))
            }
            # Mock X as a sparse matrix
            row_ind = np.array([0, 0, 1, 2, 2])
            col_ind = np.array([0, 2, 1, 0, 2])
            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            self.X = csr_matrix((data, (row_ind, col_ind)), shape=(3, 3))
            # Mock var dataframe
            import pandas as pd
            self.var = pd.DataFrame(index=pd.Index(['gene1', 'gene2', 'gene3']))
    
    # Patch anndata.read_h5ad to return our mock
    def mock_read_h5ad(*args, **kwargs):
        return MockAnnData()
    
    # Apply the patch
    monkeypatch.setattr(ad, 'read_h5ad', mock_read_h5ad)
    
    # Create dataset with neighbors
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scy", 
        h5ad_path="dummy_path.h5ad",  # This will use our mocked function
        load_neighbors=True
    )
    
    # Test that neighbor data was extracted
    assert ds._has_neighbors is True
    assert ds._neighbor_indptr is not None
    assert ds._neighbor_indices is not None
    assert ds._neighbor_data is not None
    
    # Test neighbor data content
    assert np.array_equal(ds._neighbor_indptr, np.array([0, 1, 2, 3]))
    assert np.array_equal(ds._neighbor_indices, np.array([1, 2, 0]))
    assert np.array_equal(ds._neighbor_data, np.array([1.0, 2.0, 3.0]))

def test_sample_neighbor_index(tmp_path, monkeypatch):
    # Setup mock dataset with known neighbor structure
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scn", 
        num_rows=3, 
        num_elements=5
    )
    
    # Mock the neighbor data structures
    ds.load_neighbors = True
    ds._has_neighbors = True
    ds._neighbor_indptr = np.array([0, 2, 3, 5])
    ds._neighbor_indices = np.array([1, 2, 0, 0, 1])
    ds._neighbor_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Mock numpy's random choice to always return the first element
    def mock_choice(arr):
        return arr[0]
    
    monkeypatch.setattr(np.random, 'choice', mock_choice)
    
    # Test sampling for each cell
    assert ds.sample_neighbor_index(0) == 1  # First cell's first neighbor
    assert ds.sample_neighbor_index(1) == 0  # Second cell's only neighbor
    assert ds.sample_neighbor_index(2) == 0  # Third cell's first neighbor
    
    # Test fallback behavior when no neighbors
    ds._neighbor_indptr = np.array([0, 0, 1, 2])  # Cell 0 has no neighbors
    assert ds.sample_neighbor_index(0) == 0  # Should return itself
    
    # Test when fallback_to_identity is False
    ds.fallback_to_identity = False
    with pytest.warns():
        assert ds.sample_neighbor_index(0) == 0  # Should still return itself with warning

def test_get_row_with_neighbor(tmp_path, monkeypatch):
    # Setup a dataset with mock data
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scn", 
        num_rows=3, 
        num_elements=5
    )
    
    # Mock data structures
    ds.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ds.col_index = np.array([0, 2, 1, 0, 2])
    ds.row_index = np.array([0, 2, 3, 5])
    
    # Mock neighbor functionality
    ds.load_neighbors = True
    ds._has_neighbors = True
    
    # Mock sample_neighbor_index to always return cell 2
    def mock_sample_neighbor(idx):
        return 2
    
    ds.sample_neighbor_index = mock_sample_neighbor
    
    # Test get_row_with_neighbor with neighbor
    result = ds.get_row_with_neighbor(0, include_neighbor=True)
    
    # Validate structure and content
    assert isinstance(result, dict)
    assert set(result.keys()) == {'current_cell', 'next_cell', 'current_cell_index', 'next_cell_index', 'features'}
    assert result['current_cell_index'] == 0
    assert result['next_cell_index'] == 2
    
    # Test cell data
    current_values, current_cols = result['current_cell']
    assert np.array_equal(current_values, np.array([1.0, 2.0]))
    assert np.array_equal(current_cols, np.array([0, 2]))
    
    next_values, next_cols = result['next_cell']
    assert np.array_equal(next_values, np.array([4.0, 5.0]))
    assert np.array_equal(next_cols, np.array([0, 2]))
    
    # Test get_row_with_neighbor without neighbor
    result = ds.get_row_with_neighbor(0, include_neighbor=False)
    assert isinstance(result, tuple)
    assert len(result) == 2
    values, cols = result[0]
    assert np.array_equal(values, np.array([1.0, 2.0]))
    assert np.array_equal(cols, np.array([0, 2]))

def test_get_row_padded_with_neighbor(tmp_path, monkeypatch):
    # Setup a dataset with mock data
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scn", 
        num_rows=3, 
        num_elements=5
    )
    
    # Mock data structures
    ds.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ds.col_index = np.array([0, 2, 1, 0, 2])
    ds.row_index = np.array([0, 2, 3, 5])
    
    # Mock feature index
    class MockFeatureIndex:
        def number_vars_at_row(self, idx):
            return 3  # All rows have 3 features
        
        def lookup(self, idx, select_features=None):
            return [[idx, idx+1, idx+2]], None
    
    ds._feature_index = MockFeatureIndex()
    
    # Mock neighbor functionality
    ds.load_neighbors = True
    ds._has_neighbors = True
    
    # Mock sample_neighbor_index to always return cell 2
    def mock_sample_neighbor(idx):
        return 2
    
    ds.sample_neighbor_index = mock_sample_neighbor
    
    # Test get_row_padded_with_neighbor with neighbor
    result = ds.get_row_padded_with_neighbor(0, include_neighbor=True)
    
    # Validate structure and content
    assert isinstance(result, dict)
    assert set(result.keys()) == {'current_cell', 'next_cell', 'current_cell_index', 'next_cell_index', 'features'}
    
    # Verify padded data (dense arrays)
    assert np.array_equal(result['current_cell'], np.array([1.0, 0.0, 2.0]))
    assert np.array_equal(result['next_cell'], np.array([4.0, 0.0, 5.0]))
    
    # Test without neighbor
    result = ds.get_row_padded_with_neighbor(0, include_neighbor=False)
    assert isinstance(result, tuple)
    assert np.array_equal(result[0], np.array([1.0, 0.0, 2.0]))

def test_get_neighbor_stats(tmp_path):
    # Setup a dataset with mock data
    ds = SingleCellMemMapDataset(
        data_path=tmp_path / "scn", 
        num_rows=4, 
        num_elements=5
    )
    
    # Mock neighbor data with a specific pattern:
    # Cell 0: has 2 neighbors
    # Cell 1: has 1 neighbor
    # Cell 2: has 0 neighbors
    # Cell 3: has 1 neighbor
    ds.load_neighbors = True
    ds._has_neighbors = True
    ds._neighbor_indptr = np.array([0, 2, 3, 3, 4])
    ds._neighbor_indices = np.array([1, 2, 0, 2])
    ds._neighbor_data = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Get and check stats
    stats = ds.get_neighbor_stats()
    
    assert stats["has_neighbors"] is True
    assert stats["total_connections"] == 4
    assert stats["min_neighbors_per_cell"] == 0
    assert stats["max_neighbors_per_cell"] == 2
    assert stats["avg_neighbors_per_cell"] == 1.0
    assert stats["cells_with_no_neighbors"] == 1
    
    # Test case with no neighbors
    ds._has_neighbors = False
    stats = ds.get_neighbor_stats()
    assert stats == {"has_neighbors": False}