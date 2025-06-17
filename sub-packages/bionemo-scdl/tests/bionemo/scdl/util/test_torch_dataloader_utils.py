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

import torch
from torch.utils.data import DataLoader

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch, collate_neighbor_sparse_matrix_batch


def test_sparse_collate_function_produces_correct_batch():
    columns_one = torch.tensor([2, 3, 5])
    columns_two = torch.tensor([1, 2, 5, 6])
    values_one = torch.tensor([1, 2, 3])
    values_two = torch.tensor([4, 5, 6, 7])
    sparse_tensor_one = torch.stack((values_one, columns_one))
    sparse_tensor_two = torch.stack((values_two, columns_two))
    csr_matrix = collate_sparse_matrix_batch([sparse_tensor_one, sparse_tensor_two])
    assert torch.equal(csr_matrix.to_dense(), torch.tensor([[0, 0, 1, 2, 0, 3, 0], [0, 4, 5, 0, 0, 6, 7]]))


def test_sparse_collate_function_with_one_empty_entry_correct():
    columns_one = torch.tensor([2, 3, 5])
    columns_two = torch.tensor([])
    values_one = torch.tensor([1, 2, 3])
    values_two = torch.tensor([])
    sparse_tensor_one = torch.stack((values_one, columns_one))
    sparse_tensor_two = torch.stack((values_two, columns_two))
    csr_matrix = collate_sparse_matrix_batch([sparse_tensor_one, sparse_tensor_two])
    assert torch.equal(csr_matrix.to_dense(), torch.tensor([[0, 0, 1, 2, 0, 3], [0, 0, 0, 0, 0, 0]]))


def test_sparse_collate_function_with_all_empty_entries_correct():
    columns_one = torch.tensor([])
    columns_two = torch.tensor([])
    values_one = torch.tensor([])
    values_two = torch.tensor([])
    sparse_tensor_one = torch.stack((values_one, columns_one))
    sparse_tensor_two = torch.stack((values_two, columns_two))
    csr_matrix = collate_sparse_matrix_batch([sparse_tensor_one, sparse_tensor_two])
    assert csr_matrix.to_dense().shape == torch.Size([2, 0])


def test_dataloading_batch_size_one_work_without_collate(tmp_path, test_directory):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample1.h5ad")
    dataloader = DataLoader(ds, batch_size=1, shuffle=False)
    expected_tensors = [
        torch.tensor([[[8.0], [1.0]]]),
        torch.empty(1, 2, 0),
        torch.tensor([[[7.0, 18.0], [0.0, 1.0]]]),
        torch.empty(1, 2, 0),
        torch.tensor([[[3.0, 15.0, 4.0, 3.0], [1.0, 0.0, 0.0, 1.0]]]),
        torch.tensor([[[6.0, 4.0, 9.0], [1.0, 1.0, 0.0]]]),
    ]
    for index, batch in enumerate(dataloader):
        assert torch.equal(batch, expected_tensors[index])


def test_dataloading_batch_size_one_works_with_collate(tmp_path, test_directory):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample1.h5ad")
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_sparse_matrix_batch)
    expected_tensors = [
        torch.tensor([[[8.0], [1.0]]]),
        torch.empty(1, 2, 0),
        torch.tensor([[[7.0, 18.0], [0.0, 1.0]]]),
        torch.empty(1, 2, 0),
        torch.tensor([[[3.0, 15.0, 4.0, 3.0], [1.0, 0.0, 0.0, 1.0]]]),
        torch.tensor([[[6.0, 4.0, 9.0], [1.0, 1.0, 0.0]]]),
    ]
    for index, batch in enumerate(dataloader):
        rows = torch.tensor([0, expected_tensors[index].shape[2]])
        columns = expected_tensors[index][0][1].to(torch.int32)
        values = expected_tensors[index][0][0]
        assert torch.equal(batch.to_dense(), torch.sparse_csr_tensor(rows, columns, values).to_dense())


def test_dataloading_batch_size_three_works_with_collate(tmp_path, test_directory):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample1.h5ad")
    dataloader = DataLoader(ds, batch_size=3, shuffle=False, collate_fn=collate_sparse_matrix_batch)
    expected_tensor = torch.tensor([[0, 8], [0, 0], [7, 18]])

    batch = next(iter(dataloader))
    assert torch.equal(batch.to_dense(), expected_tensor)


def test_neighbor_collate_function_produces_correct_batch():
    """Test that collate_neighbor_sparse_matrix_batch works with simple neighbor data."""
    # Create mock neighbor data similar to what get_row_with_neighbor returns
    batch = [
        {
            'current_cell': ([1.0, 2.0], [0, 1]),  # values, columns
            'next_cell': ([3.0], [2]),
            'current_cell_index': 0,
            'next_cell_index': 1
        },
        {
            'current_cell': ([4.0, 5.0, 6.0], [1, 2, 3]),
            'next_cell': ([7.0, 8.0], [0, 1]),
            'current_cell_index': 2,
            'next_cell_index': 3
        }
    ]
    
    result = collate_neighbor_sparse_matrix_batch(batch)
    
    # Check structure
    assert 'current_cells' in result
    assert 'next_cells' in result
    assert 'current_cell_indices' in result
    assert 'next_cell_indices' in result
    assert 'batch_size' in result
    
    # Check indices
    assert result['current_cell_indices'] == [0, 2]
    assert result['next_cell_indices'] == [1, 3]
    assert result['batch_size'] == 2
    
    # Check current cells sparse tensor
    current_dense = result['current_cells'].to_dense()
    expected_current = torch.tensor([[1.0, 2.0, 0.0, 0.0], [0.0, 4.0, 5.0, 6.0]])
    assert torch.equal(current_dense, expected_current)
    
    # Check next cells sparse tensor
    next_dense = result['next_cells'].to_dense()
    expected_next = torch.tensor([[0.0, 0.0, 3.0], [7.0, 8.0, 0.0]])
    assert torch.equal(next_dense, expected_next)


def test_neighbor_collate_function_with_empty_cells():
    """Test collate_neighbor_sparse_matrix_batch handles empty cells correctly."""
    batch = [
        {
            'current_cell': ([1.0], [0]),
            'next_cell': ([], []),  # Empty cell
            'current_cell_index': 0,
            'next_cell_index': 1
        },
        {
            'current_cell': ([], []),  # Empty cell
            'next_cell': ([2.0], [1]),
            'current_cell_index': 2,
            'next_cell_index': 3
        }
    ]
    
    result = collate_neighbor_sparse_matrix_batch(batch)
    
    # Check structure
    assert result['batch_size'] == 2
    assert result['current_cell_indices'] == [0, 2]
    assert result['next_cell_indices'] == [1, 3]
    
    # Check current cells - first has data, second is empty
    current_dense = result['current_cells'].to_dense()
    expected_current = torch.tensor([[1.0], [0.0]])
    assert torch.equal(current_dense, expected_current)
    
    # Check next cells - first is empty, second has data
    next_dense = result['next_cells'].to_dense()
    expected_next = torch.tensor([[0.0, 0.0], [0.0, 2.0]])
    assert torch.equal(next_dense, expected_next)


def test_neighbor_collate_function_all_empty():
    """Test collate_neighbor_sparse_matrix_batch handles all empty cells."""
    batch = [
        {
            'current_cell': ([], []),
            'next_cell': ([], []),
            'current_cell_index': 0,
            'next_cell_index': 1
        },
        {
            'current_cell': ([], []),
            'next_cell': ([], []),
            'current_cell_index': 2,
            'next_cell_index': 3
        }
    ]
    
    result = collate_neighbor_sparse_matrix_batch(batch)
    
    assert result['batch_size'] == 2
    assert result['current_cell_indices'] == [0, 2]
    assert result['next_cell_indices'] == [1, 3]
    
    # Both tensors should have shape [2, 0] for all empty data
    assert result['current_cells'].to_dense().shape == torch.Size([2, 0])
    assert result['next_cells'].to_dense().shape == torch.Size([2, 0])


def test_dataloading_neighbor_batch_with_real_data(tmp_path):
    """Test neighbor collate function with real sample data."""
    import os
    from pathlib import Path
    
    # Check if sample neighbor data exists
    sample_path = Path("/workspaces/bionemo-framework/scdl_data_local/adata_sample0_neighbors.h5ad")
    if not sample_path.exists():
        import pytest
        pytest.skip(f"Sample neighbor data not found at {sample_path}")
    
    # Create dataset with neighbors
    ds = SingleCellMemMapDataset(
        tmp_path / "neighbor_test", 
        h5ad_path=str(sample_path),
        load_neighbors=True,
        neighbor_key='next_cell_ids'
    )
    
    # Mock the sample_neighbor_index to return predictable neighbors
    def mock_sample_neighbor(cell_index):
        neighbor_map = {0: 2, 2: 0, 7: 0}
        return neighbor_map.get(cell_index, cell_index)
    
    import unittest.mock
    with unittest.mock.patch.object(ds, 'sample_neighbor_index', side_effect=mock_sample_neighbor):
        # Create a custom dataset that returns neighbor data
        class NeighborDataset:
            def __init__(self, base_dataset):
                self.base_dataset = base_dataset
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                return self.base_dataset.get_row_with_neighbor(idx, include_neighbor=True)
        
        neighbor_ds = NeighborDataset(ds)
        
        # Test with batch size 1
        dataloader = DataLoader(neighbor_ds, batch_size=1, shuffle=False, collate_fn=collate_neighbor_sparse_matrix_batch)
        
        batch = next(iter(dataloader))
        
        # Check structure
        assert 'current_cells' in batch
        assert 'next_cells' in batch
        assert 'current_cell_indices' in batch
        assert 'next_cell_indices' in batch
        assert 'batch_size' in batch
        assert batch['batch_size'] == 1
        
        # Check that we got valid sparse tensors
        assert batch['current_cells'].is_sparse_csr
        assert batch['next_cells'].is_sparse_csr
        
        # Test with batch size 2
        dataloader2 = DataLoader(neighbor_ds, batch_size=2, shuffle=False, collate_fn=collate_neighbor_sparse_matrix_batch)
        batch2 = next(iter(dataloader2))
        
        assert batch2['batch_size'] == 2
        assert len(batch2['current_cell_indices']) == 2
        assert len(batch2['next_cell_indices']) == 2
        
        # Check dimensions make sense
        current_dense = batch2['current_cells'].to_dense()
        next_dense = batch2['next_cells'].to_dense()
        assert current_dense.shape[0] == 2  # batch size
        assert next_dense.shape[0] == 2     # batch size


def test_dataloading_neighbor_batch_works_with_expected_output(tmp_path, monkeypatch):
    """Test neighbor collate function with expected output patterns using real sample data."""
    from pathlib import Path
    
    # Will update path once data is in ngc registry
    sample_path = Path("/workspaces/bionemo-framework/scdl_data_local/adata_sample0_neighbors.h5ad")
    
    # Create dataset with neighbors
    ds = SingleCellMemMapDataset(
        tmp_path / "neighbor_expected_test", 
        h5ad_path=str(sample_path),
        load_neighbors=True,
        neighbor_key='next_cell_ids'
    )
    
    # Mock the sample_neighbor_index to return predictable neighbors for testing
    def mock_sample_neighbor(cell_index):
        # Known neighbor mappings for our sample data
        neighbor_map = {0: 2, 2: 0, 7: 0}
        return neighbor_map.get(cell_index, cell_index)

    # Use monkeypatch to mock the method
    monkeypatch.setattr(ds, 'sample_neighbor_index', mock_sample_neighbor)
    
    # Create a custom dataset that returns neighbor data
    class NeighborDataset:
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            return self.base_dataset.get_row_with_neighbor(idx, include_neighbor=True)

    neighbor_ds = NeighborDataset(ds)
    dataloader = DataLoader(neighbor_ds, batch_size=1, shuffle=False, collate_fn=collate_neighbor_sparse_matrix_batch)

    expected_batches = [
        {
            'current_cell_indices': [0],
            'next_cell_indices': [2],
            'current_dense': torch.tensor([[0.0, 0.0, 6.0]]),   # Cell 0: value 6.0 at column 2 (shape [1,3])
            'next_dense': torch.tensor([[0.0, 0.0, 19.0]]),     # Cell 2: value 19.0 at column 2 (shape [1,3])
            'batch_size': 1
        },
        {
            'current_cell_indices': [1],
            'next_cell_indices': [1], 
            'current_dense': torch.tensor([[]]).reshape(1, 0),  # Cell 1: empty
            'next_dense': torch.tensor([[]]).reshape(1, 0),     # Cell 1: empty (self)
            'batch_size': 1
        },
        {
            'current_cell_indices': [2],
            'next_cell_indices': [0],
            'current_dense': torch.tensor([[0.0, 0.0, 19.0]]), # Cell 2: value 19.0 at column 2 (shape [1,3])
            'next_dense': torch.tensor([[0.0, 0.0, 6.0]]),     # Cell 0: value 6.0 at column 2 (shape [1,3])
            'batch_size': 1
        }
    ]
    
    # Test only the first three batches against expected output
    for index, batch in enumerate(dataloader):
        if index >= 3:  # Only test first 3 batches
            break
            
        expected = expected_batches[index]
        
        # Check basic structure
        assert 'current_cells' in batch
        assert 'next_cells' in batch
        assert 'current_cell_indices' in batch
        assert 'next_cell_indices' in batch
        assert 'batch_size' in batch
        
        # Check batch metadata
        assert batch['batch_size'] == expected['batch_size']
        assert batch['current_cell_indices'] == expected['current_cell_indices']
        assert batch['next_cell_indices'] == expected['next_cell_indices']
        
        # Check that tensors are sparse CSR format
        assert batch['current_cells'].is_sparse_csr
        assert batch['next_cells'].is_sparse_csr
        
        # Check dense representations match expected patterns
        current_dense = batch['current_cells'].to_dense()
        next_dense = batch['next_cells'].to_dense()
        
        assert torch.equal(current_dense, expected['current_dense']), f"Current cell mismatch at batch {index}: got {current_dense}, expected {expected['current_dense']}"
        assert torch.equal(next_dense, expected['next_dense']), f"Next cell mismatch at batch {index}: got {next_dense}, expected {expected['next_dense']}"


def test_neighbor_collate_single_batch_consistency():
    """Test that neighbor collate produces consistent results for single samples."""
    sample = {
        'current_cell': ([6.0, 19.0], [2, 5]),
        'next_cell': ([12.0], [1]),
        'current_cell_index': 0,
        'next_cell_index': 2
    }
    
    # Test single sample
    result1 = collate_neighbor_sparse_matrix_batch([sample])
    
    # Test same sample duplicated
    result2 = collate_neighbor_sparse_matrix_batch([sample, sample])
    
    # Single sample results
    assert result1['batch_size'] == 1
    current1 = result1['current_cells'].to_dense()
    next1 = result1['next_cells'].to_dense()
    
    # Duplicated sample results
    assert result2['batch_size'] == 2
    current2 = result2['current_cells'].to_dense()
    next2 = result2['next_cells'].to_dense()
    
    # Check that the first row of the duplicated batch matches the single sample
    assert torch.equal(current1[0], current2[0])
    assert torch.equal(current1[0], current2[1])  # Should be identical
    assert torch.equal(next1[0], next2[0])
    assert torch.equal(next1[0], next2[1])  # Should be identical

