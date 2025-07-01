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

from typing import Dict, List, Union

import torch


def collate_sparse_matrix_batch(batch: list[torch.Tensor]) -> torch.Tensor:
    """Collate function to create a batch out of sparse tensors.

    This is necessary to collate sparse matrices of various lengths.

    Args:
        batch: A list of Tensors to collate into a batch.

    Returns:
        The tensors collated into a CSR (Compressed Sparse Row) Format.
    """
    batch_rows = torch.cumsum(
        torch.tensor([0] + [sparse_representation.shape[1] for sparse_representation in batch]), dim=0
    )
    batch_cols = torch.cat([sparse_representation[1] for sparse_representation in batch]).to(torch.int32)
    batch_values = torch.cat([sparse_representation[0] for sparse_representation in batch])
    if len(batch_cols) == 0:
        max_pointer = 0
    else:
        max_pointer = int(batch_cols.max().item() + 1)
    batch_sparse_tensor = torch.sparse_csr_tensor(batch_rows, batch_cols, batch_values, size=(len(batch), max_pointer))
    return batch_sparse_tensor


def collate_neighbor_sparse_matrix_batch(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List[int], int]]:
    """Collates a batch of samples with neighbor data into a single batch.

    This collation function handles the output format when SingleCellMemMapDataset
    is used with load_neighbors=True and get_row_with_neighbor() returns tuples.

    Args:
        batch: List of dictionaries, each containing:
               - 'current_cell': Tuple[np.ndarray, np.ndarray] (values, columns)
               - 'next_cell': Tuple[np.ndarray, np.ndarray] (values, columns)
               - 'current_cell_index': int
               - 'next_cell_index': int

    Returns:
        Dict containing:
        - 'current_cells': Sparse tensor containing all current cells
        - 'next_cells': Sparse tensor containing all next cells
        - 'current_cell_indices': List of original indices for current cells
        - 'next_cell_indices': List of original indices for next cells
        - 'batch_size': Number of samples in the batch
    """
    # Extract components
    current_cells = [item["current_cell"] for item in batch]
    next_cells = [item["next_cell"] for item in batch]
    current_indices = [item["current_cell_index"] for item in batch]
    next_indices = [item["next_cell_index"] for item in batch]

    # Convert tuple format (values, columns) to tensors for collation
    # Each tensor should be stacked as [values, columns] to match collate_sparse_matrix_batch format
    current_tensors = [
        torch.stack([torch.tensor(values, dtype=torch.float32), torch.tensor(columns, dtype=torch.float32)])
        for values, columns in current_cells
    ]
    next_tensors = [
        torch.stack([torch.tensor(values, dtype=torch.float32), torch.tensor(columns, dtype=torch.float32)])
        for values, columns in next_cells
    ]

    # Collate the sparse tensors
    current_batch = collate_sparse_matrix_batch(current_tensors)
    next_batch = collate_sparse_matrix_batch(next_tensors)

    return {
        "current_cells": current_batch,
        "next_cells": next_batch,
        "current_cell_indices": current_indices,
        "next_cell_indices": next_indices,
        "batch_size": len(batch),
    }
