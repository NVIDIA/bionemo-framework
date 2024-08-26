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
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch


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
