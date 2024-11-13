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

from typing import List, Optional

import pytest
import torch
from torch.utils.data import Dataset, SequentialSampler


class MyDataset(Dataset):
    def __init__(self, size: int, dim: int, device: torch.device):
        self.size = size
        self.dim = dim
        self.device = device

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index >= self.size or index < 0:
            raise IndexError("Index out of range")
        data = torch.ones(self.dim, dtype=torch.float32, device=self.device) * index
        return data


class MyModel(torch.nn.Module):
    def __init__(
        self,
        dim_io: int,
        dim_hidden: int,
        dim_hidden_large: Optional[int] = None,
        ids_do_hidden_large: Optional[List[int]] = None,
    ):
        if not isinstance(dim_io, int) or dim_io <= 0:
            raise ValueError("dim_io must be a positive integer")
        if not isinstance(dim_hidden, int) or dim_hidden <= 0:
            raise ValueError("dim_hidden must be a positive integer")
        if ids_do_hidden_large is not None and dim_hidden_large is None:
            raise ValueError("dim_hidden_large must be provided if ids_do_hidden_large is not None")
        super().__init__()
        self.dim_io = dim_io
        self.dim_hidden = dim_hidden
        self.dim_hidden_large = dim_hidden_large
        self.ids_do_hidden_large = ids_do_hidden_large
        self.layer = torch.nn.Linear(self.dim_io, self.dim_io)

    def forward(self, x: torch.Tensor):
        update = self.layer(x)
        idx = int(x[0].item())
        use_large_dim = (
            self.ids_do_hidden_large is not None
            and self.dim_hidden_large is not None
            and idx in self.ids_do_hidden_large
        )
        repeat_dim = self.dim_hidden_large if use_large_dim else self.dim_hidden
        update = update.unsqueeze(-1).repeat(1, repeat_dim)
        ans = (x + update.sum(dim=-1)).sum()
        return ans


@pytest.fixture
def size_and_dim():
    size = 5
    dim_hidden = 4
    return (size, dim_hidden)


@pytest.fixture
def device():
    return torch.device("cuda:0")


@pytest.fixture
def dataset(size_and_dim, device):
    return MyDataset(*size_and_dim, device)


@pytest.fixture
def sampler(dataset):
    return SequentialSampler(dataset)


@pytest.fixture
def get_sizeof(request):
    def sizeof(i: int) -> int:
        return ((i % 3) + 1) * 10

    return sizeof


@pytest.fixture
def model_and_alloc_peak(size_and_dim, device):
    dim_io = size_and_dim[1]
    alloc_peak = 2**9 * 1024**2  # ~512MB
    dim_hidden = alloc_peak // (4 * dim_io)
    return MyModel(dim_io, dim_hidden).to(device), alloc_peak


@pytest.fixture
def model_huge_sample02(size_and_dim, device):
    dim_io = size_and_dim[1]
    alloc_peak = 2**9 * 1024**2  # ~512MB
    dim_hidden = alloc_peak // (4 * dim_io)
    mem_total = torch.cuda.get_device_properties(device).total_memory
    dim_hidden_large = mem_total // (4 * dim_io) * 2
    ids_do_hidden_large = [0, 2]
    return MyModel(dim_io, dim_hidden, dim_hidden_large, ids_do_hidden_large).to(device)
