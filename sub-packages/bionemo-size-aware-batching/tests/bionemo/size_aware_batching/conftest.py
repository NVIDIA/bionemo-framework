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

import pytest
import torch
from torch.utils.data import Dataset


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
        data = torch.ones(self.dim, dtype=torch.float32, device=self.device)
        return data


class MyModel(torch.nn.Module):
    def __init__(self, dim_io: int, dim_hidden: int):
        if not isinstance(dim_io, int) or dim_io <= 0:
            raise ValueError("dim_io must be a positive integer")
        if not isinstance(dim_hidden, int) or dim_hidden <= 0:
            raise ValueError("dim_hidden must be a positive integer")
        super().__init__()
        self.dim_io = dim_io
        self.dim_hidden = dim_hidden
        self.layer = torch.nn.Linear(self.dim_io, self.dim_io)

    def forward(self, x: torch.Tensor):
        update = self.layer(x)
        update = update.unsqueeze(-1).repeat(1, self.dim_hidden)
        # update.numel == torch.prod(x.shape[:-1]) * self.dim_io * self.dim_hidden
        ans = (x + update.sum(dim=-1)).sum()
        return ans


@pytest.fixture(scope="module")
def dataset():
    return MyDataset(4, 4, torch.device("cuda:0"))


@pytest.fixture(scope="module")
def model(dataset):
    device = dataset.device
    dim_io = dataset.dim
    alloc_peak = 2**9 * 1024**2  # ~512MB
    dim_hidden = alloc_peak // (4 * dim_io)
    return MyModel(dim_io, dim_hidden).to(device), alloc_peak


@pytest.fixture(scope="module")
def model_huge(dataset):
    device = dataset.device
    dim_io = dataset.dim
    mem_total = torch.cuda.get_device_properties(device).total_memory
    dim_hidden = mem_total // (4 * dim_io) * 2
    return MyModel(dim_io, dim_hidden).to(device)
