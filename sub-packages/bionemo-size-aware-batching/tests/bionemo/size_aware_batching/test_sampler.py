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
from torch.utils.data import Sampler

from bionemo.size_aware_batching.sampler import SizeAwareBatchSampler


@pytest.fixture
def sampler():
    return Sampler([1, 2, 3])


@pytest.mark.parametrize(
    "max_total_size, idx_to_size",
    [
        (100, {1: 10, 2: 20, 3: 30}),
    ],
)
def test_initialization_valid_inputs(sampler, max_total_size, idx_to_size):
    batch_sampler = SizeAwareBatchSampler(sampler, max_total_size=max_total_size, idx_to_size=idx_to_size)
    assert batch_sampler.num_batches == 3


@pytest.mark.parametrize(
    "sampler",
    [
        "not a sampler",
    ],
)
def test_initialization_invalid_inputs(sampler):
    with pytest.raises(ValueError):
        SizeAwareBatchSampler(sampler, max_total_size=100, idx_to_size={1: 10})


def test_edge_case_empty_dataset():
    with pytest.raises(RuntimeError):
        SizeAwareBatchSampler(Sampler([]), max_total_size=100, idx_to_size={})


def test_edge_case_single_element_dataset():
    sampler = Sampler([1])
    idx_to_size = {1: 10}
    batch_sampler = SizeAwareBatchSampler(sampler, max_total_size=100, idx_to_size=idx_to_size)
    batches = list(batch_sampler)
    assert len(batches) == 1
    assert batches[0] == [1]


def test_edge_case_large_batch_size():
    sampler = Sampler([1, 2, 3, 4, 5])
    idx_to_size = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
    batch_sampler = SizeAwareBatchSampler(sampler, max_total_size=200, idx_to_size=idx_to_size)
    batches = list(batch_sampler)
    assert len(batches) == 2
    assert batches[0] == [1, 2]
    assert batches[1] == [3, 4, 5]


def test_edge_case_max_total_size_exceeded():
    sampler = Sampler([1, 2, 3])
    idx_to_size = {1: 10, 2: 200, 3: 30}
    batch_sampler = SizeAwareBatchSampler(sampler, max_total_size=100, idx_to_size=idx_to_size)
    batches = list(batch_sampler)
    assert len(batches) == 2
    assert batches[0] == [1]
    assert batches[1] == [3]


def test_iteration():
    sampler = Sampler([1, 2, 3])
    idx_to_size = {1: 10, 2: 20, 3: 30}
    batch_sampler = SizeAwareBatchSampler(sampler, max_total_size=100, idx_to_size=idx_to_size)
    for _ in range(3):
        batches = list(batch_sampler)
        assert len(batches) == 3
        assert batches[0] == [1]
        assert batches[1] == [2]
        assert batches[2] == [3]
