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

import sys

import pytest
from torch.utils.data import SequentialSampler

from bionemo.size_aware_batching.sampler import SizeAwareBatchSampler


def test_SABS_init_valid_input(sampler, get_sizeof_dataset):
    sizeof, dataset = get_sizeof_dataset
    max_total_size = 60
    batch_sampler = SizeAwareBatchSampler(sampler, max_total_size, sizeof, dataset=dataset)
    assert batch_sampler._sampler == sampler
    assert batch_sampler._max_total_size == max_total_size
    assert batch_sampler._sizeof == sizeof
    assert batch_sampler._dataset == dataset


def test_SABS_init_invalid_max_total_size(sampler):
    with pytest.raises(ValueError):
        SizeAwareBatchSampler(sampler, -1, {})

    with pytest.raises(ValueError):
        SizeAwareBatchSampler(sampler, 0, {})


def test_SABS_init_invalid_sampler_type():
    max_total_size = 60
    sampler = "not a sampler"
    with pytest.raises(TypeError):
        SizeAwareBatchSampler(sampler, max_total_size, {})


def test_SABS_init_invalid_sizeof_type(sampler):
    max_total_size = 60
    sizeof = " invalid type"
    with pytest.raises(TypeError):
        SizeAwareBatchSampler(sampler, max_total_size, sizeof)


def test_SABS_init_callable_sizeof_without_dataset(sampler):
    max_total_size = 60

    with pytest.raises(ValueError):
        SizeAwareBatchSampler(sampler, max_total_size, lambda i: 10)


def test_SABS_init_predefined_sizeof_with_dataset(sampler):
    max_total_size = 60
    dataset = [None] * len(sampler)  # dummy dataset

    with pytest.raises(ValueError):
        SizeAwareBatchSampler(sampler, max_total_size, {}, dataset=dataset)
        SizeAwareBatchSampler(sampler, max_total_size, [], dataset=dataset)


def test_SABS_init_sizeof_seq_bounds_check(sampler):
    max_total_size = 60
    sizeof = [10] * (len(sampler) - 1)  # invalid length

    sys.gettrace = lambda: True

    with pytest.raises(ValueError):
        SizeAwareBatchSampler(sampler, max_total_size, sizeof)

    sys.gettrace = lambda: None


def test_SABS_init_max_size_exceeds_max_total_size(sampler):
    max_total_size = 100
    sizeof = {i: (1000 if i == 0 else 1) for i in sampler}

    sys.gettrace = lambda: True
    with pytest.warns(UserWarning):
        SizeAwareBatchSampler(sampler, max_total_size, sizeof)
    sys.gettrace = lambda: None


def test_SABS_init_min_size_exceeds_max_total_size(sampler):
    max_total_size = 60
    sizeof = {i: max_total_size + 1 for i in range(len(sampler))}  # invalid value

    sys.gettrace = lambda: True

    with pytest.raises(ValueError), pytest.warns(UserWarning):
        SizeAwareBatchSampler(sampler, max_total_size, sizeof)

    sys.gettrace = lambda: None


def test_SABS_iter(sampler, get_sizeof_dataset):
    sizeof, dataset = get_sizeof_dataset
    max_total_size = 29

    size_aware_sampler = SizeAwareBatchSampler(sampler, max_total_size, sizeof, dataset=dataset)

    meta_batch_ids = list(size_aware_sampler)

    def fn_sizeof(i: int):
        if callable(sizeof):
            return sizeof(dataset[i])
        else:
            return sizeof[i]

    # Check that the batches are correctly sized
    for ids_batch in meta_batch_ids:
        size_batch = sum(fn_sizeof(idx) for idx in ids_batch)
        assert size_batch <= max_total_size

    meta_batch_ids_expected = []
    ids_batch = []
    s_all = 0
    for idx in sampler:
        s = fn_sizeof(idx)
        if s > max_total_size:
            continue
        if s + s_all > max_total_size:
            meta_batch_ids_expected.append(ids_batch)
            s_all = s
            ids_batch = [idx]
            continue
        s_all += s
        ids_batch.append(idx)
    if len(ids_batch) > 0:
        meta_batch_ids_expected.append(ids_batch)

    assert meta_batch_ids == meta_batch_ids_expected

    # the 2nd pass should return the same result
    meta_batch_ids_2nd_pass = list(size_aware_sampler)
    assert meta_batch_ids == meta_batch_ids_2nd_pass


def test_SABS_iter_no_samples():
    # Test iterating over a batch of indices with no samples
    sampler = SequentialSampler([])
    size_aware_sampler = SizeAwareBatchSampler(sampler, 100, {})

    batched_indices = list(size_aware_sampler)

    assert not batched_indices


def test_SABS_iter_empty_sizeof(sampler):
    size_aware_sampler = SizeAwareBatchSampler(sampler, 1, {})

    with pytest.raises(RuntimeError):
        list(size_aware_sampler)
