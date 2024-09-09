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

from bionemo.size_aware_batching.sampler import SizeAwareBatchSampler


def test_init_valid_input(sampler, sizeof_dataset):
    sizeof, dataset = sizeof_dataset
    max_total_size = 60
    batch_sampler = SizeAwareBatchSampler(sampler, max_total_size, sizeof, dataset=dataset)
    assert batch_sampler._sampler == sampler
    assert batch_sampler._max_total_size == max_total_size
    assert batch_sampler._sizeof == sizeof
    assert batch_sampler._dataset == dataset


def test_init_invalid_max_total_size(sampler):
    max_total_size = -1
    with pytest.raises(ValueError):
        SizeAwareBatchSampler(sampler, max_total_size, {})


def test_init_invalid_sampler_type():
    max_total_size = 60
    sampler = "not a sampler"
    with pytest.raises(TypeError):
        SizeAwareBatchSampler(sampler, max_total_size, {})


def test_init_invalid_sizeof_type(sampler):
    max_total_size = 60
    sizeof = " invalid type"
    with pytest.raises(TypeError):
        SizeAwareBatchSampler(sampler, max_total_size, sizeof)


def test_init_callable_sizeof_without_dataset(sampler):
    max_total_size = 60

    with pytest.raises(ValueError):
        SizeAwareBatchSampler(sampler, max_total_size, lambda i: 10)


def test_init_predefined_sizeof_with_dataset(sampler):
    max_total_size = 60
    dataset = [None] * len(sampler)  # dummy dataset

    with pytest.raises(ValueError):
        SizeAwareBatchSampler(sampler, max_total_size, {}, dataset=dataset)
        SizeAwareBatchSampler(sampler, max_total_size, [], dataset=dataset)


def test_init_caching_with_predefined_sizeof(sampler):
    max_total_size = 60
    with pytest.raises(ValueError):
        SizeAwareBatchSampler(sampler, max_total_size, {}, do_caching=True)
        SizeAwareBatchSampler(sampler, max_total_size, [], do_caching=True)


def test_init_sizeof_seq_bounds_check(sampler):
    max_total_size = 60
    sizeof = [10] * (len(sampler) - 1)  # invalid length

    sys.gettrace = lambda: True

    with pytest.raises(ValueError):
        SizeAwareBatchSampler(sampler, max_total_size, sizeof)

    sys.gettrace = lambda: None


def test_init_max_size_exceeds_max_total_size(sampler):
    max_total_size = 100
    sizeof = {i: (1000 if i == 0 else 1) for i in sampler}

    sys.gettrace = lambda: True
    with pytest.warns(UserWarning):
        SizeAwareBatchSampler(sampler, max_total_size, sizeof)
    sys.gettrace = lambda: None


def test_init_min_size_exceeds_max_total_size(sampler):
    max_total_size = 60
    sizeof = {i: max_total_size + 1 for i in range(len(sampler))}  # invalid value

    sys.gettrace = lambda: True

    with pytest.raises(ValueError), pytest.warns(UserWarning):
        SizeAwareBatchSampler(sampler, max_total_size, sizeof)

    sys.gettrace = lambda: None
