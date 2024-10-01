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


import numpy as np
import pytest

from bionemo.size_aware_batching.bucket_batch_sampler import create_buckets


def test_create_buckets_empty_input():
    with pytest.raises(ValueError):
        create_buckets([])


def test_create_buckets_non_integers_input():
    with pytest.raises(ValueError):
        create_buckets([1.0, 2.0])


def test_create_buckets_non_1d_ndarray_input():
    with pytest.raises(ValueError):
        create_buckets(np.array([[1, 2], [3, 4]]))


def test_create_buckets_negative_max_range():
    with pytest.raises(ValueError):
        create_buckets(np.array([1, 2, 3]), max_range=-1)


def test_create_buckets_negative_min_bucket_count():
    with pytest.raises(ValueError):
        create_buckets(np.array([1, 2, 3]), min_bucket_count=-1)


def test_create_buckets_non_integer_min_bucket_count():
    with pytest.raises(ValueError):
        create_buckets(np.array([1, 2, 3]), min_bucket_count=3.5)


def test_create_buckets_single_element():
    bucket, bucket_sizes = create_buckets(np.array([1]))
    assert np.allclose(bucket, [[1, 1]])
    assert np.allclose(bucket_sizes, [1])


def test_create_buckets_multiple_elements():
    sizes = np.array([1, 2, 3, 4, 5])
    np.random.shuffle(sizes)
    bucket, bucket_sizes = create_buckets(sizes, max_range=10)
    assert np.allclose(bucket, [[1, 5]])
    assert np.allclose(bucket_sizes, [5])


def test_create_buckets_multiple_buckets():
    sizes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    np.random.shuffle(sizes)
    bucket, bucket_sizes = create_buckets(sizes, max_range=4)
    assert np.allclose(bucket, [[1, 5], [6, 10]])
    assert np.allclose(bucket_sizes, [5, 5])


def test_create_buckets_with_duplicates():
    sizes = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    np.random.shuffle(sizes)
    bucket, bucket_sizes = create_buckets(sizes)
    assert np.allclose(bucket, [[1, 3]])
    assert np.allclose(bucket_sizes, [12])


def test_create_buckets_with_max_range_zero():
    sizes = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    np.random.shuffle(sizes)
    bucket, bucket_sizes = create_buckets(sizes, max_range=0)
    assert np.allclose(bucket, [[1, 1], [2, 2], [3, 3]])
    assert np.allclose(bucket_sizes, [3, 4, 5])


def test_create_buckets_with_max_range_reached():
    sizes = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 22, 22, 22, 22])
    np.random.shuffle(sizes)
    bucket, bucket_sizes = create_buckets(sizes, max_range=20)
    assert np.allclose(bucket, [[1, 3], [22, 22]])
    assert np.allclose(bucket_sizes, [12, 4])


def test_create_buckets_with_min_bucket_count_reached():
    sizes = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 11, 11, 11, 11, 11])
    np.random.shuffle(sizes)
    bucket, bucket_sizes = create_buckets(sizes, max_range=20, min_bucket_count=10)
    assert np.allclose(bucket, [[1, 3], [11, 11]])
    assert np.allclose(bucket_sizes, [12, 5])
