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
import torch
import torch.utils

from bionemo.size_aware_batching.bucket_batch_sampler import BucketBatchSampler
from bionemo.size_aware_batching.sampler import SizeAwareBatchSampler


@pytest.fixture
def sample_data():
    sizes = np.arange(25)
    bucket_ranges = np.array([[0, 5], [6, 14], [15, 24]])
    base_batch_sampler_class = torch.utils.data.BatchSampler
    base_batch_sampler_shared_kwargs = {"drop_last": False}
    base_batch_sampler_individual_kwargs = {"batch_size": [2, 3, 5]}
    return (
        sizes,
        bucket_ranges,
        base_batch_sampler_class,
        base_batch_sampler_shared_kwargs,
        base_batch_sampler_individual_kwargs,
    )


def test_init_bucket_batch_sampler_with_invalid_sizes(sample_data):
    (
        sizes,
        bucket_ranges,
        base_batch_sampler_class,
        base_batch_sampler_shared_kwargs,
        base_batch_sampler_individual_kwargs,
    ) = sample_data
    # sizes must be a numpy array
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=list(sizes),
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        )
    # sizes dim be 1D
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=sizes.reshape(5, 5),
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        )
    # sizes data type must be integer or floating numbers
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=np.array(["a", "b", "c"]),
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        )


def test_init_bucket_batch_sampler_with_invalid_bucket_ranges(sample_data):
    (
        sizes,
        bucket_ranges,
        base_batch_sampler_class,
        base_batch_sampler_shared_kwargs,
        base_batch_sampler_individual_kwargs,
    ) = sample_data
    # bucket_ranges must be a numpy array
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges.tolist(),
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        )
    # bucket_ranges must be a 2D array
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges[None, :, :],
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        )
    # bucket_ranges has shape (num_buckets, 2)
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges.T,
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        )
    # bucket_ranges' data type must be integer or floating number
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges.astype(str),
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        )
    # bucket_ranges should not overlap
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=np.array([[0, 6], [6, 14], [15, 24]]),
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        )
    # bucket_ranges should be valid intervals
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=np.array([[5, 0], [6, 14], [15, 24]]),
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        )
    # there should be elements in the buckets
    with pytest.raises(RuntimeError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges + 25,
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        )
    # warning if some elements are outside the bucket_ranges and will be skipped.
    with pytest.warns(UserWarning):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges + 5,
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        )


def test_init_bucket_batch_sampler_with_invalid_shuffle(sample_data):
    (
        sizes,
        bucket_ranges,
        base_batch_sampler_class,
        base_batch_sampler_shared_kwargs,
        base_batch_sampler_individual_kwargs,
    ) = sample_data
    # shuffle must be a boolean
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=torch.utils.data.DataLoader,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
            shuffle=1,
        )


def test_init_bucket_batch_sampler_with_invalid_base_batch_sampler_class(sample_data):
    (
        sizes,
        bucket_ranges,
        base_batch_sampler_class,
        base_batch_sampler_shared_kwargs,
        base_batch_sampler_individual_kwargs,
    ) = sample_data
    # base_batch_sampler_class must be a class inherited from torch.utils.data.Sampler
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=torch.utils.data.DataLoader,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        )


def test_init_bucket_batch_sampler_with_invalid_base_batch_sampler_kwargs(sample_data):
    (
        sizes,
        bucket_ranges,
        base_batch_sampler_class,
        base_batch_sampler_shared_kwargs,
        base_batch_sampler_individual_kwargs,
    ) = sample_data
    # base_batch_sampler_shared_kwargs and base_batch_sampler_individual_kwargs should be keyword argument dictionary
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs={1: False},
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        )
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=[("drop_last", False)],
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        )
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs={1: [2, 3, 5]},
        )
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs=[("batch_sizes", [2, 3, 5])],
        )
    # values in base_batch_sampler_individual_kwargs should have same length as bucket_ranges.
    with pytest.raises(ValueError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs={"batch_sizes": [2, 3]},
        )
    # base_batch_sampler_shared_kwargs and base_batch_sampler_individual_kwargs should provide
    # valid and sufficient arguments for base_batch_sampler_class
    with pytest.raises(TypeError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs={"drop_last": False, "shuffle": False},
            base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        )
    with pytest.raises(TypeError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
            base_batch_sampler_individual_kwargs={"batch_size": [2, 3, 5], "shuffle": [True, True, True]},
        )
    with pytest.raises(TypeError):
        BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=base_batch_sampler_class,
            base_batch_sampler_shared_kwargs={},
            base_batch_sampler_individual_kwargs={"batch_size": [2, 3, 5]},
        )


def test_bucket_batch_sampler_attributes(sample_data):
    (
        sizes,
        bucket_ranges,
        base_batch_sampler_class,
        base_batch_sampler_shared_kwargs,
        base_batch_sampler_individual_kwargs,
    ) = sample_data

    batch_sampler = BucketBatchSampler(
        sizes=sizes,
        bucket_ranges=bucket_ranges,
        base_batch_sampler_class=base_batch_sampler_class,
        base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
        base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
    )
    assert len(batch_sampler) == 8
    assert batch_sampler.num_buckets == 3
    assert np.all(batch_sampler.bucket_sizes == np.array([6, 9, 10]))
    assert batch_sampler.num_samples == len(sizes)
    assert np.all(np.sort(batch_sampler.bucket_indices[0]) == np.arange(6))
    assert np.all(np.sort(batch_sampler.bucket_indices[1]) == np.arange(6, 15))
    assert np.all(np.sort(batch_sampler.bucket_indices[2]) == np.arange(15, 25))


def test_iter_bucket_batch_sampler(sample_data):
    (
        sizes,
        bucket_ranges,
        base_batch_sampler_class,
        base_batch_sampler_shared_kwargs,
        base_batch_sampler_individual_kwargs,
    ) = sample_data

    batch_sampler = BucketBatchSampler(
        sizes=sizes,
        bucket_ranges=bucket_ranges,
        base_batch_sampler_class=base_batch_sampler_class,
        base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
        base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        shuffle=False,
    )
    batch_lists_first_iter = list(iter(batch_sampler))
    ref_batch_lists = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
    ]
    assert batch_lists_first_iter == ref_batch_lists
    batch_lists_second_iter = list(iter(batch_sampler))
    assert batch_lists_second_iter == ref_batch_lists


def test_iter_bucket_batch_sampler_with_shuffle(sample_data):
    (
        sizes,
        bucket_ranges,
        base_batch_sampler_class,
        base_batch_sampler_shared_kwargs,
        base_batch_sampler_individual_kwargs,
    ) = sample_data
    np.random.seed(0)
    batch_sampler = BucketBatchSampler(
        sizes=sizes,
        bucket_ranges=bucket_ranges,
        base_batch_sampler_class=base_batch_sampler_class,
        base_batch_sampler_shared_kwargs=base_batch_sampler_shared_kwargs,
        base_batch_sampler_individual_kwargs=base_batch_sampler_individual_kwargs,
        shuffle=True,
    )
    batch_lists_first_iter = list(iter(batch_sampler))
    ref_batch_lists_first_iter = [
        [9, 7, 13],
        [20, 17, 18, 19, 16],
        [12, 14, 6],
        [15, 24, 23, 22, 21],
        [5, 2],
        [10, 8, 11],
        [1, 3],
        [0, 4],
    ]
    assert batch_lists_first_iter == ref_batch_lists_first_iter
    batch_lists_second_iter = list(iter(batch_sampler))
    ref_batch_lists_second_iter = [
        [6, 14, 13],
        [5, 2],
        [12, 11, 10],
        [8, 7, 9],
        [17, 21, 20, 15, 16],
        [18, 22, 24, 19, 23],
        [1, 0],
        [3, 4],
    ]

    assert batch_lists_second_iter == ref_batch_lists_second_iter
    assert batch_lists_first_iter != ref_batch_lists_second_iter


def test_bucket_batch_sampler_with_size_aware_batch_sampler(sample_data):
    sizes, bucket_ranges, _, _, _ = sample_data
    item_costs = np.copy(sizes).tolist()

    def cost_of_element(index):
        return item_costs[index]

    batch_sampler = BucketBatchSampler(
        sizes=sizes,
        bucket_ranges=bucket_ranges,
        base_batch_sampler_class=SizeAwareBatchSampler,
        base_batch_sampler_shared_kwargs={"sizeof": cost_of_element},
        base_batch_sampler_individual_kwargs={"max_total_size": [10, 30, 50]},
        shuffle=False,
    )
    batch_lists_first_iter = list(iter(batch_sampler))
    ref_batch_lists = [
        [0, 1, 2, 3, 4],
        [5],
        [6, 7, 8, 9],
        [10, 11],
        [12, 13],
        [14],
        [15, 16, 17],
        [18, 19],
        [20, 21],
        [22, 23],
        [24],
    ]
    assert batch_lists_first_iter == ref_batch_lists
    batch_lists_second_iter = list(iter(batch_sampler))
    assert batch_lists_second_iter == ref_batch_lists

    # with shuffling
    np.random.seed(0)
    batch_sampler = BucketBatchSampler(
        sizes=sizes,
        bucket_ranges=bucket_ranges,
        base_batch_sampler_class=SizeAwareBatchSampler,
        base_batch_sampler_shared_kwargs={"sizeof": cost_of_element},
        base_batch_sampler_individual_kwargs={"max_total_size": [10, 30, 50]},
        shuffle=True,
    )
    batch_lists_first_iter = list(iter(batch_sampler))
    ref_batch_lists_first_iter = [
        [9, 7, 13],
        [20, 17],
        [12, 14],
        [18, 19],
        [5, 2, 1],
        [16, 15],
        [6, 10, 8],
        [24, 23],
        [11],
        [22, 21],
        [3, 0, 4],
    ]
    assert batch_lists_first_iter == ref_batch_lists_first_iter
    batch_lists_second_iter = list(iter(batch_sampler))
    ref_batch_lists_second_iter = [
        [19, 18],
        [7, 11, 9],
        [17, 23],
        [22, 24],
        [1, 3, 5, 0],
        [15, 16],
        [6, 13, 10],
        [4, 2],
        [12, 8],
        [21, 20],
        [14],
    ]

    assert batch_lists_second_iter == ref_batch_lists_second_iter
    assert batch_lists_first_iter != ref_batch_lists_second_iter
