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

import itertools
import sys
from warnings import warn

import numpy as np
import pytest
import torch
from torch.utils.data import SequentialSampler, default_collate

from bionemo.size_aware_batching.sampler import BucketBatchSampler, SizeAwareBatchSampler, size_aware_batching


@pytest.mark.parametrize(
    "collate_fn, max_total_size, warn_logger", itertools.product([None, default_collate], [0, 15, 31], [None, warn])
)
def test_sabs_iter(dataset, collate_fn, max_total_size, warn_logger):
    def sizeof(data: torch.Tensor):
        return ((data[0].item() + 1) % 3) * 10

    if warn_logger is not None and (max_total_size == 0 or max_total_size == 15):
        with pytest.warns(UserWarning):
            meta_batch_ids = list(
                size_aware_batching(dataset, sizeof, max_total_size, collate_fn=collate_fn, warn_logger=warn_logger)
            )
    else:
        meta_batch_ids = list(
            size_aware_batching(dataset, sizeof, max_total_size, collate_fn=collate_fn, warn_logger=warn_logger)
        )

    meta_batch_ids_expected = []
    ids_batch = []
    s_all = 0
    for data in dataset:
        s = sizeof(data)
        if s > max_total_size:
            continue
        if s + s_all > max_total_size:
            meta_batch_ids_expected.append(ids_batch)
            s_all = s
            ids_batch = [data]
            continue
        s_all += s
        ids_batch.append(data)
    if len(ids_batch) > 0:
        meta_batch_ids_expected.append(ids_batch)

    if collate_fn is not None:
        meta_batch_ids_expected = [collate_fn(batch) for batch in meta_batch_ids_expected]

    for i in range(len(meta_batch_ids)):
        torch.testing.assert_close(meta_batch_ids[i], meta_batch_ids_expected[i])


def test_sabs_init_invalid_sizeof_type(sampler):
    max_total_size = 60
    sizeof = " invalid type"
    with pytest.raises(TypeError, match="sizeof must be a callable"):
        list(size_aware_batching(sampler, sizeof, max_total_size))


def test_SABS_init_valid_input(sampler, get_sizeof):
    sizeof = get_sizeof
    max_total_size = 60
    batch_sampler = SizeAwareBatchSampler(sampler, sizeof, max_total_size)
    assert batch_sampler._sampler == sampler
    assert batch_sampler._max_total_size == max_total_size

    for idx in sampler:
        if callable(sizeof):
            assert batch_sampler._sizeof(idx) == sizeof(idx)
        else:
            assert batch_sampler._sizeof(idx) == sizeof[idx]


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
    with pytest.raises(TypeError, match="sizeof must be a callable"):
        SizeAwareBatchSampler(sampler, sizeof, max_total_size)


@pytest.mark.parametrize("max_total_size, warn_logger", itertools.product([0, 31, 60], [None, warn]))
def test_SABS_iter(sampler, get_sizeof, max_total_size, warn_logger):
    sizeof = get_sizeof

    if max_total_size == 0 and not callable(sizeof):
        sys.gettrace = lambda: True
        if warn_logger is not None:
            with pytest.raises(ValueError, match=r"exceeds max_total_size"), pytest.warns(UserWarning):
                size_aware_sampler = SizeAwareBatchSampler(sampler, sizeof, max_total_size, warn_logger=warn_logger)
        else:
            with pytest.raises(ValueError):
                size_aware_sampler = SizeAwareBatchSampler(sampler, sizeof, max_total_size, warn_logger=warn_logger)
        sys.gettrace = lambda: None
    else:
        # construction should always succeed
        size_aware_sampler = SizeAwareBatchSampler(sampler, sizeof, max_total_size, warn_logger=warn_logger)

        if max_total_size == 0 and warn_logger is not None:
            with pytest.warns(UserWarning):
                meta_batch_ids = list(size_aware_sampler)
        else:
            meta_batch_ids = list(size_aware_sampler)

        def fn_sizeof(i: int):
            if callable(sizeof):
                return sizeof(i)
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
        if max_total_size == 0 and warn_logger is not None:
            with pytest.warns(UserWarning):
                meta_batch_ids_2nd_pass = list(size_aware_sampler)
        else:
            meta_batch_ids_2nd_pass = list(size_aware_sampler)
        assert meta_batch_ids == meta_batch_ids_2nd_pass


def test_SABS_iter_no_samples(get_sizeof):
    # Test iterating over a batch of indices with no samples
    sampler = SequentialSampler([])
    sizeof = get_sizeof
    size_aware_sampler = SizeAwareBatchSampler(sampler, sizeof, 100)

    batched_indices = list(size_aware_sampler)

    assert not batched_indices


def test_SABS_iter_sizeof_raises(sampler):
    def sizeof(i: int):
        raise RuntimeError("error at data")

    size_aware_sampler = SizeAwareBatchSampler(sampler, sizeof, 1)

    with pytest.raises(RuntimeError, match="sizeof raises error at data"):
        list(size_aware_sampler)


def test_SABS_iter_sizeof_invalid_return_type(sampler):
    def sizeof(i: int):
        return str(i)

    size_aware_sampler = SizeAwareBatchSampler(sampler, sizeof, 1)

    with pytest.raises(TypeError, match="'>' not supported"):
        list(size_aware_sampler)


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
