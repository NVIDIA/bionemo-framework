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


import warnings
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Sampler


def create_buckets(
    sizes: Iterable[int], max_range: Union[int, float] = 20, min_bucket_count: int = 10_000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create buckets for a list of integers with pre-defined maximal range of interval and minimal bucket sizes.

    Args:
        sizes (Iterable[int]): An iterable of integers representing sizes.
        max_range (int | float, optional): The maximum range of a bucket. Defaults to 20.
        min_bucket_count (int, optional): The minimum count of a bucket. Defaults to 10_000.
            Bucket size may be smaller than min_bucket_count if its range reaches max_range.

    Raises:
        ValueError: If the provided sizes is empty, or not integers.
        ValueError: If max_range is is not non-negative integer or min_bucket_count is not positive integer.

    Returns:
        np.ndarray, np.ndarray: bucket ranges in ascending order and the number of elements in each bucket.
        e.g. np.array([[0, 5], [7,10]]), np.array([3,2]): specifies 2 buckets: 0<= sizes <= 5, 7 <= sizes <= 10, with 3 and 2 elements.
    """
    sizes = np.array(list(sizes))
    if sizes.ndim != 1 or not np.issubdtype(sizes.dtype, np.integer):
        raise ValueError("sizes should be a list of integers")

    if len(sizes) == 0:
        raise ValueError("sizes should not be empty")

    if not isinstance(max_range, (int, float)) or max_range < 0:
        raise ValueError(f"max_range should be non-negative but got {max_range}")

    if not isinstance(min_bucket_count, int) or min_bucket_count <= 0:
        raise ValueError(f"min_bucket_count should be positive integer but got {min_bucket_count}")

    counter = Counter(sizes.tolist())
    dist_size, dist_count = zip(*sorted(counter.items()))

    bucket_ranges = []
    bucket_sizes = []
    start = 0
    end = 1

    while start < len(dist_size):
        while (
            end < len(dist_size)
            and sum(dist_count[start:end]) < min_bucket_count
            and dist_size[end] - dist_size[start] <= max_range
        ):
            end += 1
        bucket_ranges.append([dist_size[start], dist_size[end - 1]])
        bucket_sizes.append(sum(dist_count[start:end]))
        start = end
        end = start + 1

    bucket_ranges = np.array(bucket_ranges)
    bucket_sizes = np.array(bucket_sizes)

    return bucket_ranges, bucket_sizes


class BucketBatchSampler(Sampler):
    """
    A batch sampler to create batches with elements from each pre-defined bucket ranges.
    A base batch sampler will be used for each bucket.

    Modified from https://github.com/rssrwn/semla-flow/blob/main/semlaflow/data/util.py

    Args:
        sizes (np.ndarray): A 1D numpy array of real numbers representing the size of each element in the dataset.
        bucket_ranges (np.ndarray): A 2D numpy array of real numbers with shape (num_buckets, 2) with each row representing the closed boundary of each bucket interval.
        base_batch_sampler_class (Sampler): Base batch sampler class type, which will be used for each bucket.
        base_batch_sampler_shared_kwargs (Dict[str, Any], optional): Shared keyword argument dict used to initialize all base batch samplers for all buckets.
            Sufficient and valid arguments should be provided for base_batch_sampler_class with base_batch_sampler_individual_kwargs. Default to  {}.
        base_batch_sampler_individual_kwargs (Dict[str, Iterable], optional): Keyword argument dict used to initialize each bucket batch sampler with the corresponding key value pairs.
            Length of each value in this dict must be equal to len(bucket_ranges) (the number of buckets).
            e.g. {'batch_size': [8,10,12]} will be used to create 3 batch samplers with batch_size = 8, 10, 12 for 3 buckets.
            Sufficient and valid arguments should be provided for base_batch_sampler_class with base_batch_sampler_shared_kwargs.
            Default to  {}.
        shuffle (bool): A boolean indicating whether to shuffle the dataset and buckets. Defaults to True.

    Raises:
        ValueError: If sizes is not a 1D numpy array of real numbers.
        ValueError: If bucket_ranges is not a 2D numpy array with shape (num_buckets, 2), or each row is not a valid interval, or the intervals overlap.
        ValueError: If the length of values in the dict of base_batch_sampler_individual_kwargs must be equal to len(bucket_ranges).
        RuntimeError: If there is no elements with sizes inside the buckets.

    ---------
    Examples:

    ```python
    >>> import torch
    >>> from bionemo.size_aware_batching.bucket_batch_sampler import BucketBatchSampler

    >>> # Define the sizes for a dataset
    >>> import numpy as np
    >>> sizes = np.arange(25)
    >>> # Define bucket ranges
    >>> bucket_ranges = np.array([[0,5],[6,14],[15,24]])

    >>> # Create a bucket batch sampler with torch.utils.data.BatchSampler as base batch sampler
    >>> # As there are 3 buckets, there will be 3 base batch samplers with batch sizes 2, 3, and 5.
    >>> batch_sampler = BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=torch.utils.data.BatchSampler,
            base_batch_sampler_shared_kwargs={'drop_last': False},
            base_batch_sampler_individual_kwargs={'batch_size': [2,3,5]},
            shuffle=False,
        )

    >>> # Iterate over batches of indices that lies in the same bucket and with different batch sizes.
    >>> print(list(batch_sampler))
    [[0, 1], [2, 3], [4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24]]

    >>> # randomize the dataset and buckets
    >>> np.random.seed(0)
    >>> batch_sampler = BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=torch.utils.data.BatchSampler,
            base_batch_sampler_shared_kwargs={'drop_last': False},
            base_batch_sampler_individual_kwargs={'batch_size': [2,3,5]},
            shuffle=True,
        )
    >>> print(list(batch_sampler))
    [[9, 7, 13], [20, 17, 18, 19, 16], [12, 14, 6], [15, 24, 23, 22, 21], [5, 2], [10, 8, 11], [1, 3], [0, 4]]
    >>> print(list(batch_sampler))
    [[6, 14, 13], [5, 2], [12, 11, 10], [8, 7, 9], [17, 21, 20, 15, 16], [18, 22, 24, 19, 23], [1, 0], [3, 4]]
    ```
    """

    def __init__(
        self,
        sizes: np.ndarray,
        bucket_ranges: np.ndarray,
        base_batch_sampler_class: Sampler,
        base_batch_sampler_shared_kwargs: Optional[Dict[str, Any]] = {},
        base_batch_sampler_individual_kwargs: Optional[Dict[str, Iterable]] = {},
        shuffle: bool = True,
    ):
        if (
            not isinstance(sizes, np.ndarray)
            or sizes.ndim != 1
            or not (np.issubdtype(sizes.dtype, np.integer) or np.issubdtype(sizes.dtype, np.floating))
        ):
            raise ValueError(f"sizes should be a 1D numpy array of real numbers, but got sizes={sizes}")

        if (
            not isinstance(bucket_ranges, np.ndarray)
            or bucket_ranges.ndim != 2
            or bucket_ranges.shape[-1] != 2
            or not (np.issubdtype(bucket_ranges.dtype, np.integer) or np.issubdtype(bucket_ranges.dtype, np.floating))
        ):
            raise ValueError(
                f"bucket_ranges should be a 2D numpy array of real numbers with shape (num_buckets, 2), but got bucket_ranges={bucket_ranges}"
            )

        idx = np.argsort(bucket_ranges[:, 0])
        bucket_ranges = bucket_ranges[idx]

        if np.any(bucket_ranges[:, 0] > bucket_ranges[:, 1]) or (
            len(bucket_ranges) > 1 and np.any(bucket_ranges[1:, 0] <= bucket_ranges[:-1, 1])
        ):
            raise ValueError(
                "Invalid buckets, buckets specifies the closed boundary of each bucket interval, with lower boundary <= upper boundary, and no overlap between buckets"
            )

        if not isinstance(shuffle, bool):
            raise ValueError(f"shuffle should be a boolean value, but got shuffle={shuffle}")

        self.sizes = sizes
        self.bucket_ranges = bucket_ranges
        self.num_buckets = len(bucket_ranges)
        self.shuffle = shuffle

        if not issubclass(base_batch_sampler_class, Sampler):
            raise ValueError(
                f"base_batch_sampler_class should be a batch sampler class inherited from torch.utils.data.Sampler, but got base_batch_sampler_class={base_batch_sampler_class}"
            )

        if not isinstance(base_batch_sampler_shared_kwargs, dict) or not all(
            isinstance(key, str) for key in base_batch_sampler_shared_kwargs.keys()
        ):
            raise ValueError(
                f"base_batch_sampler_shared_kwargs should be a keyword argument dictionary, but got base_batch_sampler_shared_kwargs={base_batch_sampler_shared_kwargs}"
            )

        if (
            not isinstance(base_batch_sampler_individual_kwargs, dict)
            or not all(isinstance(key, str) for key in base_batch_sampler_individual_kwargs.keys())
            or not all(len(value) == self.num_buckets for value in base_batch_sampler_individual_kwargs.values())
        ):
            raise ValueError(
                f"base_batch_sampler_individual_kwargs should be a keyword argument dictionary "
                f"with each of its value having length of number of buckets={self.num_buckets}, "
                f"but got base_batch_sampler_individual_kwargs={base_batch_sampler_individual_kwargs}"
            )

        self.base_batch_sampler_class = base_batch_sampler_class
        self.base_batch_sampler_shared_kwargs = base_batch_sampler_shared_kwargs
        self.base_batch_sampler_individual_kwargs = [
            {key: base_batch_sampler_individual_kwargs[key][k] for key in base_batch_sampler_individual_kwargs}
            for k in range(self.num_buckets)
        ]

        self.bucket_sizes: List[int]  # number of elements in each bucket
        self.bucket_indices: List[np.ndarray]  # List of elements' indices for each bucket

        bucket_indices = [np.argwhere((self.sizes >= st) * (self.sizes <= ed))[:, 0] for st, ed in self.bucket_ranges]
        self.bucket_indices = [bucket for bucket in bucket_indices if len(bucket) > 0]
        self.bucket_sizes = np.array([len(bucket) for bucket in self.bucket_indices])
        self.num_samples = np.sum(self.bucket_sizes)
        if self.num_samples == 0:
            raise RuntimeError("The sizes of all elements in the dataset are outside the bucket ranges provided")
        if self.num_samples < len(self.sizes):
            warnings.warn(
                f"{len(self.sizes) - self.num_samples} elements are outside the buckets provided and will be skipped"
            )

        self.base_batch_samplers: List[Sampler] = self._init_base_batch_samplers()

    def _init_base_batch_samplers(self):
        base_batch_samplers = []
        for k in range(self.num_buckets):
            base_batch_samplers.append(
                self.base_batch_sampler_class(
                    self.bucket_indices[k],
                    **self.base_batch_sampler_shared_kwargs,
                    **self.base_batch_sampler_individual_kwargs[k],
                )
            )
        return base_batch_samplers

    def __len__(self):
        # Can only be called if the base_batch_sampler has __len__ implemented
        num_batches = sum(len(sampler) for sampler in self.base_batch_samplers)
        return num_batches

    def __iter__(self):
        if self.shuffle:
            for indices in self.bucket_indices:
                np.random.shuffle(indices)

        base_batch_sampler_iters = [iter(batch_sampler) for batch_sampler in self.base_batch_samplers]
        bucket_remaining_elements = np.copy(self.bucket_sizes)
        total_remaining_elements = self.num_samples

        while total_remaining_elements > 0:
            if self.shuffle:
                bucket_idx = np.random.choice(self.num_buckets, p=bucket_remaining_elements / total_remaining_elements)
            else:
                bucket_idx = np.argmax(bucket_remaining_elements > 0)

            try:
                batch = next(base_batch_sampler_iters[bucket_idx])
                bucket_remaining_elements[bucket_idx] -= len(batch)
                total_remaining_elements -= len(batch)
                yield batch
            except StopIteration:
                bucket_remaining_elements[bucket_idx] = 0
                total_remaining_elements = np.sum(bucket_remaining_elements)
                continue
            except Exception as e:
                raise e
