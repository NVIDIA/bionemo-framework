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
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Type, TypeVar, Union
from warnings import warn

import numpy as np
from torch.utils.data import Sampler


Data = TypeVar("Data")
Real = Union[int, float]


def size_aware_batching(
    dataset: Iterable[Data],
    sizeof: Callable[[Data], Real],
    max_total_size: Real,
    collate_fn: Optional[Callable[[Iterable[Data]], Any]] = None,
    info_logger: Optional[Callable[[str], None]] = None,
    warn_logger: Optional[Callable[[str], None]] = None,
) -> Iterator[Any]:
    """
    A generator that batches elements from an iterable while ensuring that the
    total size of each batch does not exceed a specified maximum. Here the size
    can be a measurement of memory consumption of the elements in the batch.
    This can be useful for both indexible data or non-indexible but iterable data.

    Args:
        dataset: The input iterable.
        sizeof: A function or mapping that returns the "size" of each element in `dataset`.
            E.g., this can used to determine how much memory an element consumes. Its return
            type must be comparable with `max_total_size` and it must be addable (operator `+`).
        max_total_size: The maximum total "size" of each batch. The semantics of "size"
            is defined by the `sizeof` argument. The type of this value must be comparable
            with the return type of sizeof, i.e., the operator `<` and `==` must be meaningful.
        collate_fn: An optional function to collate batches. Defaults to None.
        info_logger: A function to log info. Defaults to None.
        warn_logger: A function to log warnings. Defaults to None.

    Yields:
        A generator that yields batches from `dataset`.

    -----------
    Assumptions
    1. Linear complexity. This function consumes the given Iterable of data (`dataset`) once,
       by going over the data item one by one to build a batch and yield it as soon as the
       addition of the next data item to the batch would exceed `max_total_size` or if the
       batch is the last one (end of iteration)
    2. Additive size measurement. For the general usage case of building mini-batches with
       a threshold of the batch's memory consumption, it assumes that the size of the batch is
       the sum of all elements in the batch (additive property).
    3. Comparable type of `max_total_size` and `sizeof`'s return. `sizeof`'s return values
       must be compared with `max_total_size` to threshold the size of batches


    ------
    Caveat
    1: The generated batch sizes may have large variance
       - how to workaround: filter the output of this generator using a batch size threshold
    2: The number of batches may vary a lot across different epochs.
       - how to workaround: increase the number of steps that compose an epoch,
         e.g., in the Lightning training/validation loop, which effectively increases the input
         dataset size per epoch


    -------
    Example

    ```python
    >>> import torch
    >>> from torch.utils.data import default_collate
    >>> from bionemo.size_aware_batching.sampler import size_aware_batching

    >>> # Define a sample dataset with torch.tensor
    >>> dataset = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6]),
    ...            torch.tensor([7, 8]), torch.tensor([9, 10])]

    >>> # Define a sizeof function that returns the size of each tensor
    >>> def sizeof(x):
    ...     return x.numel()

    >>> # Create a generator with max_total_size=4 and default_collate_fn
    >>> gen = size_aware_batching(dataset, sizeof, 4, collate_fn=default_collate)
    >>> batches = list(gen)
    >>> print(batches)
        [tensor([[1, 2], [3, 4]]), tensor([[5, 6], [7, 8]]), tensor([[9, 10]])]
    ```

    """
    is_sizeof_callable = callable(sizeof)
    has_collate_fn = collate_fn is not None and callable(collate_fn)

    if not is_sizeof_callable:
        raise TypeError("sizeof must be a callable")

    batch_total_size = 0
    batch = []
    n_samples = 0
    n_samples_batched = 0
    n_batches = 0
    for data in dataset:
        n_samples += 1
        try:
            new_size = sizeof(data)
        except Exception as e:
            raise RuntimeError(f"sizeof raises error at data={data}: {e}") from e
        if new_size > max_total_size:
            if warn_logger is not None:
                warn_logger(
                    f"Size of element {data} exceeds max_total_size" f" ({new_size} > {max_total_size}), skipping"
                )
            continue
        if new_size + batch_total_size > max_total_size:
            n_batches += 1
            if has_collate_fn:
                yield collate_fn(batch)
            else:
                yield batch
            batch_total_size = 0
            batch = []
        batch.append(data)
        n_samples_batched += 1
        batch_total_size += new_size

    # return the remaining batch if there is
    if len(batch) > 0:
        n_batches += 1
        if has_collate_fn:
            yield collate_fn(batch)
        else:
            yield batch

    if warn_logger is not None and n_samples_batched < n_samples:
        warn_logger(
            f"{n_samples_batched} samples were batched from {n_samples} "
            f"of the input data. Missing samples are due to exceeding max_total_size={max_total_size})"
        )

    if info_logger is not None:
        info_logger(
            f"Batched {n_samples_batched} samples into {n_batches} batches. "
            f"If this doesn't match the your expectation, consider adjusting "
            f"max_total_size or the sizeof functor"
        )


class SizeAwareBatchSampler(Sampler[List[int]]):
    """
    A sampler that batches elements of varying sizes while ensuring
    that the total size of each batch does not exceed a specified maximum.

    This is useful when dealing with datasets where each element has a
    different size, such as graphs or sequences of varying lengths.
    The sampler uses a provided `sizeof` function to determine the size
    of each element in the dataset and ensures that the total size of
    each batch does not exceed the specified `max_total_size`.

    ---------
    Examples:

    ```python
    >>> import torch
    >>> from bionemo.size_aware_batching.sampler import SizeAwareBatchSampler


    >>> # Define a sample dataset with torch.tensor
    >>> dataset = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6]),
    ...            torch.tensor([7, 8]), torch.tensor([9, 10])]


    >>> # Define a function that returns the size of each element in the dataset.
    >>> def sizeof(index):
    ...     return dataset[index].numel()


    >>> # Create a SizeAwareBatchSampler with a maximum total batch size of 10.
    >>> batch_sampler = SizeAwareBatchSampler(
    ...     sampler=torch.utils.data.SequentialSampler(dataset),
    ...     sizeof=sizeof,
    ...     max_total_size=4
    ... )


    >>> # Iterate over batches of indices that do not exceed the maximum total size.
    >>> print(list(batch_sampler))
        [[0, 1], [2, 3], [4]]
    ```
    """

    def __init__(
        self,
        sampler: Union[Sampler[List[int]], Iterable[int]],
        sizeof: Callable[[int], Real],
        max_total_size: Real,
        info_logger: Optional[Callable[[str], None]] = lambda msg: print(msg),
        warn_logger: Optional[Callable[[str], None]] = lambda msg: warn(msg),
    ) -> None:
        """
        Initializes the SizeAwareBatchSampler.

        Args:
            sampler: The underlying sampler.
            sizeof: A function that returns the size at each index. E.g., this can used to
                determine how much memory an element consumes. Its return type must be
                comparable with `max_total_size` and it must be addable (operator `+`).
            max_total_size: The maximum total size of a mini-batch. The semantics of "size"
                is defined by the `sizeof` argument. The type of this value must be comparable
                with the return type of sizeof, i.e., the operator `<` and `==` must be meaningful.
            info_logger: A function to log info. Defaults to a lambda function that print.
            warn_logger: A function to log warnings. Defaults to a lambda function that warns.

        Raises:
            TypeError: If sampler is not an instance of Sampler or Iterable, or if sizeof is not a callable, dictionary, or sequence container.
            ValueError: If max_total_size is not a positive number.

        """
        if not (isinstance(sampler, Sampler) or (isinstance(sampler, Iterable) and not isinstance(sampler, str))):
            raise TypeError("sampler should be an instance of torch.utils.data.Sampler or Iterable")

        if not isinstance(max_total_size, Real):
            raise ValueError(f"max_total_size should be int or float but got {type(max_total_size)}")

        self._info_logger = info_logger
        self._warn_logger = warn_logger

        self._is_sizeof_callable = callable(sizeof)

        if not self._is_sizeof_callable:
            raise TypeError("sizeof must be a callable")

        self._sampler = sampler
        self._sizeof = sizeof
        self._max_total_size = max_total_size

    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterate over batches of indices.

        This function yields batches of indices that do not exceed the maximum total size.

        Yields:
            A batch of indices that do not exceed the maximum total size.
        """
        return size_aware_batching(
            self._sampler,
            self._sizeof,
            self._max_total_size,
            collate_fn=None,
            info_logger=self._info_logger,
            warn_logger=self._warn_logger,
        )


class BucketBatchSampler(Sampler[List[int]]):
    """
    A batch sampler to create batches with sizes of elements from each pre-defined bucket ranges.
    A base batch sampler will be used for each bucket.

    Modified from https://github.com/rssrwn/semla-flow/blob/main/semlaflow/data/util.py

    Args:
        sizes (np.ndarray): A 1D numpy array of real numbers representing the size of each element in the dataset.
        bucket_ranges (np.ndarray): A 2D numpy array of real numbers with shape (num_buckets, 2) with each row representing the closed boundary of each bucket interval.
        base_batch_sampler_class (Type[Sampler]): Base batch sampler class type, which will be used for each bucket.
        base_batch_sampler_shared_kwargs (Dict[str, Any], optional): Shared keyword argument dictionary used to initialize all base batch samplers for all buckets.
            Sufficient and valid arguments should be provided for `base_batch_sampler_class` with `base_batch_sampler_individual_kwargs`. Default to  {}.
        base_batch_sampler_individual_kwargs (Dict[str, Iterable], optional): Keyword argument dictionary used to initialize each bucket batch sampler with the corresponding key value pairs.
            Length of each value in this dict must be equal to len(`bucket_ranges`) (the number of buckets).
            e.g. {'batch_size': [8,10,12]} will be used to create 3 batch samplers with batch_size = 8, 10, 12 for 3 buckets.
            Sufficient and valid arguments should be provided for `base_batch_sampler_class` with `base_batch_sampler_shared_kwargs`.
            Default to  {}.
        shuffle (bool): A boolean indicating whether to shuffle the dataset and buckets. Defaults to True.

    Raises:
        ValueError: If `sizes` is not a 1D numpy array of real numbers.
        ValueError: If `bucket_ranges` is not a 2D numpy array with shape (num_buckets, 2), or each row is not a valid interval, or the intervals overlap.
        ValueError: If `base_batch_sampler_individual_kwargs` or `base_batch_sampler_individual_kwargs` is not a keyword argument dictionary.
        ValueError: If the length of values in the dict of `base_batch_sampler_individual_kwargs` must be equal to len(bucket_ranges).
        RuntimeError: If there is no elements with sizes inside the `bucket_ranges`.

    ---------
    Examples:

    ```python
    >>> import torch
    >>> from bionemo.size_aware_batching.sampler import BucketBatchSampler

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
    >>> # Combine with SizeAwareBatchSampler to control the cost of each batch
    >>> from bionemo.size_aware_batching.sampler import SizeAwareBatchSampler
    >>> item_costs = np.copy(sizes).tolist()
    >>> def cost_of_element(index):
            return item_costs[index]
    >>> np.random.seed(0)
    >>> batch_sampler = BucketBatchSampler(
            sizes=sizes,
            bucket_ranges=bucket_ranges,
            base_batch_sampler_class=SizeAwareBatchSampler,
            base_batch_sampler_shared_kwargs={"sizeof": cost_of_element, "max_total_size": 40},
            base_batch_sampler_individual_kwargs={},
            shuffle=True,
        )
    >>> print(list(iter(batch_sampler)))
    [[9, 7, 13], [20, 17], [12, 14, 6], [18, 19], [5, 2, 1, 3, 0, 4], [16, 15], [24], [23], [10, 8, 11], [22], [21]]
    """

    def __init__(
        self,
        sizes: np.ndarray,
        bucket_ranges: np.ndarray,
        base_batch_sampler_class: Type[Sampler],
        base_batch_sampler_shared_kwargs: Optional[Dict[str, Any]] = {},
        base_batch_sampler_individual_kwargs: Optional[Dict[str, Iterable]] = {},
        shuffle: bool = True,
    ) -> None:
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
            or not all(len(list(value)) == self.num_buckets for value in base_batch_sampler_individual_kwargs.values())
        ):
            raise ValueError(
                f"base_batch_sampler_individual_kwargs should be a keyword argument dictionary "
                f"with each of its value having length of number of buckets={self.num_buckets}, "
                f"but got base_batch_sampler_individual_kwargs={base_batch_sampler_individual_kwargs}"
            )

        self.base_batch_sampler_class = base_batch_sampler_class
        self.base_batch_sampler_shared_kwargs = base_batch_sampler_shared_kwargs
        self.base_batch_sampler_individual_kwargs = [
            {key: list(base_batch_sampler_individual_kwargs[key])[k] for key in base_batch_sampler_individual_kwargs}
            for k in range(self.num_buckets)
        ]

        self.bucket_sizes: np.ndarray  # number of elements in each bucket
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

    def _init_base_batch_samplers(self) -> list[Sampler]:
        """
        Initialize batch samplers for each bucket

        Returns:
            List of batch samplers.
        """
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

    def __len__(self) -> int:
        # Can only be called if the base_batch_sampler has __len__ implemented
        num_batches = sum(len(sampler) for sampler in self.base_batch_samplers)  # type: ignore
        return num_batches

    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterate over batches of indices.

        This function yields batches of indices of elements with sizes from each bucket range.

        Yields:
            List[int]: A batch of indices of elements with sizes from each bucket range.
        """
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
