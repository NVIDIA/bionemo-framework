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

from typing import Callable, Iterable, Iterator, List, Optional, Sequence, TypeVar, Union

from torch.utils.data import Sampler


__all__: Sequence[str] = (
    "size_aware_batching",
    "SizeAwareBatchSampler",
    "Real",
)

Data = TypeVar("Data")
BatchCollated = TypeVar("BatchCollated")
Real = Union[int, float]


def size_aware_batching(
    dataset: Iterable[Data],
    sizeof: Callable[[Data], Real],
    max_total_size: Real,
    collate_fn: Optional[Callable[[Iterable[Data]], BatchCollated]] = None,
    info_logger: Optional[Callable[[str], None]] = None,
    warn_logger: Optional[Callable[[str], None]] = None,
) -> Iterator[Union[List[Data], BatchCollated]]:
    """Creates a batching iterator where each batch size varries (within a max limit) according to memory consumption.

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
        collate_fn: An optional function to collate batches. Defaults to None, in which case
            each batch is a list of elements from the input dataset
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

    Example:
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
    """Varriying-size batching data sampler class that ensures batch size doesn't exceed maximum.

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
        info_logger: Optional[Callable[[str], None]] = None,
        warn_logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initializes the SizeAwareBatchSampler.

        Args:
            sampler: The underlying sampler.
            sizeof: A function that returns the size at each index. E.g., this can used to
                determine how much memory an element consumes. Its return type must be
                comparable with `max_total_size` and it must be addable (operator `+`).
            max_total_size: The maximum total size of a mini-batch. The semantics of "size"
                is defined by the `sizeof` argument. The type of this value must be comparable
                with the return type of sizeof, i.e., the operator `<` and `==` must be meaningful.
            info_logger: A function to log info. Defaults to None.
            warn_logger: A function to log warnings. Defaults None.

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
        """Iterate over batches of indices.

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
