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
from collections.abc import Sequence
from typing import Any, Callable, Dict, Generator, Iterable, List, TypeVar, Union
from warnings import warn

from git import Optional
from torch.utils.data import Sampler


Data = TypeVar("Data")
Real = Union[int, float]


def size_aware_batching(
    dataset: Iterable[Data],
    sizeof: Callable[[Data], Real],
    max_total_size: int,
    collate_fn: Optional[Callable[[Iterable[Data]], Any]] = None,
) -> Generator[Any, None, None]:
    """
    A generator that batches elements from an iterable while ensuring that the
    total size of each batch does not exceed a specified maximum.

    Args:
        dataset (Iterable[Data]): The input iterable.
        max_total_size (int): The maximum total size of each batch.
        sizeof (Callable[[int], Real]):
            A function or mapping that returns the size of each element in `dataset`.
        collate_fn (Optional[Callable[[Iterable[Data]], Any]], optional):
            An optional function to collate batches. Defaults to None.

    Yields:
        Generator[Any, None, None]: A generator that yields batches from `dataset`.
    """
    is_sizeof_callable = callable(sizeof)
    has_collate_fn = collate_fn is not None and callable(collate_fn)

    if not is_sizeof_callable:
        raise TypeError("sizeof can only be a callable")

    batch_total_size = 0
    batch = []
    for data in dataset:
        try:
            new_size = sizeof(data)
        except Exception as e:
            raise RuntimeError(f"sizeof raises error at data={data}: {e}") from e
        if not isinstance(new_size, Real):
            raise TypeError(f"Size of element is not int or float at index {data}")
        if new_size > max_total_size:
            warn(f"Size of element {data} exceeds max_total_size" f" ({new_size} > {max_total_size}), skipping")
            continue
        if new_size + batch_total_size > max_total_size:
            if has_collate_fn:
                yield collate_fn(batch)
            else:
                yield batch
            batch_total_size = 0
            batch = []
        batch.append(data)
        batch_total_size += new_size

    # return the remaining batch if there is
    if len(batch) > 0:
        if has_collate_fn:
            yield collate_fn(batch)
        else:
            yield batch


class SizeAwareBatchSampler(Sampler[List[int]]):
    """
    A sampler that batches elements of varying sizes while ensuring
    that the total size of each batch does not exceed a specified maximum.

    This is useful when dealing with datasets where each element has a
    different size, such as graphs or sequences of varying lengths.
    The sampler uses a provided `sizeof` function to determine the size
    of each element in the dataset and ensures that the total size of
    each batch does not exceed the specified `max_total_size`.

    """

    def __init__(
        self,
        sampler: Union[Sampler[List[int]], Iterable[int]],
        sizeof: Union[Dict[int, Real], Sequence[Real], Callable[[int], Real]],
        max_total_size: Real,
    ) -> None:
        """
        Initializes the SizeAwareBatchSampler.

        Args:
            sampler (Union[Sampler[List[int]], Iterable[int]]): The underlying sampler.
            sizeof (Union[Dict[int, Real], Sequence[Real], Callable[[int], Real]]):
                A function or data structure that returns the size at each index.
            max_total_size (Real): The maximum total size of a mini-batch.

        Raises:
            TypeError: If sampler is not an instance of Sampler or Iterable, or if sizeof is not a callable, dictionary, or sequence container.
            ValueError: If max_total_size is not a positive number.

        """
        if not (isinstance(sampler, Sampler) or (isinstance(sampler, Iterable) and not isinstance(sampler, str))):
            raise TypeError("sampler should be an instance of torch.utils.data.Sampler or Iterable")

        if not isinstance(max_total_size, (int, float)) or max_total_size <= 0:
            raise ValueError(f"max_total_size should be a positive number, but got max_total_size={max_total_size}")

        self._is_sizeof_callable = callable(sizeof)
        self._is_sizeof_dict = isinstance(sizeof, dict)
        self._is_sizeof_seq = isinstance(sizeof, Sequence) and not isinstance(sizeof, str)

        if not (self._is_sizeof_callable or self._is_sizeof_dict or self._is_sizeof_seq):
            raise TypeError("sizeof can only be a callable, a dictionary or a sequence container")

        is_debug = hasattr(sys, "gettrace") and sys.gettrace() is not None

        if is_debug and self._is_sizeof_seq:
            # O(n) expensive check
            # check the bounds for the sample indices
            idx_min = min(sampler)
            idx_max = max(sampler)
            size_idx = len(sizeof)
            if idx_min < 0 or idx_min >= size_idx or idx_max < 0 or idx_max >= size_idx:
                raise ValueError(
                    f"The index range of sampler [{idx_min}, {idx_max}] exceeds the index bounds of the sequence container [{0}, {size_idx-1}]"
                )

        if is_debug and (self._is_sizeof_dict or self._is_sizeof_seq):
            # O(n) expensive check
            if self._is_sizeof_dict:
                max_size = max(sizeof.values())
                min_size = min(sizeof.values())
            else:
                max_size = max(sizeof)
                min_size = min(sizeof)

            if max_size > max_total_size:
                warn(
                    "Sizes of some elements in the `sizeof` data structure exceed max_total_size "
                    f"{max_total_size}. Such elements will be skipped. max(sizeof) = {max_size}"
                )
            if min_size > max_total_size:
                raise ValueError(
                    f"Minimum element size in the `sizeof` data structure exceeds "
                    f"requested max_total_size ({min_size} > {max_total_size}). "
                    f"No samples can be generated."
                )

        self._sampler = sampler
        if not self._is_sizeof_callable:
            self._sizeof = lambda i: sizeof[i]
        else:
            self._sizeof = sizeof
        self._max_total_size = max_total_size

    def __iter__(self) -> Generator[List[int], None, None]:
        """
        Iterate over batches of indices.

        This function yields batches of indices that do not exceed the maximum total size.

        Yields:
            List[int]: A batch of indices that do not exceed the maximum total size.
        """
        return size_aware_batching(self._sampler, self._sizeof, self._max_total_size)
