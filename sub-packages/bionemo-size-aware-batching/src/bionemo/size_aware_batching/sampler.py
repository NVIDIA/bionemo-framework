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
from typing import Callable, Dict, Generator, Iterable, List, TypeVar, Union
from warnings import warn

from torch.utils.data import Sampler


Data = TypeVar("Data")


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
        max_total_size: Union[int, float],
        sizeof: Union[Dict[int, Union[int, float]], Sequence[int], Callable[[int], Union[int, float]]],
    ) -> None:
        """
        Initializes the SizeAwareBatchSampler.

        Args:
            sampler (Union[Sampler[List[int]], Iterable[int]]): The underlying sampler.
            max_total_size (Union[int, float]): The maximum total size of a mini-batch.
            sizeof (Union[Dict[int, Union[int, float]], Sequence[int], Callable[[int], Union[int, float]]]):
                A function or data structure that returns the size at each index.

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
        self._max_total_size = max_total_size
        self._sizeof = sizeof

    def __iter__(self) -> Generator[List[int], None, None]:
        """
        Iterate over batches of indices.

        This function yields batches of indices that do not exceed the maximum total size.

        Yields:
            List[int]: A batch of indices that do not exceed the maximum total size.
        """
        batch_total_size = 0
        batch = []

        for idx in self._sampler:
            try:
                if self._is_sizeof_callable:
                    new_size = self._sizeof(idx)
                else:
                    # self._sizeof is dict or sequence
                    new_size = self._sizeof[idx]
            except Exception as e:
                raise RuntimeError(f"sizeof raises error at idx={idx}: {e}") from e
            if not isinstance(new_size, int) and not isinstance(new_size, float):
                raise TypeError(f"Size of element is not int or float at index {idx}")
            if new_size > self._max_total_size:
                warn(
                    f"Size of element {idx} exceeds max_total_size" f" ({new_size} > {self._max_total_size}), skipping"
                )
                continue
            if new_size + batch_total_size > self._max_total_size:
                yield batch
                batch_total_size = 0
                batch = []
            batch.append(idx)
            batch_total_size += new_size

        # return the remaining batch if there is
        if len(batch) > 0:
            yield batch
