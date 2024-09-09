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
    r"""Wraps another sampler to yield a mini-batch of indices. Keeps track of
    the some total size of the mini-batch (eg number of nodes in
    the resulting graph), rather than the number of elements in the mini-batch.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        max_total_size (int or float): Max total size of the resulting mini-batch
          calculated as a sum of per-element sizeof, will be disobeyed if only one element in a batch
        sizeof (Iterable or Callable): sizeof[i] should be a size of i'th element in the dataset (eg number of nodes in the graph)
            if callable, will take a sample, and return the size
        batch_size_mean (int, optional): average number of samples in one batch. Default to be None.
    """

    def __init__(
        self,
        sampler: Union[Sampler[List[int]], Iterable[int]],
        max_total_size: Union[int, float],
        sizeof: Union[Dict[int, Union[int, float]], Sequence[int], Callable[[Data], Union[int, float]]],
        do_caching: bool = False,
        dataset: Sequence[Data] = None,
    ) -> None:
        if not isinstance(max_total_size, (int, float)) or max_total_size <= 0:
            raise ValueError(f"max_total_size should be a positive number, but got max_total_size={max_total_size}")

        self._is_sizeof_callable = callable(sizeof)
        self._is_sizeof_dict = isinstance(sizeof, dict)
        self._is_sizeof_seq = isinstance(sizeof, Sequence)

        if not (self._is_sizeof_callable or self._is_sizeof_dict or self._is_sizeof_seq):
            raise ValueError("sizeof can only be a callable, a dictionary or a sequence container")

        if do_caching and (self._is_sizeof_dict or self._is_sizeof_seq):
            raise ValueError("Caching is only supported for callable sizeof")

        self._do_caching = do_caching

        if self._do_caching:
            self._sizes_cache = {}
        else:
            self._sizes_cache = None

        if self._is_sizeof_callable and dataset is None:
            raise ValueError("dataset should be provided when using callable sizeof")

        if not self._is_sizeof_callable and dataset is not None:
            raise ValueError(
                "When using a predefined dict or sequenct container sizeof, dataset should not be provided"
            )

        self._dataset = dataset

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
                    "Sizes of some elements in the dataset exceed max_total_size "
                    f"{max_total_size}. Such elements will be skipped. max(sizeof) = {max_size}"
                )
            if min_size > max_total_size:
                raise RuntimeError(
                    f"Minimum element size in the dataset exceeds "
                    f"requested max_total_size ({min_size} > {max_total_size}). "
                    f"No samples can be generated."
                )

        self._sampler = sampler
        self._max_total_size = max_total_size
        self._sizeof = sizeof

    def __iter__(self) -> Generator[List[int], None, None]:
        batch_total_size = 0
        batch = []

        for idx in self._sampler:
            if self._sizes_cache is not None and idx in self._sizes_cache:
                new_size = self._sizes_cache[idx]
            elif self._is_sizeof_callable:
                new_size = self._sizeof(self._dataset[idx])
                if self._sizes_cache is not None:
                    self._sizes_cache[idx] = new_size
            else:
                # self._sizeof is dict or sequence
                new_size = self._sizeof[idx]
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
