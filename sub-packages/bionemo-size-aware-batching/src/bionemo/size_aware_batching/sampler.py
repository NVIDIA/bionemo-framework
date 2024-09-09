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
from typing import Callable, Dict, Iterable, Iterator, List, Union
from warnings import warn

from torch.utils.data import Sampler


class SizeAwareBatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices. Keeps track of
    the some total size of the mini-batch (eg number of nodes in
    the resulting graph), rather than the number of elements in the mini-batch.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        max_total_size (int or float): Max total size of the resulting mini-batch
          calculated as a sum of per-element idx_to_size, will be disobeyed if only one element in a batch
        idx_to_size (Iterable or Callable): idx_to_size[i] should be a size of i'th element in the dataset (eg number of nodes in the graph)
            if callable, will take a sample, and return the size
        batch_size_mean (int, optional): average number of samples in one batch. Default to be None.
    """

    def __init__(
        self,
        sampler: Union[Sampler[List[int]], Iterable[int]],
        max_total_size: Union[int, float],
        idx_to_size: Union[Dict[int, Union[int, float]], Sequence[int], Callable[[int], Union[int, float]]],
        batch_size_mean: int,
        do_caching: bool = False,
    ) -> None:
        if not isinstance(max_total_size, (int, float)) or max_total_size <= 0:
            raise ValueError(f"max_total_size should be a positive number, but got max_total_size={max_total_size}")
        if not isinstance(batch_size_mean, int) or batch_size_mean <= 0:
            raise ValueError(f"batch_size_mean should be a positive integer, but got batch_size_mean={batch_size_mean}")
        self._batch_size_mean = batch_size_mean
        self._num_batches = (len(sampler) + self._batch_size_mean - 1) // self._batch_size_mean

        self._is_idx_to_size_callable = callable(idx_to_size)
        self._is_idx_to_size_dict = isinstance(idx_to_size, dict)
        self._is_idx_to_size_seq = isinstance(idx_to_size, Sequence)

        if not (self._is_idx_to_size_callable or self._is_idx_to_size_dict or self._is_idx_to_size_seq):
            raise ValueError("idx_to_size can only be a callable, a dictionary or a sequence container")

        if do_caching and (self._is_idx_to_size_dict or self._is_idx_to_size_seq):
            raise ValueError("Caching is only supported for callable idx_to_size")

        self._do_caching = do_caching

        if self._do_caching:
            self._sizes_cache = {}
        else:
            self._sizes_cache = None

        is_debug = hasattr(sys, "gettrace") and sys.gettrace() is not None

        if is_debug and self._is_idx_to_size_seq:
            # O(n) expensive check
            # check the bounds for the sample indices
            idx_min = min(sampler)
            idx_max = max(sampler)
            size_idx = len(idx_to_size)
            if idx_min < 0 or idx_min >= size_idx or idx_max < 0 or idx_max >= size_idx:
                raise ValueError(
                    f"The index range of sampler [{idx_min}, {idx_max}] exceeds the index bounds of the sequence container [{0}, {size_idx-1}]"
                )

        if is_debug and (self._is_idx_to_size_dict or self._is_idx_to_size_seq):
            # O(n) expensive check
            if self._is_idx_to_size_dict:
                max_size = max(idx_to_size.values())
                min_size = min(idx_to_size.values())
            else:
                max_size = max(idx_to_size)
                min_size = min(idx_to_size)

            if max_size > max_total_size:
                warn(
                    "Sizes of some elements in the dataset exceed max_total_size "
                    f"{max_total_size}. Such elements will be skipped. max(idx_to_size) = {max_size}"
                )
            if min_size > max_total_size:
                raise RuntimeError(
                    f"Minimum element size in the dataset exceeds "
                    f"requested max_total_size ({min_size} > {max_total_size}). "
                    f"No samples can be generated."
                )

        self._sampler = sampler
        self._max_total_size = max_total_size
        self._idx_to_size = idx_to_size

    def __len__(self):
        return self._num_batches

    def __iter__(self) -> Iterator[List[int]]:
        batch_total_size = 0
        yield_num_batches = 0
        batch = []
        counter = 1
        num_samples = 0
        while yield_num_batches < self._num_batches:
            # TODO: triage the necessity of this check
            if counter > 1:
                warn(
                    f"Sampler is being used {counter} times in one epoch, try to increase batch_size_mean = {self._batch_size_mean}, "
                    f"yield_num_batches = {yield_num_batches}, num_batches = {self._num_batches}"
                )
            for idx in self._sampler:
                if self._sizes_cache is not None and idx in self._sizes_cache:
                    new_size = self._sizes_cache[idx]
                elif self._is_idx_to_size_callable:
                    new_size = self._idx_to_size(idx)
                    if self._sizes_cache is not None:
                        self._sizes_cache[idx] = new_size
                else:
                    # self._idx_to_size is dict or sequence
                    new_size = self._idx_to_size[idx]
                if new_size > self._max_total_size:
                    warn(
                        f"Size of element {idx} exceeds max_total_size"
                        f" ({new_size} > {self._max_total_size}), skipping"
                    )
                    num_samples += 1
                    continue
                if new_size + batch_total_size > self._max_total_size:
                    num_samples += len(batch)
                    yield batch
                    yield_num_batches += 1
                    batch_total_size = 0
                    batch = []
                batch.append(idx)
                batch_total_size += new_size
                if yield_num_batches >= self._num_batches:
                    if num_samples < len(self._sampler):
                        warn(
                            f"Only {num_samples} out of {len(self._sampler)} samples are used in one epoch, "
                            f"try to increase num_batches = {self._num_batches} or reduce batch_size = {self._batch_size_mean}"
                        )
                    break
            counter += 1
            if num_samples == 0:
                raise RuntimeError(
                    f"The underlying sampler has been exhausted once but no samples were generated. This could be due to "
                    f"the minimal size of elements exceeding the requested max_total_size {self._max_total_size}."
                )

        # return the last one
        if len(batch) > 0 and yield_num_batches < self._num_batches:
            num_samples += len(batch)
            yield batch
            yield_num_batches += 1

        if num_samples < len(self._sampler):
            warn(
                f"Only {num_samples} out of {len(self._sampler)} samples are used "
                f"try to increase max_total_size from {self._max_total_size} or "
                f"reduce batch_size_mean from {self._batch_size_mean}"
            )
