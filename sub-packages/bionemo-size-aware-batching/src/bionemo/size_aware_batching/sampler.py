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

from typing import Any, Callable, Generator, Iterable, List, TypeVar, Union
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
    info_logger: Optional[Callable[[str], None]] = None,
    warn_logger: Optional[Callable[[str], None]] = None,
) -> Generator[Any, None, None]:
    """
    A generator that batches elements from an iterable while ensuring that the
    total size of each batch does not exceed a specified maximum. This can be
    useful for both indexible data or non-indexible but iterable data.

    Args:
        dataset (Iterable[Data]): The input iterable.
        max_total_size (int): The maximum total size of each batch.
        sizeof (Callable[[Data], Real]):
            A function or mapping that returns the size of each element in `dataset`.
        collate_fn (Optional[Callable[[Iterable[Data]], Any]], optional):
            An optional function to collate batches. Defaults to None.
        info_logger (Optional[Callable[[str], None]], optional): A function to log info.
            Defaults to None.
        warn_logger (Optional[Callable[[str], None]], optional): A function to log warnings.
            Defaults to None.

    Yields:
        Generator[Any, None, None]: A generator that yields batches from `dataset`.
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
        if not isinstance(new_size, Real):
            raise TypeError(f"Size of element is not int or float at index {data}")
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
            sampler (Union[Sampler[List[int]], Iterable[int]]): The underlying sampler.
            sizeof (Callable[[int], Real]): A function that returns the size at each index.
            max_total_size (Real): The maximum total size of a mini-batch.
            info_logger (Optional[Callable[[str], None]], optional): A function to log info.
                Defaults to a lambda function that print.
            warn_logger (Optional[Callable[[str], None]], optional): A function to log warnings.
                Defaults to a lambda function that warns.

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

    def __iter__(self) -> Generator[List[int], None, None]:
        """
        Iterate over batches of indices.

        This function yields batches of indices that do not exceed the maximum total size.

        Yields:
            List[int]: A batch of indices that do not exceed the maximum total size.
        """
        return size_aware_batching(
            self._sampler,
            self._sizeof,
            self._max_total_size,
            collate_fn=None,
            info_logger=self._info_logger,
            warn_logger=self._warn_logger,
        )
