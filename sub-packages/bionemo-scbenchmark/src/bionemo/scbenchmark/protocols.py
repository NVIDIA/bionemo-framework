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


"""Protocols for dataloader benchmarking.

This module defines the protocols that dataloaders should implement
to be benchmarkable. Using protocols provides type safety without
requiring inheritance, allowing any iterable object to be benchmarked
with proper type checking.
"""

from typing import Any, Iterator, Optional, Protocol


class DataloaderProtocol(Protocol):
    """Protocol for dataloaders that can be benchmarked.

    This protocol defines the minimum interface that a dataloader
    must implement to be benchmarkable. Any object that implements
    these methods can be benchmarked without inheritance. The protocol
    is designed to be compatible with most existing dataloader implementations
    including PyTorch DataLoaders, custom iterators, and generators.

    The protocol requires:
    - __iter__: Return an iterator over batches
    - __len__: Return the number of batches (optional but recommended)

    Note:
        - The __len__ method is optional but recommended for better
          progress reporting and limit handling
        - Any object that supports iteration can be benchmarked
        - The protocol is structural, not nominal, so no inheritance is needed
    """

    def __iter__(self) -> Iterator[Any]:
        """Return an iterator over the dataloader.

        This method should return an iterator that yields batches
        of data. Each batch can be any type (tensor, dict, list, etc.)
        as long as it can be processed by the benchmark framework.

        Returns:
            Iterator that yields batches of data
        """
        ...

    def __len__(self) -> Optional[int]:
        """Return the number of batches in the dataloader.

        This method is optional but recommended. If implemented,
        it should return the total number of batches that will be
        yielded by the iterator. This information is used for
        progress reporting and limit handling.

        Returns:
            Number of batches, or None if not known
        """
        ...


class DatasetProtocol(Protocol):
    """Protocol for datasets that can be wrapped in a dataloader.

    This protocol defines the interface for datasets that need to be
    converted to a dataloader before benchmarking. It's useful for
    datasets that implement the standard PyTorch dataset interface
    and need to be wrapped in a DataLoader or similar container.

    The protocol requires:
    - __len__: Return the number of samples in the dataset
    - __getitem__: Get a sample at the given index

    Note:
        - This protocol follows the standard PyTorch dataset interface
        - Datasets implementing this protocol can be easily wrapped
          in DataLoaders for benchmarking
        - The protocol is structural, so no inheritance is required
    """

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Total number of samples in the dataset
        """
        ...

    def __getitem__(self, idx: int) -> Any:
        """Get a sample at the given index.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Sample data at the specified index
        """
        ...


# Convenience type aliases for easier imports
DataloaderType = DataloaderProtocol
DatasetType = DatasetProtocol
