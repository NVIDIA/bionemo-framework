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


from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

from torch.utils.data import Dataset


__all__: Sequence[str] = (
    "SingleCellRowDataset",
    "SingleCellRowDatasetCore",
)


class SingleCellRowDatasetCore(ABC):
    """Implements the actual ann data-like interface."""

    @abstractmethod
    def load_h5ad(self, h5ad_path: str) -> None:
        """Loads an H5AD file and converts it into the backing representation.

        Calls to __len__ and __getitem__ Must be valid after a call to
        this method.
        """
        raise NotImplementedError()

    @abstractmethod
    def number_nonzero_values(self) -> int:
        """Return the number of non-zero values in the data."""
        raise NotImplementedError()

    @abstractmethod
    def number_of_values(self) -> int:
        """Return the total number of values in the data."""
        raise NotImplementedError()

    @abstractmethod
    def number_of_rows(self) -> int:
        """Return the number of rows in the data."""
        raise NotImplementedError()

    @abstractmethod
    def shape(self) -> Tuple[int, List[int]]:
        """Returns the shape of the object, which may be ragged.

        A ragged dataset is where the number and dimension of features
        can be different at every row.
        """
        raise NotImplementedError()

    def sparsity(self) -> float:
        """Return the sparsity of the underlying data.

        Sparsity is defined as the fraction of zero values in the data.
        It is within the range [0, 1.0]. If there are no values, the
        sparsity is defined as 0.0.
        """
        total_values = self.number_of_values()
        if total_values == 0:
            return 0.0

        nonzero_values = self.number_nonzero_values()
        zero_values = total_values - nonzero_values
        sparsity_value = zero_values / total_values
        return sparsity_value

    @abstractmethod
    def version(self) -> str:
        """Returns a version number.

        (following <major>.<minor>.<point> convention).
        """
        pass


class SingleCellRowDataset(SingleCellRowDatasetCore, Dataset):
    """One row in an ann dataframe (hdf5 file with a spare array format)."""

    @abstractmethod
    def load(self, data_path: str) -> None:
        """Loads the data from datapath.

        Calls to __len__ and __getitem__ Must be valid after a call to
        this method.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, data_path: str) -> None:
        """Saves the class to an archive at datapath."""
        raise NotImplementedError()

    pass
