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

from __future__ import annotations

import importlib.metadata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


__all__: Sequence[str] = ("RowFeatureIndex",)


def are_dicts_equal(dict1: dict[str, np.ndarray], dict2: dict[str, np.ndarray]) -> bool:
    """Compare two dictionaries with string keys and numpy.ndarray values.

    Args:
        dict1 (dict[str, np.ndarray]): The first dictionary to compare.
        dict2 (dict[str, np.ndarray]): The second dictionary to compare.

    Returns:
        bool: True if the dictionaries have the same keys and all corresponding
              numpy arrays are equal; False otherwise.
    """
    return dict1.keys() == dict2.keys() and all(np.array_equal(dict1[k], dict2[k]) for k in dict1)


class RowFeatureIndex(ABC):
    """Maintains a mapping between a row or column and its features.

    This is a ragged dataset, where the number and dimension of features
    can be different at every row.

    Attributes:
        _cumulative_sum_index: Pointer that deliniates which entries
        correspond to a given row. For examples if the array is [-1, 200, 201],
        rows 0 to 199 correspond to _feature_arr[0] and 200 corresponds to
        _feature_arr[1]
        _feature_arr: list of feature dictionaries for each dataset
        _num_entries_per_row: list that tracks the feature length (number of genes) for each dataset.
        Extracting this information repeatedly from self._feature_arr would be cumbersome which is why we
        add this attribute.
        _labels: list of labels
        _version: The version of the dataset
    """

    def __init__(self) -> None:
        """Instantiates the index."""
        self._cumulative_sum_index: np.array = np.array([-1])
        self._feature_arr: list[dict[str, np.ndarray]] = []
        self._num_entries_per_row: list[int] = []
        self._version = importlib.metadata.version("bionemo.scdl")
        self._labels: list[str] = []

    def _get_dataset_id(self, row) -> int:
        """Gets the dataset id for a specified row index.

        Args:
            row (int): The index of the row.

        Returns:
            An int representing the dataset id the row belongs to.
        """
        if row < 0:
            raise IndexError(f"Row index {row} is not valid. It must be non-negative.")
        if len(self._cumulative_sum_index) < 2:
            raise IndexError("There are no features to lookup.")
        if row >= self._cumulative_sum_index[-1]:
            raise IndexError(
                f"Row index {row} is larger than number of rows in FeatureIndex ({self._cumulative_sum_index[-1]})."
            )

        # creates a mask for values where cumulative sum > row
        mask = ~(self._cumulative_sum_index > row)
        # Sum these to get the index of the first range > row
        # Subtract one to get the range containing row.
        d_id = sum(mask) - 1
        return d_id

    @staticmethod
    def _load_common(datapath: str, instance: "RowFeatureIndex") -> "RowFeatureIndex":
        parquet_data_paths = sorted(Path(datapath).rglob("*.parquet"))
        data_tables = [pq.read_table(csv_path) for csv_path in parquet_data_paths]
        instance._feature_arr = [
            {column: table[column].to_numpy() for column in table.column_names} for table in data_tables
        ]
        instance._num_entries_per_row = []
        for features in instance._feature_arr:
            instance._extend_num_entries_per_row(features)
        instance._cumulative_sum_index = np.load(Path(datapath) / "cumulative_sum_index.npy")
        instance._labels = np.load(Path(datapath) / "labels.npy", allow_pickle=True)
        instance._version = np.load(Path(datapath) / "version.npy").item()
        return instance

    def _filter_features(
        self, features_dict: dict[str, np.ndarray], select_features: Optional[list[str]]
    ) -> list[np.ndarray]:
        if select_features is not None:
            features: list[np.ndarray] = []
            for feature in select_features:
                if feature not in features_dict:
                    raise ValueError(f"Provided feature column {feature} in select_features not present in dataset.")
                features.append(features_dict[feature])
            return features
        return [features_dict[f] for f in features_dict]

    def version(self) -> str:
        """Returns a version number.

        (following <major>.<minor>.<point> convention).
        """
        return self._version

    def __len__(self) -> int:
        """The length is the number of rows or FeatureIndex length."""
        return len(self._feature_arr)

    @abstractmethod
    def _extend_num_entries_per_row(self, features: dict[str, np.ndarray]) -> None:
        """Extend the number of entries per row for a concrete index implementation."""
        ...

    def _check_and_append(self, n_obs: int, features: dict[str, np.ndarray], total_csum: Optional[str] = None) -> bool:
        """Subclass hook to optionally merge with last block. Return True if merged."""
        return False

    def number_vars_at_row(self, row: int) -> int:
        """Return number of variables in a given row (base: uses stored per-dataset counts)."""
        dataset_idx = self._get_dataset_id(row)
        return self._num_entries_per_row[dataset_idx]

    def column_dims(self) -> list[int]:
        """Return the number of columns in all rows.

        Args:
            length of features at every row is returned.

        Returns:
            A list containing the lengths of the features in every row
        """
        return self._num_entries_per_row

    def number_of_values(self) -> list[int]:
        """Get the total number of values in the array.

        For each row, the number of genes is counted.

        Returns:
            A list containing the lengths of the features in every block of rows
        """
        if len(self._feature_arr) == 0:
            return [0]
        rows = [
            self._cumulative_sum_index[i] - max(self._cumulative_sum_index[i - 1], 0)
            for i in range(1, len(self._cumulative_sum_index))
        ]
        vals = []
        vals = [n_rows * self._num_entries_per_row[i] for i, n_rows in enumerate(rows)]
        return vals

    def append_features(self, n_obs: int, features: object, label: Optional[str] = None) -> None:
        """Append features, delegating validation and merge behavior to subclasses."""
        if not isinstance(features, dict):
            raise TypeError(f"{self.__class__.__name__}.append_features expects a dict of arrays")

        total_csum = max(self._cumulative_sum_index[-1], 0) + n_obs

        # Optionally merge into previous block
        if self._check_and_append(n_obs, features, total_csum):
            return

        # Otherwise start a new block
        self._cumulative_sum_index = np.append(self._cumulative_sum_index, total_csum)
        self._feature_arr.append(features)
        self._labels.append(label)
        self._extend_num_entries_per_row(features)

    @abstractmethod
    def lookup(self, row: int, select_features: Optional[list[str]] = None) -> Tuple[list[np.ndarray], str]:
        """Lookup features at a given row; must be implemented by subclasses."""
        ...

    def number_of_rows(self) -> int:
        """The number of rows in the index"".

        Returns:
            An integer corresponding to the number or rows in the index
        """
        return int(max(self._cumulative_sum_index[-1], 0))

    def concat(self, other_row_index: RowFeatureIndex, fail_on_empty_index: bool = True) -> RowFeatureIndex:
        """Concatenates the other FeatureIndex to this one.

        Returns the new, updated index. Warning: modifies this index in-place.

        Args:
            other_row_index: another FeatureIndex
            fail_on_empty_index: A boolean flag that sets whether to raise an
            error if an empty row index is passed in.

        Returns:
            self, the RowIndexFeature after the concatenations.

        Raises:
            TypeError if other_row_index is not a FeatureIndex
            ValueError if an empty FeatureIndex is passed and the function is
            set to fail in this case.
        """
        # Require the exact same concrete subclass to ensure semantic compatibility
        if not isinstance(other_row_index, RowFeatureIndex):
            raise TypeError("Error: trying to concatenate something that's not a FeatureIndex.")
        if type(self) is not type(other_row_index):
            raise TypeError(
                f"Error: cannot concatenate FeatureIndex instances of different kinds: {type(self)} and {type(other_row_index)}."
            )
        if fail_on_empty_index and not len(other_row_index._feature_arr) > 0:
            raise ValueError("Error: Cannot append empty FeatureIndex.")
        for i, feats in enumerate(list(other_row_index._feature_arr)):
            c_span = other_row_index._cumulative_sum_index[i + 1]
            label = other_row_index._labels[i]
            self.append_features(c_span, feats, label)

        return self

    def save(self, datapath: str) -> None:
        """Saves the FeatureIndex to a given path.

        Args:
            datapath: path to save the index
        """
        Path(datapath).mkdir(parents=True, exist_ok=True)
        num_digits = len(str(len(self._feature_arr)))
        for index, feature_dict in enumerate(self._feature_arr):
            table = pa.table({col: pa.array(vals) for col, vals in feature_dict.items()})
            dataframe_str_index = f"{index:0{num_digits}d}"
            pq.write_table(table, f"{datapath}/dataframe_{dataframe_str_index}.parquet")
        np.save(Path(datapath) / "cumulative_sum_index.npy", self._cumulative_sum_index)
        np.save(Path(datapath) / "labels.npy", self._labels)
        np.save(Path(datapath) / "version.npy", np.array(self._version))


class VariableFeatureIndex(RowFeatureIndex):
    """This gives features for genes in a given row (.var). This is stored in a list of dictionaries."""

    def __init__(self) -> None:
        """Create a variable (column) feature index."""
        super().__init__()

    def _check_and_append(self, n_obs: int, features: dict[str, np.ndarray], total_csum: Optional[str] = None) -> bool:
        """Check if the features are the same as the last features in the index and if so, merge the current block with the last block."""
        if len(self._feature_arr) > 0 and are_dicts_equal(self._feature_arr[-1], features):
            self._cumulative_sum_index[-1] = total_csum
            return True
        return False

    def _extend_num_entries_per_row(self, features: dict[str, np.ndarray]) -> None:
        """Extend the number of entries per row by the number of features in the dictionary."""
        if len(features) == 0:
            num_entries = 0
        else:
            num_entries = len(features[next(iter(features.keys()))])
        self._num_entries_per_row.append(num_entries)

    @staticmethod
    def load(datapath: str) -> "VariableFeatureIndex":
        """Load a variable (column) feature index from a directory."""
        return RowFeatureIndex._load_common(datapath, VariableFeatureIndex())

    def lookup(self, row: int, select_features: Optional[list[str]] = None) -> Tuple[list[np.ndarray], str]:
        """Lookup features at a given row, returning full arrays for that span."""
        d_id = self._get_dataset_id(row)
        features_dict = self._feature_arr[d_id]
        features = self._filter_features(features_dict, select_features)
        return features, self._labels[d_id]
