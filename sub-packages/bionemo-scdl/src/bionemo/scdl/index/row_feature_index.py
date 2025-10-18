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

"""Feature index abstractions for SCDL.

This module defines an abstract base `RowFeatureIndex` and two concrete
implementations:

- `ObservedFeatureIndex`: row-oriented features, where a lookup returns one
  scalar per selected feature for a given row.
- `VariableFeatureIndex`: column-oriented features, where a lookup returns full
  feature arrays for the block containing a given row.

Data are stored in blocks of feature dictionaries. `_cumulative_sum_index`
tracks row boundaries between blocks. The feature dictionarires are stroed in `_feature_arr`.
"""

from __future__ import annotations

import importlib.metadata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


__all__: Sequence[str] = (
    "ObservedFeatureIndex",
    "VariableFeatureIndex",
    "are_dicts_equal",
)


def _to_arrow_array(vals):
    """Convert a list of values to an Arrow array."""
    try:
        return pa.array(vals, from_pandas=True)
    except pa.ArrowInvalid:
        return pa.array(vals, type=pa.string(), from_pandas=True)


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
    """Abstract base for ragged feature indices.

    Represents datasets where the number and/or shape of features can differ per
    row. Data are organized in blocks (feature dictionaries), with a cumulative
    sum index delineating block boundaries.

    Attributes:
        _cumulative_sum_index: Cumulative row counts that delineate block
            boundaries. For example, with `[-1, 200, 350]`, rows `0..199` are in
            block 0, which is in _feature_arr[0], and rows `200..349` are in block 1,
            which is in _feature_arr[1].
        _feature_arr: List of feature dictionaries, one per block.
        _num_entries_per_row: Per-block counts used by `number_vars_at_row` and
            `column_dims`.
        _labels: Optional label per block (e.g., dataset ID or name).
        _version: Version string for the dataset.
    """

    def __init__(self) -> None:
        """Instantiates the index."""
        self._cumulative_sum_index: np.array = np.array([-1])
        self._feature_arr: list[dict[str, np.ndarray]] = []
        self._num_entries_per_row: list[int] = []
        self._version = importlib.metadata.version("bionemo.scdl")
        self._labels: list[Optional[str]] = []

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
        """Load state common to all concrete indices from a directory.

        Reads block data from Parquet files under `datapath`, plus the saved
        cumulative sum index, labels, and version.

        Args:
            datapath: Directory containing saved index files.
            instance: An empty instance of the concrete subclass to populate.

        Returns:
            The populated `instance`.
        """
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
        """Select and order features by name from a feature dictionary.

        If `select_features` is None, all features are returned in dictionary
        iteration order. Otherwise, features are returned in the provided order,
        and a ValueError is raised if any name is missing.
        """
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
        """Return the number of blocks (feature dictionaries)."""
        return len(self._feature_arr)

    @abstractmethod
    def _extend_num_entries_per_row(self, features: dict[str, np.ndarray]) -> None:
        """Extend the number of entries per row for a concrete index implementation."""
        ...

    @abstractmethod
    def _check_if_can_merge_features_with_last_block(
        self, features: dict[str, np.ndarray], total_csum: Optional[int] = None
    ) -> bool:
        """Optionally merge a new block into the last block.

        Subclasses may coalesce compatible blocks and update `_cumulative_sum_index`.
        Return True if merged; False otherwise.
        """
        ...

    def _append_block(self, features: dict[str, np.ndarray], label: Optional[str], total_csum: int) -> None:
        """Append a new block after merge check, updating internal state.

        This centralizes the common append path used by concrete subclasses.
        """
        # Optionally merge into previous block
        if self._check_if_can_merge_features_with_last_block(features, total_csum):
            return

        # Otherwise start a new block
        self._cumulative_sum_index = np.append(self._cumulative_sum_index, total_csum)
        self._feature_arr.append(features)
        self._labels.append(label)
        self._extend_num_entries_per_row(features)

    def number_vars_at_row(self, row: int) -> int:
        """Return the number of variables for the block that contains `row`."""
        dataset_idx = self._get_dataset_id(row)
        return self._num_entries_per_row[dataset_idx]

    def column_dims(self) -> list[int]:
        """Return per-block feature counts.

        For `ObservedFeatureIndex`, this is the number of feature columns per
        block. For `VariableFeatureIndex`, this is the per-row array length per
        block.
        """
        return self._num_entries_per_row

    def _validate_features_get_entries_count(self, features: dict[str, np.ndarray]) -> int:
        """Validate feature input and return the expected number of entries for each value in the feature dictionary.

        Ensures `features` is a dictionary with arrays of equal length. Returns
        the common length (0 if empty).

        Returns:
            The length of the features.

        Raises:
            TypeError: If features is not a dictionary
            ValueError: If the features are not all the same length
        """
        if not isinstance(features, dict):
            raise TypeError(f"{self.__class__.__name__}.append_features expects a dict of arrays")

        if len(features) > 0:
            first_length = len(next(iter(features.values())))
            if any(len(v) != first_length for v in features.values()):
                raise ValueError("All feature arrays must have the same length")
            return first_length
        else:
            return 0

    def number_of_values(self) -> list[int]:
        """Return total value counts per block.

        For each block, `(rows in block) * (per-block feature count)`. Returns
        `[0]` when there are no blocks.
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

    @abstractmethod
    def concat(self, other_row_index: "RowFeatureIndex") -> "RowFeatureIndex":
        """Concatenate another FeatureIndex to this one.

        Args:
            other_row_index: another FeatureIndex

        Returns:
            self (updated).

        Raises:
            TypeError or ValueError
        """
        ...

    @abstractmethod
    def append_features(self, *args, **kwargs) -> None:
        """Append features, delegating validation and merge behavior to subclasses.

        May or may not have n_obs as an argument, depending on subclass.
        """
        ...

    def number_of_rows(self) -> int:
        """The number of rows in the index.

        Returns:
            An integer corresponding to the number or rows in the index
        """
        return int(max(self._cumulative_sum_index[-1], 0))

    def save(self, datapath: str) -> None:
        """Saves the FeatureIndex to a given path.

        Args:
            datapath: path to save the index
        """
        Path(datapath).mkdir(parents=True, exist_ok=True)
        num_digits = len(str(len(self._feature_arr)))
        for index, feature_dict in enumerate(self._feature_arr):
            table = pa.table({col: _to_arrow_array(vals) for col, vals in feature_dict.items()})
            dataframe_str_index = f"{index:0{num_digits}d}"
            pq.write_table(table, f"{datapath}/dataframe_{dataframe_str_index}.parquet")
        np.save(Path(datapath) / "cumulative_sum_index.npy", self._cumulative_sum_index)
        np.save(Path(datapath) / "labels.npy", self._labels)
        np.save(Path(datapath) / "version.npy", np.array(self._version))


class ObservedFeatureIndex(RowFeatureIndex):
    """Feature index for observed (row) features.

    Each block is a dictionary mapping feature name to the full column array. A
    lookup at a row returns one scalar per selected feature and the block label.
    Successive blocks with identical schemas (same keys) are merged by
    concatenating column arrays.
    """

    def __init__(self) -> None:
        """Create an observed (row) feature index."""
        super().__init__()

    def _check_if_can_merge_features_with_last_block(self, features: dict[str, np.ndarray], total_csum: int) -> bool:
        """Merge into last block when schemas match.

        If the last block has the same set of keys, concatenate each column to
        extend the block and update the cumulative sum index.

        Returns:
            True if merged into the last block; False otherwise.
        """
        if len(self._feature_arr) > 0:
            last_features = self._feature_arr[-1]
            if last_features.keys() == features.keys():
                merged = {k: np.concatenate([last_features[k], np.asarray(features[k])]) for k in features}
                self._feature_arr[-1] = merged
                self._cumulative_sum_index[-1] = total_csum
                return True
        return False

    def lookup(self, row: int, select_features: Optional[list[str]] = None) -> Tuple[list[np.ndarray], Optional[str]]:
        """Return scalar feature values and block label for a given row."""
        d_id = self._get_dataset_id(row)
        features_dict: dict[str, np.ndarray] = self._feature_arr[d_id]
        start = self._cumulative_sum_index[d_id] if d_id > 0 else 0
        row_idx_in_block = row - start
        if select_features is not None:
            arrays = self._filter_features(features_dict, select_features)
        else:
            arrays = [features_dict[f] for f in features_dict]
        vals = [np.asarray(arr)[row_idx_in_block] for arr in arrays]
        return vals, self._labels[d_id]

    def _extend_num_entries_per_row(self, features: dict[str, np.ndarray]) -> None:
        """Extend the number of entries per row for the observed feature index."""
        num_entries = len(features)
        self._num_entries_per_row.append(num_entries)

    @staticmethod
    def load(datapath: str) -> "ObservedFeatureIndex":
        """Load a observed (row) feature index from a directory.

        This will  load the parquet files in sorted order. In the SCDL use case, this expects a directory with
        parquet files named dataframe_<index>.parquet.
        """
        return RowFeatureIndex._load_common(datapath, ObservedFeatureIndex())

    def __getitem__(self, idx):
        """Access one row or a slice of rows for observed features (.obs).

        - If `idx` is an int, returns `(values, label)` for that row:
            - `values` is a list of scalar values (one per selected feature)
            - `label` is the block label
        - If `idx` is a slice, returns `(blocks, labels)`:
            - `blocks` is a list of feature dictionaries, one per intersected
              block, with arrays sliced to the selected rows for that block.
            - `labels` is a list of block labels in the same order.
        """
        if isinstance(idx, int):
            n = self.number_of_rows()
            if idx < 0:
                idx += n
            return self.lookup(idx)

        if isinstance(idx, slice):
            n = self.number_of_rows()
            start, stop, step = idx.indices(n)
            if step == 0:
                raise ValueError("slice step cannot be zero")
            rows = list(range(start, stop, step))
            if not rows:
                return [], []
            ends = self._cumulative_sum_index[1:]
            by_block: dict[int, list[int]] = {}
            for r in rows:
                d_id = int(np.searchsorted(ends, r, side="right"))
                by_block.setdefault(d_id, []).append(r)
            out = []
            labels = []
            for d_id, rs in by_block.items():
                rs.sort()
                prev_end = self._cumulative_sum_index[d_id] if d_id > 0 else 0
                relative_indices = [r - prev_end for r in rs]
                sub = {k: np.asarray(v)[relative_indices] for k, v in self._feature_arr[d_id].items()}
                out.append(sub)
                labels.append(self._labels[d_id])
            return out, labels

        raise TypeError("Index must be int or slice")

    def concat(
        self,
        other_row_index: ObservedFeatureIndex,
    ) -> ObservedFeatureIndex:
        """Concatenates the other ObservedFeatureIndex to this one.

        Returns the new, updated index. Warning: modifies this index in-place.

        Args:
            other_row_index: another ObservedFeatureIndex
            error if an empty row index is passed in.

        Returns:
            self, the RowIndexFeature after the concatenations.

        Raises:
            TypeError if other_row_index is not a ObservedFeatureIndex
        """
        # Require the exact same concrete subclass to ensure semantic compatibility
        if not isinstance(other_row_index, ObservedFeatureIndex):
            raise TypeError("Error: trying to concatenate something that's not a ObservedFeatureIndex.")
        for i, feats in enumerate(list(other_row_index._feature_arr)):
            label = other_row_index._labels[i]
            self.append_features(feats, label)

        return self

    def append_features(self, features: dict[str, np.ndarray], label: Optional[str] = None) -> None:
        """Append features, delegating validation and merge behavior to subclasses."""
        feature_size = self._validate_features_get_entries_count(features)
        """ There are no features to append, so we return early."""
        if feature_size == 0:
            return
        total_csum = max(self._cumulative_sum_index[-1], 0) + feature_size
        self._append_block(features, label, total_csum)


class VariableFeatureIndex(RowFeatureIndex):
    """Feature index for variables (columns).

    Lookup returns full arrays for the block that contains a given row. When
    successive blocks have identical feature dictionaries, they are merged and
    only `_cumulative_sum_index` advances.
    """

    def __init__(self) -> None:
        """Create a variable (column) feature index."""
        super().__init__()

    def _check_if_can_merge_features_with_last_block(self, features: dict[str, np.ndarray], total_csum: int) -> bool:
        """Check if the features are the same as the last features in the index and if so, merge the current block with the last block."""
        if len(self._feature_arr) > 0 and are_dicts_equal(self._feature_arr[-1], features):
            self._cumulative_sum_index[-1] = total_csum
            return True
        return False

    def _extend_num_entries_per_row(self, features: dict[str, np.ndarray]) -> None:
        """Record per-row array length for this block."""
        if self._validate_features_get_entries_count(features) == 0:
            num_entries = 0
        else:
            num_entries = len(features[next(iter(features.keys()))])
        self._num_entries_per_row.append(num_entries)

    @staticmethod
    def load(datapath: str) -> "VariableFeatureIndex":
        """Load a variable (column) feature index from a directory.

        This will  load the parquet files in sorted order. In the SCDL use case, this expects a directory with
        parquet files named dataframe_<index>.parquet.
        """
        return RowFeatureIndex._load_common(datapath, VariableFeatureIndex())

    def lookup(self, row: int, select_features: Optional[list[str]] = None) -> Tuple[list[np.ndarray], Optional[str]]:
        """Return feature arrays and the block label for the block containing `row`."""
        d_id = self._get_dataset_id(row)
        features_dict = self._feature_arr[d_id]
        features = self._filter_features(features_dict, select_features)
        return features, self._labels[d_id]

    def append_features(self, n_obs: int, features: dict[str, np.ndarray], label: Optional[str] = None) -> None:
        """Append a new block, or merge into the last block when possible."""
        self._validate_features_get_entries_count(features)
        total_csum = max(self._cumulative_sum_index[-1], 0) + n_obs
        self._append_block(features, label, total_csum)

    def concat(self, other_row_index: VariableFeatureIndex) -> VariableFeatureIndex:
        """Concatenates the other VariableFeatureIndex to this one.

        Returns the new, updated index. Warning: modifies this index in-place.

        Args:
            other_row_index: another VariableFeatureIndex

        Returns:
            self, the VariableFeatureIndex after the concatenations.

        Raises:
            TypeError if other_row_index is not a VariableFeatureIndex
            ValueError if an empty FeatureIndex is passed and the function is
            set to fail in this case.
        """
        # Require the exact same concrete subclass to ensure semantic compatibility
        if not isinstance(other_row_index, VariableFeatureIndex):
            raise TypeError("Error: trying to concatenate something that's not a VariableFeatureIndex.")
        for i, feats in enumerate(list(other_row_index._feature_arr)):
            c_span = other_row_index._cumulative_sum_index[i + 1] - max(0, other_row_index._cumulative_sum_index[i])
            label = other_row_index._labels[i]
            self.append_features(c_span, feats, label)

        return self
