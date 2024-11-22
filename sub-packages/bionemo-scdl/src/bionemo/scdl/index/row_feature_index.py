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
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


__all__: Sequence[str] = ("RowFeatureIndex",)


class RowFeatureIndex:
    """Maintains a mapping between a row and its features.

    This is a ragged dataset, where the number and dimension of features
    can be different at every row.

    Attributes:
        _cumulative_sum_index: Pointer that deliniates which entries
        correspondto a given row. For examples if the array is [-1, 200, 201],
        rows 0 to 199 correspond to _feature_arr[0] and 200 corresponds to
        _feature_arr[1]
        _feature_arr: list of feature dataframes
        _labels: list of labels
        _version: The version of the dataset
    """

    def __init__(self) -> None:
        """Instantiates the index."""
        self._cumulative_sum_index: np.array = np.array([-1])
        self._feature_arr: List[dict] = []
        self._num_genes_per_row: List[int] = []
        self._version = importlib.metadata.version("bionemo.scdl")
        self._labels: List[str] = []
        self._feature_arr_lookup: List[dict] = []

    def version(self) -> str:
        """Returns a version number.

        (following <major>.<minor>.<point> convention).
        """
        return self._version

    def __len__(self) -> int:
        """The length is the number of rows or RowFeatureIndex length."""
        return len(self._feature_arr)

    def append_features(
        self, n_obs: int, features: dict, num_genes: Optional[int], label: Optional[str] = None
    ) -> None:
        """Updates the index with the given features.

        The dataframe is inserted into the feature array by adding a
        new span to the row lookup index.

        Args:
            n_obs (int): The number of times that these feature occur in the
            class.
            features (pd.DataFrame): Corresponding features.
            num_genes (int): the length of the features for each feature key in features (i.e., number of genes)
            label (str): Label for the features.
        """
        if isinstance(features, pd.DataFrame):
            raise TypeError("Expected a dictionary, but received a Pandas DataFrame.")
        csum = max(self._cumulative_sum_index[-1], 0)
        self._cumulative_sum_index = np.append(self._cumulative_sum_index, csum + n_obs)
        self._feature_arr.append(features)
        self._num_genes_per_row.append(num_genes)
        self._labels.append(label)

    def lookup(self, row: int, select_features: Optional[List[str]] = None) -> Tuple[List[np.ndarray], str]:
        """Find the features at a given row.

        It is assumed that the row is
        non-zero._cumulative_sum_index contains pointers to which rows correspond
        to given dataframes. To obtain a specific row, we determine where it is
        located in _cumulative_sum_index and then look up that dataframe in
        _feature_arr
        Args:
            row (int): The row in the feature index.
            select_features (List[str]): a list of features to select
        Returns
            pd.DataFrame: dataframe of features in that row
            str: optional label for the row
        Raises:
            IndexError: An error occured due to input row being negative or it
            exceeding the larger row of the rows in the index. It is also raised
            if there are no entries in the index yet.
        """
        if row < 0:
            raise IndexError(f"Row index {row} is not valid. It must be non-negative.")
        if len(self._cumulative_sum_index) < 2:
            raise IndexError("There are no dataframes to lookup.")

        if row > self._cumulative_sum_index[-1]:
            raise IndexError(
                f"Row index {row} is larger than number of rows in FeatureIndex ({self._cumulative_sum_index[-1]})."
            )
        # This line does the following:
        # creates a mask for values where cumulative sum > row
        mask = ~(self._cumulative_sum_index > row)
        # Sum these to get the index of the first range > row
        # Subtract one to get the range containing row.
        d_id = sum(mask) - 1

        # Retrieve the features for the identified value.
        features_dict = self._feature_arr[d_id]

        # If specific features are to be selected, filter the features.
        if select_features is not None:
            features = []
            for feature in select_features:
                if feature not in features_dict:
                    raise ValueError(f"Provided feature column {feature} in select_features not present in dataset.")
                features.append(features_dict[feature])
        else:
            features = [features_dict[f] for f in features_dict]

        # Return the features for the identified range.
        return features, self._labels[d_id]

    def number_vars_at_row(self, row: int) -> int:
        """Return number of variables (length of the dataframe) in a given row.

        Args:
            row (int): The row in the feature index.

        Returns:
            The length of the features at the row
        """
        feats, _ = self.lookup(row=row)
        return len(feats[0])

    def column_dims(self) -> List[int]:
        """Return the number of columns in all rows.

        Args:
            length of features at every row is returned.

        Returns:
            A list containing the lengths of the features in every row
        """
        # Just take the total dim of the DataFrame(s)
        return self._num_genes_per_row

    def number_of_values(self) -> List[int]:
        """Get the total number of values in the array.

        For each row, the length of the corresponding dataframe is counted.

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
        for i, n_rows in enumerate(rows):
            num_genes = self._num_genes_per_row[i]
            vals.append(n_rows * num_genes)
        return vals

    def number_of_rows(self) -> int:
        """The number of rows in the dataframe.

        Returns:
            An integer corresponding to the number or rows in the index
        """
        return int(max(self._cumulative_sum_index[-1], 0))

    def concat(self, other_row_index: RowFeatureIndex, fail_on_empty_index: bool = True) -> RowFeatureIndex:
        """Concatenates the other FeatureIndex to this one.

        Returns the new, updated index. Warning: modifies this index in-place.

        Args:
            other_row_index: another RowFeatureIndex
            fail_on_empty_index: A boolean flag that sets whether to raise an
            error if an empty row index is passed in.

        Returns:
            self, the RowIndexFeature after the concatenations.

        Raises:
            TypeError if other_row_index is not a RowFeatureIndex
            ValueError if an empty RowFeatureIndex is passed and the function is
            set to fail in this case.
        """
        match other_row_index:
            case self.__class__():
                pass
            case _:
                raise TypeError("Error: trying to concatenate something that's not a RowFeatureIndex.")

        if fail_on_empty_index and not len(other_row_index._feature_arr) > 0:
            raise ValueError("Error: Cannot append empty FeatureIndex.")
        for i, feats in enumerate(list(other_row_index._feature_arr)):
            c_span = other_row_index._cumulative_sum_index[i + 1]
            label = other_row_index._labels[i]
            num_genes = other_row_index._num_genes_per_row[i]
            self.append_features(c_span, feats, num_genes, label)

        return self

    def save(self, datapath: str) -> None:
        """Saves the RowFeatureIndex to a given path.

        Args:
            datapath: path to save the index
        """
        Path(datapath).mkdir(parents=True, exist_ok=True)
        num_digits = len(str(len(self._feature_arr)))
        for index, feature_dict in enumerate(self._feature_arr):
            table = pa.table({column: pa.array(values) for column, values in feature_dict.items()})
            dataframe_str_index = f"{index:0{num_digits}d}"
            pq.write_table(table, f"{datapath}/dataframe_{dataframe_str_index}.parquet")

        np.save(Path(datapath) / "cumulative_sum_index.npy", self._cumulative_sum_index)
        np.save(Path(datapath) / "labels.npy", self._labels)
        np.save(Path(datapath) / "version.npy", np.array(self._version))

    @staticmethod
    def load(datapath: str) -> RowFeatureIndex:
        """Loads the data from datapath.

        Args:
            datapath: the path to load from
        Returns:
            An instance of RowFeatureIndex
        """
        new_row_feat_index = RowFeatureIndex()
        parquet_data_paths = sorted(Path(datapath).rglob("*.parquet"))
        new_row_feat_index._feature_arr = [pq.read_table(csv_path) for csv_path in parquet_data_paths]
        new_row_feat_index._feature_arr = [
            {column: table[column].to_numpy() for column in table.column_names}
            for table in new_row_feat_index._feature_arr
        ]
        new_row_feat_index._num_genes_per_row = [
            len(feats[next(iter(feats.keys()))]) for feats in new_row_feat_index._feature_arr
        ]

        new_row_feat_index._cumulative_sum_index = np.load(Path(datapath) / "cumulative_sum_index.npy")
        new_row_feat_index._labels = np.load(Path(datapath) / "labels.npy", allow_pickle=True)
        new_row_feat_index._version = np.load(Path(datapath) / "version.npy").item()
        return new_row_feat_index
