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

"""Utility functions for memory-mapped dataset operations.

This module contains helper functions for:
- Data type casting
- Sparse array manipulation
- Memory-mapped array creation
"""

from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from bionemo.scdl.util.scdl_constants import (
    FLOAT_ORDER,
    INT_ORDER,
    FileNames,
    Mode,
)


def smallest_uint_dtype(x: int):
    """Returns the smallest unsigned integer dtype that can represent the given number.

    Args:
        x: The number to represent

    Returns:
        The smallest unsigned integer dtype that can represent the given number

    Raises:
        ValueError: If x is negative or too large to represent
    """
    if x < 0:
        raise ValueError("Negative numbers can't be unsigned.")
    for dtype, bits in [("uint8", 8), ("uint16", 16), ("uint32", 32), ("uint64", 64)]:
        if x < (1 << bits):
            return dtype
    raise ValueError(f"No unsigned integer dtype can represent the given number: {x}")


def determine_dtype(dtypes: Iterable[object]) -> str:
    """Choose a common destination dtype by same-family upscaling.

    - If all source dtypes are unsigned integers: return the widest unsigned int
    - If all source dtypes are floats: return the widest float
    - Otherwise: raise (mixed families not allowed)
    """
    if len(dtypes) == 0:
        raise ValueError("No dtypes provided")
    canonical = [np.dtype(dt).name for dt in dtypes]
    if all(dt in INT_ORDER for dt in canonical):
        return max(set(canonical), key=lambda dt: INT_ORDER.index(dt))
    if all(dt in FLOAT_ORDER for dt in canonical):
        return max(set(canonical), key=lambda dt: FLOAT_ORDER.index(dt))
    raise ValueError(f"Mixed float and integer dtype families not allowed: {sorted(set(canonical))}")


def _pad_sparse_array(row_values, row_col_ptr, n_cols: int) -> np.ndarray:
    """Creates a conventional array from a sparse one.

    Convert a sparse matrix representation of a 1d matrix to a conventional
    numpy representation.

    Args:
        row_values: The row indices of the entries
        row_col_ptr: The corresponding column pointers
        n_cols: The number of columns in the dataset.

    Returns:
        The full 1d numpy array representation.
    """
    ret = np.zeros(n_cols)
    for row_ptr in range(0, len(row_values)):
        col = row_col_ptr[row_ptr]
        ret[col] = row_values[row_ptr]
    return ret


def _create_row_memmaps(
    num_rows: int,
    memmap_dir_path: Path,
    mode: Mode,
    dtypes: Dict[str, str],
) -> np.ndarray:
    """Records a pointer into the data and column arrays.

    Args:
        num_rows: Number of rows in the dataset
        memmap_dir_path: Path to directory where memmap files are stored
        mode: File opening mode
        dtypes: Dictionary mapping file names to dtypes

    Returns:
        Memory-mapped array for row pointers
    """
    return np.memmap(
        f"{str(memmap_dir_path.absolute())}/{FileNames.ROWPTR.value}",
        dtype=dtypes[FileNames.ROWPTR.value],
        shape=(num_rows + 1,),
        mode=mode.value,
    )


def _create_data_col_memmaps(
    num_elements: int,
    memmap_dir_path: Path,
    mode: Mode,
    dtypes: Dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Records a pointer into the data and column arrays.

    Args:
        num_elements: Total number of non-zero elements
        memmap_dir_path: Path to directory where memmap files are stored
        mode: File opening mode
        dtypes: Dictionary mapping file names to dtypes

    Returns:
        Tuple of (data array, column pointer array)
    """
    # Records the value at index[i]
    data_arr = np.memmap(
        f"{memmap_dir_path}/{FileNames.DATA.value}",
        dtype=dtypes[FileNames.DATA.value],
        shape=(num_elements,),
        mode=mode.value,
    )
    # Records the column the data resides in at index [i]
    col_arr = np.memmap(
        f"{memmap_dir_path}/{FileNames.COLPTR.value}",
        dtype=dtypes[FileNames.COLPTR.value],
        shape=(num_elements,),
        mode=mode.value,
    )
    return data_arr, col_arr


def _create_compressed_sparse_row_memmaps(
    num_elements: int,
    num_rows: int,
    memmap_dir_path: Path,
    mode: Mode,
    dtypes: Dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a set of CSR-format numpy arrays.

    They are saved to memmap_dir_path. This is an efficient way of indexing
    into a sparse matrix. Only non-zero values of the data are stored.

    To get the data for a specific row, slice row_idx[idx, idx+1]
    and then get the elements in data[row_idx[idx]:row_idx[idx+1]]
    which are in the corresponding columns col_index[row_idx[idx], row_idx[row_idx+1]]

    Args:
        num_elements: Total number of non-zero elements
        num_rows: Number of rows in the dataset
        memmap_dir_path: Path to directory where memmap files are stored
        mode: File opening mode
        dtypes: Dictionary mapping file names to dtypes

    Returns:
        Tuple of (data array, column pointer array, row pointer array)

    Raises:
        ValueError: If num_elements or num_rows is not positive
    """
    if num_elements <= 0:
        raise ValueError(f"n_elements is set to {num_elements}. It must be positive to create CSR matrices.")

    if num_rows <= 0:
        raise ValueError(f"num_rows is set to {num_rows}. It must be positive to create CSR matrices.")
    memmap_dir_path.mkdir(parents=True, exist_ok=True)
    data_arr, col_arr = _create_data_col_memmaps(
        num_elements,
        memmap_dir_path,
        mode,
        dtypes,
    )

    row_arr = _create_row_memmaps(
        num_rows,
        memmap_dir_path,
        mode,
        dtypes,
    )
    return data_arr, col_arr, row_arr


def _extract_features(df, feature_index_name):
    """Helper to convert a DataFrame into a features dict."""
    if df.columns.size > 0:
        return {col: np.array(df[col].values) for col in df.columns}
    elif df.index.size > 0:
        return {feature_index_name: df.index.values}
    else:
        return {}
