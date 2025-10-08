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

import os

import numpy as np

from bionemo.scdl.util.scdl_constants import VALID_DTYPE_CONVERSIONS


def extend_files(
    first: str,
    second: str,
    buffer_size_b: int = 10 * 1024 * 1024,
    delete_file2_on_complete: bool = False,
    offset: int = 0,
    source_dtype: str | None = None,
    dest_dtype: str | None = None,
):
    """Concatenates the contents of `second` into `first` using memory-efficient operations.

    Supports optional dtype conversion for upscaling:
    - uint upscaling: uint8 → uint16 → uint32 → uint64
    - float upscaling: float16 → float32 → float64

    When source_dtype and dest_dtype are provided and differ, performs element-wise
    conversion during copy. Only safe upscaling operations are allowed (no downscaling,
    no cross-type conversions like uint↔float).

    Parameters:
    - first (str): Path to the first file (will be extended).
    - second (str): Path to the second file (data will be read from here).
    - buffer_size_b (int): Size of the buffer to use for reading/writing data.
    - delete_file2_on_complete (bool): Whether to delete the second file after operation.
    - offset (int): Byte offset to skip in the second file.
    - source_dtype (str, optional): Numpy dtype of source data (e.g., 'uint32', 'float16').
        If provided, enables dtype-aware conversion.
    - dest_dtype (str, optional): Numpy dtype of destination data (e.g., 'uint64', 'float32').
        If provided, enables dtype-aware conversion.

    Raises:
    - ValueError: If conversion is not a safe upscaling operation.

    """
    # Check if dtype conversion is needed
    if source_dtype is not None and dest_dtype is not None and source_dtype != dest_dtype:
        _extend_files_with_dtype_conversion(
            first, second, source_dtype, dest_dtype, buffer_size_b, delete_file2_on_complete, offset
        )
        return

    # Original fast path for matching dtypes or no dtype specified
    with open(first, "r+b") as f1, open(second, "rb") as f2:
        size1 = os.path.getsize(first)
        size2 = os.path.getsize(second)

        # Resize file1 to the final size to accommodate both files
        f1.seek(size1 + size2 - 1 - offset)
        f1.write(b"\0")  # Extend file1

        # Move data from file2 to file1 in chunks
        read_position = offset  # Start reading from the beginning of file2
        write_position = size1  # Start appending at the end of original data1
        f2.seek(read_position)

        while read_position < size2:
            # Determine how much to read/write in this iteration
            chunk_size = min(buffer_size_b, size2 - read_position)

            # Read data from file2
            new_data = f2.read(chunk_size)

            # Write the new data into file1
            f1.seek(write_position)
            f1.write(new_data)

            # Update pointers
            read_position += chunk_size
            write_position += chunk_size
            f2.seek(read_position)

    if delete_file2_on_complete:
        os.remove(second)


def _extend_files_with_dtype_conversion(
    first: str,
    second: str,
    source_dtype: str,
    dest_dtype: str,
    buffer_size_b: int,
    delete_file2_on_complete: bool,
    offset: int,
):
    """Internal function to extend files with dtype conversion.

    Converts data from source_dtype to dest_dtype during copy. Only supports safe
    lossless conversions defined in VALID_DTYPE_CONVERSIONS:

    Supported conversions:
    - uint upscaling: uint8 → uint16 → uint32 → uint64 (lossless)
    - float upscaling: float16 → float32 → float64 (lossless)
    - Safe uint → float: uint8/uint16 → float32, uint8/uint16/uint32 → float64 (lossless)

    Rejects:
    - Downscaling (e.g., uint64 → uint32): risk of overflow
    - Unsafe uint → float (e.g., uint32 → float32, uint64 → float64): risk of precision loss
    - float → uint conversions: loss of fractional part

    Parameters:
    - first (str): Destination file
    - second (str): Source file
    - source_dtype (str): Source numpy dtype
    - dest_dtype (str): Destination numpy dtype
    - buffer_size_b (int): Buffer size in bytes
    - delete_file2_on_complete (bool): Whether to delete source after
    - offset (int): Byte offset to skip in source

    Raises:
    - ValueError: If conversion is not in VALID_DTYPE_CONVERSIONS
    """
    source_dtype_np = np.dtype(source_dtype)
    dest_dtype_np = np.dtype(dest_dtype)

    # Normalize dtype names to simple strings (e.g., 'uint32', 'float64')
    source_dtype_str = source_dtype_np.name
    dest_dtype_str = dest_dtype_np.name

    # Check if conversion is valid
    if (source_dtype_str, dest_dtype_str) not in VALID_DTYPE_CONVERSIONS:
        raise ValueError(
            f"Conversion from {source_dtype_str} to {dest_dtype_str} is not supported. "
            f"Only lossless conversions are allowed. "
            f"Valid conversions: uint upscaling (uint8→uint16→uint32→uint64), "
            f"float upscaling (float16→float32→float64), "
            f"and safe uint→float (uint8/16→float32, uint8/16/32→float64)."
        )

    # Calculate number of elements that fit in buffer
    # Use source dtype size since we're reading from source
    elements_per_chunk = buffer_size_b // source_dtype_np.itemsize

    # Get source file size and calculate number of elements
    size2 = os.path.getsize(second)
    num_elements = (size2 - offset) // source_dtype_np.itemsize

    # Open destination in append mode and source for reading
    with open(first, "ab") as f_dest, open(second, "rb") as f_source:
        # Skip offset bytes in source
        if offset > 0:
            f_source.seek(offset)

        elements_processed = 0
        while elements_processed < num_elements:
            # Read chunk
            chunk_elements = min(elements_per_chunk, num_elements - elements_processed)
            bytes_to_read = chunk_elements * source_dtype_np.itemsize

            chunk_bytes = f_source.read(bytes_to_read)
            if not chunk_bytes:
                break

            # Convert to numpy array with source dtype
            source_array = np.frombuffer(chunk_bytes, dtype=source_dtype_np)

            # Convert to destination dtype (upscaling)
            dest_array = source_array.astype(dest_dtype_np)

            # Write converted data
            f_dest.write(dest_array.tobytes())

            elements_processed += len(source_array)

    if delete_file2_on_complete:
        os.remove(second)
