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

from bionemo.scdl.util.scdl_constants import FLOAT_ORDER, INT_ORDER


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

    Supports optional dtype conversion for upscaling within the same family only:
    - uint upscaling: uint8 → uint16 → uint32 → uint64
    - float upscaling: float16 → float32 → float64

    When source_dtype and dest_dtype are provided and differ, performs element-wise
    conversion during copy. Only safe upscaling operations are allowed (no downscaling,
    no cross-family conversions like uint↔float).

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
    if offset < 0:
        raise ValueError(f"Offset {offset} must be non-negative")
    if dest_dtype is not None and source_dtype is not None and source_dtype != dest_dtype:
        if offset % np.dtype(source_dtype).itemsize != 0:
            raise ValueError(
                f"Offset {offset} must be divisible by source dtype size {np.dtype(source_dtype).itemsize}"
            )
        if buffer_size_b % np.dtype(source_dtype).itemsize != 0:
            raise ValueError(
                f"Buffer size {buffer_size_b} is not divisible by source dtype size {np.dtype(source_dtype).itemsize}"
            )
        _extend_files_with_dtype_conversion(
            first, second, source_dtype, dest_dtype, buffer_size_b, delete_file2_on_complete, offset
        )
        return

    # Fallback/raw copy when no conversion requested (or same dtype)
    _extend_files_fast_copy(first, second, buffer_size_b, delete_file2_on_complete, offset)


def _extend_files_fast_copy(
    first: str,
    second: str,
    buffer_size_b: int,
    delete_file2_on_complete: bool,
    offset: int,
):
    """Fast raw-byte copy path when no dtype conversion is needed.

    Copies in chunks, ensuring at least 8 bytes per iteration (except possibly the last chunk).
    """
    with open(first, "r+b") as f_dest, open(second, "rb") as f_source:
        size1 = os.path.getsize(first)
        size2 = os.path.getsize(second)

        # Resize file1 to the final size to accommodate both files
        f_dest.seek(size1 + size2 - 1 - offset)
        f_dest.write(b"\0")  # Extend file1

        # Move data from file2 to file1 in chunks
        read_position = offset  # Start reading from the beginning of file2
        write_position = size1  # Start appending at the end of original data1
        f_source.seek(read_position)

        min_bytes = 8
        while read_position < size2:
            # Determine how much to read/write in this iteration (at least 8 bytes if available)
            remaining = size2 - read_position
            chunk_size = min(remaining, max(min_bytes, buffer_size_b))

            # Read data from file2
            new_data = f_source.read(chunk_size)

            # Write the new data into file1
            f_dest.seek(write_position)
            f_dest.write(new_data)

            # Update pointers
            read_position += chunk_size
            write_position += chunk_size
            f_source.seek(read_position)

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
    same-family upscaling (as ordered by INT_ORDER/FLOAT_ORDER):

    Supported conversions:
    - uint upscaling and float upscaling

    Rejects:
    - Downscaling (e.g., uint64 → uint32): risk of overflow
    - Any uint ↔ float conversions

    Parameters:
    - first (str): Destination file
    - second (str): Source file
    - source_dtype (str): Source numpy dtype
    - dest_dtype (str): Destination numpy dtype
    - buffer_size_b (int): Buffer size in bytes
    - delete_file2_on_complete (bool): Whether to delete source after
    - offset (int): Byte offset to skip in source

    Raises:
    - ValueError: If conversion violates same-family or non-decreasing order
    """
    # Determine family/order and enforce same-family upscaling
    family_orders = [INT_ORDER, FLOAT_ORDER]
    for order in family_orders:
        if source_dtype in order and dest_dtype in order:
            if order.index(dest_dtype) < order.index(source_dtype):
                raise ValueError(f"Downscaling not allowed: {source_dtype} → {dest_dtype}.")
            break
    else:
        raise ValueError(
            f"Unsupported dtype conversion: {source_dtype} → {dest_dtype}. Only same-family upscaling allowed."
        )
    # Resolve dtypes once (native endianness) and sizes
    sd = np.dtype(source_dtype).newbyteorder("=")
    dd = np.dtype(dest_dtype).newbyteorder("=")
    src_item = sd.itemsize
    dst_item = dd.itemsize

    # Elements per chunk
    min_bytes = 8
    elements_per_chunk = max(1, max(buffer_size_b, min_bytes) // src_item)

    # Source sizing
    size2 = os.path.getsize(second)
    remaining = size2 - offset
    if remaining % src_item != 0:
        raise ValueError(
            f"Source size minus offset ({remaining} bytes) not divisible by source dtype size ({src_item})."
        )
    num_elements = remaining // src_item

    # Pre-extend destination to final size
    extend_bytes = num_elements * dst_item
    size1 = os.path.getsize(first)
    with open(first, "r+b") as f_dest:
        if extend_bytes > 0:
            f_dest.seek(size1 + extend_bytes - 1)
            f_dest.write(b"\0")

        write_position = size1

        # Reusable output buffer
        out_buf = bytearray(elements_per_chunk * dst_item)

        with open(second, "rb") as f_source:
            if offset > 0:
                f_source.seek(offset)

            elements_processed = 0
            while elements_processed < num_elements:
                chunk_elements = min(elements_per_chunk, num_elements - elements_processed)
                bytes_to_read = chunk_elements * src_item

                chunk_bytes = f_source.read(bytes_to_read)
                if not chunk_bytes:
                    break

                # Zero-copy view of source and writable view of prealloc'd dest buffer
                src = np.frombuffer(chunk_bytes, dtype=sd, count=chunk_elements)
                dst_mv = memoryview(out_buf)[: chunk_elements * dst_item]
                dst = np.frombuffer(dst_mv, dtype=dd, count=chunk_elements)

                # Single-pass safe upcast into dst buffer
                np.copyto(dst, src, casting="safe")

                # Write the exact slice
                f_dest.seek(write_position)
                f_dest.write(dst_mv)
                write_position += chunk_elements * dst_item

                elements_processed += src.size

    if delete_file2_on_complete:
        os.remove(second)
