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
    source_dtype: str,
    dest_dtype: str,
    elements_per_chunk: int = 10 * 1024 * 1024,
    delete_file2_on_complete: bool = False,
    offset: int = 0,
    add_value: int | None = None,
    allow_downscaling: bool = False,
):
    """Concatenates the contents of `second` into `first` using memory-efficient operations.

    Supports optional dtype conversion for upscaling within the same family only:
    - uint upscaling: uint8 → uint16 → uint32 → uint64
    - float upscaling:  float32 → float64

    Additionally, supports adding a scalar value to each converted element during copy.

    Parameters:
    - first (str): Destination file path (will be extended).
    - second (str): Source file path (data read from here).
    - source_dtype (str): Source numpy dtype (e.g., 'uint32', 'float32').
    - dest_dtype (str): Destination numpy dtype (e.g., 'uint64', 'float32').
    - elements_per_chunk (int): Number of elements to read/write per chunk.
    - delete_file2_on_complete (bool): Whether to delete the source after completion.
    - offset (int): Byte offset to start reading within the source file.
    - add_value (int | None): Optional scalar added to each converted element (after casting).
    - allow_downscaling (bool): Whether to allow downscaling of the data dtype.

    Raises:
    - ValueError: If conversion is not a safe upscaling operation.

    """
    if offset < 0 or offset % np.dtype(source_dtype).itemsize != 0:
        raise ValueError(
            f"Offset {offset} must be non-negative and divisible by source dtype size {np.dtype(source_dtype).itemsize}"
        )
    if not allow_downscaling:
        if source_dtype in INT_ORDER and dest_dtype in INT_ORDER:
            order = INT_ORDER
        elif source_dtype in FLOAT_ORDER and dest_dtype in FLOAT_ORDER:
            order = FLOAT_ORDER
        else:
            raise ValueError(
                f"Unsupported dtype conversion: {source_dtype} → {dest_dtype}. Only same-family upscaling allowed."
            )
        if order.index(dest_dtype) < order.index(source_dtype):
            raise ValueError(f"Downscaling not allowed: {source_dtype} → {dest_dtype}.")

    # Resolve dtypes once (native endianness) and sizes
    source_dtype = np.dtype(source_dtype).newbyteorder("=")
    dest_dtype = np.dtype(dest_dtype).newbyteorder("=")
    src_item = source_dtype.itemsize
    dst_item = dest_dtype.itemsize
    # Pre-cast scalar once to destination dtype for speed
    add_scalar = None
    if add_value is not None and add_value != 0:
        add_scalar = np.array(add_value, dtype=dest_dtype).item()

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
                target_elements = min(elements_per_chunk, num_elements - elements_processed)
                bytes_to_read = target_elements * src_item

                chunk_bytes = f_source.read(bytes_to_read)
                if not chunk_bytes:
                    # Unexpected EOF
                    raise OSError(f"Short read at element {elements_processed}: expected {bytes_to_read} bytes, got 0")

                # Derive actual elements from bytes read to tolerate partial reads
                actual_elements = len(chunk_bytes) // src_item
                if actual_elements == 0:
                    continue

                if source_dtype == dest_dtype and add_scalar is None and len(chunk_bytes) == bytes_to_read:
                    dst_mv = chunk_bytes
                else:
                    src = np.frombuffer(chunk_bytes, dtype=source_dtype, count=actual_elements)
                    dst_mv = memoryview(out_buf)[: actual_elements * dst_item]
                    dst = np.frombuffer(dst_mv, dtype=dest_dtype, count=actual_elements)
                    if add_scalar is not None:
                        np.add(src.astype(dest_dtype, copy=False), add_scalar, out=dst)
                    else:
                        safe_casting = "unsafe" if allow_downscaling else "safe"
                        np.copyto(dst, src, casting=safe_casting)

                f_dest.seek(write_position)
                f_dest.write(dst_mv)
                write_position += len(dst_mv)
                elements_processed += actual_elements

    if delete_file2_on_complete:
        os.remove(second)
