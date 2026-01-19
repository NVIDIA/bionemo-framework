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

"""Partition a monolithic SCDL dataset into chunks for efficient remote access.

This script takes an existing SCDL dataset and splits it into smaller chunks,
each containing a subset of rows. The chunked format enables:
- Efficient remote storage access (fetch only needed chunks)
- Local caching with LRU eviction
- Parallel prefetching during training

The script handles large files (5TB+) efficiently by:
- Using streaming binary I/O to avoid loading entire arrays into memory
- Processing data in configurable buffer sizes
- Creating raw memmap files consistent with SCDL format

Usage:
    partition-scdl --input /path/to/scdl --output /path/to/chunked --chunk-size 100000
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.schema.header import ChunkedInfo, SCDLHeader
from bionemo.scdl.util.scdl_constants import Backend, FileNames


logger = logging.getLogger(__name__)


def partition_scdl(
    input_path: Path,
    output_path: Path,
    chunk_size: int = 100_000,
    buffer_elements: int = 10 * 1024 * 1024,  # ~10M elements per buffer
    delete_original: bool = False,
) -> SCDLHeader:
    """Partition an SCDL dataset into chunks.

    Uses streaming binary I/O to handle large files (5TB+) efficiently without
    loading entire arrays into memory.

    Args:
        input_path: Path to existing monolithic SCDL dataset.
        output_path: Path where chunked dataset will be created.
        chunk_size: Number of rows per chunk (default: 100,000).
        buffer_elements: Number of elements to read/write per buffer (default: ~10M).
        delete_original: If True, delete the original dataset after successful partitioning.

    Returns:
        SCDLHeader with Backend.CHUNKED_MEMMAP_V0 and ChunkedInfo.

    Raises:
        FileNotFoundError: If input_path doesn't exist or is missing required files.
        FileExistsError: If output_path already exists.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Validate input
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    required_files = [FileNames.DATA.value, FileNames.ROWPTR.value, FileNames.COLPTR.value]
    for fname in required_files:
        if not (input_path / fname).exists():
            raise FileNotFoundError(f"Missing required file: {input_path / fname}")

    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")

    output_path.mkdir(parents=True)

    # Load the source SCDL dataset (uses memmaps, doesn't load data into RAM)
    source_ds = SingleCellMemMapDataset(str(input_path))
    total_rows = len(source_ds)
    rowptr = source_ds.row_index
    num_chunks = (total_rows + chunk_size - 1) // chunk_size

    # Get dtypes from source
    rowptr_dtype = np.dtype(source_ds.dtypes[FileNames.ROWPTR.value])
    data_dtype = np.dtype(source_ds.dtypes[FileNames.DATA.value])
    colptr_dtype = np.dtype(source_ds.dtypes[FileNames.COLPTR.value])

    # Create chunks using streaming binary I/O
    for chunk_id in range(num_chunks):
        row_start = chunk_id * chunk_size
        row_end = min(row_start + chunk_size, total_rows)
        chunk_dir = output_path / f"chunk_{chunk_id:05d}"
        chunk_dir.mkdir()

        # Calculate data slice for this chunk
        data_start = int(rowptr[row_start])
        data_end = int(rowptr[row_end])
        num_elements = data_end - data_start

        # Create adjusted row pointers (relative to chunk start)
        chunk_rowptr = (rowptr[row_start : row_end + 1] - data_start).astype(rowptr_dtype)

        # Write rowptr as raw binary (small enough to do in one go)
        with open(chunk_dir / FileNames.ROWPTR.value, "wb") as f:
            f.write(chunk_rowptr.tobytes())

        # Stream data and colptr using buffered binary I/O
        _stream_copy_slice(
            input_path / FileNames.DATA.value,
            chunk_dir / FileNames.DATA.value,
            data_start,
            num_elements,
            data_dtype,
            buffer_elements,
        )

        _stream_copy_slice(
            input_path / FileNames.COLPTR.value,
            chunk_dir / FileNames.COLPTR.value,
            data_start,
            num_elements,
            colptr_dtype,
            buffer_elements,
        )

        logger.info(f"Created chunk {chunk_id}: rows {row_start}-{row_end} ({num_elements} elements)")

    # Copy feature indices
    _copy_global_features(input_path, output_path)

    # Copy metadata files
    for fname in [FileNames.VERSION.value, FileNames.METADATA.value]:
        src = input_path / fname
        if src.exists():
            shutil.copy(src, output_path / fname)

    # Copy original header and add chunked info
    header = source_ds.header if source_ds.header else SCDLHeader()
    header.backend = Backend.CHUNKED_MEMMAP_V0
    header.chunked_info = ChunkedInfo(chunk_size=chunk_size, num_chunks=num_chunks, total_rows=total_rows)
    header.save(str(output_path / FileNames.HEADER.value))

    logger.info(f"Created {num_chunks} chunks from {total_rows} rows")
    logger.info(f"SCDLHeader saved to: {output_path / FileNames.HEADER.value}")

    # Delete original dataset if requested
    if delete_original:
        # Close memmap references before deleting
        del source_ds
        shutil.rmtree(input_path)
        logger.info(f"Deleted original dataset: {input_path}")

    return header


def _stream_copy_slice(
    src_path: Path,
    dst_path: Path,
    start_element: int,
    num_elements: int,
    dtype: np.dtype,
    buffer_elements: int,
) -> None:
    """Copy a slice of a binary array file using streaming I/O.

    Avoids loading the entire slice into memory by reading/writing in buffers.

    Args:
        src_path: Source file path.
        dst_path: Destination file path.
        start_element: Starting element index in source.
        num_elements: Number of elements to copy.
        dtype: Numpy dtype of elements.
        buffer_elements: Number of elements per buffer.
    """
    itemsize = dtype.itemsize
    start_byte = start_element * itemsize

    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        src.seek(start_byte)
        elements_remaining = num_elements

        while elements_remaining > 0:
            to_read = min(buffer_elements, elements_remaining) * itemsize
            chunk = src.read(to_read)
            if not chunk:
                raise IOError(f"Unexpected EOF reading {src_path}")
            dst.write(chunk)
            elements_remaining -= len(chunk) // itemsize


def _copy_global_features(input_path: Path, output_path: Path) -> None:
    """Copy global features (var_features and obs_features) across all chunks."""
    for name in [FileNames.VAR_FEATURES.value, FileNames.OBS_FEATURES.value]:
        src = input_path / name
        if src.exists():
            shutil.copytree(src, output_path / name)
            logger.info(f"Copied {name} to output")


def main():
    """CLI entry point for partitioning SCDL datasets."""
    parser = argparse.ArgumentParser(description="Partition a monolithic SCDL dataset into chunks for remote access.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input SCDL dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where the chunked dataset will be created.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of rows per chunk (default: 100,000).",
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Delete the original dataset after successful partitioning.",
    )

    args = parser.parse_args()

    partition_scdl(
        input_path=Path(args.input),
        output_path=Path(args.output),
        chunk_size=args.chunk_size,
        delete_original=args.delete_original,
    )


if __name__ == "__main__":
    main()
