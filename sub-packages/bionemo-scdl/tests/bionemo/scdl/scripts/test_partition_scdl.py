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

"""Tests for the SCDL partition script."""

import os

import numpy as np
import pytest

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.scripts.partition_scdl import ChunkedSCDLMetadata, partition_scdl
from bionemo.scdl.util.scdl_constants import FileNames


def _load_chunk_memmap(chunk_dir, fname, dtype):
    """Load a raw binary memmap file from a chunk directory."""
    path = chunk_dir / fname
    itemsize = np.dtype(dtype).itemsize
    num_elements = os.path.getsize(path) // itemsize
    return np.memmap(path, dtype=dtype, mode="r", shape=(num_elements,))


@pytest.fixture
def partitioned_scdl(tmp_path, make_h5ad_with_raw):
    """Create an SCDL dataset and partition it into chunks.

    Uses a fixed CHUNK_SIZE for clarity between tests and metadata calculation.
    """
    CHUNK_SIZE = 50
    h5ad_path = make_h5ad_with_raw(tmp_path)
    scdl_path = tmp_path / "scdl"
    original = SingleCellMemMapDataset(scdl_path, h5ad_path=h5ad_path)

    chunked_path = tmp_path / "chunked"
    metadata = partition_scdl(scdl_path, chunked_path, chunk_size=CHUNK_SIZE)

    return original, metadata, chunked_path, CHUNK_SIZE


def test_metadata_total_rows_matches_original(partitioned_scdl):
    """Metadata total_rows matches number of rows in original dataset."""
    original, metadata, *_ = partitioned_scdl
    assert metadata.total_rows == len(original)


def test_metadata_num_chunks_formula(partitioned_scdl):
    """Metadata reports correct number of chunks using ceiling division.

    Ensures the calculation matches both the test and the script.
    """
    original, metadata, _, CHUNK_SIZE = partitioned_scdl
    expected_chunks = (len(original) + (CHUNK_SIZE - 1)) // CHUNK_SIZE
    assert metadata.num_chunks == expected_chunks


def test_partition_row_data_correctness(partitioned_scdl):
    """Partitioned chunk's row data matches original for all rows."""
    original, metadata, chunked_path, _ = partitioned_scdl

    for global_idx in range(metadata.total_rows):
        chunk_id, local_idx = metadata.get_chunk_for_row(global_idx)
        chunk_dir = chunked_path / f"chunk_{chunk_id:05d}"

        rowptr = _load_chunk_memmap(chunk_dir, FileNames.ROWPTR.value, metadata.dtypes["rowptr"])
        data = _load_chunk_memmap(chunk_dir, FileNames.DATA.value, metadata.dtypes["data"])
        colptr = _load_chunk_memmap(chunk_dir, FileNames.COLPTR.value, metadata.dtypes["colptr"])

        (orig_vals, orig_cols), _, _ = original.get_row(global_idx)
        chunk_vals = data[rowptr[local_idx] : rowptr[local_idx + 1]]
        chunk_cols = colptr[rowptr[local_idx] : rowptr[local_idx + 1]]

        np.testing.assert_array_equal(orig_vals, chunk_vals)
        np.testing.assert_array_equal(orig_cols, chunk_cols)


def test_metadata_save_load_roundtrip(partitioned_scdl, tmp_path):
    """Metadata saves and loads correctly from disk."""
    _, metadata, _, _ = partitioned_scdl
    metadata_path = tmp_path / "metadata.json"
    metadata.save(metadata_path)
    loaded = ChunkedSCDLMetadata.load(metadata_path)
    assert loaded.total_rows == metadata.total_rows
    assert loaded.chunk_size == metadata.chunk_size
    assert loaded.num_chunks == metadata.num_chunks
    assert loaded.dtypes == metadata.dtypes
    assert loaded.version == metadata.version
