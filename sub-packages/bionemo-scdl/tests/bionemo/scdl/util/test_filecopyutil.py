# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import numpy as np
import pytest

from bionemo.scdl.util.filecopyutil import extend_files
from bionemo.scdl.util.scdl_constants import FLOAT_ORDER, INT_ORDER


# All supported dtypes (order preserved)
_ALL_DTYPES = list(INT_ORDER + FLOAT_ORDER)
_BUFFER_SIZES = [8, 64, 1024, 4096, 65536]


def _same_family_upscale_pairs(order):
    pairs = []
    for i, s in enumerate(order):
        for j, d in enumerate(order):
            if j > i:  # strict upscaling only; same-dtype covered by a separate test
                pairs.append((s, d))
    return pairs


# All valid source→dest pairs within same family (no cross-family, no same-dtype)
SOURCE_DEST_PAIRS = _same_family_upscale_pairs(INT_ORDER) + _same_family_upscale_pairs(FLOAT_ORDER)


@pytest.mark.parametrize("dtype", _ALL_DTYPES)
@pytest.mark.parametrize("buffer_size_b", _BUFFER_SIZES)
def test_extend_files_same_dtype(tmp_path, dtype, buffer_size_b):
    """Extend first file with second when both share the same dtype, using memmap files.

    Uses small integer values for uint types and fractional values for float types.
    """
    if dtype.startswith("float"):
        a = np.array([0.0, 1.5, -2.0, 3.25], dtype=dtype)
        b = np.array([-1.25, 2.75], dtype=dtype)
    else:
        a = np.array([1, 2, 3, 4, 5], dtype=dtype)
        b = np.array([10, 20, 30], dtype=dtype)

    f1 = tmp_path / "first.npy"
    f2 = tmp_path / "second.npy"

    # Create memmap files and write data
    mm1 = np.memmap(f1, dtype=dtype, mode="w+", shape=(a.size,))
    mm1[:] = a
    mm1.flush()
    del mm1

    mm2 = np.memmap(f2, dtype=dtype, mode="w+", shape=(b.size,))
    mm2[:] = b
    mm2.flush()
    del mm2

    # Extend without dtype parameters (fast byte-copy path)
    extend_files(str(f1), str(f2), buffer_size_b=buffer_size_b, delete_file2_on_complete=False)

    # Read back and validate
    merged = np.fromfile(f1, dtype=dtype)
    expected = np.concatenate([a, b])
    np.testing.assert_array_equal(merged, expected)

    # Ensure the second file is unchanged if delete flag was False
    original_second = np.fromfile(f2, dtype=dtype)
    np.testing.assert_array_equal(original_second, b)


@pytest.mark.parametrize(
    "src_dtype,dest_dtype",
    SOURCE_DEST_PAIRS,
    ids=[f"{s}->{d}" for (s, d) in SOURCE_DEST_PAIRS],
)
@pytest.mark.parametrize("buffer_size_b", _BUFFER_SIZES)
def test_extend_files_all_valid_source_pairs_memmap(tmp_path, src_dtype, dest_dtype, buffer_size_b):
    """Verify each valid source→destination conversion with memmap append and dtype conversion."""
    # Representative source values per family
    if src_dtype.startswith("uint"):
        src_vals = np.array([0, 1, 2, 3, 10, 100], dtype=src_dtype)
    else:
        src_vals = np.array([0.0, 1.5, -2.0, 3.25], dtype=src_dtype)

    # Small initial destination to verify append semantics
    if dest_dtype.startswith("uint"):
        dest_initial = np.array([11, 22], dtype=dest_dtype)
    else:
        dest_initial = np.array([1.25, -0.5], dtype=dest_dtype)

    f1 = tmp_path / f"dest_{dest_dtype}.npy"
    f2 = tmp_path / f"src_{src_dtype}.npy"

    mm_dest = np.memmap(f1, dtype=dest_dtype, mode="w+", shape=(dest_initial.size,))
    mm_dest[:] = dest_initial
    mm_dest.flush()
    del mm_dest

    mm_src = np.memmap(f2, dtype=src_dtype, mode="w+", shape=(src_vals.size,))
    mm_src[:] = src_vals
    mm_src.flush()
    del mm_src

    extend_files(
        str(f1),
        str(f2),
        buffer_size_b=buffer_size_b,
        delete_file2_on_complete=False,
        source_dtype=src_dtype,
        dest_dtype=dest_dtype,
    )
    expected = np.concatenate([dest_initial, src_vals.astype(dest_dtype)])
    merged = np.fromfile(f1, dtype=dest_dtype)
    np.testing.assert_allclose(merged, expected, rtol=0, atol=0)
