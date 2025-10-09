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
import scipy.sparse as sp

from bionemo.scdl.util.memmap_utils import check_integer_valued_and_cast


def test_integer_and_float_compression_and_cast(tmp_path):
    """Test integer detection and dtype casting helper."""

    # Integer-valued floats (0-255 range) → uint8
    data = np.array([1.0, 5.0, 23.0, 156.0, 200.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3, 4])
    indptr = np.array([0, 2, 5])
    sparse_matrix = sp.csr_matrix((data, indices, indptr), shape=(2, 5))
    result_dtype = check_integer_valued_and_cast(sparse_matrix.data)
    assert result_dtype == "uint8", f"Expected uint8, got {result_dtype}"
    converted_data = sparse_matrix.data.astype(result_dtype)
    assert np.allclose(converted_data, data)

    # Larger integer values → uint16
    data_large = np.array([1.0, 500.0, 2000.0, 30000.0], dtype=np.float32)
    sparse_large = sp.csr_matrix((data_large, [0, 1, 2, 3], [0, 2, 4]), shape=(2, 4))
    result_dtype_large = check_integer_valued_and_cast(sparse_large)
    assert result_dtype_large == "uint16", f"Expected uint16, got {result_dtype_large}"

    # Non-integer values → keep float32
    data_float = np.array([1.5, 2.7, 3.14], dtype=np.float32)
    sparse_float = sp.csr_matrix((data_float, [0, 1, 2], [0, 2, 3]), shape=(2, 3))
    result_dtype_float = check_integer_valued_and_cast(sparse_float)
    assert result_dtype_float == "float32" or result_dtype_float == np.float32
