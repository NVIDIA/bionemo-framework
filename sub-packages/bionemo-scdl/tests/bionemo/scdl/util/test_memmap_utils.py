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

import re

import numpy as np
import pytest

from bionemo.scdl.util.memmap_utils import determine_dtype, smallest_uint_dtype


def test_smallest_uint_dtype():
    assert smallest_uint_dtype(1) == "uint8"
    assert smallest_uint_dtype(256) == "uint16"
    assert smallest_uint_dtype(65536) == "uint32"
    with pytest.raises(ValueError):
        smallest_uint_dtype(2**64)


def test_determine_dtype_finds_correct_dtype():
    # scatter the order of the input dtypes for more robust tests

    # mix order for integer types
    assert determine_dtype(dtypes=[np.uint64, np.uint16, np.uint32, np.uint8]) == "uint64"

    # mix order for mixed family (should raise to float32)
    assert determine_dtype(dtypes=["float32", "float64"]) == "float64"


def test_determine_dtype_raises_error_for_mixed_families():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Mixed float and integer dtype families not allowed: ['float32', 'float64', 'uint16', 'uint32', 'uint64', 'uint8']"
        ),
    ):
        determine_dtype(dtypes=[np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Mixed float and integer dtype families not allowed: ['float32', 'float64', 'uint16', 'uint32', 'uint64', 'uint8']"
        ),
    ):
        determine_dtype(dtypes=[np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64])
