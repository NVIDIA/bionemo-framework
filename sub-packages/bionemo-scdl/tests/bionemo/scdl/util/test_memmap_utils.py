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

from bionemo.scdl.util.memmap_utils import determine_dtype, smallest_uint_dtype


def test_smallest_uint_dtype():
    assert smallest_uint_dtype(1) == "uint8"
    assert smallest_uint_dtype(256) == "uint16"
    assert smallest_uint_dtype(65536) == "uint32"
    with pytest.raises(ValueError):
        smallest_uint_dtype(18446744073709551616)


def test_determine_dtype():
    # scatter the order of the input dtypes for more robust tests

    # mix order for integer types
    assert determine_dtype([np.uint64, np.uint16, np.uint32, np.uint8]) == "uint64"

    # mix order for mixed family (should raise to float32)
    assert determine_dtype(["float32", "float16"]) == "float32"
    with pytest.raises(ValueError):
        determine_dtype([np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64])
    with pytest.raises(ValueError):
        determine_dtype([np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64])
