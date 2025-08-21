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


from pathlib import Path

import pytest

from bionemo.scdl.schema.header import SCDLHeader
from bionemo.scdl.schema.magic import SCDL_MAGIC_NUMBER
from bionemo.scdl.schema.version import CurrentSCDLVersion


@pytest.skip("Skipping test_header_file.py because test has not been updated.", allow_module_level=True)
@pytest.mark.parametrize("header_filename", ["header.sch"])
def test_scdl_header_file_valid(test_directory: Path, header_filename: str):
    """Verify header exists, has correct magic, current version, and required arrays.

    Given a path to a SCDL archive (directory), this test checks that:
      - The header file exists
      - The header starts with the SCDL magic number
      - The header version matches the current SCDL schema version
      - The header contains array descriptors for DATA, COLPTR, and ROWPTR (any order)
    """
    header_path = test_directory / header_filename

    # Header file must exist
    assert header_path.exists(), f"Header file not found at {header_path}"

    # Magic number must match
    with open(header_path, "rb") as fh:
        magic = fh.read(4)
    assert magic == SCDL_MAGIC_NUMBER, "Header magic number mismatch"

    # Deserialize and validate version
    header = SCDLHeader.load(str(header_path))
    current_version = CurrentSCDLVersion()
    assert (
        header.version.major == current_version.major
        and header.version.minor == current_version.minor
        and header.version.point == current_version.point
    ), f"Header version {header.version} != current schema version {current_version}"

    # Required arrays must be present (order-agnostic)
    array_names = {arr.name for arr in header.arrays}
    required = {"DATA", "COLPTR", "ROWPTR"}
    missing = required.difference(array_names)
    assert not missing, f"Required arrays missing from header: {missing} (present: {sorted(array_names)})"
