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

from bionemo.utils.remote import RemoteResource


"""
RemoteResource tests and justification

We are not testing downloading behavior, because that is difficult and requires resources to achieve.
We are assuming the code 'works', and validating the rest of the methods.

We test the branchpoints on check_exists because we cannot guarantee correctness without types


No tests for ABCs, as they are just stubs.
Implementors of the ABCs we have created are


Regression/functional tests exist in their own file.

Last set of tests, which should be marked to NOT run in automated tests, and only in a manual manner.
These download and test the behavior of each method, explicitly.
"""


# Test remote resources
def test_RemoteResource_check_exists():
    """Core functionality"""

    checksum = "a95c530a7af5f492a74499e70578d150"
    data = "asdfasdfasdf"
    with open("test-file", "w") as fd:
        fd.write(data)

    result = RemoteResource(
        checksum=checksum,
        dest_filename="test-file",
        dest_directory=".",
        root_directory=".",
        url=None,
    )

    with open("test-file", "rb") as fd:
        # NOTE: checksum was taken from this.
        # expected = md5(data).hexdigest()
        data = fd.read()
    is_true = result.check_exists()
    assert is_true

    bad_checksum = "a95c530a7af5f492a74499e70578d151"
    bad_result = RemoteResource(
        checksum=bad_checksum,
        dest_filename="test-file",
        dest_directory=".",
        root_directory=".",
        url=None,
    )
    is_false = bad_result.check_exists()
    os.remove("test-file")
    assert not is_false


def test_RemoteResource_nochecksum():
    """Has to not fail, when no checksum is provided"""
    checksum = None
    data = "asdfasdfasdf"
    with open("test-file", "w") as fd:
        fd.write(data)

    # File exists, checksum is None, behavior is to greedily assume it 'exists.'
    result = RemoteResource(
        checksum=checksum,
        dest_filename="test-file",
        dest_directory=".",
        root_directory=".",
        url=None,
    )
    assert result.check_exists()
    os.remove("test-file")

    # File is removed, check_exists() should return False.
    result = RemoteResource(
        checksum=checksum,
        dest_filename="test-file",
        dest_directory=".",
        root_directory=".",
        url=None,
    )
    assert not result.check_exists()
