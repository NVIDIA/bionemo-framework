# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
