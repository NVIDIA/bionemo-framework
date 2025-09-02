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


class Version:
    """Generic version class (used throughout SCDL including for new backing implementations)."""

    def __init__(self, major: int = 0, minor: int = 0, point: int = 0):
        """Initialize a version.

        Args:
            major (int): Major version number.
            minor (int): Minor version number.
            point (int): Patch/point version number.
        """
        self.major = major
        self.minor = minor
        self.point = point


class SCDLVersion(Version):
    """Represent the SCDL schema version.

    This class models the version of the schema used to store data in an archive.
    """

    def __init__(self, major: int = 0, minor: int = 0, point: int = 0):
        """Initialize an SCDL schema version.

        Args:
            major (int): Major version number.
            minor (int): Minor version number.
            point (int): Patch/point version number.
        """
        super().__init__(major, minor, point)

    def __str__(self) -> str:
        """Return the semantic version string.

        Returns:
            str: Version formatted as "major.minor.point".
        """
        return f"{self.major}.{self.minor}.{self.point}"

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            str: Representation including field names and values.
        """
        return f"SCDLVersion(major={self.major}, minor={self.minor}, point={self.point})"

    def __eq__(self, other: "SCDLVersion") -> bool:
        """Return whether two versions are equal.

        Args:
            other (SCDLVersion): The version to compare to.

        Returns:
            bool: True if ``major``, ``minor``, and ``point`` are equal; otherwise False.
        """
        return self.major == other.major and self.minor == other.minor and self.point == other.point

    def __ne__(self, other: "SCDLVersion") -> bool:
        """Return whether two versions are not equal.

        Args:
            other (SCDLVersion): The version to compare to.

        Returns:
            bool: True if any of ``major``, ``minor``, or ``point`` differ; otherwise False.
        """
        return not self == other


class CurrentSCDLVersion(SCDLVersion):
    """Current version of the SCDL schema."""

    def __init__(self):
        """Initialize with the current SCDL schema version: 0.1.0."""
        super().__init__(major=0, minor=1, point=0)


# Note: Backend enums are defined in header.py to maintain consistency
# with binary serialization format which requires integer enum values
