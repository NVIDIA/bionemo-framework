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
# ruff: noqa: D101, D102, D103, D107
from pathlib import Path

from setuptools import setup


# **NO** LOCAL_REQS allowed for bionemo-core !!!

# Therefore, no need for BIONEMO_PUBLISH_MODE

# Therefore, we can use the requirements.txt file as-is for dependencies.
# Meaning we can just fill it in dynamically from the pyproject.toml !!!

if __name__ == "__main__":
    repo_root = Path(__file__).absolute().parent.parent.parent

    _version_file = repo_root / "VERSION"
    if not _version_file.is_file():
        raise ValueError(f"ERROR: cannot find VERSION file! {str(_version_file)}")
    with open(str(_version_file), "rt") as rt:
        BIONEMO_VERSION: str = rt.read().strip()
    if len(BIONEMO_VERSION) == 0:
        raise ValueError(f"ERROR: no version specified in VERSION file! {str(_version_file)}")

    setup(
        version=BIONEMO_VERSION,
    )
