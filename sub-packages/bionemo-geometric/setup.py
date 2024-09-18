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
import os
from pathlib import Path

from setuptools import setup


LOCAL_REQS: list[str] = [
    "bionemo-core",
]


def is_true(x) -> bool:
    if isinstance(x, str):
        x = x.strip().lower()
        return x == "true" or x == "yes" or x == "y" or x == "1"
    else:
        return x is True


BIONEMO_PUBLISH_MODE: bool = is_true(os.environ.get("BIONEMO_PUBLISH_MODE"))


def read_reqs(f: str) -> list[str]:
    lines = []
    with open(f, "rt") as rt:
        for l in rt:
            l = l.strip()
            if len(l) == 0 or l.startswith("#"):
                continue
            lines.append(l)
    return lines


if __name__ == "__main__":
    repo_root = Path(__file__).absolute().parent.parent.parent

    _version_file = repo_root / "VERSION"
    if not _version_file.is_file():
        raise ValueError(f"ERROR: cannot find VERSION file! {str(_version_file)}")
    with open(str(_version_file), "rt") as rt:
        BIONEMO_VERSION: str = rt.read().strip()
    if len(BIONEMO_VERSION) == 0:
        raise ValueError(f"ERROR: no version specified in VERSION file! {str(_version_file)}")

    _reqs_file = Path(__file__).absolute().parent / "requirements.txt"
    if not _reqs_file.is_file():
        raise ValueError(f"ERROR: no requirements.txt file present! {str(_reqs_file)}")

    reqs = read_reqs(str(_reqs_file))
    if BIONEMO_PUBLISH_MODE:
        for x in LOCAL_REQS:
            reqs.append(f"{x}=={BIONEMO_VERSION}")
    else:
        for x in LOCAL_REQS:
            sub_package_root = repo_root / "sub-packages" / x
            if not sub_package_root.is_dir():
                raise ValueError(f'ERROR: sub-package "{x}" does not exist locally! ({sub_package_root})')
            reqs.append(f"{x} @ {sub_package_root.as_uri()}")

    setup(
        version=BIONEMO_VERSION,
        install_requires=reqs,
    )
