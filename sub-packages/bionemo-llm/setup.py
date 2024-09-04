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


from setuptools import setup


from pathlib import Path
from setuptools import setup


_version_file = Path(__file__).absolute().parent.parent.parent / 'VERSION'
if not _version_file.is_file():
    raise ValueError(f"ERROR: cannot find VERSION file! {str(_version_file)}")
with open(str(_version_file), 'rt') as rt:
    BIONEMO_VERSION: str = rt.read().strip()
if len(BIONEMO_VERSION) == 0:
    raise ValueError(f"ERROR: no verison specified in VERSION file! {str(_version_file)}")

_reqs_file = Path(__file__).absolute().parent / 'requirements.txt'
if not _reqs_file.is_file():
    raise ValueError(f"ERROR: no requirements.txt file present! {str(_reqs_file)}")


def read_reqs(f: str) -> list[str]:
    lines = []
    with open(f, 'rt') as rt:
        for l in rt:
            l = l.strip()
            if len(l) == 0 or l.startswith("#"):
                continue
            lines.append(l)
    return lines


LOCAL_REQS: list[str] = [
    'bionemo-core',
]


if __name__ == "__main__":
    # L = dict(**locals())
    # G = dict(**globals())
    #
    # def write(wt, mapping):
    #     for k, v in mapping.items():
    #         wt.write(f"\t{k} ({type(k)}):  ({type(v)}) {v}\n")
    #         wt.write('-'*80)
    #         wt.write("\n")
    #
    # with open('here.txt', 'wt') as wt:
    #     wt.write("globals()=\n")
    #     write(wt, G)
    #
    #     wt.write("\n\nlocals()=\n")
    #     write(wt, L)
    # import ipdb
    #
    # ipdb.set_trace()

    setup(
        version=BIONEMO_VERSION,
        install_requires=LOCAL_REQS + read_reqs(str(_reqs_file)),
    )
