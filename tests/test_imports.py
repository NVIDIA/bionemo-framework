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


import importlib
from pathlib import Path

import pytest

import bionemo


package_path = bionemo.__file__.replace("__init__.py", "")

imports = []
for path in Path(package_path).rglob("*.py"):
    import_str = (
        str(path)
        .replace(package_path, "bionemo.")
        .replace("__init__.py", "")
        .replace(".py", "")
        .replace("/", ".")
        .strip(".")
    )
    imports.append(import_str)


@pytest.mark.parametrize("import_str", imports)
def test_import(import_str):
    print(import_str)
    try:
        importlib.import_module(import_str)
        assert True
    except Exception as e:
        print(e)
        assert False
