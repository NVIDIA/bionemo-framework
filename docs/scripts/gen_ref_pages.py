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


"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files


root = Path(__file__).parent.parent.parent
sub_package_srcs = (root / "sub-packages").rglob("src")

for src in sub_package_srcs:
    for path in sorted(src.rglob("*.py")):
        module_path = path.relative_to(src).with_suffix("")
        doc_path = path.relative_to(src).with_suffix(".md")
        full_doc_path = Path("API_reference", doc_path)

        parts = tuple(module_path.parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
        elif parts[-1] == "__main__":
            continue

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            identifier = ".".join(parts)
            print("::: " + identifier, file=fd)

        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))
