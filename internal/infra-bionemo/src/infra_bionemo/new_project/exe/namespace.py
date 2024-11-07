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

import shutil
from pathlib import Path
from typing import Sequence

import click

from infra_bionemo.new_project.api import check, create_on_filesystem, namespace_py_project_structure


__all__: Sequence[str] = ()


def main(
    *,
    namespace: str,
    module: str,
    location: str,
    no_test_append: bool,
) -> None:
    loc = Path(location)
    project_name = f"{namespace}-{module}"
    print(f"🔨 Creating namespaced project {loc}/{project_name}")

    if loc.is_file():
        raise ValueError("❌ --location is a file!")

    loc.mkdir(parents=True, exist_ok=True)

    check(namespace)
    check(module)

    new_project_representation = namespace_py_project_structure(
        base_name=namespace,
        project_module_name=module,
        dependencies=[],
        add_setup_py=False,
        add_test_reqs=True,
        add_dev_reqs=True,
        prefix_test_dirs=not no_test_append,
    )

    print("🔨 Creating new namespace Python project on file system.")
    try:
        # create the Python project
        create_on_filesystem(loc, new_project_representation)

    except Exception:  # pragma: no cover
        print("❌ ERROR: failed to create! Cleaning up.")
        shutil.rmtree(str(loc / project_name))
        raise

    print(f"✅ Created namespaced {project_name} in {loc} 🎉")


@click.command(help="Create a new bionemo sub-package project")
@click.option("--namespace", "-n", type=str, required=True, help="Name of new Python base namespace.")
@click.option("--module", "-m", type=str, required=True, help="Name of new Python subpackage in the namespace.")
@click.option("--location", "-l", type=str, required=True, help="Location to create new project.", default=".")
@click.option(
    "--no-test-append",
    is_flag=True,
    help="If present, do not append 'test_' to the name of each directory created under 'tests/'",
)
def entrypoint(
    namespace: str,
    module: str,
    location: str,
    no_test_append: bool,
) -> None:
    main(**locals())  # pragma: no cover


if __name__ == "__main__":
    entrypoint()  # pragma: no cover
