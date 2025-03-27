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
import tomli
import tomli_w

from infra_bionemo.new_project.api import bionemo_subproject_structure, check, create_on_filesystem
from infra_bionemo.new_project.utils import ask_yes_or_no


__all__: Sequence[str] = ()


@click.command(help="Create a new bionemo sub-package project")
@click.option("--project-name", "-p", type=str, required=True, help="Name of new bionemo sub-package project")
@click.option(
    "--loc-sub-pack",
    "-l",
    type=str,
    required=True,
    help="Location to sub-packages/ directory",
    default="./sub-packages",
)
@click.option(
    "--relax-name-check",
    "-r",
    is_flag=True,
    help="If present, allows --loc-sub-pack to not be exactly 'sub-packages/'.",
)
def entrypoint(project_name: str, loc_sub_pack: str, relax_name_check: bool) -> None:
    main(**locals())  # pragma: no cover


def main(*, project_name: str, loc_sub_pack: str, relax_name_check: bool) -> None:
    location_sub_packages = Path(loc_sub_pack)

    if project_name.startswith("bionemo-"):
        project_name = project_name.split("bionemo-", maxsplit=1)[1]

    full_project_name = f"bionemo-{project_name}"
    print(f"🔨 Creating {location_sub_packages}/{full_project_name}")

    if not location_sub_packages.is_dir():
        raise ValueError(
            f"❌ Need to specify location of sub-packages/ with --loc-sub-pack. Does not exist: {location_sub_packages}"
        )

    if not relax_name_check and location_sub_packages.name != "sub-packages":
        raise ValueError(
            f"❌ Must specify sub-packages/ as --loc-sub-pack, not: {location_sub_packages} "
            f"Otherwise, specify --relax-name-check to skip this check."
        )

    bionemo_fw = location_sub_packages / "bionemo-fw"
    if not bionemo_fw.is_dir():
        raise ValueError(
            "❌ bionemo-fw is missing from sub-packages! "
            f"Check that this exists: {location_sub_packages / 'bionemo-fw'}"
        )
    bionemo_fw_pyproject_toml = bionemo_fw / "pyproject.toml"
    if not bionemo_fw_pyproject_toml.is_file():
        raise ValueError(
            f"❌ bionemo-fw is missing its pyproject.toml file. Cannot add {full_project_name} as a dependency!"
        )

    check(project_name)

    internal_deps = []
    # UPDATE THIS LIST WITH NEW bionemo-* COMPONENT LIBRARIES!
    for component in ["bionemo-llm"]:
        if ask_yes_or_no(f"🤔 Do you want to depend on {component} ?"):
            internal_deps.append(component)

    new_project_representation = bionemo_subproject_structure(
        subproject_name=project_name,
        internal_dependencies=internal_deps,
    )

    print("🔨 Creating new project on file system.")
    try:
        # create the bionemo subpackage project
        create_on_filesystem(location_sub_packages, new_project_representation)

        # add to bionemo-fw's requirements
        _add_dependency(bionemo_fw_pyproject_toml, full_project_name)

    except Exception:  # pragma: no cover
        print("❌ ERROR: failed to create! Cleaning up.")
        shutil.rmtree(str(location_sub_packages / full_project_name))
        raise

    print(f"✅ Created {full_project_name} and added as a dependency to the bionemo-fw package 🎉")


def _add_dependency(bionemo_fw_pyproject_toml: Path, full_project_name: str) -> None:
    with open(str(bionemo_fw_pyproject_toml), "rb") as rb:
        fw_toml = tomli.load(rb)

    if "project" not in fw_toml:
        raise ValueError(
            f"bionemo-fw's pyproject.toml is invalid! No project section found in: {bionemo_fw_pyproject_toml}"
        )
    if "dependencies" not in fw_toml["project"]:
        raise ValueError(
            "bionemo-fw's pyproject.toml is invalid! No project.dependencies section found in: "
            f"{bionemo_fw_pyproject_toml}"
        )
    if not isinstance(fw_toml["project"]["dependencies"], list):
        raise ValueError(
            "bionemo-fw's pyproject.toml is invalid! The project.dependencies section is not a list, it is a "
            f'{type(fw_toml["project"]["dependencies"])=}, found in: '
            f"{bionemo_fw_pyproject_toml}"
        )
    fw_toml["project"]["dependencies"].append(full_project_name)

    fw_toml_s = tomli_w.dumps(fw_toml)
    with open(str(bionemo_fw_pyproject_toml), "wt") as wt:
        wt.write(fw_toml_s)


if __name__ == "__main__":
    entrypoint()  # pragma: no cover
