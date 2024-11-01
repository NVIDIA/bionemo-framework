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
    bionemo_fw_reqs = bionemo_fw / "requirements.txt"
    if not bionemo_fw_reqs.is_file():
        raise ValueError(
            "❌ bionemo-fw is missing its requirements.txt file. " f"Cannot add {full_project_name} as a dependency!"
        )

    check(project_name)

    internal_deps = []
    # UPDATE THIS LIST WITH NEW bionemo-* COMPONENT LIBRARIES!
    for component in ["bionemo-llm"]:
        if ask_yes_or_no(f"🤔 Do you want to depend on {component} ?"):
            internal_deps.append(f"-e ../{component}")

    new_project_representation = namespace_py_project_structure(
        base_name="bionemo",
        project_module_name=project_name,
        dependencies=internal_deps,
        add_setup_py=False,
        add_test_reqs=False,
        add_dev_reqs=False,
        prefix_test_dirs=False,
    )

    print("🔨 Creating new project on file system.")
    try:
        # create the bionemo subpackage project
        create_on_filesystem(location_sub_packages, new_project_representation)

        # add to bionemo-fw's requirements
        with open(str(bionemo_fw_reqs), "at") as wt:
            wt.write(f"\n-e ../{full_project_name}\n")

    except Exception:  # pragma: no cover
        print("❌ ERROR: failed to create! Cleaning up.")
        shutil.rmtree(str(location_sub_packages / full_project_name))
        raise

    print(f"✅ Created {full_project_name} and added as a dependency to the bionemo-fw package 🎉")


if __name__ == "__main__":
    entrypoint()  # pragma: no cover
