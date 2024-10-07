# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import shutil
from pathlib import Path
from typing import Sequence

import click

from infra_bionemo.new_project.api import check, create_on_filesystem, py_project_structure


__all__: Sequence[str] = ()


@click.command(help="Create a Python project")
@click.option("--project-name", "-p", type=str, required=True, help="Name of new Python project & module.")
@click.option("--location", "-l", type=str, required=True, help="Location to create new project.", default=".")
def entrypoint(project_name: str, location: str) -> None:
    main(**locals())  # pragma: no cover


def main(*, project_name: str, location: str) -> None:
    loc = Path(location)
    print(f"🔨 Creating {loc}/{project_name}")

    if loc.is_file():
        raise ValueError("❌ --location is a file!")

    loc.mkdir(parents=True, exist_ok=True)

    check(project_name)

    new_project_representation = py_project_structure(
        project_name=project_name,
        dependencies=[],
        add_setup_py=False,
        add_test_reqs=True,
        add_dev_reqs=True,
        prefix_test_dirs=True,
    )

    print("🔨 Creating new project on file system.")
    try:
        # create the Python project
        create_on_filesystem(loc, new_project_representation)

    except Exception:  # pragma: no cover
        print("❌ ERROR: failed to create! Cleaning up.")
        shutil.rmtree(str(loc / project_name))
        raise

    print(f"✅ Created {project_name} in {loc} 🎉")


if __name__ == "__main__":
    entrypoint()  # pragma: no cover
