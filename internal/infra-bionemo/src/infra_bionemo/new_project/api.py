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

from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from infra_bionemo.license_check import is_valid_python
from infra_bionemo.new_project.templates import (
    pyproject_toml_setuptools,
    pyproject_toml_subproject,
    pytest_example,
    readme_md,
    requirements_txt,
    setup_py,
)


__all__: Sequence[str] = (
    "File",
    "Dir",
    "create_on_filesystem",
    "bionemo_subproject_structure",
    "namespace_py_project_structure",
    "py_project_structure",
    "check",
    "convert",
)


@dataclass(frozen=True)
class File:
    """Virtual representation of a text file."""

    name: str
    contents: str


@dataclass(frozen=True)
class Dir:
    """Virtual representation of a directory."""

    name: str
    contents: List["Dir" | File]


def create_on_filesystem(starting_location: Path, representation: Dir | File) -> None:
    """Recursively constructs files and directories as specified.

    This function creates files and directories according to the supplied virtual representation. The relationship
    between them is encoded using the :class:`Dir` and :class:`File` classes and specified via `contents` of a
    :class:`Dir`. Files may have contents by populating the `contents` field of the :class:`File` class. Any contents
    will be written to disk. These files and directories are written to disk from the :param:`starting_location`. Note
    that as this function recursively executes, any :class:`Dir` specified in the contents of another :class:`Dir`
    becomes the new :param:`starting_location`.

    Args:
        starting_location: Where the contents will be written to on-disk.
        representation: The virtual filesystem representation being created.

    Raises:
        TypeError if the :param:`representation` is not a :class:`Dir` or :class:`File`.
    """
    if not starting_location.is_dir():
        raise ValueError(f"❌ Starting location must be a directory. This is not: {starting_location}")

    match representation:
        case File(name, contents):
            fi = starting_location / name
            fi.touch(exist_ok=False)
            if len(contents) > 0:
                with open(str(fi), "wt") as wt:
                    wt.write(contents)

        case Dir(name, contents):
            d = starting_location / name
            d.mkdir(exist_ok=False)
            for child in contents:
                create_on_filesystem(d, child)

        case _:
            raise TypeError(f"😱 Expecting a {File} or a {Dir} but obtained a {type(representation)}")


def namespace_py_project_structure(
    base_name: str,
    project_module_name: str,
    dependencies: List[str],
    add_setup_py: bool = True,
    add_test_reqs: bool = False,
    add_dev_reqs: bool = False,
    prefix_test_dirs: bool = False,
) -> Dir:
    """Virtual representation of files and folders for a namespaced Python project.

    The returned `Dir` represents the entire directory containing a namespaced Python project. Such a project needs
    things like a place to store the Python packages and modules (`src/`), a place for unit tests (`tests/`),
    files to list project infrastructure (`requirements*.txt`, `pyproject.toml`, `setup.py`), and documentation
    (`README.md`).

    It also needs to have the right directory setup to support PEP 420 namespace packages. Of note, the `src/`
    directory will contain a directory for the namespace (`base_name`) that will **not** have a `__init__.py` file.
    However, its sub-package directories (first, the Python-friendly version of `project_module_name`) **will** have
    `__init__.py` files like regular Python packages do.

    Note, unlike :func:`py_project_structure`, this function defaults to exclude the test & development dependencies
    under `requirements-test.txt` and `requirements-dev.txt`, respectively. Additionally, this function will include the
    `setup.py` file by default.

    Args:
        base_name: The namespace package name. The import name for the project will follow `base_name.*`.
                   Note, however, that when used as a Python name, this value will have `-` converted to `_`.
        project_module_name: Used in the project infrastructure & documentation files. It's also used to create the
                             first differentiated namespaced Python package and initial unit test file. This will be
                             the first sub-package created under the `base_name` namespace. Note, however, that when
                             used as a Python name, this value will have `-` converted to `_`.
        dependencies: populates the generated `requirements.txt` file.
        add_setup_py: if true, includes a `File` for `setup.py`.
        add_test_reqs: If true, includes a `File` for `requirements-test.txt` populated with `pytest`.
        add_dev_reqs: If true, includes a `File` for `requirements-dev.txt` populated with `ruff` & `ipdb`.
        prefix_test_dirs: If present, then "test_" is prefixed to the name of each directory created under `tests/`
                          with "_" as the word separator.

    Returns:
        Virtual representation of simple Python project on a filesystem.

    Raises:
        ValueError If the :param:`base_name` or :param:`project_module_name` is not a valid Python identifier.
    """
    check(base_name)
    check(project_module_name)

    project_name = f"{base_name}-{project_module_name}"

    base_module = convert(base_name)
    module_name = convert(project_module_name)

    test_dir_prefix = "test_" if prefix_test_dirs else ""

    project = Dir(
        name=project_name,
        contents=[
            File("README.md", readme_md(base_module, project_name)),
            File(
                "pyproject.toml",
                pyproject_toml_setuptools(module_name, project_name),
            ),
            File("requirements.txt", requirements_txt(dependencies)),
            Dir(
                "src",
                contents=[
                    Dir(
                        name=base_module,
                        contents=[
                            Dir(
                                name=module_name,
                                contents=[
                                    File("__init__.py", contents=""),
                                ],
                            )
                        ],
                    )
                ],
            ),
            Dir(
                "tests",
                contents=[
                    Dir(
                        name=f"{test_dir_prefix}{base_module}",
                        contents=[
                            Dir(
                                name=f"{test_dir_prefix}{module_name}",
                                contents=[
                                    File(f"test_TODO_{base_module}_{module_name}.py", contents=pytest_example())
                                ],
                            )
                        ],
                    )
                ],
            ),
        ],
    )

    if add_setup_py:
        project.contents.append(File("setup.py", setup_py()))
    if add_test_reqs:
        project.contents.append(File("requirements-test.txt", requirements_txt(["pytest-cov"])))
    if add_dev_reqs:
        project.contents.append(File("requirements-dev.txt", requirements_txt(["ruff", "ipython", "ipdb"])))

    return project


def bionemo_subproject_structure(
    subproject_name: str,
    internal_dependencies: List[str],
) -> Dir:
    """Virtual representation of files and folders for a bionemo sub-project Python project.

    Very similar to :func:`namespace_py_project_structure`, but specific for creating new sub-projects in
    the bionemo framework repository. Like that function, the returned `Dir` represents the entire directory
    containing a namespaced Python project, with files and subdirectories set up for  PEP 420 namespace packages.

    Args:
        subproject_name: The bionemo sub-package name. Note the directory will be `bionemo-<this value>` and the
                         Python import path will be `import bionemo.<this value>`.
                         When used as a Python name, this value will have `-` converted to `_`.
        internal_dependencies: Other bionemo subprojects to depend on.

    Returns:
        Virtual representation of simple Python project on a filesystem.

    Raises:
        ValueError If the :param:`base_name` or :param:`project_module_name` is not a valid Python identifier.
        ValueError If the :param:`internal_dependencies` are not all bionemo sub-projects.
    """
    # TODO some mild refactoring necessary to this and namespace project creation
    #      most logic is the same, but we want to have a private function to do it and then
    #      call that with the right checking from these 2 public-facing functions
    base_name = "bionemo"
    check(base_name)
    check(subproject_name)

    project_name = f"{base_name}-{subproject_name}"

    base_module = convert(base_name)
    module_name = convert(subproject_name)

    project = Dir(
        name=project_name,
        contents=[
            File("README.md", readme_md(base_module, project_name)),
            File(
                "pyproject.toml",
                pyproject_toml_subproject(subproject_name, internal_dependencies),
            ),
            Dir(
                "src",
                contents=[
                    Dir(
                        name=base_module,
                        contents=[
                            Dir(
                                name=module_name,
                                contents=[
                                    File("__init__.py", contents=""),
                                ],
                            )
                        ],
                    )
                ],
            ),
            Dir(
                "tests",
                contents=[
                    Dir(
                        name=f"{base_module}",
                        contents=[
                            Dir(
                                name=f"{module_name}",
                                contents=[
                                    File(f"test_TODO_{base_module}_{module_name}.py", contents=pytest_example())
                                ],
                            )
                        ],
                    )
                ],
            ),
        ],
    )

    return project


def py_project_structure(
    project_name: str,
    dependencies: List[str],
    add_setup_py: bool = False,
    add_test_reqs: bool = True,
    add_dev_reqs: bool = True,
    prefix_test_dirs: bool = True,
) -> Dir:
    """Virtual representation of files and folders for a simple, non-namespaced Python project.

    The returned `Dir` represents the entire directory containing a Python project. Such a project needs
    things like a place to store the Python packages and modules (`src/`), a place for unit tests (`tests/`),
    files to list project infrastructure (`requirements*.txt`, `pyproject.toml`, `setup.py`), and documentation
    (`README.md`).

    Note, unlike :func:`namespace_py_project_structure`, this function defaults to include the test & development
    dependencies under `requirements-test.txt` and `requirements-dev.txt`, respectively. Additionally, this function
    will not include the `setup.py` file by default.

    Args:
        project_name: Used in the project infrastructure & documentation files. It's also used to create the first
                        Python package and initial unit test file.
        dependencies: Populates the generated `requirements.txt` file.
        add_setup_py: If true, includes a `File` for `setup.py`.
        add_test_reqs: If true, includes a `File` for `requirements-test.txt` populated with `pytest`.
        add_dev_reqs: If true, includes a `File` for `requirements-dev.txt` populated with `ruff` & `ipdb`.
        prefix_test_dirs: If present, then "test_" is prefixed to the name of each directory created under `tests/`
                           with "_" as the word separator.

    Returns:
        Virtual representation of simple Python project on a filesystem.

    Raises:
        ValueError If the project name is not a valid Python package or module name.
    """
    check(project_name)

    module_name = convert(project_name)

    test_dir_prefix = "test_" if prefix_test_dirs else ""

    project = Dir(
        name=project_name,
        contents=[
            File("README.md", readme_md(module_name, project_name)),
            File(
                "pyproject.toml",
                pyproject_toml_setuptools(module_name, project_name),
            ),
            File("requirements.txt", requirements_txt(dependencies)),
            Dir(
                "src",
                contents=[
                    Dir(
                        name=module_name,
                        contents=[
                            File("__init__.py", contents=""),
                        ],
                    )
                ],
            ),
            Dir(
                "tests",
                contents=[
                    Dir(
                        f"{test_dir_prefix}{module_name}",
                        contents=[File(f"test_TODO_{module_name}.py", contents=pytest_example())],
                    )
                ],
            ),
        ],
    )

    if add_setup_py:
        project.contents.append(File("setup.py", setup_py()))
    if add_test_reqs:
        project.contents.append(File("requirements-test.txt", requirements_txt(["pytest-cov"])))
    if add_dev_reqs:
        project.contents.append(File("requirements-dev.txt", requirements_txt(["ruff", "ipython", "ipdb"])))

    return project


def check(project_module_name: str) -> None:
    """Checks whether or not the input is acceptable as a Python module or package name.

    Raises:
        ValueError if the input is invalid. Error message will contain specific reason.
    """
    project_module_name = project_module_name.strip()

    if len(project_module_name) == 0:
        raise ValueError(f"❌ Must be non-empty: {project_module_name=}")

    if " " in project_module_name:
        raise ValueError(f"❌ No empty spaces allowed in {project_module_name=}")

    if project_module_name[0] in string.digits:
        raise ValueError(f"❌ Cannot start with number: {project_module_name=}")

    for i, c in enumerate(project_module_name):
        if c in string.ascii_uppercase:
            raise ValueError(
                f'❌  Cannot have capital letters: character {i} ("{c}") is not allowed for {project_module_name=}'
            )

    if "_" in project_module_name:
        raise ValueError(f"❌ Use '-' instead of '_' as a word separator for {project_module_name=}")

    as_python = convert(project_module_name)
    # We want to know if the name will be a valid python module.
    # A quick hack is to try and see if it would work as the name of a value.
    # So we make it equal to an int in a small Python program and try to parse it w/ the ast package.
    if is_valid_python(f"{as_python} = 10") is not None:
        raise ValueError(
            f"❌ {project_module_name=} is invalid as a python module name ({as_python=}): "
            "it contains one or more not allowed characters"
        )


def convert(project_module_name: str) -> str:
    """Replaces hyphens with underscores and removes surrounding whitespace."""
    return project_module_name.strip().replace("-", "_")
