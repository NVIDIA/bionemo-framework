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
import sys
from string import Template
from typing import Sequence


__all__: Sequence[str] = (
    "pyproject_toml_setuptools",
    "pyproject_toml_subproject",
    "setup_py",
    "requirements_txt",
    "readme_md",
    "pytest_example",
)


def pyproject_toml_setuptools(package_name: str, project_name: str) -> str:
    """A pyproject.toml file contents that configures a Python project according to PEP-517 & PEP-518 with setuptools.

    Args:
        package_name: name of the project's Python package.
        project_name: name of the Python project.

    Returns:
        pyproject.toml contents that configure all aspects of the Python project. Uses setuptools.

    Raises:
        ValueError wrapping any encountered exception.
    """
    try:
        return Template(_pyproject_toml_setuptools).substitute(
            package_name=package_name,
            project_name=project_name,
        )
    except Exception as e:  # pragma: no cover
        raise ValueError("😱 Creation of pyproject.toml failed!") from e


def pyproject_toml_subproject(subproject_name: str, internal_dependencies: Sequence[str]) -> str:
    """A pyproject.toml suitable as a bionemo sub-project.

    Args:
        subproject_name: name of the project's Python package, not the top-level namespaced one.
        internal_dependencies: list of other bionemo sub-projects to depend on.

    Returns:
        pyproject.toml contents that configure all aspects of the Python project. Uses setuptools and uv.

    Raises:
        ValueError wrapping any encountered exception.
        ValueError if providing a non-bionemo internal dependency.
    """
    ok_internal_deps = []
    for x in internal_dependencies:
        x = x.strip()
        if len(x) == 0 or not x.startswith("bionemo-"):
            raise ValueError(f"Invalid internal dependency: {x}")
        if x == "bionemo-core":
            print("bionemo-core is always a dependency, ignoring redundant inclusion", file=sys.stderr)
        else:
            ok_internal_deps.append(x)

    try:
        return Template(_pyproject_toml_subproject).substitute(
            subproject_name=subproject_name,
            internal_deps=",".join(ok_internal_deps),
        )
    except Exception as e:  # pragma: no cover
        raise ValueError("😱 Creation of pyproject.toml for bionemo sub-project failed!") from e


def setup_py() -> str:
    """Contents of a minimal setup.py file that works with a pyproject.toml configured project."""
    return _setup_py


def requirements_txt(packages: Sequence[str]) -> str:
    """Contents of a simple requirements.txt style list of Python package dependencies."""
    return "\n".join(packages)


def readme_md(package_name: str, project_name: str) -> str:
    """Contents for the start of a Python project's README in Markdown format.

    Args:
        package_name: name of the project's Python package.
        project_name: name of the Python project.

    Returns:
        Basic README contents.

    Raises:
        ValueError wrapping any encountered exception.
    """
    try:
        return Template(_readme_md).substitute(
            package_name=package_name,
            project_name=project_name,
        )
    except Exception as e:  # pragma: no cover
        raise ValueError("😱 Creation of README.md failed!") from e


def pytest_example() -> str:
    """Contents of an example pytest based Python file."""
    return _pytest_example


_pyproject_toml_subproject: str = """
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "bionemo-${subproject_name}"
readme = "README.md"
description = ""
authors = [{ name = "BioNeMo Team", email = "bionemofeedback@nvidia.com" }]
requires-python = ">=3.10"
license = { file = "LICENSE" }
version = { file = "VERSION" }
dependencies = [
    # internal
    'bionemo-core', ${internal_deps}
    # external
]

[tool.setuptools.packages.find]
where = ["src"]
include = ["bionemo.*"]
namespaces = true
exclude = ["test*."]

[tool.uv]
cache-keys = [{ git = true }]
"""


_pyproject_toml_setuptools: str = """
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

# For guidance, see: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
[project]
name = "${project_name}"
version = "0.0.0"
authors = []
description = ""
readme = "README.md"
requires-python = ">=3.10"
keywords = []
license = {file = "LICENSE"}
classifiers = [
    "Programming Language :: Python :: 3.10",
    "Private :: Do Not Upload",
]
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = {file = ["requirements.txt"]}

[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = [ "ignore::DeprecationWarning",]

[tool.coverage.run]
source = ["${package_name}"]

[tool.black]
line-length = 120
target-version = ['py310']

[tool.ruff]
lint.ignore = ["C901", "E741", "E501",]
# Run `ruff linter` for a description of what selection means.
lint.select = ["C", "E", "F", "I", "W",]
line-length = 120

# Ignore import violations in all `__init__.py` files.
[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["E402", "F401", "F403", "F811",]

[tool.ruff.lint.isort]
lines-after-imports = 2
known-first-party = ["${package_name}"]

[tool.ruff.lint.pydocstyle]
convention = "google"

""".strip()

_setup_py: str = """
from setuptools import setup


if __name__ == "__main__":
    setup()
""".strip()


_readme_md: str = """
# ${project_name}

To install, execute the following:
```bash
pip install -e .
```

To run unit tests, execute:
```bash
pytest -v .
```

""".strip()


_pytest_example: str = """
import pytest
from pytest import fixture, raises, mark


def test_todo() -> None:
    raise ValueError(f"Implement tests! Make use of {fixture} for data, {raises} to check for "
                     f"exceptional cases, and {mark} as needed")

""".strip()
