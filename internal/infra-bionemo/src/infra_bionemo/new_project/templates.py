# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from string import Template
from typing import List, Sequence


__all__: Sequence[str] = (
    "pyproject_toml",
    "setup_py",
    "requirements_txt",
    "readme_md",
    "pytest_example",
)


def pyproject_toml(package_name: str, project_name: str) -> str:
    """Contents of a pyproject.toml file that configures a Python project according to PEP-517 & PEP-518.

    Args:
        package_name: name of the project's Python package.
        project_name: name of the Python project.

    Returns:
        pyproject.toml contents that configure all aspects of the Python project.

    Raises:
        ValueError wrapping any encountered exception.
    """
    try:
        return Template(_pyproject_toml).substitute(
            package_name=package_name,
            project_name=project_name,
        )
    except Exception as e:  # pragma: no cover
        raise ValueError("😱 Creation of pyproject.toml failed!") from e


def setup_py() -> str:
    """Contents of a minimal setup.py file that works with a pyproject.toml configured project."""
    return _setup_py


def requirements_txt(packages: List[str]) -> str:
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


_pyproject_toml: str = """
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
