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
"""Generate the code reference pages and copy Jupyter notebooks and README files."""

import logging
from pathlib import Path

import mkdocs_gen_files


# log stuff
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_api_reference() -> None:
    """Generate API reference documentation for a given source directory.

    This function iterates through all 'src' directories in the sub-packages,
    generating API reference documentation for Python files and copying Markdown files.

    Returns:
        None
    """
    root = Path(__file__).parent.parent.parent
    sub_package_srcs = (root / "sub-packages").rglob("src")

    for src in sub_package_srcs:
        # Process Python files
        for path in sorted(src.rglob("*.py")):
            module_path = path.relative_to(src).with_suffix("")
            doc_path = path.relative_to(src).with_suffix(".md")
            full_doc_path = Path("API_reference") / doc_path
            parts = tuple(module_path.parts)

            if parts[-1] == "__init__":
                continue  # Don't generate ref pages for __init__.py
            elif parts[-1] == "__main__":
                continue  # Don't generate ref pages for __main__.py

            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                identifier = ".".join(parts)
                print("::: " + identifier, file=fd)

            mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

        # Process Markdown files
        for path in sorted(src.rglob("*.md")):
            doc_path = path.relative_to(src)
            full_doc_path = Path("API_reference") / doc_path
            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                fd.write(path.read_text())
            logger.info(f"Added Markdown file: {full_doc_path}")
            mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))


def get_subpackage_notebooks(sub_package: Path, root: Path) -> None:
    """Copy Jupyter notebooks from a sub-package to the examples directory.

    Args:
        sub_package (Path): The path to the sub-package directory.
        root (Path): The root directory of the project.

    Returns:
        None
    """
    examples_dir = sub_package / "examples"
    if examples_dir.exists():
        for notebook in examples_dir.glob("*.ipynb"):
            dest_dir = Path("user-guide/examples") / sub_package.name
            dest_file = dest_dir / notebook.name

            with mkdocs_gen_files.open(dest_file, "wb") as fd:
                fd.write(notebook.read_bytes())
            logger.info(f"Added notebook: {dest_file}")
            mkdocs_gen_files.set_edit_path(dest_file, notebook.relative_to(root))


def get_subpackage_readmes(sub_package: Path, root: Path) -> None:
    """Copy README file from a sub-package to the user guide's developer guide directory.

    Args:
        sub_package (Path): The path to the sub-package directory.
        root (Path): The root directory of the project.

    Returns:
        None
    """
    readme_file = sub_package / "README.md"
    if readme_file.exists():
        dest_dir = Path("user-guide/developer-guide") / sub_package.name
        dest_file = dest_dir / f"{sub_package.name}-Overview.md"

        with mkdocs_gen_files.open(dest_file, "w") as fd:
            fd.write(readme_file.read_text())
        logger.info(f"Added README: {dest_file}")
        mkdocs_gen_files.set_edit_path(dest_file, readme_file.relative_to(root))


def generate_pages() -> None:
    """Generate pages for documentation.

    This function orchestrates the entire process of generating API references,
    copying notebooks, and copying README files for all sub-packages.

    Returns:
        None
    """
    root = Path(__file__).parent.parent.parent
    sub_packages_dir = root / "sub-packages"

    # generate api docs
    generate_api_reference()

    for sub_package in sub_packages_dir.glob("bionemo-*"):
        if sub_package.is_dir():
            get_subpackage_notebooks(sub_package, root)
            get_subpackage_readmes(sub_package, root)


if __name__ in {"__main__", "<run_path>"}:
    # Check if name is either '__main__', or the equivalent default in `runpy.run_path(...)`, which is '<run_path>'
    generate_pages()
