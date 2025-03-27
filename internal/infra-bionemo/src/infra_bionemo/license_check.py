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


import ast
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import click


__all__: Sequence[str] = (
    "LICENSE_HEADER",
    "Checked",
    "HeaderNotFound",
    "LicenseCheckError",
    "append_license_header",
    "check_license_project_files",
    "ensure_license_starts_with_pound",
    "has_header",
    "is_valid_python",
    "license_check",
    "main",
    "remove_existing_license_header",
)

NVIDIA_COPYRIGHT: str = (
    "# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
)
APACHE_BLOCK: str = """
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
""".strip()

# default header (split to allow for intermediate copyright headers)
LICENSE_HEADER = f"{NVIDIA_COPYRIGHT}\n{APACHE_BLOCK}"


@dataclass(frozen=True)
class HeaderNotFound(ValueError):
    """Error that indicates the pointed-to file does not have a valid license header."""

    pyfile: Path

    def __str__(self) -> str:  # noqa: D105
        return f"{self.pyfile.name} does not have the license header!"


LicenseCheckError = IOError | SyntaxError | HeaderNotFound
"""Errors that can be encountered during the license check process.

Specific errors and their underlying causes:
  - IOError: problem reading file
  - SyntaxError: the input file for license checking is not valid Python
  - HeaderNotFound: the input file was valid Python, but did not have the right license @ the header
"""


def license_check(
    pyfile: Path, *, license_header: str = LICENSE_HEADER, modify: bool, replace: bool = False
) -> Optional[LicenseCheckError]:
    """Check Python file for license header, returning nothing on success or an error describing the failure."""
    if not pyfile.is_file():
        return IOError(f"{pyfile.name} file does not exist!")

    with open(str(pyfile), "rt") as rt:
        pyfile_contents: str = rt.read()

    maybe_err = is_valid_python(pyfile_contents)
    if maybe_err is not None:
        return maybe_err

    if has_header(pyfile_contents, license_header=license_header):
        return None

    elif modify:
        # `pyfile` doesn't start with `license_header` text

        if replace:
            # does it start with some other license header?
            # if so, then we delete that before appending our new `license_header` text
            pyfile_contents = remove_existing_license_header(pyfile_contents)
            maybe_err = is_valid_python(pyfile_contents)
            if maybe_err is not None:
                return maybe_err

        pyfile_contents = append_license_header(pyfile_contents, license_header=license_header)
        maybe_err = is_valid_python(pyfile_contents)
        if maybe_err is not None:
            return maybe_err

        with open(str(pyfile), "wt") as wt:
            wt.write(pyfile_contents)
        return None

    else:
        return HeaderNotFound(pyfile)


def is_valid_python(pyfile_contents: str) -> Optional[SyntaxError]:
    """Returns a SyntaxError detailing why the input string is not valid Python. Returns None if it is valid."""
    try:
        _ = ast.parse(pyfile_contents)
    except SyntaxError as error:
        return error
    else:
        return None


def has_header(pyfile_contents: str, *, license_header: str = LICENSE_HEADER) -> bool:
    """Check if file has valid license header.

    First checks if file has multiple copyright lines - if so, validates structure only.
    If not, and custom license_header provided, does exact string match.
    Otherwise validates basic structure.
    """
    lines = pyfile_contents.split("\n")

    # Count copyright lines at start of file
    copyright_count = 0
    for line in lines:
        if line.strip().startswith("# SPDX-FileCopyrightText: Copyright"):
            copyright_count += 1
        else:
            break

    # If file has multiple copyrights, only validate structure
    if copyright_count > 1:
        # Must start with NVIDIA copyright
        if not lines or not lines[0].strip() == NVIDIA_COPYRIGHT:
            return False

        # Find where Apache block starts
        apache_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("# SPDX-License-Identifier: LicenseRef-Apache2"):
                apache_start = i
                break

        if apache_start is None:
            return False

        # All lines between NVIDIA copyright and Apache block must be valid SPDX copyright lines
        for line in lines[1:apache_start]:
            if line.strip() and not line.strip().startswith("# SPDX-FileCopyrightText: Copyright"):
                return False

        # Check Apache block matches exactly
        apache_lines = APACHE_BLOCK.split("\n")
        if len(lines[apache_start:]) < len(apache_lines):
            return False

        for actual, expected in zip(lines[apache_start : apache_start + len(apache_lines)], apache_lines):
            if actual.strip() != expected.strip():
                return False

        return True

    # Otherwise, if custom header provided, use exact match
    if license_header != LICENSE_HEADER:
        return pyfile_contents.startswith(license_header)

    # Otherwise do basic structure validation
    return lines[0].strip() == NVIDIA_COPYRIGHT and pyfile_contents.startswith(LICENSE_HEADER)


def append_license_header(pyfile_contents: str, *, license_header: str = LICENSE_HEADER, n_sep_lines: int = 2) -> str:
    """Appends the :param:`license_header` to the beginning of the input Python code (:param:`pyfile_contents`).

    Inserts :param:`n_sep_lines` newlines between the license header & Python file contents.
    """
    spacer = "\n" * n_sep_lines
    return f"{license_header}{spacer}{pyfile_contents}"


def remove_existing_license_header(pyfile_contents: str) -> str:
    """Heuristically removes the license header from a Python file's contents.

    Assumes that a license header is identified by a span of commented-out lines from the beginning of the file.
    I.e. a big initial block of lines starting with "#" ==> a license header.

    Will always return the input without this "license header" block.
    """
    if not pyfile_contents.startswith("#") or len(pyfile_contents) == 0:
        return pyfile_contents
    lines: List[str] = pyfile_contents.split("\n")
    non_header_lines = lines[_last_index_of_header_comment_line(lines) + 1 :]
    return "\n".join(non_header_lines)


def _last_index_of_header_comment_line(lines: List[str]) -> int:
    """Returns positive int: index into `lines` with the first line that doesn't start as a comment."""
    if len(lines) == 0:
        raise ValueError()
    last_index_of_line_that_started_with_hash_from_beginning: int = -1
    for i, line in enumerate(lines):
        if line.startswith("#"):
            last_index_of_line_that_started_with_hash_from_beginning = i
        else:
            break
    if last_index_of_line_that_started_with_hash_from_beginning < 0:
        raise ValueError("Must supply non-empty lines of Python!")
    return last_index_of_line_that_started_with_hash_from_beginning


@dataclass(frozen=True)
class Checked:
    """Result of running license check across a collection of Python files."""

    noncompliant_files: Mapping[Path, LicenseCheckError]
    """Files that either don't have a license header for some reason or another.
    """

    n_files: int
    """Total number of Python files checked.
    """


def check_license_project_files(
    python_package_directory: Path, *, license_header: str, modify: bool, replace: bool
) -> Checked:
    """Recursively checks all Python files in a given directory. Returns all noncompliant files.

    Each returned file will be associated with the specific :class:`LicenseCheckError`.
    For more details,
    see :func:`license_check`.
    """
    assert python_package_directory.is_dir(), (
        f"Input must be a directory of Python files, not a directory: {python_package_directory}"
    )
    noncompliant_files = {}
    n_files = 0
    for pyfile in python_package_directory.rglob("*.py"):
        n_files += 1
        maybe_error = license_check(pyfile, license_header=license_header, modify=modify, replace=replace)
        if maybe_error is not None:
            noncompliant_files[pyfile] = maybe_error
    return Checked(noncompliant_files, n_files)


def ensure_license_starts_with_pound(license_header_contents: str) -> str:
    """Ensures that each line of the license headers starts with "# "; adds if necessary."""
    if len(license_header_contents) == 0:
        raise ValueError("License header must be non-empty!")
    safe_license_header_lines: List[str] = []
    for line in license_header_contents.split("\n"):
        if not line.startswith("#"):
            line = f"# {line}"
        safe_license_header_lines.append(line)
    return "\n".join(safe_license_header_lines)


@click.command(help="Check that Python files start with a license header.")
@click.option(
    "--check",
    "-c",
    required=True,
    multiple=True,
    type=str,
    help="Either a file or directory. If a directory, then all files that are accessible will be included (directories)"
    " are searched recursively). Only files that end with *.py will be included. Acceptable to use multiple "
    "times. All --check files will be included. Must specify at least one *.py file.",
)
@click.option(
    "--modify",
    "-m",
    is_flag=True,
    help="If present, modifies files that don't have the license header. "
    "Otherwise, will error-out if it finds any non-compliant files.",
)
@click.option(
    "--license-header",
    "-l",
    required=False,
    help="If present, loads the license header from this file. Defaults to use standard license header.",
)
@click.option(
    "--add-leading",
    "-a",
    is_flag=True,
    help="If present, will ensure that each line of the license header starts with '#'. "
    "If any line doesn't, then this option will make the program append '# ' to the start of each line.",
)
@click.option(
    "--replace",
    "-r",
    is_flag=True,
    help="If present, will replace an existing license header. By default, this program simply appends the"
    "license header text to each .py file. This option will employ a heuristic to detect if a .py file "
    "starts with a license header. If detected, then this text is removed before the normal license "
    "header appending logic runs.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="If present, will perform extra (verbose) logging. ",
)
def entrypoint(
    check: Tuple[str], modify: bool, license_header: Optional[str], add_leading: bool, replace: bool, verbose: bool
) -> None:
    print(f"Files/directories for finding .py files: {check}")
    print(f"Modify .py files with license header?:   {modify}")
    print(f"Overriding standard license header?:     {license_header}")
    print(f"Force each line to start with '# '?:     {add_leading}")
    print(f"Check for and replace existing header?:  {replace}")
    print(f"Verbose (extra) logging?:                {verbose}")
    print("-" * 100)

    if len(check) == 0:
        raise ValueError("Must supply at least one file or directory via --check !")

    if replace and not modify:
        raise ValueError("Must use --modify if also using --replace !")

    if modify and not replace:
        print(
            "WARNING: existing license headers are ignored. "
            "To replace any existing header text, re-run with --replace. "
        )

    # get all files / directories from --check
    files: List[Path] = []
    directories: List[Path] = []
    unknown: List[Path] = []
    for f in check:
        p = Path(f).absolute()
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            directories.append(p)
        else:
            unknown.append(p)

    # check that they all exist
    if len(unknown) > 0:
        raise ValueError(f"Found {len(unknown)} --check things that do not exist!\n\n".join(map(str, unknown)))

    # check that files passed in explicitly from --check end with .py
    non_py_files = [f for f in files if not f.name.endswith(".py")]
    if len(non_py_files) > 0:
        raise ValueError(
            f"Found {len(non_py_files)} files from --check that are not Python files! (They don't end with .py):\n"
            "\n".join(map(str, non_py_files))
        )

    # resolve the license header: either the default or user override .txt file
    if license_header is not None:
        lic_file = Path(license_header).absolute()
        if not lic_file.is_file():
            raise ValueError(f"Supplied a --license-header, but {license_header} is not a file!")
        with open(str(lic_file), "rt") as rt:
            license_header_contents: str = rt.read()
        msg_license: str = "Using custom license header"
    else:
        license_header_contents = LICENSE_HEADER
        msg_license = "Using default license header"

    if add_leading:
        msg_license += ". Ensuring each line of the license header starts with '#'"
        license_header_contents = ensure_license_starts_with_pound(license_header_contents)

    print(f"{msg_license}:\n{license_header_contents}" if verbose else f"{msg_license}.")

    # run license check
    try:
        checked_n_files: int = main(
            modify,
            license_header_contents,
            files=files,
            directories=directories,
            replace=replace,
        )
    except ValueError as error:
        print(str(error))
        sys.exit(1)
    else:
        print(f"Success! All {checked_n_files} checked have the required license header!")


def main(
    modify: bool,
    license_header_contents: str,
    *,
    files: List[Path],
    directories: List[Path],
    replace: bool,
) -> int:
    """Runs license check on all files & files accessible from the directories.

    On failure, raises an error with all noncompliant files. Returns nothing on success.
    See :func:`check_license_project_files` for details.

    Returns the number of files checked on success. On failure, :raises:`ValueError` with message
    containing the # of non-compliant files & their specific :class:`LicenseCheckError` errors.

    The :param:`replace` option will heuristically check for an existing license header. It will remove
    and replace this with the :param:`license_header_contents`.
    """
    if len(files) == 0 and len(directories) == 0:
        raise ValueError("Must supply at least one file or directory!")
    if len(license_header_contents) == 0:
        raise ValueError("Must supply non-empty license header!")
    checked = _main(modify, license_header_contents, files, directories, replace)
    if len(checked.noncompliant_files) > 0:
        raise _error(checked.noncompliant_files, checked.n_files, modify)
    else:
        return checked.n_files


def _main(
    modify: bool,
    license_header_contents: str,
    files: List[Path],
    directories: List[Path],
    replace: bool,
) -> Checked:
    check_file: Callable[[Path], Optional[LicenseCheckError]] = partial(
        license_check,
        license_header=license_header_contents,
        modify=modify,
        replace=replace,
    )
    check_dir: Callable[[Path], Checked] = partial(
        check_license_project_files,
        modify=modify,
        license_header=license_header_contents,
        replace=replace,
    )

    n_files_checked: int = 0
    noncompliant_files: Dict[Path, LicenseCheckError] = {}

    # license check all individual files
    for f in files:
        maybe_err = check_file(f)
        if maybe_err is not None:
            noncompliant_files[f] = maybe_err
    n_files_checked += len(files)

    # license check all directories and their contents, recursively
    for d in directories:
        checked = check_dir(d)
        noncompliant_files.update(checked.noncompliant_files)
        n_files_checked += checked.n_files

    return Checked(noncompliant_files=noncompliant_files, n_files=n_files_checked)


def _error(noncompliant_files: Mapping[Path, LicenseCheckError], n_files_checked: int, modify: bool) -> ValueError:
    maybe_modify_msg: str = (
        " You can re-run with '--modify' to automatically add the required license header." if not modify else ""
    )
    error_message: str = (
        f"ERROR: There are {len(noncompliant_files)} / {n_files_checked} "
        f"files that do not have the license header!{maybe_modify_msg}\n"
    )
    for pyfile, error in noncompliant_files.items():
        error_message += f"  {str(pyfile)}: {error}\n"
    return ValueError(error_message)


if __name__ == "__main__":
    entrypoint()
