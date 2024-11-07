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


from pathlib import Path
from typing import Tuple

from pytest import fixture, raises

from infra_bionemo.license_check import (
    Checked,
    HeaderNotFound,
    append_license_header,
    check_license_project_files,
    ensure_license_starts_with_pound,
    has_header,
    is_valid_python,
    license_check,
    remove_existing_license_header,
)
from infra_bionemo.license_check import (
    main as license_check_cli_main,
)


@fixture(scope="module")
def valid() -> str:
    return """
def double(x: int) -> int:
  return x * 2
print(f"{double(10)=}")
    """.strip()


@fixture(scope="module")
def invalid() -> str:
    return """
print "nope, this is no longer ok!"
    """.strip()


@fixture(scope="module")
def license_text() -> str:
    return """
# The license would be here.
# And continue on.
# For maybe a few more
# lines.
    """.strip()


@fixture(scope="module")
def license_text_not_escaped() -> str:
    return """
The license would be here.
And continue on.
For maybe a few more
lines.
    """.strip()


@fixture(scope="module")
def full(license_text: str, valid: str) -> str:
    return f"{license_text}\n\n{valid}"


def test_is_valid_python(valid: str, invalid: str):
    assert is_valid_python(valid) is None
    assert isinstance(is_valid_python(invalid), SyntaxError)


def test_has_header(full: str, license_text: str):
    assert has_header(full, license_header=license_text) is True


def test_append_license_header(valid: str, license_text: str, full: str):
    actual_full = append_license_header(valid, license_header=license_text, n_sep_lines=2)
    assert actual_full == full


def test_license_check(valid: str, invalid: str, full: str, license_text: str, tmp_path: Path):
    pyfile = tmp_path / "_testing_pyfile_89712652015.py"

    # does not exist
    assert isinstance(license_check(pyfile, modify=False), IOError)

    # invalid python
    with open(str(pyfile), "wt") as wt:
        wt.write(invalid)
    assert isinstance(license_check(pyfile, modify=False), SyntaxError)

    # valid, but w/o license header
    with open(str(pyfile), "wt") as wt:
        wt.write(valid)
    assert isinstance(license_check(pyfile, modify=False), HeaderNotFound)

    # valid w/o license header, but automatic fix works
    assert license_check(pyfile, modify=True) is None
    assert license_check(pyfile, modify=False) is None

    # works as expected on valid python w/ header
    with open(str(pyfile), "wt") as wt:
        wt.write(full)
    assert license_check(pyfile, modify=False, license_header=license_text) is None


def test_check_license_project_files(valid: str, invalid: str, full: str, license_text: str, tmp_path: Path):
    project_dir = tmp_path / "python_package_for_testing_1245"

    # not a directory
    with raises(AssertionError):
        check_license_project_files(project_dir, modify=False, replace=False, license_header=license_text)

    # create directory & populate .py files
    _, invalid_fi, _ = _create_py_project_and_files(project_dir, valid, invalid, full)

    # checking w/o modifying --> one invalid file + 1 file w/o license
    checked: Checked = check_license_project_files(
        project_dir, modify=False, replace=False, license_header=license_text
    )
    assert checked.n_files == 3
    assert len(checked.noncompliant_files) == 2

    # remove invalid file
    invalid_fi.unlink()
    # now, checking w/o modification will result in no non-compliant files
    checked = check_license_project_files(project_dir, modify=True, replace=False, license_header=license_text)
    assert checked.n_files == 2
    assert len(checked.noncompliant_files) == 0


def test_ensure_license_starts_with_pound(license_text: str, license_text_not_escaped: str):
    assert ensure_license_starts_with_pound(license_text_not_escaped) == license_text


def test_remove_existing_license_header(valid: str, license_text: str):
    assert (
        len(remove_existing_license_header(license_text)) == 0
    ), "Removing from a header-only file should result in an empty string."

    pyfile_with_header = append_license_header(valid, license_header=license_text, n_sep_lines=1)
    removed_pyfile = remove_existing_license_header(pyfile_with_header)
    assert removed_pyfile == valid


def test_main(valid: str, invalid: str, full: str, license_text: str, tmp_path: Path):
    project_dir = tmp_path / "different_python_package_for_testing_1245"

    valid_fi, invalid_fi, full_fi = _create_py_project_and_files(
        project_dir,
        valid,
        invalid,
        full,
    )

    full_fi_2 = tmp_path / "another_full.py"
    with open(str(full_fi_2), "wt") as wt:
        wt.write(full)

    invalid_fi_2 = tmp_path / "another_invalid.py"
    with open(str(invalid_fi_2), "wt") as wt:
        wt.write(invalid)

    # check: len(files) + len(directories) > 0
    with raises(ValueError):
        license_check_cli_main(
            modify=False,
            replace=False,
            license_header_contents=license_text,
            files=[],
            directories=[],
        )

    # check: non-empty license contents
    with raises(ValueError):
        license_check_cli_main(
            modify=False,
            replace=False,
            license_header_contents="",
            files=[full_fi_2],
            directories=[project_dir],
        )

    # invalid file
    # valid file w/o license header
    with raises(ValueError):
        license_check_cli_main(
            modify=False,
            replace=False,
            license_header_contents=license_text,
            files=[full_fi_2, invalid_fi_2],
            directories=[project_dir],
        )

    # can fix if there are no invalid files
    invalid_fi.unlink()
    # and modify=True
    checked_n_files: int = license_check_cli_main(
        modify=True,
        replace=False,
        license_header_contents=license_text,
        files=[full_fi_2],
        directories=[project_dir],
    )
    assert checked_n_files == 3


def _create_py_project_and_files(project_dir: Path, valid: str, invalid: str, full: str) -> Tuple[Path, Path, Path]:
    """Creates Python project dir w/ valid, invalid, and full .py file contents.
    Returns (valid, invalid, full) filepaths.
    """
    project_dir.mkdir()
    # add the valid, invalid, and full python code to the directory
    # make use of nested directories to ensure that recursive logic works
    valid_fi = project_dir / "valid.py"
    invalid_fi = project_dir / "another_package" / "invalid.py"
    invalid_fi.parent.mkdir()
    full_fi = project_dir / "different" / "nested" / "packages" / "full.py"
    full_fi.parent.mkdir(parents=True)
    for fi, contents in [(valid_fi, valid), (invalid_fi, invalid), (full_fi, full)]:
        with open(str(fi), "wt") as wt:
            wt.write(contents)
    return valid_fi, invalid_fi, full_fi
