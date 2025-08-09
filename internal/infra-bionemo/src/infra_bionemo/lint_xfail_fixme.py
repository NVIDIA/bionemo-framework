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

"""Linter enforcing issue tracking for XFAILs and technical debt markers.

This script scans source code files to ensure proper issue tracking by:
1. Verifying pytest.mark.xfail decorators have associated GitHub issue links
2. Ensuring FIXME/XXX/HACK comments have corresponding issue links

Usage:
    python lint_xfail_fixme.py

Exit codes:
    0: All checks passed
    1: Issues found

Examples:
    # Run from project root
    $ python lint_xfail_fixme.py
    Scanning files from: /path/to/project

    Linter found 2 issues:
      Line 45: XFAIL missing required issue link
        > @pytest.mark.xfail(reason="Known issue with batch processing")

      Line 123: FIXME missing required issue link
        > # FIXME: Handle edge case with empty input

Good practices:
    @pytest.mark.xfail(reason="Known issue https://github.com/NVIDIA/bionemo-framework/issues/123")
    # FIXME: Need better error handling https://github.com/NVIDIA/bionemo-framework/issues/456
"""

import os
import re
import sys
from typing import List, Optional, Set, Tuple


THIS_FILENAME = os.path.abspath(__file__)
ISSUE_PATTERN = r"https://github\.com/NVIDIA/bionemo-framework/issues/\d+"
FIXME_PATTERN = r"\b(FIXME)\b"
SKIP_DIRS = [".git", "__pycache__", ".pytest_cache", "build", "dist", "3rdparty"]
SOURCE_EXTENSIONS = (".py", ".sh", "Dockerfile", ".yaml", ".yml")


def find_all_files(root_dir: str) -> Set[str]:
    """Find all source code files recursively from the root directory."""
    all_files = []
    for root, _, files in os.walk(root_dir):
        if any(skip in root.split(os.path.sep) for skip in SKIP_DIRS):
            continue
        for file in files:
            if file.endswith(SOURCE_EXTENSIONS):
                all_files.append(os.path.join(root, file))
    return set(all_files)


def check_xfail_issues(file_path: str) -> List[Tuple[int, str, str]]:
    """Check if pytest.mark.xfail has associated issue links."""
    errors = []
    if not os.path.basename(file_path).startswith("test_"):
        return errors

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if "pytest.mark.xfail" in line:
                if not re.search(ISSUE_PATTERN, line):
                    errors.append(
                        (line_num, f"XFAIL in {file_path}:{line_num} missing required issue link", line.strip())
                    )
    return errors


def check_fixme_issues(file_path: str) -> List[Tuple[int, str, str]]:
    """Check if FIXME comments have associated issue links."""
    errors = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if re.search(FIXME_PATTERN, line, re.IGNORECASE):
                if not re.search(ISSUE_PATTERN, line):
                    errors.append(
                        (line_num, f"FIXME in {file_path}:{line_num} missing required issue link", line.strip())
                    )
    return errors


def main(files: Optional[Set[str]] = None) -> None:
    """Main entry point for the linter script."""
    root_dir = os.getcwd()
    if files is None:
        print(f"Scanning files from: {root_dir}")
        files = find_all_files(root_dir)
    errors = []

    for file_path in files:
        if file_path in THIS_FILENAME:
            continue
        rel_path = os.path.relpath(file_path, root_dir)
        try:
            file_errors = check_xfail_issues(file_path)
            errors.extend(
                [
                    (line_num, msg.replace(file_path, rel_path), line_content)
                    for line_num, msg, line_content in file_errors
                ]
            )

            file_errors = check_fixme_issues(file_path)
            errors.extend(
                [
                    (line_num, msg.replace(file_path, rel_path), line_content)
                    for line_num, msg, line_content in file_errors
                ]
            )
        except UnicodeDecodeError:
            print(f"Warning: Skipping binary file {rel_path}")
            continue

    if errors:
        for line_num, error, line_content in sorted(errors, key=lambda x: x[1]):
            print(f"  Line {line_num}: {error}")
            print(f"    > {line_content}")
            print()
        sys.exit(1)
    sys.exit(0)


def get_files_from_precommit() -> Optional[Set[str]]:
    """Captures filenames passed from command line or by pre-commit hook."""
    files = None
    if len(sys.argv) > 1:
        files = set(sys.argv[1:])
    if "PRE_COMMIT_FILES" in os.environ:
        files = set(os.environ["PRE_COMMIT_FILES"].split())
    return files


def entrypoint():
    """Entrypoint of the linter script."""
    files = get_files_from_precommit()
    main(files)


if __name__ == "__main__":
    entrypoint()
