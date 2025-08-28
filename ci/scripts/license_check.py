#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import logging
import re
import textwrap
from datetime import datetime
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

year = datetime.now().year

license_text = textwrap.dedent("""\
    SPDX-License-Identifier: LicenseRef-Apache2

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """)

default_copyright_text = (
    f"SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES. All rights reserved."
)
default_combined_license = "\n".join([default_copyright_text, license_text])

copyright_regex_pattern = (
    r"SPDX-FileCopyrightText: Copyright \(c\) \d{4} NVIDIA CORPORATION & AFFILIATES\. All rights reserved\.\n?\r?"
    r"(?:SPDX-FileCopyrightText: Copyright \(c\) \d{4}.*\n?\r?)*"
)
license_regex_pattern = copyright_regex_pattern + re.escape(license_text)
license_regex = re.compile(license_regex_pattern)

# Basic regex sanity checks.
assert re.compile(copyright_regex_pattern).match(default_copyright_text), "Default copyright text not valid"
assert license_regex.match(default_combined_license), "Default license text or regex is not valid"


def process_file(filepath: Path, dry_run: bool):
    """Process a file to ensure it has a valid license block, or add a new one if it doesn't."""
    comment_start = get_comment_delimiter(filepath)
    lines = filepath.read_text().splitlines()

    start_line = 0
    if lines and lines[0].startswith("#!"):
        # Make sure there's a blank line after the shebang
        if lines[1] != "":
            lines.insert(1, "")
        start_line = 2

    license_block = []
    for line in lines[start_line:]:
        if line.startswith(comment_start):
            license_block.append(line)
        else:
            break

    def uncomment(text: str) -> str:
        return re.sub(rf"^{comment_start}\ ?", "", text)

    if "# noqa: license-check" in license_block:
        logger.info(f"Skipping {filepath} because it contains `# noqa: license-check`.")
        return

    license_block_text = "\n".join(uncomment(line) for line in license_block) + "\n"
    if len(license_block) != 0 and license_regex.match(license_block_text):
        logger.info(f"Skipping {filepath} because it contains a valid license block.")
        return

    logger.info(f"Adding license block to {filepath}.")
    license_lines = "\n".join([default_copyright_text, license_text])
    license_lines = textwrap.indent(license_lines, comment_start + " ", predicate=lambda _: True)
    license_lines = "\n".join(line.rstrip() for line in license_lines.splitlines()) + "\n"
    lines.insert(start_line, license_lines)

    if not dry_run:
        filepath.write_text("\n".join(lines) + "\n")


def get_comment_delimiter(filepath: Path) -> str:
    """Get the comment delimiter for a file based on its extension."""
    match filepath.suffix:
        case ".py":
            return "#"
        case ".rs":
            return "//"
        case _:
            raise ValueError(f"Unsupported file type: {filepath}")


def main():
    """Main entry point for the license check script."""
    parser = argparse.ArgumentParser(description="Ensure files have proper license headers")
    parser.add_argument("files", nargs="+", help="Files to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")

    args = parser.parse_args()

    for filename in args.files:
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filename} does not exist")

        if filepath.suffix not in [".py", ".rs"]:
            raise ValueError(f"Unsupported file type: {filepath}")

        process_file(filepath, args.dry_run)


if __name__ == "__main__":
    main()
