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

# Script to re-copy files from a given source filepath to destination filepaths, used as a pre-commit hook to ensure
# that copied files between recipe folders stay up-to-date.

import argparse
import functools
import logging
import operator
import shutil
from pathlib import Path


logger = logging.getLogger(__name__)


SOURCE_TO_DESTINATION_MAP: dict[str, list[str]] = {
    "bionemo-recipes/models/esm2/src/esm/modeling_esm_te.py": [
        "bionemo-recipes/recipes/esm2_native_te/example_8m_checkpoint/esm_nv.py",
        "bionemo-recipes/recipes/esm2_peft_te/example_8m_checkpoint/esm_nv.py",
        "bionemo-recipes/recipes/esm2_accelerate_te/example_8m_checkpoint/esm_nv.py",
    ],
    "bionemo-recipes/models/esm2/src/esm/collator.py": [
        "bionemo-recipes/recipes/esm2_native_te/collator.py",
        "bionemo-recipes/recipes/llama3_native_te/collator.py",
    ],
    "bionemo-recipes/models/esm2/src/esm/state.py": [
        "bionemo-recipes/models/amplify/src/amplify/state.py",
        "bionemo-recipes/models/llama3/state.py",
    ],
    "bionemo-recipes/models/llama3/modeling_llama_te.py": [
        "bionemo-recipes/recipes/llama3_native_te/example_checkpoint/llama3_nv.py",
    ],
}


def main():
    """Copy files from the source to the destinations."""
    parser = argparse.ArgumentParser(description="Ensure copied files are synchronized across recipe folders")
    parser.add_argument("files", nargs="*", help="Files to process", default=[])
    parser.add_argument("--fix", action="store_true", help="Copy the files from source to destinations")

    args = parser.parse_args()

    # Check if the script needs to run.
    all_files = set(SOURCE_TO_DESTINATION_MAP.keys()) | set(
        functools.reduce(operator.iadd, SOURCE_TO_DESTINATION_MAP.values(), [])
    )
    relevant_files = [f for f in args.files if f in all_files]
    # If pre-commit passed a list of files and none are relevant, skip.
    if args.files and not relevant_files:
        return

    for source, destinations in SOURCE_TO_DESTINATION_MAP.items():
        if not Path(source).exists():
            raise ValueError(
                f"Source file {source} does not exist -- if this file was removed, please update the "
                f"source-to-destination map in {Path(__file__).relative_to(Path.cwd())}"
            )

        for destination in destinations:
            if not Path(destination).exists():
                raise ValueError(
                    f"Destination file {destination} does not exist -- if this file was removed, please update the "
                    f"source-to-destination map in {Path(__file__).relative_to(Path.cwd())}"
                )

            if args.fix:
                shutil.copy(source, destination)
                logger.info(f"Copied {source} to {destination}")

            else:
                with open(source, "rb") as f1, open(destination, "rb") as f2:
                    if f1.read() != f2.read():
                        raise ValueError(
                            f"Files {source} and {destination} do not match. Run "
                            f"{Path(__file__).relative_to(Path.cwd())} --fix to fix."
                        )


if __name__ == "__main__":
    main()
