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


import logging
import subprocess
import sys
from subprocess import CompletedProcess
from typing import List


logger = logging.getLogger(__name__)


def run_subprocess_safely(command: List[str], capture_output: bool = True) -> CompletedProcess:
    """Run a subprocess and raise an error if it fails.

    Args:
        command: The command to run.
        capture_output: Whether to capture the output of the command.

    Returns:
        The result of the subprocess.
    """
    result = subprocess.run(command, capture_output=capture_output)

    if result.returncode != 0:
        logger.error(
            "Command failed with exit code",
            result.returncode,
            "\nstdout:\n",
            result.stdout.decode(),
            "\nstderr:\n",
            result.stderr.decode(),
        )
        sys.exit(result.returncode)

    return result
