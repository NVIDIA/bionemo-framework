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
    nvidia_smi_output = subprocess.run(["nvidia-smi"], capture_output=True)
    with open("/root/output.txt", "a") as f:
        f.write("before nvidia-smi output:\n")
        f.write(nvidia_smi_output.stdout.decode())
        f.write("--------------------------------\n\n")

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
        with open("/root/output.txt", "a") as f:
            f.write(f"Command failed with exit code {result.returncode}\n")
            f.write("stdout:\n")
            f.write(result.stdout.decode())
            f.write("\nstderr:\n")
            f.write(result.stderr.decode())

        with open("/root/output.txt", "a") as f:
            f.write("--------------------------------\n")
            f.write("after failed nvidia-smi output:\n")
            f.write(nvidia_smi_output.stdout.decode())
            f.write("--------------------------------\n\n")

        sys.exit(result.returncode)

    with open("/root/output.txt", "a") as f:
        f.write(f"Command succeded with exit code {result.returncode}\n")
        f.write("stdout:\n")
        f.write(result.stdout.decode())
        f.write("\nstderr:\n")
        f.write(result.stderr.decode())

    with open("/root/output.txt", "a") as f:
        f.write("--------------------------------\n")
        f.write("after successful nvidia-smi output:\n")
        f.write(nvidia_smi_output.stdout.decode())
        f.write("--------------------------------\n\n")

    return result
