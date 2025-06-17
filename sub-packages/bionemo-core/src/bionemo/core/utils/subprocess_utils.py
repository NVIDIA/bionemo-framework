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
import os
import signal
import subprocess
import sys
from subprocess import CompletedProcess
from typing import List


logger = logging.getLogger(__name__)


def run_subprocess_safely(command: List[str], std_name: str, capture_output: bool = True) -> CompletedProcess:
    """Run a subprocess and raise an error if it fails.

    Args:
        command: The command to run.
        std_name: The name of the standard output and error files.
        capture_output: Whether to capture the output of the command.

    Returns:
        The result of the subprocess.
    """
    nvidia_smi_output = subprocess.run(["nvidia-smi"], capture_output=True)
    with open("/root/output.txt", "a") as f:
        f.write("before nvidia-smi output:\n")
        f.write(nvidia_smi_output.stdout.decode())
        f.write("--------------------------------\n\n")

    def preexec():
        # Completely detach from Jupyter's process tree
        try:
            os.setsid()  # New session
            os.setpgrp()  # New process group
            # Reset signal handlers to default
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        except Exception as e:
            with open("/root/output.txt", "a") as f:
                f.write(f"Error in preexec: {e}\n")
                f.write("--------------------------------\n\n")
                # print environment variables
                f.write("--------------------------------\n")
                f.write("Environment variables of child process:\n")
                for key, value in os.environ.items():
                    f.write(f"{key}={value}\n")
                f.write("--------------------------------\n\n")

        with open("/root/output.txt", "a") as f:
            # print environment variables
            f.write("--------------------------------\n")
            f.write("Environment variables of child process:\n")
            for key, value in os.environ.items():
                f.write(f"{key}={value}\n")
            f.write("--------------------------------\n\n")

    command = ["setsid"] + command
    result = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
        preexec_fn=preexec,
    )

    try:
        stdout, stderr = result.communicate(timeout=1500)
    except subprocess.TimeoutExpired as e:
        with open("/root/output.txt", "a") as f:
            f.write("Command timeout\n")
            f.write("stdout:\n")
            f.write(e.stdout.decode())
            f.write("\nstderr:\n")
            f.write(e.stderr.decode())
            sys.exit(1)

    if result.returncode != 0:
        logger.error(
            "Command failed with exit code",
            result.returncode,
            "\nstdout:\n",
            stdout.decode(),
            "\nstderr:\n",
            stderr.decode(),
        )
        with open("/root/output.txt", "a") as f:
            f.write(f"Command failed with exit code {result.returncode}\n")
            f.write("stdout:\n")
            f.write(stdout.decode())
            f.write("\nstderr:\n")
            f.write(stderr.decode())

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
