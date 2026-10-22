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
import shlex
import subprocess
import sys
from threading import Thread
from typing import Any, Dict


logger = logging.getLogger(__name__)


def run_subprocess_safely(command: str, timeout: int = 2000) -> Dict[str, Any]:
    """Run a subprocess with live output and return its result or error details.

    Args:
        command: The command to run.
        timeout: Maximum runtime in seconds; timed-out processes are killed and reaped.

    Returns:
        The result of the subprocess.
    """
    try:
        # Use Popen to enable real-time output while still capturing it
        process = subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        stdout_lines = []
        stderr_lines = []

        # Drain both pipes concurrently so a partial line or a full pipe cannot
        # prevent the main thread from enforcing the process timeout.
        def stream_output(pipe, lines, destination):
            try:
                for line in pipe:
                    lines.append(line)
                    print(line.rstrip(), file=destination, flush=True)
            finally:
                pipe.close()

        readers = [
            Thread(target=stream_output, args=(process.stdout, stdout_lines, sys.stdout)),
            Thread(target=stream_output, args=(process.stderr, stderr_lines, sys.stderr)),
        ]
        for reader in readers:
            reader.start()

        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            for reader in readers:
                reader.join()
            raise subprocess.TimeoutExpired(
                command, timeout, output="".join(stdout_lines), stderr="".join(stderr_lines)
            ) from None
        finally:
            for reader in readers:
                reader.join()

        # Check return code
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, command, output="".join(stdout_lines), stderr="".join(stderr_lines)
            )

        # Create result object similar to subprocess.run
        class Result:
            def __init__(self, stdout, stderr, returncode):
                self.stdout = stdout
                self.stderr = stderr
                self.returncode = returncode

        result = Result("".join(stdout_lines), "".join(stderr_lines), process.returncode)
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    except subprocess.TimeoutExpired as e:
        logger.error(f"Command timed out. Command: {command}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}")
        return {"error": "timeout", "stdout": e.stdout, "stderr": e.stderr, "returncode": None}

    except subprocess.CalledProcessError as e:
        logger.error(
            f"Command failed. Command: {command}\nreturncode: {e.returncode}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
        )
        return {"error": "non-zero exit", "stdout": e.stdout, "stderr": e.stderr, "returncode": e.returncode}

    except FileNotFoundError as e:
        logger.error(f"Command not found. Command: {command}\nstderr:\n{e!s}")
        return {"error": "not found", "stdout": "", "stderr": str(e), "returncode": None}

    except Exception as e:
        # catch-all for other unexpected errors
        return {"error": "other", "message": str(e), "stdout": "", "stderr": "", "returncode": None}
