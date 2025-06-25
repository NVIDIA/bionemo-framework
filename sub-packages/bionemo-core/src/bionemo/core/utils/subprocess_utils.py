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
from typing import Any, Dict


logger = logging.getLogger(__name__)


def run_subprocess_safely(command: str, timeout: int = 2000) -> Dict[str, Any]:
    """Run a subprocess and raise an error if it fails.

    Args:
        command: The command to run.
        timeout: The timeout for the command.

    Returns:
        The result of the subprocess.
    """
    try:
        result = subprocess.run(shlex.split(command), capture_output=True, timeout=timeout, check=True, text=True)
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
        logger.error(f"Command not found. Command: {command}\nstderr:\n{str(e)}")
        return {"error": "not found", "stdout": "", "stderr": str(e), "returncode": None}

    except Exception as e:
        # catch-all for other unexpected errors
        return {"error": "other", "message": str(e), "stdout": "", "stderr": "", "returncode": None}
