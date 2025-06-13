import logging
import subprocess
import sys
from subprocess import CompletedProcess
from typing import List


logger = logging.getLogger(__name__)


def run_subprocess_safely(command: List[str], capture_output: bool = True) -> CompletedProcess:
    result = subprocess.run(command, capture_output=capture_output)

    if result.returncode != 0:
        logger.error("Command failed with exit code", result.returncode,
                     "\nstdout:\n", result.stdout.decode(),
                     "\nstderr:\n", result.stderr.decode()
        )
        sys.exit(result.returncode)

    return result