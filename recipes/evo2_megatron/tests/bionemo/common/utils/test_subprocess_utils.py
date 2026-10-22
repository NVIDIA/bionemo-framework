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


import shlex
import sys

import pytest

from bionemo.common.utils.subprocess_utils import run_subprocess_safely


def command(code):
    return shlex.join([sys.executable, "-c", code])


@pytest.mark.parametrize("output", ["", "print('partial', flush=True);", "print('partial', end='', flush=True);"])
def test_timeout(output):
    result = run_subprocess_safely(command(f"import time; {output} time.sleep(2)"), timeout=1)
    assert result.get("error") == "timeout"
    assert result["returncode"] is None
    if output:
        assert "partial" in result["stdout"]


def test_output(capsys):
    result = run_subprocess_safely(command("import sys; print('out'); print('err', file=sys.stderr)"))
    assert result == {"stdout": "out\n", "stderr": "err\n", "returncode": 0}
    captured = capsys.readouterr()
    assert captured.out == "out\n"
    assert captured.err == "err\n"


def test_failure():
    result = run_subprocess_safely(command("import sys; print('failure', file=sys.stderr); sys.exit(7)"))
    assert result["error"] == "non-zero exit"
    assert result["returncode"] == 7
    assert result["stderr"] == "failure\n"


def test_missing_command():
    assert run_subprocess_safely("/nonexistent/bionemo-test-command")["error"] == "not found"


def test_large_output():
    result = run_subprocess_safely(command("import sys; print('x' * 100000); print('y' * 100000, file=sys.stderr)"))
    assert result["stdout"] == "x" * 100000 + "\n"
    assert result["stderr"] == "y" * 100000 + "\n"
    assert result["returncode"] == 0
