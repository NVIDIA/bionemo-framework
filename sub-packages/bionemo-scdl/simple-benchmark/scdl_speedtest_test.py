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


import os
import subprocess
import sys


def test_scdl_speedtest_runs():
    """Test that the scdl_speedtest.py script runs successfully in various invocation modes.

    This test attempts to locate the script in the current directory and run it directly,
    or as a module if not found. It checks that the script runs with no arguments,
    with --help, with --csv, and with --generate-baseline, and that all invocations
    complete successfully (exit code 0).
    """
    script_path = os.path.join(os.path.dirname(__file__), "scdl_speedtest.py")
    if os.path.exists(script_path):
        cmd = [sys.executable, script_path]
    else:
        # Try running as a module if not found as a script
        cmd = [sys.executable, "-m", "scdl_speedtest"]
    # Also test running with some common arguments
    # Test with --help
    help_cmd = cmd + ["--help"]
    help_result = subprocess.run(help_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert help_result.returncode == 0, (
        f"scdl_speedtest --help did not run successfully.\nstdout: {help_result.stdout.decode()}\nstderr: {help_result.stderr.decode()}"
    )

    # Test with --csv (should not error, even if no output file is checked)
    csv_cmd = cmd + ["--csv"]
    csv_result = subprocess.run(csv_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert csv_result.returncode == 0, (
        f"scdl_speedtest --csv did not run successfully.\nstdout: {csv_result.stdout.decode()}\nstderr: {csv_result.stderr.decode()}"
    )

    # Test with --generate-baseline
    baseline_cmd = cmd + ["--generate-baseline"]
    baseline_result = subprocess.run(baseline_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert baseline_result.returncode == 0, (
        f"scdl_speedtest --generate-baseline did not run successfully.\nstdout: {baseline_result.stdout.decode()}\nstderr: {baseline_result.stderr.decode()}"
    )

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, (
        f"scdl_speedtest did not run successfully.\nstdout: {result.stdout.decode()}\nstderr: {result.stderr.decode()}"
    )
