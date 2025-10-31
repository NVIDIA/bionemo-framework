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

import glob
import os
import subprocess
import sys
import tempfile
import time

import pytest


@pytest.fixture
def scdl_speedtest_cmd():
    """Fixture to provide the command to run scdl_speedtest as a script."""
    script_path = os.path.join(os.path.dirname(__file__), "../../../../simple-benchmark/scdl_speedtest.py")
    script_path = os.path.abspath(script_path)
    return [sys.executable, script_path]


def test_scdl_speedtest_help_runs(scdl_speedtest_cmd):
    """Test that scdl_speedtest.py runs with --help and exits successfully."""
    help_cmd = scdl_speedtest_cmd + ["--help"]
    help_result = subprocess.run(help_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert help_result.returncode == 0, (
        f"scdl_speedtest --help did not run successfully.\nstdout: {help_result.stdout.decode()}\nstderr: {help_result.stderr.decode()}"
    )


def test_scdl_speedtest_csv_creates_files(scdl_speedtest_cmd):
    """Test that scdl_speedtest.py with --csv produces CSV files in a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_cmd = scdl_speedtest_cmd + ["--csv"]
        csv_result = subprocess.run(csv_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdir)
        assert csv_result.returncode == 0, (
            f"scdl_speedtest --csv did not run successfully.\nstdout: {csv_result.stdout.decode()}\nstderr: {csv_result.stderr.decode()}"
        )
        time.sleep(1)
        csv_files = glob.glob(os.path.join(tmpdir, "*.csv"))
        assert csv_files, (
            f"scdl_speedtest --csv did not produce any CSV files in {tmpdir}.\nstdout: {csv_result.stdout.decode()}\nstderr: {csv_result.stderr.decode()}"
        )


def test_scdl_speedtest_generate_baseline_runs(scdl_speedtest_cmd):
    """Test that scdl_speedtest.py runs with --generate-baseline and exits successfully."""
    baseline_cmd = scdl_speedtest_cmd + ["--generate-baseline"]
    baseline_result = subprocess.run(baseline_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert baseline_result.returncode == 0, (
        f"scdl_speedtest --generate-baseline did not run successfully.\nstdout: {baseline_result.stdout.decode()}\nstderr: {baseline_result.stderr.decode()}"
    )


def test_scdl_speedtest_runs_no_args(scdl_speedtest_cmd):
    """Test that scdl_speedtest.py runs with no arguments and exits successfully."""
    result = subprocess.run(scdl_speedtest_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, (
        f"scdl_speedtest did not run successfully.\nstdout: {result.stdout.decode()}\nstderr: {result.stderr.decode()}"
    )


def test_scdl_speedtest_num_runs(scdl_speedtest_cmd):
    """Test that scdl_speedtest.py runs with --num-runs option and exits successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use very short runtime and small number of runs for test efficiency
        num_runs_cmd = scdl_speedtest_cmd + ["--num-runs", "2", "--max-time", "5", "--warmup-time", "0"]
        result = subprocess.run(num_runs_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdir)
        assert result.returncode == 0, (
            f"scdl_speedtest --num-runs 2 did not run successfully.\nstdout: {result.stdout.decode()}\nstderr: {result.stderr.decode()}"
        )

        # Check that output contains evidence of multiple runs
        stdout = result.stdout.decode()
        assert "run 1/2" in stdout or "run 2/2" in stdout or "averaged 2 runs" in stdout, (
            f"Output should indicate multiple runs were performed.\nstdout: {stdout}"
        )


def test_scdl_speedtest_num_runs_invalid(scdl_speedtest_cmd):
    """Test that scdl_speedtest.py rejects invalid --num-runs values."""
    invalid_cmd = scdl_speedtest_cmd + ["--num-runs", "0"]
    result = subprocess.run(invalid_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode != 0, (
        f"scdl_speedtest should reject --num-runs 0.\nstdout: {result.stdout.decode()}\nstderr: {result.stderr.decode()}"
    )

    # Check that error message is about num-runs
    stderr = result.stderr.decode()
    stdout = result.stdout.decode()
    assert "--num-runs must be a positive integer" in stdout or "--num-runs must be a positive integer" in stderr, (
        f"Should show validation error for invalid --num-runs.\nstdout: {stdout}\nstderr: {stderr}"
    )
