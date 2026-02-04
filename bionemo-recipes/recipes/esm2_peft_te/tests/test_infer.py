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

import subprocess
import sys
from pathlib import Path

import pytest
import torch


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def run_infer_cmd(cmd, recipe_path):
    """Run an inference command and check for errors."""
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
        cwd=str(recipe_path),
    )

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command:\n{' '.join(cmd)}\nfailed with exit code {result.returncode}")


@requires_cuda
def test_infer_runs(recipe_path):
    """Test that the infer script runs with default config."""
    output_path = Path(recipe_path) / "preds.csv"

    run_infer_cmd(
        [
            sys.executable,
            "infer.py",
            "--config-name",
            "L0_sanity_infer",
        ],
        recipe_path,
    )

    if output_path.exists():
        output_path.unlink()
