# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The base import stays framework-free."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import nemotron_stitch

FRAMEWORKS = {"nemo_automodel", "nemo_rl", "peft", "transformers", "vllm"}


def test_base_import_has_no_framework_imports() -> None:
    # Run in a fresh interpreter: importing the package must neither require
    # nor pull in any framework, installed or not.
    code = (
        "import sys, nemotron_stitch; "
        f"leaked = sorted(m for m in sys.modules if m.split('.')[0] in {FRAMEWORKS!r}); "
        "assert not leaked, leaked"
    )
    # Propagate the import path so the check also works from an uninstalled
    # source tree, not only against an installed wheel.
    package_parent = Path(nemotron_stitch.__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": f"{package_parent}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
