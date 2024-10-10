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


from pathlib import Path

import pytest
from testbook import testbook


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
@pytest.mark.parametrize(
    "notebook",
    [
        "examples/protein/prott5nv/nbs/Inference_interactive.ipynb",
        "examples/protein/esm1nv/nbs/Inference_interactive.ipynb",
        "examples/protein/esm2nv/nbs/Inference_interactive.ipynb",
        "examples/molecule/megamolbart/nbs/Inference_interactive.ipynb",
    ],
)
def test_example_notebook(bionemo_home: Path, notebook: str):
    notebook_path = bionemo_home / notebook
    assert notebook_path.is_file(), f"Expected interactive notebook to exist: {notebook_path=}"
    with testbook(str(notebook_path), execute=False, timeout=60 * 5) as tb:
        tb.execute()  # asserts are in notebook
