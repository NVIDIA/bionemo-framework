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

import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import _mesh_resources


sys.path.append(Path(__file__).parent.parent.as_posix())
sys.path.append(Path(__file__).parent.as_posix())


@pytest.fixture
def recipe_path() -> Path:
    """Return the root directory of the recipe."""
    return Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def distributed_cleanup():
    yield

    # Try to destroy the process group, but don't fail if it's not available.
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except (AssertionError, RuntimeError):
        pass

    # Clear ALL mesh resources (for both MFSDP and FSDP2) to avoid issues re-running in the same process.
    _mesh_resources.mesh_stack.clear()
    _mesh_resources.child_to_root_mapping.clear()
    _mesh_resources.root_to_flatten_mapping.clear()
    _mesh_resources.flatten_name_to_root_dims.clear()
    _mesh_resources.mesh_dim_group_options.clear()

    # Clear CUDA cache to prevent memory issues between tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Reset CUDA device to ensure clean state
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        # Reset all accumulated values in CUDA memory stats
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_accumulated_memory_stats(device)
