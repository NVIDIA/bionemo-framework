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
from unittest import mock

import pytest
import torch
from torch.distributed.device_mesh import _mesh_resources, init_device_mesh


sys.path.append(Path(__file__).parent.parent.as_posix())
sys.path.append(Path(__file__).parent.as_posix())

from distributed_config import DistributedConfig


@pytest.fixture
def recipe_path() -> Path:
    """Return the root directory of the recipe."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session", autouse=True)
def device_mesh():
    """Create a re-usable device mesh for testing.

    Megatron-FSDP throws issues when re-creating the torch device mesh in the same process, starting in the 25.09 NGC
    pytorch container release. To work around this, we create a re-usable device mesh that use in all single-process
    tests.
    """
    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)
    device_mesh = init_device_mesh("cuda", mesh_shape=(1, 1), mesh_dim_names=("dp", "tp"))

    with (
        mock.patch("torch.distributed.device_mesh.init_device_mesh", return_value=device_mesh),
        mock.patch("torch.distributed.init_process_group", return_value=None),
        mock.patch("torch.distributed.destroy_process_group", return_value=None),
    ):
        yield

    torch.distributed.destroy_process_group()
    _mesh_resources.mesh_stack.clear()
    _mesh_resources.child_to_root_mapping.clear()
    _mesh_resources.root_to_flatten_mapping.clear()
    _mesh_resources.flatten_name_to_root_dims.clear()
    _mesh_resources.mesh_dim_group_options.clear()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
