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

from pathlib import Path

import pytest
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from torch.distributed.device_mesh import _mesh_resources

from train_ddp import main as main_ddp
from train_fsdp2 import main as main_fsdp2
from train_nvfsdp import main as main_nvfsdp


# Get the recipe directory
recipe_dir = Path(__file__).parent


@pytest.fixture
def mock_distributed_config(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    yield

    # Try to destroy the process group, but don't fail if it's not available.
    try:
        dist.destroy_process_group()
    except AssertionError:
        pass

    # For nvFSDP, clear mesh resources to avoid issues re-running in the same process.
    _mesh_resources.mesh_stack.clear()
    _mesh_resources.child_to_root_mapping.clear()
    _mesh_resources.root_to_flatten_mapping.clear()
    _mesh_resources.flatten_name_to_root_dims.clear()
    _mesh_resources.mesh_dim_group_options.clear()


def test_main_invocation_nvfsdp(mock_distributed_config, tmp_path):
    """Test that the main function can be invoked with the correct arguments."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(config_name="L0_sanity", overrides=[f"+wandb_init_args.dir={tmp_path}"])

    main_nvfsdp(sanity_config)


def test_main_invocation_ddp(mock_distributed_config, tmp_path):
    """Test that the main function can be invoked wrapping the model in DDP."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(config_name="L0_sanity", overrides=[f"+wandb_init_args.dir={tmp_path}"])

    main_ddp(sanity_config)


def test_main_invocation_fsdp2(mock_distributed_config, tmp_path):
    """Test that the main function can be invoked wrapping the model in FSDP2."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(config_name="L0_sanity", overrides=[f"+wandb_init_args.dir={tmp_path}"])

    main_fsdp2(sanity_config)


def test_main_invocation_nvfsdp_eager(mock_distributed_config, tmp_path):
    """Test that the main function can be invoked with the correct arguments."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[f"+wandb_init_args.dir={tmp_path}", "model_name=facebook/esm2_t6_8M_UR50D"],
        )

    main_nvfsdp(sanity_config)


def test_main_invocation_ddp_eager(mock_distributed_config, tmp_path):
    """Test that the main function can be invoked wrapping the model in DDP."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[f"+wandb_init_args.dir={tmp_path}", "model_name=facebook/esm2_t6_8M_UR50D"],
        )

    main_ddp(sanity_config)


def test_main_invocation_fsdp2_eager(mock_distributed_config, tmp_path):
    """Test that the main function can be invoked wrapping the model in FSDP2."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[f"+wandb_init_args.dir={tmp_path}", "model_name=facebook/esm2_t6_8M_UR50D"],
        )

    main_fsdp2(sanity_config)
