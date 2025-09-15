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
import torch
from hydra import compose, initialize_config_dir

from train import main


@pytest.mark.xfail(
    torch.cuda.get_device_capability() == (12, 0),
    reason="CUDNN padded packed sequences not supported on all hardware currently (nvbugs/5458694).",
)
def test_main_invocation(monkeypatch, tmp_path):
    """Test that the main function can be invoked with the correct arguments."""

    # Get the recipe directory
    recipe_dir = Path(__file__).parent

    # Set required environment variables for distributed training
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(config_name="L0_sanity", overrides=[f"+wandb_init_args.dir={tmp_path}"])

    main(sanity_config)


@pytest.mark.xfail(
    torch.cuda.get_device_capability() == (12, 0),
    reason="CUDNN padded packed sequences not supported on all hardware currently (nvbugs/5458694).",
)
def test_main_invocation_ddp(monkeypatch, tmp_path):
    """Test that the main function can be invoked wrapping the model in DDP."""

    # Get the recipe directory
    recipe_dir = Path(__file__).parent

    # Set required environment variables for distributed training
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "use_fsdp=false",
            ],
        )

    main(sanity_config)
