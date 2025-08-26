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

import logging
from copy import deepcopy
from pathlib import Path

from hydra import compose, initialize_config_dir

from train import main


def test_train(monkeypatch, tmp_path):
    """
    Test training vit_base_patch16_224.
    """

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
    with initialize_config_dir(config_dir=str(recipe_dir / "config"), version_base="1.2"):
        vit_config = compose(
            config_name="vit_base_patch16_224",
            overrides=[
                f"++training.steps=25",
                f"++training.val_interval=10",
                f"++training.log_interval=1",
                f"++training.checkpoint.path={Path(tmp_path) / 'ckpt'}",
                f"++profiling.torch_memory_profile=false",
                f"++profiling.wandb=false",
            ],
        )
        vit_resume_config = deepcopy(vit_config)
        vit_resume_config.training.steps = 50

    main(vit_config)

    # Verify checkpoints were created.
    assert sum(1 for item in (Path(tmp_path) / "ckpt").iterdir() if item.is_dir()) == 2, (
        "Expected 2 checkpoints with 25 training steps and validation interval of 10."
    )

    # Auto-resume training from checkpoint. For this test, we auto-resume from the best checkpoint,
    # so depending on what the best checkpoint is, we may have more than 5 checkpoints.
    main(vit_resume_config)


def test_train_te(monkeypatch, tmp_path):
    """
    Test training vit_te_base_patch16_224.
    """

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
    with initialize_config_dir(config_dir=str(recipe_dir / "config"), version_base="1.2"):
        vit_config = compose(
            config_name="vit_te_base_patch16_224",
            overrides=[
                f"++training.steps=25",
                f"++training.val_interval=10",
                f"++training.log_interval=1",
                f"++training.checkpoint.path={Path(tmp_path) / 'ckpt'}",
                f"++profiling.torch_memory_profile=false",
                f"++profiling.wandb=false",
            ],
        )
        vit_resume_config = deepcopy(vit_config)
        vit_resume_config.training.steps = 50

    main(vit_config)

    # Verify checkpoints were created.
    assert sum(1 for item in (Path(tmp_path) / "ckpt").iterdir() if item.is_dir()) == 2, (
        "Expected 2 checkpoints with 25 training steps and validation interval of 10."
    )

    # Auto-resume training from checkpoint. For this test, we auto-resume from the best checkpoint,
    # so depending on what the best checkpoint is, we may have more than 5 checkpoints.
    main(vit_resume_config)
