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
from hydra import compose, initialize_config_dir

from train import main


@pytest.mark.parametrize("config_name", ["vit_base_patch16_224", "vit_te_base_patch16_224"])
@pytest.mark.parametrize("init_model_with_meta_device", [True, False])
def test_train(monkeypatch, tmp_path, config_name, init_model_with_meta_device):
    """
    Test training.
    """
    # Set required environment variables for distributed training
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29500")

    # Initialize training config.
    recipe_dir = Path(__file__).parent
    training_ckpt_path = Path(tmp_path) / "test_train_checkpoints"
    with initialize_config_dir(config_dir=str(recipe_dir / "config"), version_base="1.2"):
        vit_config = compose(
            config_name=config_name,
            overrides=[
                "++training.steps=5",
                "++training.val_interval=5",
                "++training.log_interval=1",
                f"++training.checkpoint.path={training_ckpt_path}",
                "++profiling.torch_memory_profile=false",
                "++profiling.wandb=false",
                f"++fsdp.init_model_with_meta_device={init_model_with_meta_device}",
            ],
        )

    main(vit_config)

    # Verify checkpoints were created.
    assert sum(1 for item in training_ckpt_path.iterdir() if item.is_dir()) == 1, (
        "Expected 1 checkpoint with 5 training steps and validation interval of 5."
    )

    # Auto-resume training from checkpoint. For this test, we auto-resume from the best checkpoint.
    main(vit_config)
