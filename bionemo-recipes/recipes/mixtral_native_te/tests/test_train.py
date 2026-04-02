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

import gc
import random

import torch
from hydra import compose, initialize_config_dir
from train_fsdp2 import main as main_fsdp2


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_set_seed():
    random.seed(42)
    torch.manual_seed(42)


def test_sanity_convergence_fsdp2_te_bshd(tmp_path, recipe_path):
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                "num_train_steps=40",
                "config_kwargs.attn_input_format=bshd",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    _cleanup()

    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_convergence_fsdp2_te_thd(tmp_path, recipe_path):
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                "num_train_steps=40",
                "use_sequence_packing=true",
                "config_kwargs.attn_input_format=thd",
                "config_kwargs.self_attn_mask_type=padding_causal",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    _cleanup()

    assert final_loss < 8.5, f"Final loss {final_loss} is too high, expected < 8.5"
