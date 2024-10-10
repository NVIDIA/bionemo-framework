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


"""
This file tests the lora-related utilities for ESM2 finetuning.
"""

import logging
import os
import tempfile
from pathlib import Path

import pytest
import torch
from nemo.collections.nlp.parts.peft_config import LoraPEFTConfig
from omegaconf.dictconfig import DictConfig

from bionemo.model.lora_utils import extract_and_strip_fine_tuned_esm2_lora
from bionemo.model.protein.esm1nv.esm1nv_model import ESM2nvLoRAModel
from bionemo.model.utils import setup_trainer
from bionemo.utils.connectors import BioNeMoSaveRestoreConnector
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import (
    Deterministic,
    check_model_exists,
    distributed_model_parallel_state,
)


UPDATE_GOLDEN_VALUES = os.environ.get("UPDATE_GOLDEN_VALUES", "0") == "1"


@pytest.fixture(scope="module")
def cfg(config_path_for_tests):
    cfg = load_model_config(config_name="esm2nv_lora_finetune_test_8M", config_path=config_path_for_tests)
    return cfg


# ---- Golden Val Tests -----


log = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def lora_checkpoint_path(bionemo_home: Path) -> Path:
    return bionemo_home / "models" / "protein" / "esm2nv" / "esm2nv_sec_str_lora.nemo"


@pytest.fixture(scope="module")
def esm2_checkpoint_path(bionemo_home: Path) -> Path:
    return bionemo_home / "models" / "protein" / "esm2nv" / "esm2nv_8M_untrained.nemo"


@pytest.fixture(scope="module")
def output_modified_ckpt_state_dict_fi() -> Path:
    with tempfile.TemporaryDirectory() as tdir:
        yield Path(tdir) / "stripped_state_dict--esm2nv_sec_str_lora.ckpt"


@pytest.fixture(scope="module")
def golden_output_path(bionemo_home: Path) -> Path:
    # Check for architecture
    gpu_name = torch.cuda.get_device_name(0)

    if "A6000" in gpu_name:
        file_n = "esm2_lora_golden_value_output_on_ones_1_5_a6000.tensor"
    elif "A100" in gpu_name:
        file_n = "esm2_lora_golden_values_a100.tensor"
    elif "A10G" in gpu_name:
        file_n = "ci_esm2_lora_golden_values_a10g.tensor"
    else:
        file_n = "ci_esm2_lora_golden_values_a10g.tensor"

    return (
        bionemo_home
        / "examples"
        / "tests"
        / "test_data"
        / "expected_outputs"
        / "forward_golden_values"
        / "esm2nv_lora"
        / file_n
    )


@pytest.fixture(scope="module")
def test_model(
    cfg: DictConfig, lora_checkpoint_path: Path, esm2_checkpoint_path: Path, output_modified_ckpt_state_dict_fi: Path
) -> ESM2nvLoRAModel:
    peft_cfg = LoraPEFTConfig(cfg.model)
    with Deterministic(), distributed_model_parallel_state():
        trainer = setup_trainer(cfg)

        # Remove task head layers from checkpoint
        extract_and_strip_fine_tuned_esm2_lora(
            lora_checkpoint_path,
            output_modified_ckpt_state_dict_fi,
            ("elmo", "class_heads"),
            verbose=False,
        )

        model_lora = ESM2nvLoRAModel.restore_from(
            str(esm2_checkpoint_path),
            cfg.model,
            trainer=trainer,
            save_restore_connector=BioNeMoSaveRestoreConnector(),
        )

        # NeMo expects lora checkpoint to only contain lora adapters
        model_lora.load_adapters(
            str(output_modified_ckpt_state_dict_fi),
            peft_cfg,
        )

        model_lora = model_lora.to("cuda").eval()

        yield model_lora


@pytest.mark.needs_checkpoint
def test_lora_model_exists(lora_checkpoint_path):
    check_model_exists(str(lora_checkpoint_path))


@pytest.mark.needs_checkpoint
def test_esm2_model_exists(esm2_checkpoint_path):
    check_model_exists(str(esm2_checkpoint_path))


@pytest.mark.lora_golden_val
def test_model_forward(golden_output_path: Path, test_model: ESM2nvLoRAModel):
    if UPDATE_GOLDEN_VALUES:
        golden_output_path.parent.mkdir(exist_ok=True, parents=True)
    else:
        assert (
            golden_output_path.exists()
        ), f"Expected values file not found at {golden_output_path}. Rerun with UPDATE_GOLDEN_VALUES=1 to create it."

    with torch.no_grad():
        embeddings, _ = test_model.forward(
            input_ids=torch.ones((1, 5), dtype=int).to(test_model.device),
            attention_mask=torch.ones((1, 5), dtype=int).to(test_model.device),
            token_type_ids=None,
        )
    if UPDATE_GOLDEN_VALUES:
        torch.save(embeddings, str(golden_output_path))
        assert False, f"Updated expected values at {golden_output_path}, rerun with UPDATE_GOLDEN_VALUES=0"
    else:
        golden_embeddings = torch.load(str(golden_output_path), map_location="cuda")
        assert torch.allclose(embeddings, golden_embeddings, atol=1e-6), "Difference detected!"
