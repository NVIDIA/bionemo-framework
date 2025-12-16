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

import pytest
import torch
import torch.distributed.checkpoint as dcp
import transformer_engine
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from transformer_engine.common import recipe as recipe_module
from transformer_engine.pytorch import fp8
from transformers import DataCollatorForLanguageModeling

from esm.collator import DataCollatorWithFlattening
from esm.modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM


try:
    from transformer_engine.pytorch.tensor.quantized_tensor import QuantizedTensor
except ImportError:  # TE nightly uses a new import path for QuantizedTensor
    from transformer_engine.pytorch.quantized_tensor import QuantizedTensor

ALL_RECIPES = [
    recipe_module.DelayedScaling(),
    recipe_module.Float8CurrentScaling(),
    recipe_module.Float8BlockScaling(),
    recipe_module.MXFP8BlockScaling(),
    # recipe_module.NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True),
]


def _check_recipe_support(recipe: recipe_module.Recipe):
    """Check if a recipe is supported and return (supported, reason)."""
    if isinstance(recipe, recipe_module.DelayedScaling):
        recipe_supported, reason = fp8.check_fp8_support()
    elif isinstance(recipe, recipe_module.Float8CurrentScaling):
        recipe_supported, reason = fp8.check_fp8_support()
    elif isinstance(recipe, recipe_module.Float8BlockScaling):
        recipe_supported, reason = fp8.check_fp8_block_scaling_support()
    elif isinstance(recipe, recipe_module.MXFP8BlockScaling):
        recipe_supported, reason = fp8.check_mxfp8_support()
    elif isinstance(recipe, recipe_module.NVFP4BlockScaling):
        recipe_supported, reason = fp8.check_nvfp4_support()
    else:
        recipe_supported = False
        reason = "Unsupported recipe"
    return recipe_supported, reason


def requires_recipe_support(recipe: recipe_module.Recipe):
    """Decorator to skip tests that require recipe support."""

    def requires_recipe_support_inner(func):
        recipe_supported, reason = _check_recipe_support(recipe)
        return pytest.mark.skipif(not recipe_supported, reason=reason)(func)

    return requires_recipe_support_inner


def parametrize_recipes_with_support(recipes):
    """Generate pytest.param objects with skip marks for unsupported recipes."""
    parametrized_recipes = []
    for recipe in recipes:
        recipe_supported, reason = _check_recipe_support(recipe)
        parametrized_recipes.append(
            pytest.param(
                recipe,
                id=recipe.__class__.__name__,
                marks=pytest.mark.skipif(
                    not recipe_supported,
                    reason=reason,
                ),
            )
        )
    return parametrized_recipes


@pytest.fixture
def input_data_thd(tokenizer, tokenized_proteins):
    mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, seed=42)
    data_collator = DataCollatorWithFlattening(
        collator=mlm_collator,
        pad_to_multiple_of=32,  # MXFP8 requires the sequence length to be divisible by 32, regular FP8 requires 16.
    )

    return data_collator(tokenized_proteins)


@pytest.mark.parametrize("fp8_recipe", parametrize_recipes_with_support(ALL_RECIPES))
def test_fp8_forward_and_backward_pass(te_model_checkpoint, input_data, fp8_recipe):
    model_te = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)
    model_te.to("cuda")

    input_data = {k: v.to("cuda") for k, v in input_data.items()}
    outputs = model_te(**input_data)

    with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        outputs_fp8 = model_te(**input_data)
    outputs_fp8.loss.backward()

    if isinstance(fp8_recipe, recipe_module.NVFP4BlockScaling):
        atol = 0.2
        rtol = 0.05
    else:
        atol = None
        rtol = None

    torch.testing.assert_close(outputs_fp8.loss, outputs.loss, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fp8_recipe", parametrize_recipes_with_support(ALL_RECIPES))
def test_fp8_forward_and_backward_pass_thd(te_model_checkpoint, input_data_thd, fp8_recipe, monkeypatch):
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    model_te = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, attn_input_format="thd", dtype=torch.bfloat16)
    model_te.to("cuda")

    input_data = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}
    outputs = model_te(**input_data)

    with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        outputs_fp8 = model_te(**input_data)
    outputs_fp8.loss.backward()

    if isinstance(fp8_recipe, recipe_module.NVFP4BlockScaling):
        atol = 0.2
        rtol = 0.05
    elif isinstance(fp8_recipe, recipe_module.DelayedScaling):
        atol = 0.1
        rtol = 0.03
    else:
        atol = None
        rtol = None

    torch.testing.assert_close(outputs_fp8.loss, outputs.loss, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fp8_recipe", parametrize_recipes_with_support(ALL_RECIPES))
def test_fp8_model_init_forward_and_backward(te_model_checkpoint, input_data, fp8_recipe):
    config = NVEsmConfig.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)
    with transformer_engine.pytorch.fp8_model_init(enabled=True, recipe=fp8_recipe):
        model_te = NVEsmForMaskedLM(config)

    assert isinstance(model_te.lm_head.dense.weight, QuantizedTensor)

    model_te.to("cuda")
    input_data = {k: v.to("cuda") for k, v in input_data.items()}

    with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        outputs_fp8 = model_te(**input_data)

    outputs_fp8.loss.backward()


@pytest.mark.xfail(reason="BIONEMO-3055: fp8 model init and pretrained loading is not currently supported.")
@pytest.mark.parametrize("fp8_recipe", parametrize_recipes_with_support(ALL_RECIPES))
def test_fp8_model_init_from_pretrained(te_model_checkpoint, fp8_recipe):
    # TODO: this will be renamed to quantized_model_init in the future, fp8_model_init will be removed in 3.0
    with transformer_engine.pytorch.fp8_model_init(enabled=True, recipe=fp8_recipe):
        model_te = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)

    assert isinstance(model_te.esm.encoder.layers[0].layernorm_mlp.fc2_weight, QuantizedTensor)
    assert isinstance(model_te.lm_head.dense.weight, QuantizedTensor)


@pytest.mark.xfail(reason="BIONEMO-3055: fp8 model init and pretrained saving is not currently supported.")
@pytest.mark.parametrize("fp8_recipe", parametrize_recipes_with_support(ALL_RECIPES))
def test_fp8_model_init_save_pretrained(te_model_checkpoint, tmp_path, fp8_recipe):
    config = NVEsmConfig.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)
    with transformer_engine.pytorch.fp8_model_init(enabled=True, recipe=fp8_recipe):
        model_fp8 = NVEsmForMaskedLM(config)

    assert isinstance(model_fp8.esm.encoder.layers[0].layernorm_mlp.fc2_weight, QuantizedTensor)
    assert isinstance(model_fp8.lm_head.dense.weight, QuantizedTensor)

    model_fp8.save_pretrained(tmp_path / "fp8_checkpoint")
    del model_fp8
    NVEsmForMaskedLM.from_pretrained(tmp_path / "fp8_checkpoint", dtype=torch.bfloat16)


@pytest.mark.parametrize("fp8_recipe", parametrize_recipes_with_support(ALL_RECIPES))
def test_fp8_model_distributed_checkpointing_save_and_load(te_model_checkpoint, tmp_path, input_data, fp8_recipe):
    config = NVEsmConfig.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)
    with transformer_engine.pytorch.fp8_model_init(enabled=True, recipe=fp8_recipe):
        model_fp8 = NVEsmForMaskedLM(config)

    model_fp8.to("cuda")
    input_data = {k: v.to("cuda") for k, v in input_data.items()}
    with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        outputs = model_fp8(**input_data)
    outputs.loss.backward()

    state_dict = get_model_state_dict(model_fp8)
    state_dict = {key: val for key, val in state_dict.items() if not key.endswith("_extra_state")}
    dcp.save(state_dict, checkpoint_id=tmp_path / "fp8_checkpoint")

    del model_fp8, state_dict

    with transformer_engine.pytorch.fp8_model_init(enabled=True, recipe=fp8_recipe):
        model_fp8 = NVEsmForMaskedLM(config)

    state_dict = model_fp8.state_dict()
    state_dict = {key: val for key, val in state_dict.items() if not key.endswith("_extra_state")}
    dcp.load(state_dict, checkpoint_id=tmp_path / "fp8_checkpoint")
