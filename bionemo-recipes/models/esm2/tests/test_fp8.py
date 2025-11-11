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
from transformer_engine.common.recipe import DelayedScaling, MXFP8BlockScaling
from transformer_engine.pytorch.fp8 import check_fp8_support, check_mxfp8_support
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

from esm.collator import MLMDataCollatorWithFlattening
from esm.modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM


def requires_fp8(func):
    """Decorator to skip tests that require FP8 support."""
    fp8_available, reason = check_fp8_support()
    return pytest.mark.skipif(not fp8_available, reason=f"FP8 is not supported on this GPU: {reason}")(func)


def requires_mxfp8(func):
    """Decorator to skip tests that require MXFP8 support."""
    mxfp8_available, reason = check_mxfp8_support()
    if torch.cuda.get_device_capability() == (12, 0):
        mxfp8_available = False
        reason = "MXFP8 is not supported on sm120"
    return pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8 is not supported on this GPU: {reason}")(func)


@pytest.fixture
def input_data_thd(tokenizer, tokenized_proteins):
    data_collator = MLMDataCollatorWithFlattening(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=32,  # MXFP8 requires the sequence length to be divisible by 32, regular FP8 requires 16.
        seed=42,
    )

    return data_collator(tokenized_proteins)


@requires_fp8
def test_fp8_forward_and_backward_pass(te_model_checkpoint, input_data):
    model_te = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)
    model_te.to("cuda")

    input_data = {k: v.to("cuda") for k, v in input_data.items()}
    outputs = model_te(**input_data)

    fp8_recipe = DelayedScaling()
    with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        outputs_fp8 = model_te(**input_data)
    outputs_fp8.loss.backward()

    torch.testing.assert_close(outputs_fp8.loss, outputs.loss)


@requires_fp8
def test_fp8_forward_and_backward_pass_thd(te_model_checkpoint, input_data_thd, monkeypatch):
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    model_te = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, attn_input_format="thd", dtype=torch.bfloat16)
    model_te.to("cuda")

    input_data = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}
    outputs = model_te(**input_data)

    fp8_recipe = DelayedScaling()
    with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        outputs_fp8 = model_te(**input_data)
    outputs_fp8.loss.backward()

    torch.testing.assert_close(outputs_fp8.loss, outputs.loss)


@requires_mxfp8
def test_mxfp8_forward_and_backward_pass(te_model_checkpoint, input_data):
    model_te = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)
    model_te.to("cuda")

    input_data = {k: v.to("cuda") for k, v in input_data.items()}
    outputs = model_te(**input_data)

    mxfp8_recipe = MXFP8BlockScaling()
    with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=mxfp8_recipe):
        outputs_fp8 = model_te(**input_data)
    outputs_fp8.loss.backward()

    torch.testing.assert_close(outputs_fp8.loss, outputs.loss)


@requires_mxfp8
def test_mxfp8_forward_and_backward_pass_thd(te_model_checkpoint, input_data_thd):
    model_te = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, attn_input_format="thd", dtype=torch.bfloat16)
    model_te.to("cuda")

    input_data = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}
    outputs = model_te(**input_data)

    mxfp8_recipe = MXFP8BlockScaling()
    with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=mxfp8_recipe):
        outputs_fp8 = model_te(**input_data)
    outputs_fp8.loss.backward()

    torch.testing.assert_close(outputs_fp8.loss, outputs.loss)


@requires_fp8
def test_fp8_model_init_forward_and_backward(te_model_checkpoint, input_data):
    fp8_recipe = DelayedScaling()
    config = NVEsmConfig.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)
    with transformer_engine.pytorch.fp8_model_init(enabled=True, recipe=fp8_recipe):
        model_te = NVEsmForMaskedLM(config)

    assert isinstance(model_te.lm_head.dense.weight, Float8Tensor)

    model_te.to("cuda")
    input_data = {k: v.to("cuda") for k, v in input_data.items()}

    with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        outputs_fp8 = model_te(**input_data)

    outputs_fp8.loss.backward()


@requires_fp8
@pytest.mark.xfail(reason="BIONEMO-3055: fp8 model init and pretrained loading is not currently supported.")
def test_fp8_model_init_from_pretrained(te_model_checkpoint, input_data):
    fp8_recipe = DelayedScaling()

    # TODO: this will be renamed to quantized_model_init in the future, fp8_model_init will be removed in 3.0
    with transformer_engine.pytorch.fp8_model_init(enabled=True, recipe=fp8_recipe):
        model_te = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint)

    assert isinstance(model_te.esm.encoder.layers[0].layernorm_mlp.fc2_weight, Float8Tensor)
    assert isinstance(model_te.lm_head.dense.weight, Float8Tensor)


@requires_fp8
@pytest.mark.xfail(reason="BIONEMO-3055: fp8 model init and pretrained saving is not currently supported.")
def test_fp8_model_init_save_pretrained(te_model_checkpoint, tmp_path):
    fp8_recipe = DelayedScaling()
    config = NVEsmConfig.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)
    with transformer_engine.pytorch.fp8_model_init(enabled=True, recipe=fp8_recipe):
        model_fp8 = NVEsmForMaskedLM(config)

    assert isinstance(model_fp8.esm.encoder.layers[0].layernorm_mlp.fc2_weight, Float8Tensor)
    assert isinstance(model_fp8.lm_head.dense.weight, Float8Tensor)

    model_fp8.save_pretrained(tmp_path / "fp8_checkpoint")
    del model_fp8
    NVEsmForMaskedLM.from_pretrained(tmp_path / "fp8_checkpoint", dtype=torch.bfloat16)


@requires_fp8
@pytest.mark.xfail(reason="BIONEMO-3055: fp8 model init and distributed checkpointing is not currently supported.")
def test_fp8_model_distributed_checkpointing_save_and_load(te_model_checkpoint, tmp_path, input_data):
    fp8_recipe = DelayedScaling()
    config = NVEsmConfig.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)
    with transformer_engine.pytorch.fp8_model_init(enabled=True, recipe=fp8_recipe):
        model_fp8 = NVEsmForMaskedLM(config)

    model_fp8.to("cuda")
    input_data = {k: v.to("cuda") for k, v in input_data.items()}
    with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        outputs = model_fp8(**input_data)
    outputs.loss.backward()

    state_dict = get_model_state_dict(model_fp8)
    dcp.save(state_dict, checkpoint_id=tmp_path / "fp8_checkpoint")

    del model_fp8, state_dict

    with transformer_engine.pytorch.fp8_model_init(enabled=True, recipe=fp8_recipe):
        model_fp8 = NVEsmForMaskedLM(config)

    state_dict = model_fp8.state_dict()
    dcp.load(state_dict, checkpoint_id=tmp_path / "fp8_checkpoint")
