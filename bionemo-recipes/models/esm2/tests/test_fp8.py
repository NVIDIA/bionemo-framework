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
import transformer_engine
from transformer_engine.common.recipe import DelayedScaling, MXFP8BlockScaling
from transformer_engine.pytorch.fp8 import check_fp8_support, check_mxfp8_support

from esm.modeling_esm_te import NVEsmForMaskedLM


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
