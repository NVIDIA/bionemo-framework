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

"""Minimal reproducer for TransformerEngine qk_norm dtype mismatch.

When qk_norm_type='RMSNorm' is enabled and params_dtype=bfloat16, the RMSNorm
weight is created in float32. During forward, Q and K are cast to float32 by the
norm while V stays in bfloat16, causing DotProductAttention to fail with:

    AssertionError: Queries, keys and values must have the same data type!

torch.autocast masks the issue by casting everything back to bfloat16.
"""

import pytest
import torch
import transformer_engine.pytorch as te


@pytest.fixture
def qk_norm_layer():
    """Minimal TransformerLayer with qk_norm enabled."""
    return te.TransformerLayer(
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        num_gqa_groups=2,
        normalization="RMSNorm",
        activation="swiglu",
        bias=False,
        attn_input_format="bshd",
        self_attn_mask_type="causal",
        qk_norm_type="RMSNorm",
        qk_norm_before_rope=True,
        hidden_dropout=0,
        attention_dropout=0,
        layer_number=1,
        params_dtype=torch.bfloat16,
        device="cuda",
    )


@pytest.fixture
def input_tensor():
    return torch.randn(1, 8, 64, dtype=torch.bfloat16, device="cuda")


@pytest.mark.xfail(reason="qk_norm RMSNorm casts Q/K to float32 while V stays bfloat16", strict=True)
def test_qk_norm_forward_without_autocast(qk_norm_layer, input_tensor):
    """Forward pass without torch.autocast fails due to Q/K vs V dtype mismatch."""
    qk_norm_layer(input_tensor)


def test_qk_norm_forward_with_autocast(qk_norm_layer, input_tensor):
    """Forward pass with torch.autocast works (masks the dtype issue)."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = qk_norm_layer(input_tensor)
    assert out.dtype == torch.bfloat16
