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

"""Tests for quantized_model_init with FL1 (first/last layer BF16) and all-FP8.

Verifies that:
1. All-FP8 + qinit: all decoder layer weights are QuantizedTensors with high-precision init vals
2. FL1 + qinit: FP8 layers have QuantizedTensor weights, BF16 layers have regular BF16 weights
3. BF16 layers don't lose precision from an outer quantized_model_init context

Uses Float8BlockScaling instead of MXFP8BlockScaling so tests run on non-Blackwell GPUs.
The quantized_model_init behavior is recipe-agnostic.
"""

import sys
from pathlib import Path

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8BlockScaling
from transformer_engine.pytorch.tensor import QuantizedTensor


sys.path.append(Path(__file__).parent.parent.as_posix())

from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM


# Small model config for fast testing
_SMALL_CONFIG_KWARGS = {
    "num_hidden_layers": 4,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "vocab_size": 1024,
    "max_position_embeddings": 128,
}

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _has_quantized_weights(layer) -> bool:
    """Check if a TE TransformerLayer has any QuantizedTensor parameters."""
    for param in layer.parameters():
        if isinstance(param.data, QuantizedTensor):
            return True
    return False


def _has_high_precision_init_val(layer) -> bool:
    """Check if any parameter in the layer has a high-precision init val."""
    for param in layer.parameters():
        if hasattr(param, "get_high_precision_init_val") and param.get_high_precision_init_val() is not None:
            return True
    return False


@requires_gpu
def test_all_fp8_qinit():
    """All layers FP8 with quantized_model_init: all weights should be QuantizedTensors."""
    recipe = Float8BlockScaling()
    config = NVLlamaConfig(
        **_SMALL_CONFIG_KWARGS,
        attn_input_format="bshd",
        dtype=torch.bfloat16,
    )

    with te.quantized_model_init(recipe=recipe, enabled=True, preserve_high_precision_init_val=True):
        model = NVLlamaForCausalLM(config, fp8_recipe=recipe)

    # All decoder layers should have quantized weights
    for i, layer in enumerate(model.model.layers):
        assert _has_quantized_weights(layer), f"Layer {i} should have QuantizedTensor weights"
        assert _has_high_precision_init_val(layer), f"Layer {i} should have high-precision init vals"


@requires_gpu
def test_fl1_qinit_bf16_layers_not_quantized():
    """FL1 + qinit: BF16 layers (first/last) should NOT have quantized weights."""
    recipe = Float8BlockScaling()
    # FL1: layers 2,3 in FP8 (1-indexed), layers 1,4 in BF16
    layer_precision = [None, "fp8", "fp8", None]
    config = NVLlamaConfig(
        **_SMALL_CONFIG_KWARGS,
        attn_input_format="bshd",
        dtype=torch.bfloat16,
        layer_precision=layer_precision,
        use_quantized_model_init=True,
    )

    with te.quantized_model_init(recipe=recipe, enabled=True, preserve_high_precision_init_val=True):
        model = NVLlamaForCausalLM(config, fp8_recipe=recipe)

    # BF16 layers (0 and 3, 0-indexed) should NOT have quantized weights
    assert not _has_quantized_weights(model.model.layers[0]), "First layer (BF16) should not have QuantizedTensors"
    assert not _has_quantized_weights(model.model.layers[3]), "Last layer (BF16) should not have QuantizedTensors"

    # FP8 layers (1 and 2, 0-indexed) should have quantized weights
    assert _has_quantized_weights(model.model.layers[1]), "FP8 layer 1 should have QuantizedTensors"
    assert _has_quantized_weights(model.model.layers[2]), "FP8 layer 2 should have QuantizedTensors"


@requires_gpu
def test_fl1_qinit_fp8_layers_preserve_high_precision():
    """FL1 + qinit: FP8 layers should preserve high-precision init vals for master weights."""
    recipe = Float8BlockScaling()
    layer_precision = [None, "fp8", "fp8", None]
    config = NVLlamaConfig(
        **_SMALL_CONFIG_KWARGS,
        attn_input_format="bshd",
        dtype=torch.bfloat16,
        layer_precision=layer_precision,
        use_quantized_model_init=True,
    )

    with te.quantized_model_init(recipe=recipe, enabled=True, preserve_high_precision_init_val=True):
        model = NVLlamaForCausalLM(config, fp8_recipe=recipe)

    # FP8 layers should have high-precision init values
    assert _has_high_precision_init_val(model.model.layers[1]), "FP8 layer should have high-precision init vals"
    assert _has_high_precision_init_val(model.model.layers[2]), "FP8 layer should have high-precision init vals"

    # BF16 layers should NOT have high-precision init values (they're already BF16)
    assert not _has_high_precision_init_val(model.model.layers[0]), (
        "BF16 layer should not have high-precision init vals"
    )
    assert not _has_high_precision_init_val(model.model.layers[3]), (
        "BF16 layer should not have high-precision init vals"
    )


@requires_gpu
def test_fl1_no_qinit_baseline():
    """FL1 without qinit: all weights should be regular BF16 tensors (baseline)."""
    recipe = Float8BlockScaling()
    layer_precision = [None, "fp8", "fp8", None]
    config = NVLlamaConfig(
        **_SMALL_CONFIG_KWARGS,
        attn_input_format="bshd",
        dtype=torch.bfloat16,
        layer_precision=layer_precision,
    )

    # No quantized_model_init context — default behavior
    model = NVLlamaForCausalLM(config, fp8_recipe=recipe)

    # No layers should have quantized weights
    for i, layer in enumerate(model.model.layers):
        assert not _has_quantized_weights(layer), f"Layer {i} should not have QuantizedTensors without qinit"
