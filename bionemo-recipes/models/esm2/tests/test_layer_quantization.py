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

"""Unit tests for NVEsmEncoder.set_recipes and get_layer_autocast."""

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import transformer_engine.common.recipe
import transformer_engine.pytorch

from modeling_esm_te import NVEsmConfig, NVEsmEncoder


@pytest.fixture
def encoder():
    """Create a small NVEsmEncoder on CUDA for testing."""
    config = NVEsmConfig(
        hidden_size=320,
        intermediate_size=1280,
        num_hidden_layers=6,
        num_attention_heads=20,
        max_position_embeddings=1026,
    )
    return NVEsmEncoder(config)


# -- set_recipes --


def test_all_fp8(encoder):
    encoder.config.layer_precision = ["fp8"] * 6
    fp8_recipe = transformer_engine.common.recipe.DelayedScaling()
    encoder.set_recipes(fp8_recipe=fp8_recipe, fp4_recipe=None)
    assert encoder._fp8_recipe is fp8_recipe
    assert encoder._fp4_recipe is None
    assert all(p == "fp8" for p in encoder.config.layer_precision)


def test_all_fp4(encoder):
    encoder.config.layer_precision = ["fp4"] * 6
    fp4_recipe = transformer_engine.common.recipe.NVFP4BlockScaling()
    encoder.set_recipes(fp8_recipe=None, fp4_recipe=fp4_recipe)
    assert encoder._fp8_recipe is None
    assert encoder._fp4_recipe is fp4_recipe
    assert all(p == "fp4" for p in encoder.config.layer_precision)


def test_all_bf16(encoder):
    encoder.config.layer_precision = [None] * 6
    encoder.set_recipes(fp8_recipe=None, fp4_recipe=None)
    assert all(p is None for p in encoder.config.layer_precision)


def test_mixed_fp8_fp4(encoder):
    encoder.config.layer_precision = ["fp8", "fp8", "fp8", "fp4", "fp4", "fp4"]
    fp8_recipe = transformer_engine.common.recipe.DelayedScaling()
    fp4_recipe = transformer_engine.common.recipe.NVFP4BlockScaling()
    encoder.set_recipes(fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)
    assert encoder.config.layer_precision == ["fp8", "fp8", "fp8", "fp4", "fp4", "fp4"]


def test_mixed_fp8_bf16(encoder):
    encoder.config.layer_precision = ["fp8", None, "fp8", None, "fp8", None]
    fp8_recipe = transformer_engine.common.recipe.DelayedScaling()
    encoder.set_recipes(fp8_recipe=fp8_recipe, fp4_recipe=None)
    assert encoder.config.layer_precision == ["fp8", None, "fp8", None, "fp8", None]


def test_mixed_all_three(encoder):
    encoder.config.layer_precision = ["fp8", "fp8", None, None, "fp4", "fp4"]
    fp8_recipe = transformer_engine.common.recipe.DelayedScaling()
    fp4_recipe = transformer_engine.common.recipe.NVFP4BlockScaling()
    encoder.set_recipes(fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)
    assert encoder.config.layer_precision == ["fp8", "fp8", None, None, "fp4", "fp4"]


def test_covers_all_layers(encoder):
    encoder.config.layer_precision = ["fp8"] + [None] * 5
    encoder.set_recipes(fp8_recipe=transformer_engine.common.recipe.DelayedScaling(), fp4_recipe=None)
    assert len(encoder.config.layer_precision) == 6


def test_recipes_stored_as_attributes(encoder):
    encoder.config.layer_precision = ["fp8", "fp4", None, None, None, None]
    fp8_recipe = transformer_engine.common.recipe.DelayedScaling()
    fp4_recipe = transformer_engine.common.recipe.NVFP4BlockScaling()
    encoder.set_recipes(fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)
    assert encoder._fp8_recipe is fp8_recipe
    assert encoder._fp4_recipe is fp4_recipe
    # The precision list only contains strings/None, not recipe objects.
    for v in encoder.config.layer_precision:
        assert v is None or isinstance(v, str)


# -- get_layer_autocast --


def test_fp8_layer_returns_nullcontext(encoder):
    encoder.config.layer_precision = ["fp8"] + [None] * 5
    encoder.set_recipes(fp8_recipe=transformer_engine.common.recipe.DelayedScaling(), fp4_recipe=None)
    ctx = encoder.get_layer_autocast(0)
    assert isinstance(ctx, nullcontext)


def test_fp4_layer_returns_te_autocast(encoder):
    fp4_recipe = transformer_engine.common.recipe.NVFP4BlockScaling()
    encoder.config.layer_precision = ["fp4"] + [None] * 5
    encoder.set_recipes(fp8_recipe=None, fp4_recipe=fp4_recipe)
    with patch.object(transformer_engine.pytorch, "autocast") as mock_autocast:
        mock_autocast.return_value = "fp4_context"
        ctx = encoder.get_layer_autocast(0)
        mock_autocast.assert_called_once_with(enabled=True, recipe=fp4_recipe)
        assert ctx == "fp4_context"


def test_bf16_layer_returns_te_autocast_disabled(encoder):
    encoder.config.layer_precision = [None] * 6
    encoder.set_recipes(fp8_recipe=None, fp4_recipe=None)
    with patch.object(transformer_engine.pytorch, "autocast") as mock_autocast:
        mock_autocast.return_value = "bf16_context"
        ctx = encoder.get_layer_autocast(0)
        mock_autocast.assert_called_once_with(enabled=False)
        assert ctx == "bf16_context"


def test_uninitialized_defaults_to_bf16(encoder):
    """When layer_precision is None (default), all layers default to BF16."""
    assert encoder.config.layer_precision is None
    with patch.object(transformer_engine.pytorch, "autocast") as mock_autocast:
        mock_autocast.return_value = "bf16_context"
        ctx = encoder.get_layer_autocast(0)
        mock_autocast.assert_called_once_with(enabled=False)
        assert ctx == "bf16_context"


def test_mixed_layers_return_correct_contexts(encoder):
    fp8_recipe = transformer_engine.common.recipe.DelayedScaling()
    fp4_recipe = transformer_engine.common.recipe.NVFP4BlockScaling()
    encoder.config.layer_precision = ["fp8", "fp8", "fp4", "fp4", None, None]
    encoder.set_recipes(fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)

    # FP8 layers -> nullcontext
    assert isinstance(encoder.get_layer_autocast(0), nullcontext)
    assert isinstance(encoder.get_layer_autocast(1), nullcontext)

    # FP4 layers -> te.pytorch.autocast
    with patch.object(transformer_engine.pytorch, "autocast") as mock_autocast:
        mock_autocast.return_value = "fp4_context"
        encoder.get_layer_autocast(2)
        mock_autocast.assert_called_with(enabled=True, recipe=fp4_recipe)

    # BF16 layers -> te.pytorch.autocast(enabled=False)
    with patch.object(transformer_engine.pytorch, "autocast") as mock_autocast:
        mock_autocast.return_value = "bf16_context"
        encoder.get_layer_autocast(4)
        mock_autocast.assert_called_with(enabled=False)


def test_layer_precision_is_pickleable(encoder):
    """The config.layer_precision list should be trivially pickleable."""
    import pickle

    encoder.config.layer_precision = ["fp8", "fp8", "fp4", "fp4", None, None]
    roundtripped = pickle.loads(pickle.dumps(encoder.config.layer_precision))
    assert roundtripped == encoder.config.layer_precision
