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

"""Tests for state dict transformation utilities."""

import numpy as np
import pytest
import torch
from torch import nn

from amplify.state import (
    TransformCTX,
    TransformFns,
    _ModelState,
    _default_transform,
    _match_keys,
    apply_transforms,
    extract_dtypes,
    state_transform,
    StateDictTransform,
)


class MockConfig:
    """Mock configuration for testing."""

    def __init__(
        self,
        num_attention_heads=8,
        num_key_value_heads=4,
        hidden_size=256,
        num_query_groups=4,
        kv_channels=32,
        vocab_size=1000,
    ):
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.num_query_groups = num_query_groups
        self.kv_channels = kv_channels
        self.vocab_size = vocab_size


class TestMatchKeys:
    """Tests for _match_keys function."""

    def test_match_keys_simple_pattern(self):
        """Test matching keys with simple pattern without wildcards."""
        keys = ["layer.weight", "layer.bias", "other.weight"]
        pattern = "layer.weight"
        result = _match_keys(keys, pattern)
        assert result.shape == (1,)
        assert result[0] == "layer.weight"

    def test_match_keys_single_wildcard(self):
        """Test matching keys with single wildcard."""
        keys = ["layer.0.weight", "layer.1.weight", "layer.2.weight"]
        pattern = "layer.*.weight"
        result = _match_keys(keys, pattern)
        assert result.shape == (3,)
        assert result[0] == "layer.0.weight"
        assert result[1] == "layer.1.weight"
        assert result[2] == "layer.2.weight"

    def test_match_keys_multiple_wildcards(self):
        """Test matching keys with multiple wildcards."""
        keys = [
            "layer.0.attn.weight",
            "layer.0.mlp.weight",
            "layer.1.attn.weight",
            "layer.1.mlp.weight",
        ]
        pattern = "layer.*.*.weight"
        result = _match_keys(keys, pattern)
        assert result.shape == (2, 2)
        assert result[0, 0] == "layer.0.attn.weight"
        assert result[0, 1] == "layer.0.mlp.weight"
        assert result[1, 0] == "layer.1.attn.weight"
        assert result[1, 1] == "layer.1.mlp.weight"

    def test_match_keys_double_wildcard(self):
        """Test matching keys with double wildcard."""
        keys = ["model.layer.0.weight", "model.layer.1.weight", "other.weight"]
        pattern = "model.**"
        result = _match_keys(keys, pattern)
        assert result.shape == (2,)
        assert "model.layer.0.weight" in result
        assert "model.layer.1.weight" in result

    def test_match_keys_numeric_sorting(self):
        """Test that numeric wildcards are sorted correctly."""
        keys = ["layer.10.weight", "layer.2.weight", "layer.1.weight"]
        pattern = "layer.*.weight"
        result = _match_keys(keys, pattern)
        assert result.shape == (3,)
        assert result[0] == "layer.1.weight"
        assert result[1] == "layer.2.weight"
        assert result[2] == "layer.10.weight"

    def test_match_keys_no_matches(self):
        """Test matching with no matching keys."""
        keys = ["layer.0.weight", "layer.1.weight"]
        pattern = "model.*.weight"
        result = _match_keys(keys, pattern)
        assert result.shape == (0,)

    def test_match_keys_with_none_values(self):
        """Test matching with None values in keys list."""
        keys = ["layer.0.weight", None, "layer.1.weight"]
        pattern = "layer.*.weight"
        result = _match_keys(keys, pattern)
        assert result.shape == (2,)
        assert result[0] == "layer.0.weight"
        assert result[1] == "layer.1.weight"


class TestModelState:
    """Tests for _ModelState class."""

    def test_model_state_init(self):
        """Test _ModelState initialization."""
        state_dict = {"weight": torch.randn(10, 10)}
        config = MockConfig()
        model_state = _ModelState(state_dict, config)
        assert model_state.config == config
        assert model_state.state_dict() == state_dict

    def test_model_state_to_dtype(self):
        """Test _ModelState dtype conversion."""
        state_dict = {
            "weight": torch.randn(10, 10, dtype=torch.float32),
            "bias": torch.randn(10, dtype=torch.float32),
        }
        model_state = _ModelState(state_dict)
        model_state.to(torch.float16)
        assert state_dict["weight"].dtype == torch.float16
        assert state_dict["bias"].dtype == torch.float16


class TestStateDictTransform:
    """Tests for StateDictTransform class."""

    def test_default_transform(self):
        """Test _default_transform function."""
        value = torch.randn(10, 10)
        result = _default_transform(value)
        assert torch.equal(result, value)

    def test_state_dict_transform_simple_mapping(self):
        """Test simple key mapping without transformation."""
        source = nn.Linear(10, 5)
        target = nn.Linear(10, 5)
        source_state = source.state_dict()
        target_state = target.state_dict()

        ctx = TransformCTX(
            source=source, source_state=source_state, target=target, target_state=target_state
        )

        transform = StateDictTransform("weight", "weight")
        new_ctx = transform(ctx)

        assert "weight" in new_ctx.target_state
        assert torch.equal(new_ctx.target_state["weight"], source_state["weight"])

    def test_state_dict_transform_with_function(self):
        """Test state dict transform with custom transformation function."""
        source = nn.Linear(10, 5)
        target = nn.Linear(10, 5)
        source_state = source.state_dict()
        target_state = target.state_dict()

        ctx = TransformCTX(
            source=source, source_state=source_state, target=target, target_state=target_state
        )

        def double_weights(x):
            return x * 2

        transform = StateDictTransform("weight", "weight", double_weights)
        new_ctx = transform(ctx)

        assert torch.allclose(new_ctx.target_state["weight"], source_state["weight"] * 2)

    def test_state_dict_transform_with_wildcard(self):
        """Test state dict transform with wildcard patterns."""

        class SourceModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

        class TargetModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

        source = SourceModel()
        target = TargetModel()
        source_state = source.state_dict()
        target_state = target.state_dict()

        ctx = TransformCTX(
            source=source, source_state=source_state, target=target, target_state=target_state
        )

        transform = StateDictTransform("layers.*.weight", "blocks.*.weight")
        new_ctx = transform(ctx)

        for i in range(3):
            assert f"blocks.{i}.weight" in new_ctx.target_state
            assert torch.equal(
                new_ctx.target_state[f"blocks.{i}.weight"],
                source_state[f"layers.{i}.weight"],
            )

    def test_state_transform_decorator(self):
        """Test state_transform decorator."""

        @state_transform(source_key="weight", target_key="weights")
        def scale_weights(x):
            return x * 3

        assert isinstance(scale_weights, StateDictTransform)
        assert scale_weights.source_key == "weight"
        assert scale_weights.target_key == "weights"


class TestTransformFns:
    """Tests for TransformFns class methods."""

    def test_split_qkv(self):
        """Test split_qkv transformation."""
        config = MockConfig(
            num_attention_heads=8,
            num_key_value_heads=4,
            hidden_size=256,
        )
        target = nn.Linear(256, 256)
        target.config = config

        # Create a dummy QKV tensor: (num_heads + 2 * num_kv_heads) * head_size = 12 * 32 = 384
        head_size = config.hidden_size // config.num_attention_heads  # 32
        qkv_dim = (config.num_attention_heads + 2 * config.num_key_value_heads) * head_size  # 384
        linear_qkv = torch.randn(qkv_dim, config.hidden_size)

        ctx = TransformCTX(source=None, source_state={}, target=target, target_state={})

        q_proj, k_proj, v_proj = TransformFns.split_qkv(ctx, linear_qkv)

        # Check shapes
        assert q_proj.shape == (config.num_attention_heads * head_size, config.hidden_size)
        assert k_proj.shape == (config.num_key_value_heads * head_size, config.hidden_size)
        assert v_proj.shape == (config.num_key_value_heads * head_size, config.hidden_size)

    def test_merge_qkv(self):
        """Test merge_qkv transformation."""
        config = MockConfig(
            num_attention_heads=8,
            num_key_value_heads=4,
            hidden_size=256,
        )
        target = nn.Linear(256, 256)
        target.config = config

        head_size = config.hidden_size // config.num_attention_heads
        q = torch.randn(config.num_attention_heads * head_size, config.hidden_size)
        k = torch.randn(config.num_key_value_heads * head_size, config.hidden_size)
        v = torch.randn(config.num_key_value_heads * head_size, config.hidden_size)

        ctx = TransformCTX(source=None, source_state={}, target=target, target_state={})

        qkv_weights = TransformFns.merge_qkv(ctx, q, k, v)

        # Check shape
        expected_dim = head_size * (config.num_attention_heads + 2 * config.num_key_value_heads)
        assert qkv_weights.shape == (expected_dim, config.hidden_size)

    def test_split_and_merge_qkv_roundtrip(self):
        """Test that split and merge QKV are inverse operations."""
        config = MockConfig(
            num_attention_heads=8,
            num_key_value_heads=4,
            hidden_size=256,
        )
        target = nn.Linear(256, 256)
        target.config = config

        head_size = config.hidden_size // config.num_attention_heads
        qkv_dim = (config.num_attention_heads + 2 * config.num_key_value_heads) * head_size
        original_qkv = torch.randn(qkv_dim, config.hidden_size)

        ctx = TransformCTX(source=None, source_state={}, target=target, target_state={})

        # Split then merge
        q_proj, k_proj, v_proj = TransformFns.split_qkv(ctx, original_qkv)
        reconstructed_qkv = TransformFns.merge_qkv(ctx, q_proj, k_proj, v_proj)

        assert torch.allclose(original_qkv, reconstructed_qkv, atol=1e-5)

    def test_merge_fc1(self):
        """Test merge_fc1 transformation."""
        gate = torch.randn(100, 50)
        up = torch.randn(100, 50)

        result = TransformFns.merge_fc1(gate, up)

        assert result.shape == (200, 50)
        assert torch.equal(result[:100], gate)
        assert torch.equal(result[100:], up)

    def test_split_fc1(self):
        """Test split_fc1 transformation."""
        linear_fc1 = torch.randn(200, 50)

        gate_proj, up_proj = TransformFns.split_fc1(linear_fc1)

        assert gate_proj.shape == (100, 50)
        assert up_proj.shape == (100, 50)
        assert torch.equal(gate_proj, linear_fc1[:100])
        assert torch.equal(up_proj, linear_fc1[100:])

    def test_split_and_merge_fc1_roundtrip(self):
        """Test that split and merge fc1 are inverse operations."""
        original = torch.randn(200, 50)

        gate_proj, up_proj = TransformFns.split_fc1(original)
        reconstructed = TransformFns.merge_fc1(gate_proj, up_proj)

        assert torch.equal(original, reconstructed)

    def test_duplicate2(self):
        """Test duplicate2 transformation."""
        param = torch.randn(10, 10)
        p1, p2 = TransformFns.duplicate2(param)

        assert torch.equal(p1, param)
        assert torch.equal(p2, param)
        assert p1 is param
        assert p2 is param

    def test_duplicate3(self):
        """Test duplicate3 transformation."""
        param = torch.randn(10, 10)
        p1, p2, p3 = TransformFns.duplicate3(param)

        assert torch.equal(p1, param)
        assert torch.equal(p2, param)
        assert torch.equal(p3, param)

    def test_prune_padding(self):
        """Test prune_padding transformation."""
        config = MockConfig(vocab_size=100)
        target = nn.Linear(128, 10)
        target.config = config

        embedding = torch.randn(128, 10)  # Padded to 128
        ctx = TransformCTX(source=None, source_state={}, target=target, target_state={})

        result = TransformFns.prune_padding(ctx, embedding)

        assert result.shape == (100, 10)
        assert torch.equal(result, embedding[:100])


class TestApplyTransforms:
    """Tests for apply_transforms function."""

    def test_apply_transforms_simple_mapping(self):
        """Test apply_transforms with simple key mapping."""
        source = nn.Linear(10, 5)
        target = nn.Linear(10, 5)

        # Set different weights to verify transformation
        with torch.no_grad():
            source.weight.fill_(1.0)
            target.weight.fill_(0.0)

        mapping = {"weight": "weight", "bias": "bias"}

        result = apply_transforms(source, target, mapping)

        assert torch.allclose(result.weight, torch.ones_like(result.weight))

    def test_apply_transforms_with_transforms(self):
        """Test apply_transforms with transformation functions."""
        source = nn.Linear(10, 5)
        target = nn.Linear(10, 5)

        with torch.no_grad():
            source.weight.fill_(1.0)
            source.bias.fill_(2.0)

        def double_weight(x):
            return x * 2

        transforms = [StateDictTransform("weight", "weight", double_weight)]
        mapping = {"bias": "bias"}

        result = apply_transforms(source, target, mapping, transforms=transforms)

        assert torch.allclose(result.weight, torch.ones(5, 10) * 2)
        assert torch.allclose(result.bias, torch.ones(5) * 2)

    def test_apply_transforms_shape_mismatch_error(self):
        """Test that shape mismatch raises an error."""
        source = nn.Linear(10, 5)
        target = nn.Linear(10, 3)  # Different output size

        mapping = {"weight": "weight"}

        with pytest.raises(ValueError, match="Shape mismatch"):
            apply_transforms(source, target, mapping)

    def test_apply_transforms_missing_key_error(self):
        """Test that missing keys in target raise an error."""

        class CustomModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(5, 10))
                self.extra_param = nn.Parameter(torch.randn(5))

        source = nn.Linear(10, 5)
        target = CustomModule()

        mapping = {"weight": "weight"}

        with pytest.raises(RuntimeError, match="Additional keys"):
            apply_transforms(source, target, mapping)

    def test_apply_transforms_with_cast_dtype(self):
        """Test apply_transforms with dtype casting."""
        source = nn.Linear(10, 5)
        target = nn.Linear(10, 5)

        with torch.no_grad():
            source.weight.fill_(1.0)
            source.bias.fill_(2.0)

        mapping = {"weight": "weight", "bias": "bias"}

        result = apply_transforms(source, target, mapping, cast_dtype=torch.float16)

        assert result.weight.dtype == torch.float16
        assert result.bias.dtype == torch.float16

    def test_apply_transforms_with_model_state(self):
        """Test apply_transforms with _ModelState as source."""
        source_state = {
            "weight": torch.randn(5, 10),
            "bias": torch.randn(5),
        }
        source = _ModelState(source_state)
        target = nn.Linear(10, 5)

        mapping = {"weight": "weight", "bias": "bias"}

        result = apply_transforms(source, target, mapping)

        assert torch.equal(result.weight, source_state["weight"])
        assert torch.equal(result.bias, source_state["bias"])


class TestExtractDtypes:
    """Tests for extract_dtypes function."""

    def test_extract_dtypes_from_parameters(self):
        """Test extracting dtypes from named_parameters."""
        model = nn.Linear(10, 5)
        dtypes = extract_dtypes(model.named_parameters())

        assert "weight" in dtypes
        assert "bias" in dtypes
        assert dtypes["weight"] == torch.float32
        assert dtypes["bias"] == torch.float32

    def test_extract_dtypes_from_state_dict(self):
        """Test extracting dtypes from state_dict."""
        model = nn.Linear(10, 5)
        dtypes = extract_dtypes(model.state_dict().items())

        assert "weight" in dtypes
        assert "bias" in dtypes

    def test_extract_dtypes_with_different_dtypes(self):
        """Test extracting different dtypes."""

        class MixedDtypeModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight_fp32 = nn.Parameter(torch.randn(5, 10, dtype=torch.float32))
                self.weight_fp16 = nn.Parameter(torch.randn(5, 10, dtype=torch.float16))

        model = MixedDtypeModule()
        dtypes = extract_dtypes(model.named_parameters())

        assert dtypes["weight_fp32"] == torch.float32
        assert dtypes["weight_fp16"] == torch.float16


class TestStateDictTransformCallTransform:
    """Tests for StateDictTransform.call_transform method."""

    def test_call_transform_with_ctx_parameter(self):
        """Test call_transform with ctx parameter in transform function."""

        def transform_with_ctx(ctx, x):
            assert isinstance(ctx, TransformCTX)
            return x * 2

        transform = StateDictTransform("weight", "weight", transform_with_ctx)
        ctx = TransformCTX(source=None, source_state={}, target=None, target_state={})

        result = transform.call_transform(ctx, torch.ones(5, 10))
        assert torch.equal(result, torch.ones(5, 10) * 2)

    def test_call_transform_without_ctx_parameter(self):
        """Test call_transform without ctx parameter in transform function."""

        def transform_no_ctx(x):
            return x * 3

        transform = StateDictTransform("weight", "weight", transform_no_ctx)
        ctx = TransformCTX(source=None, source_state={}, target=None, target_state={})

        result = transform.call_transform(ctx, torch.ones(5, 10))
        assert torch.equal(result, torch.ones(5, 10) * 3)

    def test_call_transform_with_var_args(self):
        """Test call_transform with variable arguments."""

        def transform_var_args(*args):
            return sum(args)

        transform = StateDictTransform(("w1", "w2"), "weight", transform_var_args)
        ctx = TransformCTX(source=None, source_state={}, target=None, target_state={})

        result = transform.call_transform(ctx, torch.ones(5, 10), torch.ones(5, 10))
        assert torch.equal(result, torch.ones(5, 10) * 2)

    def test_call_transform_argument_count_mismatch(self):
        """Test that argument count mismatch raises an error."""

        def transform_two_args(x, y):
            return x + y

        transform = StateDictTransform("weight", "weight", transform_two_args)
        ctx = TransformCTX(source=None, source_state={}, target=None, target_state={})

        # Should raise ValueError when providing only one argument
        with pytest.raises(ValueError, match="Expected 2 arguments"):
            transform.call_transform(ctx, torch.ones(5, 10))


class TestTransformCTX:
    """Tests for TransformCTX dataclass."""

    def test_transform_ctx_creation(self):
        """Test creating a TransformCTX object."""
        source = nn.Linear(10, 5)
        target = nn.Linear(10, 5)
        source_state = source.state_dict()
        target_state = target.state_dict()

        ctx = TransformCTX(
            source=source,
            source_state=source_state,
            target=target,
            target_state=target_state,
        )

        assert ctx.source is source
        assert ctx.target is target
        assert ctx.source_state == source_state
        assert ctx.target_state == target_state


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_mapping(self):
        """Test apply_transforms with empty mapping."""
        source = nn.Linear(10, 5)
        target = nn.Linear(10, 5)

        # This should fail because target has keys that aren't mapped
        with pytest.raises(RuntimeError, match="Additional keys"):
            apply_transforms(source, target, {})

    def test_nested_module_transformation(self):
        """Test transformation with nested modules."""

        class NestedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 5)
                self.layer2 = nn.Linear(5, 3)

        source = NestedModule()
        target = NestedModule()

        mapping = {
            "layer1.weight": "layer1.weight",
            "layer1.bias": "layer1.bias",
            "layer2.weight": "layer2.weight",
            "layer2.bias": "layer2.bias",
        }

        result = apply_transforms(source, target, mapping)

        assert torch.equal(result.layer1.weight, source.layer1.weight)
        assert torch.equal(result.layer2.weight, source.layer2.weight)

    def test_transform_with_buffers(self):
        """Test transformation with buffers."""

        class ModuleWithBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(5, 10))
                self.register_buffer("running_mean", torch.zeros(5))

        source = ModuleWithBuffer()
        target = ModuleWithBuffer()

        with torch.no_grad():
            source.running_mean.fill_(1.0)

        mapping = {"weight": "weight", "running_mean": "running_mean"}

        result = apply_transforms(source, target, mapping)

        assert torch.equal(result.running_mean, source.running_mean)


class TestTransformFnsSplitQKVBias:
    """Tests for TransformFns split_qkv_bias and merge_qkv_bias methods."""

    def test_split_qkv_bias(self):
        """Test split_qkv_bias transformation."""
        config = MockConfig(
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=32,
        )
        source = nn.Linear(256, 256)
        source.config = config

        heads_per_group = config.num_attention_heads // config.num_query_groups
        qkv_total_dim = config.num_attention_heads + 2 * config.num_query_groups
        qkv_bias = torch.randn(qkv_total_dim * config.kv_channels)

        ctx = TransformCTX(source=source, source_state={}, target=None, target_state={})

        q_bias, k_bias, v_bias = TransformFns.split_qkv_bias(ctx, qkv_bias)

        assert q_bias.shape == (config.num_attention_heads * config.kv_channels,)
        assert k_bias.shape == (config.num_query_groups * config.kv_channels,)
        assert v_bias.shape == (config.num_query_groups * config.kv_channels,)

    def test_merge_qkv_bias(self):
        """Test merge_qkv_bias transformation."""
        config = MockConfig(
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=32,
        )
        target = nn.Linear(256, 256)
        target.config = config

        qb = torch.randn(config.num_attention_heads * config.kv_channels)
        kb = torch.randn(config.num_query_groups * config.kv_channels)
        vb = torch.randn(config.num_query_groups * config.kv_channels)

        ctx = TransformCTX(source=None, source_state={}, target=target, target_state={})

        qkv_bias = TransformFns.merge_qkv_bias(ctx, qb, kb, vb)

        qkv_total_dim = config.num_attention_heads + 2 * config.num_query_groups
        expected_size = config.kv_channels * qkv_total_dim
        assert qkv_bias.shape == (expected_size,)


class TestTransformFnsMergeConcatVariants:
    """Tests for merge_qkv_concat and merge_qkv_bias_concat methods."""

    def test_merge_qkv_concat(self):
        """Test merge_qkv_concat transformation."""
        config = MockConfig(
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=32,
        )
        target = nn.Linear(256, 256)
        target.config = config

        head_size = config.kv_channels
        q_size = config.num_attention_heads * head_size
        kv_size = config.num_query_groups * head_size

        # Concatenated QKV tensor
        qkv = torch.randn(q_size + kv_size + kv_size, 256)

        ctx = TransformCTX(source=None, source_state={}, target=target, target_state={})

        result = TransformFns.merge_qkv_concat(ctx, qkv)

        # Check that shape is correct for interleaved format
        expected_dim = head_size * (config.num_attention_heads + 2 * config.num_query_groups)
        assert result.shape[0] == expected_dim

    def test_merge_qkv_bias_concat(self):
        """Test merge_qkv_bias_concat transformation."""
        config = MockConfig(
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=32,
        )
        target = nn.Linear(256, 256)
        target.config = config

        head_size = config.kv_channels
        q_size = config.num_attention_heads * head_size
        kv_size = config.num_query_groups * head_size

        # Concatenated QKV bias tensor
        qkv_bias = torch.randn(q_size + kv_size + kv_size)

        ctx = TransformCTX(source=None, source_state={}, target=target, target_state={})

        result = TransformFns.merge_qkv_bias_concat(ctx, qkv_bias)

        # Check that shape is correct for interleaved format
        expected_size = head_size * (config.num_attention_heads + 2 * config.num_query_groups)
        assert result.shape == (expected_size,)