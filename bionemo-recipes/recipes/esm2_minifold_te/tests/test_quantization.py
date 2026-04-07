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

"""Tests for block-wise quantization and quant stats logging for MiniFold TE.

Adapted from esm2_native_te/tests/test_quantization.py with regex patterns
updated for MiniFold's module hierarchy (fold.miniformer.blocks.N.{triangular,transition}).
"""

import re
import sys
from pathlib import Path

import pytest
import yaml


sys.path.insert(0, str(Path(__file__).parent.parent))

from quantization import generate_layer_regex, resolve_layer_precision, update_quant_stats_config


# -- resolve_layer_precision --


def test_fp8_enabled_no_layers_defaults_all():
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=True, fp4_enabled=False, fp8_layers=None, fp4_layers=None
    )
    assert result == ["fp8"] * 6


def test_fp4_enabled_no_layers_defaults_all():
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=False, fp4_enabled=True, fp8_layers=None, fp4_layers=None
    )
    assert result == ["fp4"] * 6


def test_fp8_explicit_layers():
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=True, fp4_enabled=False, fp8_layers=[1, 3, 5], fp4_layers=None
    )
    assert result == ["fp8", None, "fp8", None, "fp8", None]


def test_fp4_explicit_layers():
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=False, fp4_enabled=True, fp8_layers=None, fp4_layers=[2, 4, 6]
    )
    assert result == [None, "fp4", None, "fp4", None, "fp4"]


def test_mixed_fp8_fp4_explicit():
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=True, fp4_enabled=True, fp8_layers=[1, 3, 4], fp4_layers=[2, 5]
    )
    assert result == ["fp8", "fp4", "fp8", "fp8", "fp4", None]


def test_both_enabled_no_layers_raises():
    with pytest.raises(ValueError, match="Both fp8_config and fp4_config are enabled"):
        resolve_layer_precision(num_layers=6, fp8_enabled=True, fp4_enabled=True, fp8_layers=None, fp4_layers=None)


def test_overlapping_layers_raises():
    with pytest.raises(ValueError, match="fp8_layers and fp4_layers cannot have overlapping"):
        resolve_layer_precision(
            num_layers=6, fp8_enabled=True, fp4_enabled=True, fp8_layers=[1, 2, 3], fp4_layers=[3, 4, 5]
        )


def test_disabled_ignores_layers():
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=False, fp4_enabled=False, fp8_layers=[1, 2, 3], fp4_layers=[4, 5, 6]
    )
    assert result == [None] * 6


def test_both_disabled():
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=False, fp4_enabled=False, fp8_layers=None, fp4_layers=None
    )
    assert result == [None] * 6


def test_48_block_model_defaults_all():
    result = resolve_layer_precision(
        num_layers=48, fp8_enabled=True, fp4_enabled=False, fp8_layers=None, fp4_layers=None
    )
    assert result == ["fp8"] * 48


def test_fp8_enabled_empty_list():
    result = resolve_layer_precision(num_layers=6, fp8_enabled=True, fp4_enabled=False, fp8_layers=[], fp4_layers=None)
    assert result == [None] * 6


def test_both_enabled_fp8_specified_fp4_defaults_to_remaining():
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=True, fp4_enabled=True, fp8_layers=[1, 2, 3], fp4_layers=None
    )
    assert result == ["fp8", "fp8", "fp8", "fp4", "fp4", "fp4"]


def test_returns_correct_length():
    for n in [1, 8, 48]:
        result = resolve_layer_precision(
            num_layers=n, fp8_enabled=False, fp4_enabled=False, fp8_layers=None, fp4_layers=None
        )
        assert len(result) == n


# -- generate_layer_regex (MiniFold-specific patterns) --


def test_single_block():
    """Single block (1-indexed=3, 0-indexed=2) should match block 2 in module names."""
    regex = generate_layer_regex([3])
    assert re.search(regex, "fold.miniformer.blocks.2.triangular.pi")
    assert re.search(regex, "fold.miniformer.blocks.2.transition.fc1")
    assert not re.search(regex, "fold.miniformer.blocks.1.triangular.pi")
    assert not re.search(regex, "fold.miniformer.blocks.3.triangular.pi")


def test_multiple_blocks():
    """Multiple blocks should match any of them (converted to 0-indexed)."""
    regex = generate_layer_regex([1, 2, 3])
    # 1-indexed [1,2,3] -> 0-indexed [0,1,2]
    assert re.search(regex, "fold.miniformer.blocks.0.triangular.pi")
    assert re.search(regex, "fold.miniformer.blocks.1.transition.fc1")
    assert re.search(regex, "fold.miniformer.blocks.2.triangular.go")
    assert not re.search(regex, "fold.miniformer.blocks.3.triangular.pi")


def test_matches_correct_sublayers():
    """Regex should match pi, gi, po, go, fc1, fc2."""
    regex = generate_layer_regex([1])
    # Block 0 (1-indexed=1)
    assert re.search(regex, "fold.miniformer.blocks.0.triangular.pi")
    assert re.search(regex, "fold.miniformer.blocks.0.triangular.gi")
    assert re.search(regex, "fold.miniformer.blocks.0.triangular.po")
    assert re.search(regex, "fold.miniformer.blocks.0.triangular.go")
    assert re.search(regex, "fold.miniformer.blocks.0.transition.fc1")
    assert re.search(regex, "fold.miniformer.blocks.0.transition.fc2")
    # Should not match unrelated names
    assert not re.search(regex, "fold.miniformer.blocks.0.triangular.input_norm")
    assert not re.search(regex, "fold.miniformer.blocks.0.transition.norm")


def test_none_returns_disabled_pattern():
    regex = generate_layer_regex(None)
    assert "DISABLED" in regex
    assert not re.search(regex, "fold.miniformer.blocks.0.triangular.pi")


def test_empty_list_returns_disabled_pattern():
    regex = generate_layer_regex([])
    assert "DISABLED" in regex


def test_1indexed_to_0indexed_conversion():
    """User specifies 1-indexed, but module names are 0-indexed."""
    regex = generate_layer_regex([1])
    # Should match block 0 (0-indexed)
    assert re.search(regex, "fold.miniformer.blocks.0.triangular.pi")
    # Should NOT match block 1 (that would be user's block 2)
    assert not re.search(regex, "fold.miniformer.blocks.1.triangular.pi")


def test_large_block_numbers():
    """High block numbers (e.g., 48-block model) should convert correctly."""
    regex = generate_layer_regex([47, 48])
    # 1-indexed [47,48] -> 0-indexed [46,47]
    assert re.search(regex, "fold.miniformer.blocks.46.transition.fc2")
    assert re.search(regex, "fold.miniformer.blocks.47.triangular.gi")
    assert not re.search(regex, "fold.miniformer.blocks.45.transition.fc1")


# -- update_quant_stats_config --


@pytest.fixture
def fp8_only_config(tmp_path):
    """Create an FP8-only stats config file."""
    config = {
        "example_fp8_tensor_stat_collection": {
            "enabled": True,
            "layers": {
                "layer_name_regex_pattern": "PLACEHOLDER",
            },
            "transformer_engine": {
                "LogFp8TensorStats": {
                    "enabled": True,
                    "tensors_struct": [{"tensor": "activation", "stats": ["underflows%"], "freq": 10}],
                }
            },
        }
    }
    config_path = tmp_path / "fp8_stats.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


@pytest.fixture
def fp4_fp8_config(tmp_path):
    """Create a combined FP4+FP8 stats config file."""
    config = {
        "example_fp4_tensor_stat_collection": {
            "enabled": True,
            "layers": {"layer_name_regex_pattern": "PLACEHOLDER"},
            "transformer_engine": {"LogNvfp4TensorStats": {"enabled": True}},
        },
        "example_fp8_tensor_stat_collection": {
            "enabled": True,
            "layers": {"layer_name_regex_pattern": "PLACEHOLDER"},
            "transformer_engine": {"LogFp8TensorStats": {"enabled": True}},
        },
    }
    config_path = tmp_path / "fp4_fp8_stats.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


def test_fp8_layers_updates_regex(fp8_only_config):
    """FP8 block list should update the regex in the output config."""
    output_path = update_quant_stats_config(config_file=fp8_only_config, fp4_layers=None, fp8_layers=[1, 2, 3])
    with open(output_path) as f:
        result = yaml.safe_load(f)
    regex = result["example_fp8_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"]
    # 1-indexed [1,2,3] -> 0-indexed [0,1,2]
    assert re.search(regex, "fold.miniformer.blocks.0.triangular.pi")
    assert re.search(regex, "fold.miniformer.blocks.2.transition.fc2")
    assert not re.search(regex, "fold.miniformer.blocks.3.triangular.pi")


def test_none_layers_disables_matching(fp8_only_config):
    output_path = update_quant_stats_config(config_file=fp8_only_config, fp4_layers=None, fp8_layers=None)
    with open(output_path) as f:
        result = yaml.safe_load(f)
    regex = result["example_fp8_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"]
    assert "DISABLED" in regex


def test_fp4_section_disabled_fp8_still_updated(fp4_fp8_config):
    output_path = update_quant_stats_config(config_file=fp4_fp8_config, fp4_layers=[1, 2, 3], fp8_layers=[4, 5, 6])
    with open(output_path) as f:
        result = yaml.safe_load(f)

    assert result["example_fp4_tensor_stat_collection"]["enabled"] is False

    fp8_regex = result["example_fp8_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"]
    # 1-indexed [4,5,6] -> 0-indexed [3,4,5]
    assert re.search(fp8_regex, "fold.miniformer.blocks.4.triangular.pi")
    assert not re.search(fp8_regex, "fold.miniformer.blocks.1.triangular.pi")


def test_original_file_not_modified(fp8_only_config):
    with open(fp8_only_config) as f:
        original_content = f.read()

    output_path = update_quant_stats_config(config_file=fp8_only_config, fp4_layers=None, fp8_layers=[1, 2])

    assert output_path != fp8_only_config
    with open(fp8_only_config) as f:
        assert f.read() == original_content


def test_preserves_other_config_fields(fp8_only_config):
    output_path = update_quant_stats_config(config_file=fp8_only_config, fp4_layers=None, fp8_layers=[1])
    with open(output_path) as f:
        result = yaml.safe_load(f)
    assert result["example_fp8_tensor_stat_collection"]["transformer_engine"]["LogFp8TensorStats"]["enabled"] is True


def test_missing_section_is_skipped(fp8_only_config):
    output_path = update_quant_stats_config(config_file=fp8_only_config, fp4_layers=[1, 2], fp8_layers=[3, 4])
    with open(output_path) as f:
        result = yaml.safe_load(f)
    assert "example_fp4_tensor_stat_collection" not in result
    regex = result["example_fp8_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"]
    # 1-indexed [3,4] -> 0-indexed [2,3]
    assert re.search(regex, "fold.miniformer.blocks.2.triangular.pi")


def test_with_real_fp8_config():
    """Test with the actual fp8_debugging_stats.yaml file."""
    config_path = Path(__file__).parent.parent / "fp8_debugging_stats.yaml"
    if not config_path.exists():
        pytest.skip("fp8_debugging_stats.yaml not found")

    output_path = update_quant_stats_config(config_file=str(config_path), fp4_layers=None, fp8_layers=[1, 4, 8])
    with open(output_path) as f:
        result = yaml.safe_load(f)

    fp8_regex = result["example_fp8_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"]
    # 1-indexed [1,4,8] -> 0-indexed [0,3,7]
    assert re.search(fp8_regex, "fold.miniformer.blocks.0.transition.fc1")
    assert re.search(fp8_regex, "fold.miniformer.blocks.3.triangular.go")
    assert re.search(fp8_regex, "fold.miniformer.blocks.7.triangular.pi")
    assert not re.search(fp8_regex, "fold.miniformer.blocks.1.transition.fc1")
