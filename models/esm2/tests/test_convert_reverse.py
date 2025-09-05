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

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import AutoModelForMaskedLM


def test_esm_model_has_all_te_layers():
    """Test that the converted TE model doesn't contain vanilla PyTorch layers."""
    from esm.convert import convert_esm_hf_to_te

    model_hf = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model_te = convert_esm_hf_to_te(model_hf)
    vanilla_layers_found = []
    for name, module in model_te.named_modules():
        if isinstance(module, nn.Linear):
            vanilla_layers_found.append(f"Linear layer found in {name}")
        if isinstance(module, nn.LayerNorm):
            vanilla_layers_found.append(f"LayerNorm layer found in {name}")
    if vanilla_layers_found:
        print("ERROR: Found vanilla PyTorch layers in converted TE model:")
        for error in vanilla_layers_found:
            print(f"WARNING: {error}")
        assert not vanilla_layers_found, f"Found {len(vanilla_layers_found)} vanilla layers in converted model"


def test_convert_te_to_hf_roundtrip():
    """Test that converting HF -> TE -> HF produces the same model."""
    from esm.convert import convert_esm_hf_to_te, convert_esm_te_to_hf

    model_hf_original = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")

    model_te = convert_esm_hf_to_te(model_hf_original)
    model_hf_converted = convert_esm_te_to_hf(model_te)

    original_state_dict = model_hf_original.state_dict()
    converted_state_dict = model_hf_converted.state_dict()
    original_keys = {k for k in original_state_dict.keys() if "contact_head" not in k}
    converted_keys = set(converted_state_dict.keys())
    assert original_keys == converted_keys

    for key in original_state_dict.keys():
        if not key.endswith("_extra_state") and not key.endswith("inv_freq") and "contact_head" not in key:
            torch.testing.assert_close(original_state_dict[key], converted_state_dict[key], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("model_name", ["esm2_t6_8M_UR50D"])
def test_export_te_checkpoint_to_hf(model_name):
    """Test the export function that saves TE checkpoint as HF format."""
    from esm.export import export_hf_checkpoint, export_te_checkpoint

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        model_hf_original = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_name}")

        # Use export_hf_checkpoint to create TE checkpoint
        te_checkpoint_path = temp_path / "te_checkpoint"
        export_hf_checkpoint(model_name, te_checkpoint_path)
        te_model_path = te_checkpoint_path / model_name

        hf_export_path = temp_path / "hf_export"
        export_te_checkpoint(str(te_model_path), str(hf_export_path))

        model_hf_exported = AutoModelForMaskedLM.from_pretrained(str(hf_export_path))

        original_state_dict = model_hf_original.state_dict()
        exported_state_dict = model_hf_exported.state_dict()

        # assert original_state_dict.keys() == exported_state_dict.keys()
        original_keys = {k for k in original_state_dict.keys() if "contact_head" not in k}
        exported_keys = {k for k in exported_state_dict.keys() if "contact_head" not in k}
        assert original_keys == exported_keys

        for key in original_state_dict.keys():
            if not key.endswith("_extra_state") and not key.endswith("inv_freq") and "contact_head" not in key:
                torch.testing.assert_close(original_state_dict[key], exported_state_dict[key], atol=1e-5, rtol=1e-5)


def test_qkv_unpacking():
    """Test that QKV unpacking works correctly."""
    from esm.convert import convert_esm_hf_to_te, convert_esm_te_to_hf

    model_hf = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model_te = convert_esm_hf_to_te(model_hf)
    model_hf_converted = convert_esm_te_to_hf(model_te)

    for i in range(model_hf.config.num_hidden_layers):
        hf_query = model_hf.state_dict()[f"esm.encoder.layer.{i}.attention.self.query.weight"]
        hf_key = model_hf.state_dict()[f"esm.encoder.layer.{i}.attention.self.key.weight"]
        hf_value = model_hf.state_dict()[f"esm.encoder.layer.{i}.attention.self.value.weight"]

        converted_query = model_hf_converted.state_dict()[f"esm.encoder.layer.{i}.attention.self.query.weight"]
        converted_key = model_hf_converted.state_dict()[f"esm.encoder.layer.{i}.attention.self.key.weight"]
        converted_value = model_hf_converted.state_dict()[f"esm.encoder.layer.{i}.attention.self.value.weight"]

        torch.testing.assert_close(hf_query, converted_query, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(hf_key, converted_key, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(hf_value, converted_value, atol=1e-5, rtol=1e-5)


def test_config_conversion():
    """Test that config conversion works correctly."""
    from esm.convert import convert_esm_hf_to_te, convert_esm_te_to_hf

    model_hf = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model_te = convert_esm_hf_to_te(model_hf)
    model_hf_converted = convert_esm_te_to_hf(model_te)

    assert model_hf_converted.config.model_type == "esm"
    assert model_hf_converted.config.hidden_size == model_hf.config.hidden_size
    assert model_hf_converted.config.num_hidden_layers == model_hf.config.num_hidden_layers
    assert model_hf_converted.config.num_attention_heads == model_hf.config.num_attention_heads
    assert model_hf_converted.config.intermediate_size == model_hf.config.intermediate_size
    assert model_hf_converted.config.vocab_size == model_hf.config.vocab_size

    # assert not hasattr(model_hf_converted.config, 'qkv_weight_interleaved')
    # assert not hasattr(model_hf_converted.config, 'encoder_activation')
    # assert not hasattr(model_hf_converted.config, 'attn_input_format')
    # assert not hasattr(model_hf_converted.config, 'fuse_qkv_params')
    # assert not hasattr(model_hf_converted.config, 'micro_batch_size')
    # assert not hasattr(model_hf_converted.config, 'max_seq_length')
