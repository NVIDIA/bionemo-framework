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


import torch
from transformers import AutoModelForMaskedLM


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

    original_config_dict = model_hf.config.to_dict()
    converted_config_dict = model_hf_converted.config.to_dict()

    for key, value in original_config_dict.items():
        assert key in converted_config_dict, f"Config field '{key}' missing in converted model"
        assert converted_config_dict[key] == value, (
            f"Config field '{key}' differs: original={value}, converted={converted_config_dict[key]}"
        )

    assert model_hf_converted.config.model_type == "esm"

    te_specific_fields = [
        "qkv_weight_interleaved",
        "encoder_activation",
        "attn_input_format",
        "fuse_qkv_params",
        "micro_batch_size",
        "auto_map",
    ]
    for field in te_specific_fields:
        assert not hasattr(model_hf_converted.config, field), (
            f"TE-specific field '{field}' should not be present in converted model"
        )


def test_padding_unpadding_operations():
    """Test that padding and unpadding operations work correctly for embeddings and decoder weights."""
    from esm.convert import convert_esm_hf_to_te, convert_esm_te_to_hf

    model_hf = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model_te = convert_esm_hf_to_te(model_hf)
    model_hf_converted = convert_esm_te_to_hf(model_te)

    # Test word embeddings
    original_embeddings = model_hf.state_dict()["esm.embeddings.word_embeddings.weight"]
    converted_embeddings = model_hf_converted.state_dict()["esm.embeddings.word_embeddings.weight"]
    assert original_embeddings.shape == converted_embeddings.shape, (
        f"Embedding shapes don't match: {original_embeddings.shape} vs {converted_embeddings.shape}"
    )
    torch.testing.assert_close(original_embeddings, converted_embeddings, atol=1e-5, rtol=1e-5)

    # Test decoder weights
    original_decoder = model_hf.state_dict()["lm_head.decoder.weight"]
    converted_decoder = model_hf_converted.state_dict()["lm_head.decoder.weight"]
    assert original_decoder.shape == converted_decoder.shape, (
        f"Decoder shapes don't match: {original_decoder.shape} vs {converted_decoder.shape}"
    )
    torch.testing.assert_close(original_decoder, converted_decoder, atol=1e-5, rtol=1e-5)

    # Test bias
    original_bias = model_hf.state_dict()["lm_head.bias"]
    converted_bias = model_hf_converted.state_dict()["lm_head.bias"]
    assert original_bias.shape == converted_bias.shape, (
        f"Bias shapes don't match: {original_bias.shape} vs {converted_bias.shape}"
    )
    torch.testing.assert_close(original_bias, converted_bias, atol=1e-5, rtol=1e-5)

    # Test that TE model has padded dimensions
    te_embeddings = model_te.state_dict()["esm.embeddings.word_embeddings.weight"]
    te_decoder = model_te.state_dict()["lm_head.decoder.weight"]
    assert te_embeddings.shape[0] >= original_embeddings.shape[0], "TE embeddings should be padded"
    assert te_decoder.shape[0] >= original_decoder.shape[0], "TE decoder should be padded"

    # The padded parts should be zeros (for embeddings) or min values (for bias)
    if te_embeddings.shape[0] > original_embeddings.shape[0]:
        padding_rows = te_embeddings[original_embeddings.shape[0] :]
        torch.testing.assert_close(padding_rows, torch.zeros_like(padding_rows), atol=1e-6, rtol=1e-6)


def test_weight_initialization_matches_hf():
    from transformers import AutoConfig
    from transformers.models.esm.modeling_esm import EsmForMaskedLM

    from esm.convert import convert_esm_hf_to_te
    from esm.modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM

    config_hf = AutoConfig.from_pretrained("facebook/esm2_t6_8M_UR50D", vocab_size=64)
    model_hf = EsmForMaskedLM(config_hf)
    model_te_converted = convert_esm_hf_to_te(model_hf)

    config = NVEsmConfig(**model_hf.config.to_dict())
    model_te = NVEsmForMaskedLM(config)
    model_te.to("cuda")
    model_te_converted.to("cuda")

    state_dict_hf = model_te_converted.state_dict()
    state_dict_te = model_te.state_dict()

    for name in state_dict_hf.keys():
        if name.endswith("_extra_state"):
            continue

        torch.testing.assert_close(
            state_dict_hf[name].mean(),
            state_dict_te[name].mean(),
            atol=1e-2,
            rtol=1e-4,
            msg=lambda x: f"Mean mismatch for parameter {name}: {x}",
        )

        torch.testing.assert_close(
            state_dict_hf[name].std(),
            state_dict_te[name].std(),
            atol=1e-2,
            rtol=1e-4,
            msg=lambda x: f"Std mismatch for parameter {name}: {x}",
        )
