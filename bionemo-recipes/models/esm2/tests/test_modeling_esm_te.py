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

from unittest.mock import MagicMock

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForMaskedLM


def test_esm_model_for_masked_lm(input_data):
    from esm.modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM

    config = NVEsmConfig(**AutoConfig.from_pretrained("facebook/esm2_t6_8M_UR50D").to_dict())
    model = NVEsmForMaskedLM(config)
    model.to("cuda")
    input_data = {k: v.to("cuda") for k, v in input_data.items()}

    with torch.no_grad():
        outputs = model(**input_data)
        assert outputs.loss


def test_esm_model_has_all_te_layers(input_data):
    from esm.modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM

    config = NVEsmConfig(**AutoConfig.from_pretrained("facebook/esm2_t6_8M_UR50D").to_dict())
    model = NVEsmForMaskedLM(config)
    for name, module in model.named_modules():
        assert not isinstance(module, nn.Linear), f"Vanilla linear layer found in {name}"
        assert not isinstance(module, nn.LayerNorm), f"Vanilla LayerNorm layer found in {name}"


def test_convert_state_dict(input_data):
    from esm.convert import _pack_qkv_bias, _pack_qkv_weight, _pad_bias, _pad_weights, convert_esm_hf_to_te, mapping

    model_hf = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model_te = convert_esm_hf_to_te(model_hf)
    model_hf.to("cuda")
    model_te.to("cuda")
    input_data = {k: v.to("cuda") for k, v in input_data.items()}

    with torch.no_grad():
        outputs = model_te(**input_data)
        assert outputs.loss

    te_state_dict_keys = {
        k for k in model_te.state_dict().keys() if not k.endswith("_extra_state") and not k.endswith("inv_freq")
    }

    for k, v in mapping.items():
        if "*" in k:
            for i in range(model_hf.config.num_hidden_layers):
                k_sub = k.replace("*", str(i))
                v_sub = v.replace("*", str(i))
                torch.testing.assert_close(
                    model_te.state_dict()[v_sub],
                    model_hf.state_dict()[k_sub],
                    msg=lambda x: f"{k} {i} is not close: {x}",
                )
                te_state_dict_keys.remove(v_sub)
        else:
            torch.testing.assert_close(
                model_te.state_dict()[v],
                model_hf.state_dict()[k],
                msg=lambda x: f"{k} is not close: {x}",
            )
            te_state_dict_keys.remove(v)

    # # We untie these weights so we need to compare and remove manually
    # torch.testing.assert_close(
    #     model_te.state_dict()["lm_head.layer_norm_decoder.weight"],
    #     model_hf.state_dict()["lm_head.decoder.weight"],
    # )
    # te_state_dict_keys.remove("lm_head.layer_norm_decoder.weight")

    for i in range(model_hf.config.num_hidden_layers):
        k = f"esm.encoder.layers.{i}.self_attention.layernorm_qkv.weight"
        v = [
            f"esm.encoder.layer.{i}.attention.self.query.weight",
            f"esm.encoder.layer.{i}.attention.self.key.weight",
            f"esm.encoder.layer.{i}.attention.self.value.weight",
        ]

        ctx_mock = MagicMock()
        ctx_mock.target.config.num_attention_heads = model_hf.config.num_attention_heads

        packed_weight = _pack_qkv_weight.transform(
            ctx_mock,
            model_hf.state_dict()[v[0]],
            model_hf.state_dict()[v[1]],
            model_hf.state_dict()[v[2]],
        )

        torch.testing.assert_close(packed_weight, model_te.state_dict()[k])
        te_state_dict_keys.remove(k)

    for i in range(model_hf.config.num_hidden_layers):
        k = f"esm.encoder.layers.{i}.self_attention.layernorm_qkv.bias"
        v = [
            f"esm.encoder.layer.{i}.attention.self.query.bias",
            f"esm.encoder.layer.{i}.attention.self.key.bias",
            f"esm.encoder.layer.{i}.attention.self.value.bias",
        ]

        ctx_mock = MagicMock()
        ctx_mock.target.config.num_attention_heads = model_hf.config.num_attention_heads

        packed_weight = _pack_qkv_bias.transform(
            ctx_mock,
            model_hf.state_dict()[v[0]],
            model_hf.state_dict()[v[1]],
            model_hf.state_dict()[v[2]],
        )

        torch.testing.assert_close(packed_weight, model_te.state_dict()[k])
        te_state_dict_keys.remove(k)

    ctx_mock = MagicMock()
    ctx_mock.target.config.padded_vocab_size = model_te.config.padded_vocab_size

    torch.testing.assert_close(
        _pad_weights(ctx_mock, model_hf.state_dict()["esm.embeddings.word_embeddings.weight"]),
        model_te.state_dict()["esm.embeddings.word_embeddings.weight"],
    )
    torch.testing.assert_close(
        _pad_weights(ctx_mock, model_hf.state_dict()["lm_head.decoder.weight"]),
        model_te.state_dict()["lm_head.decoder.weight"],
    )
    torch.testing.assert_close(
        _pad_bias.transform(ctx_mock, model_hf.state_dict()["lm_head.bias"]),
        model_te.state_dict()["lm_head.decoder.bias"],
    )

    te_state_dict_keys.remove("esm.embeddings.word_embeddings.weight")
    te_state_dict_keys.remove("lm_head.decoder.weight")
    te_state_dict_keys.remove("lm_head.decoder.bias")

    assert len(te_state_dict_keys) == 0

    # Check that the tied weights are the same
    assert (
        model_hf.state_dict()["esm.embeddings.word_embeddings.weight"].data_ptr()
        == model_hf.state_dict()["lm_head.decoder.weight"].data_ptr()
    )

    assert (
        model_te.state_dict()["esm.embeddings.word_embeddings.weight"].data_ptr()
        == model_te.state_dict()["lm_head.decoder.weight"].data_ptr()
    )


def test_golden_values(input_data):
    from esm.convert import convert_esm_hf_to_te

    model_hf = AutoModelForMaskedLM.from_pretrained(
        "facebook/esm2_t6_8M_UR50D", attn_implementation="flash_attention_2"
    )
    model_te = convert_esm_hf_to_te(model_hf)
    model_te.to(torch.bfloat16)
    model_hf.to(torch.bfloat16)

    model_te.to("cuda")
    model_hf.to("cuda")
    input_data = {k: v.to("cuda") for k, v in input_data.items()}

    with torch.no_grad():
        te_outputs = model_te(**input_data, output_hidden_states=True)
        hf_outputs = model_hf(**input_data, output_hidden_states=True)

    torch.testing.assert_close(te_outputs.loss, hf_outputs.loss, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(
        te_outputs.logits[input_data["attention_mask"].to(bool)],
        hf_outputs.logits[input_data["attention_mask"].to(bool)],
        atol=2,  # This seems high, needed to increase after https://github.com/huggingface/transformers/pull/40370
        rtol=1e-4,
    )


def test_converted_model_roundtrip(tmp_path, input_data):
    from transformer_engine.pytorch import TransformerLayer

    from esm.convert import convert_esm_hf_to_te
    from esm.modeling_esm_te import NVEsmConfig, NVEsmEncoder, NVEsmForMaskedLM, NVEsmLMHead, NVEsmModel

    model_hf = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model_te = convert_esm_hf_to_te(model_hf)

    model_te.save_pretrained(tmp_path / "esm2_t6_8M_UR50D_te")
    del model_te

    model_te = NVEsmForMaskedLM.from_pretrained(tmp_path / "esm2_t6_8M_UR50D_te")

    # Ensure our custom classes are still there
    assert isinstance(model_te, NVEsmForMaskedLM)
    assert isinstance(model_te.config, NVEsmConfig)
    assert isinstance(model_te.lm_head, NVEsmLMHead)
    assert isinstance(model_te.esm, NVEsmModel)
    assert isinstance(model_te.esm.encoder, NVEsmEncoder)
    assert isinstance(model_te.esm.encoder.layers[0], TransformerLayer)
    assert model_te.config.model_type == "nv_esm"

    model_te.to("cuda")
    model_hf.to("cuda")
    input_data = {k: v.to("cuda") for k, v in input_data.items()}

    with torch.no_grad():
        te_outputs = model_te(**input_data, output_hidden_states=True)
        hf_outputs = model_hf(**input_data, output_hidden_states=True)

    torch.testing.assert_close(te_outputs.loss, hf_outputs.loss, atol=1e-1, rtol=1e-3)
