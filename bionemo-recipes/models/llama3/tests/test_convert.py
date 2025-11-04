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

import logging

import torch
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from convert import convert_llama_hf_to_te, convert_llama_te_to_hf
from modeling_llama_te import NVLlamaForCausalLM


def test_convert_llama_hf_to_te_roundtrip(caplog):
    # Here we use a randomly-initialized model just to test the conversion, this avoids downloading the model from
    # Hugging Face during CI.
    config = AutoConfig.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8", dtype=torch.float32)
    config.num_hidden_layers = 1  # Just to make this faster.
    model_hf = LlamaForCausalLM(config)

    # Alternatively, we can use a model from Hugging Face, but this requires authenticating.
    # model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    with caplog.at_level(logging.WARNING):
        model_te = convert_llama_hf_to_te(model_hf)

        # Assert no warnings were logged
        assert not any(record.levelno == logging.WARNING for record in caplog.records), (
            f"Unexpected warnings logged: {[r.message for r in caplog.records if r.levelno == logging.WARNING]}"
        )

    with caplog.at_level(logging.WARNING):
        model_hf_converted = convert_llama_te_to_hf(model_te)

        # Assert no warnings were logged
        assert not any(record.levelno == logging.WARNING for record in caplog.records), (
            f"Unexpected warnings logged: {[r.message for r in caplog.records if r.levelno == logging.WARNING]}"
        )

    original_state_dict = model_hf.state_dict()
    converted_state_dict = model_hf_converted.state_dict()
    original_keys = set(original_state_dict.keys())
    converted_keys = set(converted_state_dict.keys())
    assert original_keys == converted_keys

    torch.testing.assert_close(
        model_hf.model.rotary_emb.inv_freq, model_hf_converted.model.rotary_emb.inv_freq, atol=1e-8, rtol=1e-8
    )

    for key in original_state_dict.keys():
        torch.testing.assert_close(original_state_dict[key], converted_state_dict[key], atol=1e-8, rtol=1e-8)


def test_convert_hf_to_te_with_bf16():
    config = AutoConfig.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8", dtype=torch.bfloat16, num_hidden_layers=2)
    model_hf = LlamaForCausalLM(config)
    model_hf.to(dtype=torch.bfloat16)  # I think the original llama3 model doesn't initialize in bf16.
    convert_llama_hf_to_te(model_hf)


def test_convert_te_to_hf_with_bf16():
    config = AutoConfig.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8", dtype=torch.bfloat16, num_hidden_layers=2)
    model_te = NVLlamaForCausalLM(config)
    model_te.to(dtype=torch.float32)  # I think the original llama3 model doesn't initialize in bf16.
    convert_llama_te_to_hf(model_te)
