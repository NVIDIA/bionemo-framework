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

import gc
import os

import pytest
import torch
from transformer_engine.pytorch.attention import InferenceParams
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithFlattening,
)

from convert import convert_llama_hf_to_te
from modeling_llama_te import HFInferenceParams, NVLlamaConfig, NVLlamaForCausalLM


@pytest.fixture
def input_text():
    return (
        """Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.""",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore "
        "et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip "
        "ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu "
        "fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
        "mollit anim id est laborum.",
    )


def test_llama_model_forward_pass(input_text):
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    config = NVLlamaConfig.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8", num_hidden_layers=2)
    model = NVLlamaForCausalLM(config)

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, padding_side="right")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    assert outputs.logits is not None
    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == config.num_hidden_layers + 1


def test_llama_model_forward_pass_thd_inputs(input_text):
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    config = NVLlamaConfig.from_pretrained(
        "nvidia/Llama-3.1-8B-Instruct-FP8",
        attn_input_format="thd",
        num_hidden_layers=2,
    )
    model = NVLlamaForCausalLM(config)

    inputs = [tokenizer(text) for text in input_text]
    data_collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)
    collated_inputs = data_collator(inputs)
    collated_inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in collated_inputs.items()}
    model.to("cuda")
    with torch.no_grad():
        outputs = model(**collated_inputs, output_hidden_states=True)

    assert outputs.logits is not None
    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == config.num_hidden_layers + 1


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skipping test in CI not download llama3 model.")
def test_llama_model_golden_values(input_text):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)

    model_te = convert_llama_hf_to_te(model_hf)

    tokenizer.pad_token = tokenizer.eos_token
    # TODO: figure out padding_side="left" with TE, make this several tests with different input types.
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, padding_side="right")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    labels = inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    model_hf.to("cuda")
    with torch.no_grad():
        outputs_hf = model_hf(**inputs, labels=labels, output_hidden_states=True)

    del model_hf
    gc.collect()
    torch.cuda.empty_cache()

    model_te.to("cuda")
    with torch.no_grad():
        outputs_te = model_te(**inputs, labels=labels, output_hidden_states=True)

    torch.testing.assert_close(outputs_te.loss, outputs_hf.loss, atol=5e-3, rtol=2e-3)
    torch.testing.assert_close(
        outputs_te.logits[inputs["attention_mask"].to(bool)],
        outputs_hf.logits[inputs["attention_mask"].to(bool)],
        atol=1.5,
        rtol=0.01,
    )


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skipping test in CI not download llama3 model.")
def test_llama_model_golden_values_thd_inputs(input_text):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
    model_te = convert_llama_hf_to_te(model_hf, attn_input_format="thd")
    model_te_bshd = convert_llama_hf_to_te(model_hf, attn_input_format="bshd")
    del model_hf

    tokenizer.pad_token = tokenizer.eos_token
    # TODO: figure out padding_side="left" with TE, make this several tests with different input types.
    inputs_bshd = tokenizer(input_text, return_tensors="pt", padding=True, padding_side="right")
    inputs_bshd = {k: v.to("cuda") for k, v in inputs_bshd.items()}
    labels = inputs_bshd["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    model_te_bshd.to("cuda")
    with torch.no_grad():
        outputs_bshd = model_te_bshd(**inputs_bshd, labels=labels, output_hidden_states=True)

    del model_te_bshd
    gc.collect()
    torch.cuda.empty_cache()

    inputs_thd = [tokenizer(text) for text in input_text]
    data_collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)
    collated_inputs = data_collator(inputs_thd)
    collated_inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in collated_inputs.items()}

    model_te.to("cuda")
    with torch.no_grad():
        outputs_thd = model_te(**collated_inputs, output_hidden_states=True)

    torch.testing.assert_close(outputs_thd.loss, outputs_bshd.loss, atol=5e-3, rtol=2e-3)
    torch.testing.assert_close(
        outputs_thd.logits,
        outputs_bshd.logits[inputs_bshd["attention_mask"].to(bool)],
        atol=1.0,
        rtol=0.01,
    )


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skipping test in CI not download llama3 model.")
def test_llama_model_golden_values_padding_left(input_text):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)

    model_te = convert_llama_hf_to_te(model_hf)

    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, padding_side="left")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    labels = inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    model_hf.to("cuda")
    with torch.no_grad():
        outputs_hf = model_hf(**inputs, labels=labels, output_hidden_states=True)

    del model_hf
    gc.collect()
    torch.cuda.empty_cache()

    model_te.to("cuda")
    with torch.no_grad():
        outputs_te = model_te(**inputs, labels=labels, output_hidden_states=True)

    torch.testing.assert_close(outputs_te.loss, outputs_hf.loss, atol=0.02, rtol=0.03)  # Higher than I'd like.
    torch.testing.assert_close(
        outputs_te.logits[inputs["attention_mask"].to(bool)],
        outputs_hf.logits[inputs["attention_mask"].to(bool)],
        atol=1.5,
        rtol=0.01,
    )


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skipping test in CI not download llama3 model.")
def test_hf_llama_model_generate_bshd():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)

    prompt = (
        """
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at""",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore",
    )

    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="left")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model_hf.to("cuda")

    with torch.no_grad():
        output_ids = model_hf.generate(**inputs, max_new_tokens=16, use_cache=False)

    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    assert "http://www.apache.org/licenses/LICENSE-2.0" in generated_text[0]
    assert "et dolore magna aliqua. Ut enim ad minim " in generated_text[1]


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skipping test in CI not download llama3 model.")
def test_te_llama_model_generate_with_cache():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
    model_te = convert_llama_hf_to_te(model_hf)

    prompt = """
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at"""

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model_te.to("cuda")

    past_key_values = InferenceParams(
        max_batch_size=1,
        max_sequence_length=256,
        num_heads_kv=model_te.config.num_key_value_heads,
        head_dim_k=model_te.config.hidden_size // model_te.config.num_attention_heads,
        dtype=torch.bfloat16,
        qkv_format="thd",
        max_ctx_len=256,
    )

    for layer_number in range(1, model_te.config.num_hidden_layers + 1):
        past_key_values.allocate_memory(layer_number)

    with torch.no_grad():
        output_ids = model_te.generate(**inputs, max_new_tokens=16, use_cache=True, past_key_values=past_key_values)

    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    assert "http://www.apache.org/licenses/LICENSE-2.0" in generated_text[0]


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skipping test in CI not download llama3 model.")
def test_te_llama_model_generate_with_cache_bshd():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
    model_te = convert_llama_hf_to_te(model_hf)

    prompt = (
        """
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at""",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore",
    )

    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="left")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model_te.to("cuda")

    past_key_values = InferenceParams(
        max_batch_size=2,
        max_sequence_length=256,
        num_heads_kv=model_te.config.num_key_value_heads,
        head_dim_k=model_te.config.hidden_size // model_te.config.num_attention_heads,
        dtype=torch.bfloat16,
        qkv_format="thd",
        max_ctx_len=256,
    )

    for layer_number in range(1, model_te.config.num_hidden_layers + 1):
        past_key_values.allocate_memory(layer_number)

    with torch.no_grad():
        output_ids = model_te.generate(
            **inputs,
            max_new_tokens=16,
            use_cache=True,
            past_key_values=past_key_values,
        )

    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    assert "http://www.apache.org/licenses/LICENSE-2.0" in generated_text[0]
    assert "et dolore magna aliqua. Ut enim ad minim " in generated_text[1]


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skipping test in CI not download llama3 model.")
def test_te_llama_model_generate_with_cache_bshd_beam_search():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
    model_te = convert_llama_hf_to_te(model_hf)

    prompt = (
        """
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at""",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore",
    )

    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="left")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model_te.to("cuda")

    num_beams = 2

    past_key_values = HFInferenceParams(
        max_batch_size=2 * num_beams,
        max_sequence_length=256,
        num_heads_kv=model_te.config.num_key_value_heads,
        head_dim_k=model_te.config.hidden_size // model_te.config.num_attention_heads,
        dtype=torch.bfloat16,
        qkv_format="thd",
        max_ctx_len=256,
    )

    for layer_number in range(1, model_te.config.num_hidden_layers + 1):
        past_key_values.allocate_memory(layer_number)

    with torch.no_grad():
        output_ids = model_te.generate(
            **inputs,
            max_new_tokens=16,
            use_cache=True,
            past_key_values=past_key_values,
            num_beams=num_beams,
            do_sample=True,
        )

    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    assert "http://www.apache.org/licenses/LICENSE-2.0" in generated_text[0]
    assert "et dolore magna aliqua. Ut enim ad minim " in generated_text[1]
