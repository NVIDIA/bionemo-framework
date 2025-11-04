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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from convert import convert_llama_hf_to_te
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM


@pytest.fixture
def input_text():
    return """Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License."""


def test_llama_model_forward_pass(input_text):
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    config = NVLlamaConfig.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8", num_hidden_layers=2)
    model = NVLlamaForCausalLM(config)

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    assert outputs.logits is not None
    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == config.num_hidden_layers + 1


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skipping test in CI not download llama3 model.")
def test_llama_model_golden_values(input_text):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)

    model_te = convert_llama_hf_to_te(model_hf)

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model_hf.to("cuda")
    with torch.no_grad():
        outputs_hf = model_hf(**inputs, labels=inputs["input_ids"], output_hidden_states=True)

    del model_hf
    gc.collect()
    torch.cuda.empty_cache()

    model_te.to("cuda")
    with torch.no_grad():
        outputs_te = model_te(**inputs, labels=inputs["input_ids"], output_hidden_states=True)

    torch.testing.assert_close(outputs_te.loss, outputs_hf.loss, atol=5e-3, rtol=2e-3)
    torch.testing.assert_close(outputs_te.logits, outputs_hf.logits, atol=1.0, rtol=0.01)


def test_llama_model_can_generate():
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    config = NVLlamaConfig.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8", num_hidden_layers=2)
    model = NVLlamaForCausalLM(config)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda")
    generator("Hello, how are you?", max_new_tokens=16)


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skipping test in CI not download llama3 model.")
def test_hf_llama_model_generate_golden_values():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)

    prompt = """
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at"""

    # TODO: this doesn't work with the te model?
    generator = pipeline("text-generation", model=model_hf, tokenizer=tokenizer, device="cuda")

    outputs = generator(prompt, max_new_tokens=16)
    assert "http://www.apache.org/licenses/LICENSE-2.0" in outputs[0]["generated_text"]


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skipping test in CI not download llama3 model.")
@pytest.mark.xfail(reason="This doesn't work with the te model?")
def test_te_llama_model_generate_golden_values():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
    model_te = convert_llama_hf_to_te(model_hf)

    prompt = """
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at"""

    # TODO: this doesn't work with the te model?
    generator = pipeline("text-generation", model=model_te, tokenizer=tokenizer, device="cuda")

    outputs = generator(prompt, max_new_tokens=16)
    assert "http://www.apache.org/licenses/LICENSE-2.0" in outputs[0]["generated_text"]
