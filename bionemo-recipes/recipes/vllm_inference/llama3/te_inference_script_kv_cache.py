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

import torch
from convert import convert_llama_hf_to_te
from transformer_engine.pytorch.attention import InferenceParams
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFCompatibleInferenceParams(InferenceParams):
    """Thin wrapper that adds HuggingFace Cache interface compatibility.

    HF's generate() (transformers >= 5.0) calls cache.get_seq_length()
    during _prefill to determine how many tokens are already cached.
    InferenceParams doesn't implement this method, so we read from TE's
    internal self.sequences OrderedDict (populated by pre_step() during
    each forward pass).
    """

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the cached sequence length from TE's internal tracking."""
        if not self.sequences:
            return 0
        return next(iter(self.sequences.values()))


model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
model_te = convert_llama_hf_to_te(model_hf, attn_input_format="thd", self_attn_mask_type="padding_causal")
model_te.to("cuda")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

inputs = tokenizer("The quick brown fox", return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

past_key_values = HFCompatibleInferenceParams(
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
    output_ids = model_te.generate(
        **inputs,
        max_new_tokens=16,
        use_cache=True,
        past_key_values=past_key_values,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
