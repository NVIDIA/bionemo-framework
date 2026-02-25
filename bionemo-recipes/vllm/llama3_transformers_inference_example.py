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

"""Quick smoke-test: load the round-tripped checkpoint and generate text."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "./llama3_hf_roundtrip_checkpoint"

model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, torch_dtype=torch.bfloat16)
model.to("cuda")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer("The quick brown fox", return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=16, use_cache=False)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
