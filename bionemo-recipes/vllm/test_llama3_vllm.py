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

"""Quick smoke-test: load the round-tripped checkpoint in vLLM and generate text."""

from vllm import LLM, SamplingParams


if __name__ == "__main__":
    engine = LLM(
        model="./llama3_hf_roundtrip_checkpoint",
        runner="generate",
        dtype="bfloat16",
    )
    prompts = ["The quick brown fox"]
    sampling_params = SamplingParams(max_tokens=16)
    outputs = engine.generate(prompts, sampling_params)
    print(outputs[0].outputs[0].text)
