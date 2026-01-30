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

"""Simple vLLM inference example.

This script demonstrates vLLM's text generation API with a small decoder model.

Note: vLLM's transformers backend requires transformers>=5.0.0.dev0 for encoder-only
models (like ESM2, BERT). Until then, encoder models won't work with vLLM's
transformers backend regardless of whether they use TransformerEngine or not.
"""

import torch
from vllm import LLM, SamplingParams


if __name__ == "__main__":
    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)")

    # Use a small decoder model that vLLM natively supports
    # tensor_parallel_size splits the model across GPUs
    model = LLM(
        model="Qwen/Qwen2.5-0.5B",
        max_model_len=512,
        tensor_parallel_size=num_gpus,  # Use all available GPUs
    )

    prompts = [
        "The capital of France is",
        "Machine learning is",
    ]

    sampling_params = SamplingParams(max_tokens=32, temperature=0.7)
    outputs = model.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        generated = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}\n")

    # Cleanup: explicitly delete model to trigger proper shutdown
    del model

    # Cleanup distributed process group if initialized
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
