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
from transformers import AutoModelForMaskedLM
from vllm import LLM


models = {"nvidia": "nvidia/esm2_t6_8M_UR50D", "facebook": "facebook/esm2_t6_8M_UR50D"}

MODEL_SWITCH = "facebook"


def convert_to_hf(model):
    """Convert the given model to a HuggingFace AutoModelForMaskedLM."""
    model_hf = AutoModelForMaskedLM.from_pretrained(models[MODEL_SWITCH])
    return model_hf


if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)")

    model = LLM(
        model=models[MODEL_SWITCH],
        runner="pooling",
        trust_remote_code=True,
        dtype="float32",
        # TransformerEngine layers use pydantic (ArgsKwargs) which torch.compile
        # cannot trace. Use eager mode to avoid the dynamo error.
        enforce_eager=True,
        # vLLM's profiling run packs all tokens into a single batch-1 sequence.
        # Cap batched tokens to max_position_embeddings (1026) so the rotary
        # embeddings don't run out of positions.
        max_num_batched_tokens=1026,
    )

    prompts = [
        "LKGHAMCLGCLHMLMCGLLAGAMCGLMKLLKCCGKCLMHLMKAMLGLKCACHHHHLLLHACAAKKLCLGAKLAMGLKLLGAHGKGLKMACGHHMLHLHMH",
        "CLLCCMHMHAHHCHGHGHKCKCLMMGMALMCAGCCACGMKGGCHCCLLAHCAHAKAGKGKCKLMCKKKHGLHAGLHAMLLCHLGLGCGHHHKKCKKHKCA",
    ]

    outputs = model.embed(prompts)
    breakpoint()

    del model
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
