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

"""Direct vLLM gate for the projected plugin: payloads up to 49 soft tokens
accepted, wrong width/over-budget count rejected. Usage: python scripts/qualify_vllm_projected.py
"""

from __future__ import annotations

# The pinned NeMo RL stack pairs vLLM 0.25.1 with openai 2.6.1, whose
# responses types predate namespace tools; NeMo RL's own source-compat patch
# must land before anything imports vllm.tool_parsers (its generation workers
# do this internally; this script drives vLLM directly so it calls the hook
# itself). Safe on venvs where the hook is absent.
try:
    from nemo_rl.models.generation.vllm.patches import ensure_vllm_source_compat
except ImportError:
    pass
else:
    ensure_vllm_source_compat()

import torch  # noqa: I001
from llava_example import (  # noqa: I001
    CLIP_PATCH_TOKENS,
    NANO_4B,
    PLACEHOLDER_SPEC,
    PROJECTED_ARCHITECTURE,
    PROJECTOR_NAME,
    register_vllm,
)

register_vllm()  # the plugin must register before vllm resolves the architecture

from vllm import LLM, SamplingParams  # noqa: E402

HF_OVERRIDES = {
    "architectures": [PROJECTED_ARCHITECTURE],
    "mm_encoder_start": PLACEHOLDER_SPEC.start_token,
    "mm_encoder_placeholder": PLACEHOLDER_SPEC.placeholder_token,
    "mm_encoder_end": PLACEHOLDER_SPEC.end_token,
    "mm_encoder_max_tokens": CLIP_PATCH_TOKENS,
}


def main() -> None:
    """Acceptance probe: a well-formed payload generates, and wrong width or
    over-budget payloads are rejected by the plugin's geometry checks."""
    llm = LLM(
        model="artifacts/models/nemotron-nano-4b-bf16",
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.45,
        max_model_len=512,
        enable_mm_embeds=True,
        trust_remote_code=True,
        hf_overrides=HF_OVERRIDES,
        limit_mm_per_prompt={PROJECTOR_NAME: 1},
    )

    prompt = f"Describe the input.<{PROJECTOR_NAME}>"
    embeds = torch.randn(1, CLIP_PATCH_TOKENS, NANO_4B.hidden_size, dtype=torch.bfloat16)
    params = SamplingParams(max_tokens=8, temperature=0.0)

    out = llm.generate([{"prompt": prompt, "multi_modal_data": {PROJECTOR_NAME: embeds}}], params)[0].outputs[0]
    print("GENERATED:", repr(out.text))

    for name, bad in (
        ("wrong width", torch.randn(1, CLIP_PATCH_TOKENS, 64, dtype=torch.bfloat16)),
        ("over budget", torch.randn(1, CLIP_PATCH_TOKENS + 1, NANO_4B.hidden_size, dtype=torch.bfloat16)),
    ):
        try:
            llm.generate([{"prompt": prompt, "multi_modal_data": {PROJECTOR_NAME: bad}}], params)
        except Exception as error:  # noqa: BLE001 — qualification prints the rejection
            print(f"REJECTED ({name}):", type(error).__name__, str(error)[:120])
        else:
            raise SystemExit(f"FAIL: {name} payload was accepted")
    print("VLLM-PROJECTED-OK")


if __name__ == "__main__":
    main()
