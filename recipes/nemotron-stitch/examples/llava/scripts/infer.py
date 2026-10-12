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

"""Infer the trained model on any image: encode it online with the pinned
frozen CLIP (the same encoder ``llava_example.py`` ran over the training data),
project to soft tokens with the trained projector, merge the trained LoRA
adapter into the base snapshot — the same merge GRPO's refit performs in
memory at every update — and serve the merged weights with the example's
vLLM plugin.

Runs in the vLLM worker venv (torch + transformers + safetensors; no peft):

    /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python \
        scripts/infer.py --adapter outputs/grpo/checkpoints \
            --input path/to/image.jpg --question "How many objects are in the image?"

``--adapter`` may be the PEFT directory itself (SFT:
``outputs/sft/checkpoints/epoch_<K>_step_<N>/model``; GRPO:
``outputs/grpo/checkpoints/step_<N>/policy/weights``) or any ancestor of one —
with several checkpoints below it, the highest step wins and the selection is
printed. The merge holds the 4B's ~8 GiB of bf16 weights in host memory.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import torch
from llava_example import (
    CLIP_PATCH_TOKENS,
    PLACEHOLDER_SPEC,
    PROJECTED_ARCHITECTURE,
    PROJECTOR_NAME,
    build_encoder,
    load_input,
    register_vllm,
)

# Same hf_overrides as configs/grpo.yaml's policy.hf_config_overrides (and the
# probe script): the projected architecture plus the sentinel geometry.
HF_OVERRIDES = {
    "architectures": [PROJECTED_ARCHITECTURE],
    "mm_encoder_start": PLACEHOLDER_SPEC.start_token,
    "mm_encoder_placeholder": PLACEHOLDER_SPEC.placeholder_token,
    "mm_encoder_end": PLACEHOLDER_SPEC.end_token,
    "mm_encoder_max_tokens": CLIP_PATCH_TOKENS,
}


def find_adapter(root: Path) -> Path:
    """Locate the PEFT adapter directory under ``root``.

    Several checkpoints below ``root`` resolve to the highest step number in
    their path (the latest adapter); unparseable step numbers fail closed.
    """
    candidates = sorted(root.rglob("adapter_model.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"no adapter_model.safetensors under {root}")

    def step(path: Path) -> int:
        """The highest ``step_<N>`` component in the path — the latest
        checkpoint. A path with no step number is a contract violation (the
        trainer's checkpoint layout always carries one), so it fails closed."""
        steps = [int(match.group(1)) for part in path.parts if (match := re.search(r"step_(\d+)", part))]
        if not steps:
            raise ValueError(f"no step number in checkpoint path: {path}")
        return max(steps)

    selected = max(candidates, key=step).parent
    if len(candidates) > 1:
        print(f"selected {selected} (latest of {len(candidates)} adapters under {root})")
    return selected


def merge_lora(base: Path, adapter_dir: Path, out: Path) -> None:
    """``W += (alpha / r) * B @ A`` on the adapter's named base weights.

    Mirrors NeMo RL's refit-time merge (``_maybe_merge_lora_weight``): A and B
    are cast to the base dtype before the matmul. Pure safetensors — no model
    instantiation, no peft.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    if config.get("peft_type") != "LORA":
        raise ValueError(f"unsupported adapter type: {config.get('peft_type')!r}")
    scale = float(config["lora_alpha"]) / float(config["r"])

    # Adapter keys are PEFT's: base_model.model.<module FQN>.lora_{A,B}.weight;
    # the merged base tensor is <module FQN>.weight.
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            module, dot, kind = key.removeprefix("base_model.model.").rpartition(".lora_")
            if not dot or kind not in ("A.weight", "B.weight"):
                raise ValueError(f"unexpected adapter tensor name: {key!r}")
            pairs.setdefault(f"{module}.weight", {})[kind[0]] = handle.get_tensor(key)

    tensors: dict[str, torch.Tensor] = {}
    shards = sorted(base.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards in {base}")
    for shard in shards:
        with safe_open(str(shard), framework="pt") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)

    for fqn, pair in sorted(pairs.items()):
        if fqn not in tensors:
            raise KeyError(f"adapter target {fqn!r} is not a base weight")
        if set(pair) != {"A", "B"}:
            raise ValueError(f"incomplete LoRA pair for {fqn!r}: found {sorted(pair)}")
        base_weight = tensors[fqn]
        delta = pair["B"].to(base_weight.dtype) @ pair["A"].to(base_weight.dtype)
        tensors[fqn] = (base_weight + delta * scale).contiguous()
    print(f"merged {len(pairs)} LoRA pairs from {adapter_dir} (scale {scale:g})")

    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    for entry in base.iterdir():
        # The single merged shard supersedes the base's shards and index;
        # dotfiles are hf download's .cache metadata (--local-dir layout).
        if entry.name.startswith(".") or entry.name.endswith(".safetensors") or entry.name.endswith(".index.json"):
            continue
        if entry.is_file():
            shutil.copy2(entry, out / entry.name)


def main() -> None:
    """Orchestrate: select and merge the adapter, encode the image with the
    pinned frozen CLIP, project with the trained projector, then serve the
    merged weights through the example's vLLM plugin and generate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="artifacts/models/nemotron-nano-4b-bf16")
    parser.add_argument("--adapter", required=True, help="PEFT adapter directory (SFT or GRPO), or an ancestor of it")
    parser.add_argument(
        "--projector", default="outputs/sft/projector", help="the SFT-exported projector (final trained state)"
    )
    parser.add_argument(
        "--input", required=True, help="any input path the container can read (outputs/ is host-mounted)"
    )
    parser.add_argument("--question", default="Describe the image.")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    base = Path(args.base)
    adapter_dir = find_adapter(Path(args.adapter))
    merged = Path(tempfile.mkdtemp(prefix="llava-example-merged-"))
    try:
        merge_lora(base, adapter_dir, merged)

        import numpy as np
        from transformers import AutoTokenizer

        from nemotron_stitch.nemo_rl.data import prepend_bos_if_needed
        from nemotron_stitch.nemo_rl.transport import project_features

        encode = build_encoder("cpu")
        features = torch.from_numpy(np.asarray(encode(load_input(args.input))))
        projected = project_features(features, args.projector, PROJECTOR_NAME)

        tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
        # Byte-identical to the GRPO rollout prompt (encoder_rl_processor).
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"<{PROJECTOR_NAME}>\n" + args.question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt = prepend_bos_if_needed(prompt, tokenizer, add_bos=True)

        register_vllm()  # the plugin must register before vllm resolves the architecture
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(merged),
            dtype="bfloat16",
            enforce_eager=True,
            gpu_memory_utilization=0.45,
            max_model_len=512,
            enable_mm_embeds=True,
            trust_remote_code=True,
            hf_overrides=HF_OVERRIDES,
            limit_mm_per_prompt={PROJECTOR_NAME: 1},
        )
        output = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {PROJECTOR_NAME: projected.unsqueeze(0)}}],
            SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
        )[0].outputs[0]
        print(f"PROMPT: {args.question}")
        print(f"GENERATED: {output.text!r}")
    finally:
        shutil.rmtree(merged, ignore_errors=True)


if __name__ == "__main__":
    main()
