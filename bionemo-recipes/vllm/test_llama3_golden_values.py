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

"""Golden-value tests for the Llama-3 train-then-serve workflow.

Validates that a round-tripped (HF -> TE -> HF) checkpoint served via vLLM
produces the same outputs as the original HuggingFace model.

Three tests:
- **test_greedy_text_match**: greedy-decoded text must be identical.
- **test_logprob_similarity**: per-token log-probabilities must be close.
- **test_top_token_overlap**: top-K most likely tokens must overlap at every step.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


LLAMA3_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "llama3"
sys.path.insert(0, str(LLAMA3_MODEL_DIR))

from export_llama3 import HF_MODEL_ID, convert_te_to_vllm, create_te_checkpoint  # noqa: E402


PROMPTS = [
    "The quick brown fox",
    "In a hole in the ground there lived",
]
MAX_TOKENS = 16
TOP_K = 10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vllm_checkpoint(tmp_path_factory):
    """Create the round-tripped HF checkpoint (HF -> TE -> HF) once per module."""
    base_dir = tmp_path_factory.mktemp("llama3_export")
    te_path = create_te_checkpoint(base_dir)

    hf_path = base_dir / "hf_roundtrip"
    convert_te_to_vllm(te_path, hf_path)

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    tokenizer.save_pretrained(hf_path)

    return str(hf_path)


@pytest.fixture(scope="module")
def hf_reference_outputs():
    """Run HF reference model: greedy text + per-token log-probs for each prompt."""
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    results = []
    with torch.no_grad():
        for prompt in PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            output_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False, use_cache=False)
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            log_probs = []
            top_k_ids = []
            current_ids = inputs["input_ids"]
            for token_id in generated_ids:
                outputs = model(current_ids)
                logits = outputs.logits[0, -1, :]
                lp = F.log_softmax(logits.float(), dim=-1)
                log_probs.append(lp[token_id].cpu().item())
                top_k_ids.append(torch.topk(logits, TOP_K).indices.cpu().tolist())
                current_ids = torch.cat([current_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=1)

            results.append(
                {
                    "text": text,
                    "log_probs": np.array(log_probs),
                    "token_ids": generated_ids.cpu().tolist(),
                    "top_k_ids": top_k_ids,
                }
            )

    del model, tokenizer
    torch.cuda.empty_cache()
    return results


@pytest.fixture(scope="module")
def vllm_outputs(vllm_checkpoint):
    """Run vLLM on the exported checkpoint: greedy text + per-token log-probs + top-K."""
    engine = LLM(model=vllm_checkpoint, runner="generate", dtype="bfloat16")
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0, logprobs=TOP_K)
    raw_outputs = engine.generate(PROMPTS, params)

    results = []
    for prompt, output in zip(PROMPTS, raw_outputs):
        text = prompt + output.outputs[0].text
        token_ids = list(output.outputs[0].token_ids)
        log_probs = []
        top_k_ids = []
        for tid, step in zip(token_ids, output.outputs[0].logprobs):
            log_probs.append(step[tid].logprob)
            top_k_ids.append(sorted(step.keys(), key=lambda t: step[t].logprob, reverse=True))
        results.append(
            {
                "text": text,
                "log_probs": np.array(log_probs),
                "token_ids": token_ids,
                "top_k_ids": top_k_ids,
            }
        )

    del engine
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_greedy_text_match(hf_reference_outputs, vllm_outputs):
    """Greedy-decoded text from the round-tripped checkpoint must match the reference."""
    for i, prompt in enumerate(PROMPTS):
        hf_text = hf_reference_outputs[i]["text"]
        vllm_text = vllm_outputs[i]["text"]
        assert hf_text == vllm_text, f"Prompt {i} ({prompt!r}):\n  HF:   {hf_text!r}\n  vLLM: {vllm_text!r}"


def test_logprob_similarity(hf_reference_outputs, vllm_outputs):
    """Per-token log-probabilities must be close between HF reference and vLLM."""
    # bfloat16 numerical differences between vLLM's flash attention kernels
    # and PyTorch's native SDPA compound through 16 transformer layers.
    # Longer prompts accumulate more drift; 0.1 accommodates the worst case.
    atol = 0.1

    for i, prompt in enumerate(PROMPTS):
        hf_lp = hf_reference_outputs[i]["log_probs"]
        vllm_lp = vllm_outputs[i]["log_probs"]
        hf_ids = hf_reference_outputs[i]["token_ids"]
        vllm_ids = vllm_outputs[i]["token_ids"]

        n = min(len(hf_lp), len(vllm_lp))
        assert n > 0, f"Prompt {i}: no tokens generated"

        assert hf_ids[:n] == vllm_ids[:n], (
            f"Prompt {i} ({prompt!r}): token ID mismatch\n  HF:   {hf_ids[:n]}\n  vLLM: {vllm_ids[:n]}"
        )

        max_diff = float(np.abs(hf_lp[:n] - vllm_lp[:n]).max())
        mean_diff = float(np.abs(hf_lp[:n] - vllm_lp[:n]).mean())
        assert max_diff < atol, (
            f"Prompt {i} ({prompt!r}): log-prob max |diff| = {max_diff:.6f} exceeds atol={atol}\n"
            f"  HF log-probs:   {hf_lp[:n]}\n"
            f"  vLLM log-probs: {vllm_lp[:n]}"
        )
        print(f"  Prompt {i}: max |diff| = {max_diff:.6f}, mean |diff| = {mean_diff:.6f}")


def test_top_token_overlap(hf_reference_outputs, vllm_outputs):
    """Top-K most likely tokens must overlap at every generation step.

    Unlike atol-based log-prob checks, this is naturally robust to bfloat16
    numerical noise: small logit perturbations only affect the ranking at
    tie boundaries, so the top-K set is stable.
    """
    min_overlap = 0.9

    for i, prompt in enumerate(PROMPTS):
        hf_topk = hf_reference_outputs[i]["top_k_ids"]
        vllm_topk = vllm_outputs[i]["top_k_ids"]
        n = min(len(hf_topk), len(vllm_topk))

        for step in range(n):
            hf_set = set(hf_topk[step][:TOP_K])
            vllm_set = set(vllm_topk[step][:TOP_K])
            overlap = len(hf_set & vllm_set) / TOP_K
            assert overlap >= min_overlap, (
                f"Prompt {i} ({prompt!r}), step {step}: "
                f"top-{TOP_K} overlap = {overlap:.0%} < {min_overlap:.0%}\n"
                f"  HF top-{TOP_K}:   {sorted(hf_set)}\n"
                f"  vLLM top-{TOP_K}: {sorted(vllm_set)}"
            )

        overlaps = []
        for step in range(n):
            hf_set = set(hf_topk[step][:TOP_K])
            vllm_set = set(vllm_topk[step][:TOP_K])
            overlaps.append(len(hf_set & vllm_set) / TOP_K)
        print(f"  Prompt {i}: mean top-{TOP_K} overlap = {np.mean(overlaps):.0%}, min = {np.min(overlaps):.0%}")
