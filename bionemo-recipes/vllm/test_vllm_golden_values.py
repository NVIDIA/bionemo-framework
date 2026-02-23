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

"""End-to-end golden-value test for ESM2 vLLM compatibility.

Performs a fresh facebook -> TE export, then cross-validates embeddings across
three backends on the same protein sequences:

1. **vLLM**          - freshly exported model loaded via ``LLM(runner="pooling")``.
2. **HF (exported)** - same exported checkpoint loaded via ``AutoModel``.
3. **HF (reference)**- nvidia Hub model loaded via ``AutoModel`` (ground truth).

vLLM's pooling runner returns *last-token, L2-normalised* embeddings by default,
so the HuggingFace runs replicate that post-processing for an apples-to-apples comparison.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from vllm import LLM


# ---- Fresh export ----
# The export script uses relative paths (modeling_esm_te.py, esm_fast_tokenizer, etc.)
# so we need to run it from the esm2 model directory.
ESM2_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "esm2"
EXPORT_DIR = Path(__file__).resolve().parent / "exported_checkpoint"
EXPORT_TAG = "esm2_t6_8M_UR50D"

sys.path.insert(0, str(ESM2_MODEL_DIR))


def fresh_export() -> str:
    """Run the full facebook -> TE export and return the path to the exported checkpoint."""
    from export import export_hf_checkpoint

    # export_hf_checkpoint uses relative paths, so temporarily chdir
    original_cwd = os.getcwd()
    os.chdir(ESM2_MODEL_DIR)
    try:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Exporting facebook/{EXPORT_TAG} -> {EXPORT_DIR / EXPORT_TAG}")
        export_hf_checkpoint(EXPORT_TAG, EXPORT_DIR)
    finally:
        os.chdir(original_cwd)

    return str(EXPORT_DIR / EXPORT_TAG)


# ---- Configuration ----
REFERENCE_MODEL_ID = "nvidia/esm2_t6_8M_UR50D"

SEQUENCES = [
    "LKGHAMCLGCLHMLMCGLLAGAMCGLMKLLKCCGKCLMHLMKAMLGLKCACHHHHLLLHACAAKKLCLGAKLAMGLKLLGAHGKGLKMACGHHMLHLHMH",
    "CLLCCMHMHAHHCHGHGHKCKCLMMGMALMCAGCCACGMKGGCHCCLLAHCAHAKAGKGKCKLMCKKKHGLHAGLHAMLLCHLGLGCGHHHKKCKKHKCA",
]

RTOL, ATOL = 0, 0


# ---- Helpers ----


def last_token_l2(hidden_state: torch.Tensor) -> np.ndarray:
    """Extract last-token hidden state and L2-normalise (matches vLLM pooling defaults)."""
    vec = hidden_state[0, -1, :].cpu().float().numpy()
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec = vec / norm
    return vec


def hf_embed(model_id: str, sequences: list[str], dtype=torch.float32) -> np.ndarray:
    """Run HuggingFace inference and return last-token L2-normalised embeddings."""
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to("cuda", dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    vecs = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            out = model(**inputs)
            vecs.append(last_token_l2(out.last_hidden_state))

    del model, tokenizer
    torch.cuda.empty_cache()
    return np.stack(vecs)


def vllm_embed(model_id: str, sequences: list[str]) -> np.ndarray:
    """Run vLLM pooling inference and return embeddings."""
    engine = LLM(
        model=model_id,
        runner="pooling",
        trust_remote_code=True,
        dtype="float32",
        enforce_eager=True,
        max_num_batched_tokens=1026,
    )
    outputs = engine.embed(sequences)

    vecs = []
    for output in outputs:
        emb = output.outputs.embedding
        if isinstance(emb, list):
            emb = np.array(emb)
        vecs.append(emb)

    del engine
    return np.stack(vecs)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Element-wise maximum absolute difference between two arrays."""
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity across rows."""
    sims = []
    for va, vb in zip(a, b):
        dot = np.dot(va, vb)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        sims.append(dot / max(na * nb, 1e-12))
    return float(np.mean(sims))


# ---- Main ----

if __name__ == "__main__":
    print(f"GPUs: {torch.cuda.device_count()}")

    # Step 0: fresh export (facebook HF -> our TE format)
    print("\n[0/3] Exporting checkpoint ...")
    MODEL_ID = fresh_export()

    print(f"MODEL_ID:           {MODEL_ID}")
    print(f"REFERENCE_MODEL_ID: {REFERENCE_MODEL_ID}")
    print(f"Sequences:          {len(SEQUENCES)}")

    # 1) vLLM on exported model
    print("\n[1/3] vLLM inference (exported model) ...")
    emb_vllm = vllm_embed(MODEL_ID, SEQUENCES)

    # 2) HuggingFace on exported model
    print("\n[2/3] HuggingFace inference (exported model) ...")
    emb_hf_exported = hf_embed(MODEL_ID, SEQUENCES)

    # 3) HuggingFace on reference Hub model
    print("\n[3/3] HuggingFace inference (reference model) ...")
    emb_hf_reference = hf_embed(REFERENCE_MODEL_ID, SEQUENCES)

    # ---- Pairwise comparisons ----
    pairs = [
        ("vLLM (exported)", "HF (exported)", emb_vllm, emb_hf_exported),
        ("vLLM (exported)", "HF (reference)", emb_vllm, emb_hf_reference),
        ("HF (exported)", "HF (reference)", emb_hf_exported, emb_hf_reference),
    ]

    # ---- Summary table ----
    header = f"{'Pair':<35} {'max |diff|':>14} {'mean |diff|':>14} {'cos sim':>12} {'exact':>7}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for name_a, name_b, a, b in pairs:
        diffs = np.abs(a.astype(np.float64) - b.astype(np.float64))
        label = f"{name_a}  vs  {name_b}"
        exact = np.array_equal(a, b)
        print(
            f"{label:<35} {diffs.max():>14.8e} {diffs.mean():>14.8e} "
            f"{cosine_sim(a, b):>12.10f} {'YES' if exact else 'NO':>7}"
        )

    print(sep)
    print(f"Tolerance: rtol={RTOL}, atol={ATOL} (0 = exact match required)")

    # Per-sequence breakdown
    short = {"vLLM (exported)": "vllm", "HF (exported)": "hf_exp", "HF (reference)": "hf_ref"}
    print("\nPer-sequence max |diff|:")
    for i in range(len(SEQUENCES)):
        row = f"  seq {i}:"
        for name_a, name_b, a, b in pairs:
            d = float(np.abs(a[i].astype(np.float64) - b[i].astype(np.float64)).max())
            row += f"  {short[name_a]}_vs_{short[name_b]}={d:.8e}"
        print(row)

    print(sep)

    # Cleanup
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
