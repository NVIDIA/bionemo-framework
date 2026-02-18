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

"""vLLM ESM2 embedding example -- loads directly from HuggingFace.

The NVEsm model on HuggingFace follows the standard HuggingFace convention where
wrapper classes (NVEsmForMaskedLM, NVEsmForTokenClassification) store the base
model as ``self.model``, with ``base_model_prefix = "model"``.

This means checkpoint weight keys align with vLLM's TransformersForEmbedding
wrapper out of the box:

- Checkpoint bare keys: ``embeddings.*``, ``encoder.*``
- vLLM mapper adds ``model.``: ``model.embeddings.*``, ``model.encoder.*``
- Wrapper module tree:          ``model.embeddings.*``, ``model.encoder.*``

No conversion scripts or weight renaming needed.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from vllm import LLM


# MODEL_ID = "nvidia/esm2_t6_8M_UR50D"
# To test with a local re-exported checkpoint before pushing to HuggingFace, use a path:
MODEL_ID = "/workspace/bionemo-framework/bionemo-recipes/models/esm2/exported/esm2_t6_8M_UR50D"  # after running export from models/esm2

# Reference: nvidia model on HuggingFace Hub (same as MODEL_ID when using Hub) — check local/conversion against it.
REFERENCE_MODEL_ID = "nvidia/esm2_t6_8M_UR50D"


if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)")

    # Load ESM2 directly from HuggingFace as a pooling/embedding model.
    # No checkpoint conversion needed -- the model code and checkpoint are
    # aligned so that vLLM's generic weight mapper works out of the box.
    print(f"\nLoading model: {MODEL_ID}")
    model = LLM(
        model=MODEL_ID,
        runner="pooling",
        trust_remote_code=True,
        # TransformerEngine layers use pydantic (ArgsKwargs) which torch.compile
        # cannot trace. Use eager mode to avoid the dynamo error.
        enforce_eager=True,
        # vLLM's profiling run packs all tokens into a single batch-1 sequence.
        # Cap batched tokens to max_position_embeddings (1026) so the rotary
        # embeddings don't run out of positions.
        max_num_batched_tokens=1026,
    )

    # Example protein sequences
    prompts = [
        "LKGHAMCLGCLHMLMCGLLAGAMCGLMKLLKCCGKCLMHLMKAMLGLKCACHHHHLLLHACAAKKLCLGAKLAMGLKLLGAHGKGLKMACGHHMLHLHMH",
        "CLLCCMHMHAHHCHGHGHKCKCLMMGMALMCAGCCACGMKGGCHCCLLAHCAHAKAGKGKCKLMCKKKHGLHAGLHAMLLCHLGLGCGHHHKKCKKHKCA",
    ]

    print(f"\nGenerating embeddings for {len(prompts)} sequences...")
    outputs = model.embed(prompts)

    # Collect vLLM embeddings for comparison (one vector per sequence from pooling)
    vllm_embeddings = []
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        embedding = output.outputs.embedding
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        vllm_embeddings.append(embedding)
        print(f"\nSequence {i + 1}:")
        print(f"  Length: {len(prompt)} amino acids")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  First 5 dims: {embedding[:5].tolist()}")

    vllm_embeddings = np.stack(vllm_embeddings)

    print("\nSUCCESS: ESM2 embeddings generated with vLLM!")

    # ---- Native HuggingFace inference on the same sequences ----
    # Run the same model via transformers and compare outputs to vLLM.
    print(f"\n--- HuggingFace (native) inference on same {len(prompts)} sequences ---")
    hf_model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Use float16 to match vLLM's default dtype for numerical comparison
    hf_model = hf_model.to("cuda", dtype=torch.float16)
    hf_model.eval()

    # Exported checkpoints use add_pooling_layer=False (no pooler weights).
    # vLLM pooling runner uses seq_pooling_type='LAST' (last token). Use pooler_output when
    # present, else take the last non-padding token's hidden state to match vLLM.
    hf_embeddings_list = []
    with torch.no_grad():
        for seq in prompts:
            inputs = tokenizer(seq, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            out = hf_model(**inputs)
            if out.pooler_output is not None:
                vec = out.pooler_output.cpu().numpy().squeeze(0)
            else:
                # Last-token hidden state to match vLLM's seq_pooling_type='LAST'
                last_hidden = out.last_hidden_state  # (1, seq_len, hidden_size)
                vec = last_hidden[0, -1, :].cpu().float().numpy()
                # vLLM pooling runner L2-normalizes the embedding; do the same for comparison
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
            hf_embeddings_list.append(vec)

    hf_embeddings = np.stack(hf_embeddings_list)

    # Compare vLLM vs HuggingFace (same sequences, same model).
    # Use relaxed tolerance for fp16/FlashAttention vs fp32/PyTorch attention differences.
    rtol, atol = 1e-2, 5e-4
    match = np.allclose(vllm_embeddings, hf_embeddings, rtol=rtol, atol=atol)
    max_diff = np.abs(vllm_embeddings.astype(np.float64) - hf_embeddings.astype(np.float64)).max()
    print("\nComparison (vLLM vs HuggingFace embedding):")
    print(f"  allclose(rtol={rtol}, atol={atol}): {match}")
    print(f"  max |diff|: {max_diff}")
    if not match:
        raise AssertionError(
            "vLLM and HuggingFace outputs differ. "
            f"max |diff| = {max_diff}; expected allclose(rtol={rtol}, atol={atol})."
        )
    print("  Match: vLLM and HuggingFace results are the same.")

    del hf_model
    del tokenizer

    # ---- Reference: nvidia model from HuggingFace Hub (check our conversion / local export against it) ----
    # Load nvidia/esm2_t6_8M_UR50D from Hub and run same sequences; compare to MODEL_ID (local or Hub).
    print(f"\n--- Reference: HuggingFace Hub {REFERENCE_MODEL_ID} (same {len(prompts)} sequences) ---")
    ref_model = AutoModel.from_pretrained(REFERENCE_MODEL_ID, trust_remote_code=True)
    ref_tokenizer = AutoTokenizer.from_pretrained(REFERENCE_MODEL_ID, trust_remote_code=True)
    ref_model = ref_model.to("cuda", dtype=torch.float16)
    ref_model.eval()

    def _embed_from_output(out, use_pooler=True, last_token_and_l2=True):
        """Get one embedding per batch item: pooler_output or last-token hidden state (L2-normalized)."""
        if use_pooler and out.pooler_output is not None:
            return out.pooler_output.cpu().numpy().squeeze(0)
        last_hidden = out.last_hidden_state  # (batch, seq_len, hidden_size)
        vec = last_hidden[0, -1, :].cpu().float().numpy()
        if last_token_and_l2:
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
        return vec

    ref_embeddings_list = []
    with torch.no_grad():
        for seq in prompts:
            inputs = ref_tokenizer(seq, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            out = ref_model(**inputs)
            # Use last-token + L2 to match our model (no pooler); validates conversion.
            ref_embeddings_list.append(_embed_from_output(out, use_pooler=False))

    ref_embeddings = np.stack(ref_embeddings_list)

    # Compare our model (HF path) vs reference (facebook Hub): conversion should match.
    rtol_ref, atol_ref = 1e-2, 5e-4
    match_ref = np.allclose(hf_embeddings, ref_embeddings, rtol=rtol_ref, atol=atol_ref)
    max_diff_ref = np.abs(hf_embeddings.astype(np.float64) - ref_embeddings.astype(np.float64)).max()
    print(f"\nComparison (our model vs reference {REFERENCE_MODEL_ID}):")
    print(f"  allclose(rtol={rtol_ref}, atol={atol_ref}): {match_ref}")
    print(f"  max |diff|: {max_diff_ref}")
    if not match_ref:
        raise AssertionError(
            "Our model and reference (Hub) outputs differ. "
            f"max |diff| = {max_diff_ref}; expected allclose(rtol={rtol_ref}, atol={atol_ref})."
        )
    print("  Match: conversion matches reference HuggingFace Hub model.")

    del ref_model
    del ref_tokenizer

    # Cleanup
    del model
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
