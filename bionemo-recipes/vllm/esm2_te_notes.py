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

"""Analysis: vLLM + TransformerEngine ESM2 Compatibility Gap.

This file documents the exact gaps preventing vLLM from loading the NVIDIA ESM2
TransformerEngine model, and a proof-of-concept inference script.

## The Problem (Two Issues)

### Issue 1: vLLM requires transformers>=5.0.0 for encoder models

When running ANY encoder-only model (ESM2, BERT, etc.) with vLLM's transformers backend:

    LLM(model="facebook/esm2_t6_8M_UR50D", task="embed", model_impl="transformers", ...)

vLLM fails with:
    ImportError: Transformers modeling backend requires transformers>=5.0.0.dev0
    for encoder models support, but got 4.57.1

This is a vLLM limitation - encoder-only models are not yet supported in the
transformers backend until transformers 5.0+ is released and integrated.

### Issue 2: NVIDIA TE model has weight naming mismatch (additional issue)

Even if transformers 5.0+ support was available, the NVIDIA ESM2 TE model would
still fail due to weight naming issues:

When running:
    LLM(model="nvidia/esm2_t6_8M_UR50D", task="embed", model_impl="transformers", ...)

vLLM fails with:
    ValueError: There is no module or parameter named 'model.esm' in TransformersForEmbedding

## Root Cause Analysis

vLLM's Transformers modeling backend wraps custom models in adapter classes like
`TransformersForEmbedding`. These adapters expect:

1. Weights named with a `model.` prefix (e.g., `model.encoder.layers.*`)
2. The custom model to satisfy certain interface requirements

But NVEsmForMaskedLM has:
- Weights named WITHOUT `model.` prefix: `esm.embeddings.*`, `esm.encoder.*`, `lm_head.*`
- Uses TransformerEngine's `TransformerLayer` which doesn't use HuggingFace's attention interface

## Specific Gaps (from vLLM docs: "Writing custom models")

For vLLM Transformers backend compatibility, a model needs:

1. ❌ `_supports_attention_backend = True` on base model class
   - NVEsmModel doesn't have this attribute

2. ❌ Attention must use `ALL_ATTENTION_FUNCTIONS` interface
   - NVEsmEncoder uses `transformer_engine.pytorch.TransformerLayer`
   - TE's TransformerLayer has its own attention implementation (FlashAttention, FusedAttention)
   - It does NOT use HuggingFace's `ALL_ATTENTION_FUNCTIONS` registry

3. ❌ `is_causal = False` on attention class for encoder-only models
   - TE's TransformerLayer doesn't expose this interface

4. ❌ Weight naming convention mismatch
   - vLLM wraps model in `TransformersForEmbedding(model=NVEsmForMaskedLM(...))`
   - This adds `model.` prefix when loading weights
   - Checkpoint has: `esm.encoder.layers.0.*`
   - vLLM expects: `model.esm.encoder.layers.0.*`

## Solutions

### Option A: Modify NVEsm to be vLLM-compatible (RECOMMENDED for POC)

Add a vLLM-compatible wrapper that:
1. Adds `_supports_attention_backend = True`
2. Maps weight names at load time
3. Provides the expected attention interface (may require replacing TE attention)

### Option B: Write a native vLLM model implementation

Create a custom vLLM model class (like they have for Qwen2, Llama, etc.) that:
1. Uses vLLM's optimized attention kernels
2. Handles TE weight loading directly
3. Registered in vLLM's model registry

### Option C: Use vLLM's `hf_to_vllm_mapper`

Implement a custom weight mapper that handles the `model.` prefix.

## Proof of Concept: Direct TransformerEngine Inference

For now, here's working inference using the TE model directly (without vLLM):
"""

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def get_embeddings(model, tokenizer, sequences: list[str], device: str = "cuda") -> torch.Tensor:
    """Get per-token embeddings from the model."""
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=1022)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**inputs)

    return outputs.last_hidden_state


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over non-padded tokens."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


if __name__ == "__main__":
    model_tag = "nvidia/esm2_t6_8M_UR50D"

    print(f"Loading TransformerEngine ESM2 model: {model_tag}")
    print("=" * 60)

    # Load with TransformerEngine layers
    config = AutoConfig.from_pretrained(model_tag, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            model_tag,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .cuda()
        .eval()
    )

    tokenizer = AutoTokenizer.from_pretrained(model_tag, trust_remote_code=True)

    # Print model structure to show the weight naming
    print("\nModel structure (showing weight naming issue):")
    for name, _ in list(model.named_parameters())[:10]:
        print(f"  {name}")
    print("  ...")
    print("\nvLLM expects weights prefixed with 'model.' but checkpoint has no prefix.\n")

    sequences = [
        "DLEAQWDNPHLWTYRWVLTKVTDDPGIPNVDIFVCKPDYPAHNPVFEPSVRTPLMFINEPRMKNMHLPWRIWIAYTMRWQKEVSSYIQHVNHMVYPVFKDTNPAQPSRGWSTETCFNWTEIGEKLRTQFYSNDMCGVRVHTRYGDGNEHIYCNFNQNCMQFASGNYQKDMSGGQCHIATLVTDKPIIGMMWDKRILHCIC",
        "VSTMIKFKNRNTQAGWTTYWFHSWNVCWHARYKAKCHGPPYAFWCTHRFWNKCCRRTTMVFTQGEYQGDIWHIWIMWHLLTPQEEQFCDLGPQSLHIPMMDDWVKYHAETCSEWSYDDNMSFTMVKCQTLEVWQWLWENEGYICILGPTCCFKPHASYVEEDIDNCWDFVYMCFTTIHMFRHYVCKGEQAMGGRTGQHGS",
    ]

    print(f"Running TE-accelerated inference on {len(sequences)} sequences...")

    hidden_states = get_embeddings(model, tokenizer, sequences)
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=1022)
    attention_mask = inputs["attention_mask"].to(hidden_states.device)
    sequence_embeddings = mean_pooling(hidden_states, attention_mask)

    for i, seq in enumerate(sequences):
        print(f"\nSequence {i + 1}:")
        print(f"  Length: {len(seq)} amino acids")
        print(f"  Per-token embedding shape: {hidden_states[i].shape}")
        print(f"  Sequence embedding shape: {sequence_embeddings[i].shape}")
        print(f"  First 5 dims: {sequence_embeddings[i][:5].tolist()}")

    print("\n" + "=" * 60)
    print("SUCCESS: TransformerEngine model works directly with HuggingFace.")
    print("To make this work with vLLM, we need to bridge the gaps listed above.")
