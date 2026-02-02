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

"""vLLM ESM2 embedding example with weight name remapping.

This script demonstrates how to run ESM2 embeddings with vLLM by monkey-patching
the weight loading to fix the naming mismatch between the checkpoint and vLLM's
TransformersForEmbedding wrapper.

## The Problem

vLLM wraps models in adapter classes (e.g., TransformersForEmbedding) which adds
a "model." prefix to weight names. ESM2 checkpoints have weights like:
    - esm.embeddings.word_embeddings.weight
    - esm.encoder.layer.0.attention.self.query.weight

But vLLM expects:
    - model.esm.embeddings.word_embeddings.weight
    - model.esm.encoder.layer.0.attention.self.query.weight

## The Solution

We monkey-patch vLLM's DefaultModelLoader.get_all_weights to add the "model."
prefix when loading ESM2 weights. This must be done BEFORE importing vLLM's LLM class.
"""

import os


# =============================================================================
# MONKEY-PATCH: Fix ESM2 weight naming for vLLM compatibility
# This MUST be done before importing vLLM's LLM class
# =============================================================================
def _apply_esm2_weight_patch():
    """Patch vLLM's weight loading to add 'model.' prefix for ESM2 models."""
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    _original_get_all_weights = DefaultModelLoader.get_all_weights

    def _patched_get_all_weights(self, model_config, model):
        """Wrap get_all_weights to add 'model.' prefix for ESM models."""
        for name, weight in _original_get_all_weights(self, model_config, model):
            # ESM2 weights start with "esm." or "lm_head." but vLLM expects "model." prefix
            if name.startswith("esm.") or name.startswith("lm_head.") or name.startswith("contact_head."):
                yield f"model.{name}", weight
            else:
                yield name, weight

    DefaultModelLoader.get_all_weights = _patched_get_all_weights
    print("Applied ESM2 weight naming patch for vLLM compatibility")


# Apply the patch before importing LLM
_apply_esm2_weight_patch()

# Now import vLLM components (must be after patch, hence noqa)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from vllm import LLM  # noqa: E402


if __name__ == "__main__":
    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)")

    # Load ESM2 as a pooling/embedding model
    model = LLM(
        model="facebook/esm2_t12_35M_UR50D",
        runner="pooling",
        hf_token=os.getenv("HF_TOKEN"),
        trust_remote_code=True,
    )

    # Example protein sequences
    prompts = [
        "LKGHAMCLGCLHMLMCGLLAGAMCGLMKLLKCCGKCLMHLMKAMLGLKCACHHHHLLLHACAAKKLCLGAKLAMGLKLLGAHGKGLKMACGHHMLHLHMH",
        "CLLCCMHMHAHHCHGHGHKCKCLMMGMALMCAGCCACGMKGGCHCCLLAHCAHAKAGKGKCKLMCKKKHGLHAGLHAMLLCHLGLGCGHHHKKCKKHKCA",
    ]

    print(f"\nGenerating embeddings for {len(prompts)} sequences...")
    outputs = model.embed(prompts)

    # Display results
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        embedding = output.outputs.embedding
        # Handle both list and numpy array types
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        print(f"\nSequence {i + 1}:")
        print(f"  Length: {len(prompt)} amino acids")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  First 5 dims: {embedding[:5].tolist()}")

    print("\nSUCCESS: ESM2 embeddings generated with vLLM!")

    # Cleanup
    del model
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
