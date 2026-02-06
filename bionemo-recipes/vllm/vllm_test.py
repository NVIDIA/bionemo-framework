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

"""vLLM ESM2 embedding example with offline weight conversion.

This script demonstrates how to run ESM2 embeddings with vLLM by pre-converting
the checkpoint weights to fix the naming mismatch.

## The Problem

ESM2 checkpoints are saved from NVEsmForMaskedLM which has `self.esm = NVEsmModel()`.
So weights have names like: `esm.embeddings.*`, `esm.encoder.*`

But vLLM's AutoModel.from_config() returns NVEsmModel directly, which has:
`embeddings.*`, `encoder.*` (no `esm.` prefix in the structure).

vLLM then wraps this in TransformersForEmbedding and its generic weight mapper
prepends `model.`, so it expects: `model.embeddings.*`, `model.encoder.*`

## The Solution

We pre-convert the checkpoint by stripping the `esm.` prefix:
- `esm.embeddings.*` -> `embeddings.*`
- `esm.encoder.*` -> `encoder.*`
- `lm_head.*` -> dropped (not needed for embedding)

vLLM's mapper then correctly adds `model.` to get `model.embeddings.*`, etc.
This script handles the conversion automatically on first run.
"""

import os
from pathlib import Path

import numpy as np
import torch
from convert_esm2_for_vllm import convert_esm2_checkpoint
from vllm import LLM


# Configuration
# MODEL_ID = "facebook/esm2_t12_35M_UR50D"
MODEL_ID = "nvidia/esm2_t6_8M_UR50D"
CONVERTED_MODEL_DIR = "./esm2_vllm_converted"


def ensure_converted_model(model_id: str, output_dir: str) -> str:
    """Ensure the converted model exists, converting if necessary."""
    output_path = Path(output_dir)
    marker_file = output_path / "vllm_conversion_info.json"

    if marker_file.exists():
        print(f"Using existing converted model at: {output_path}")
        return str(output_path)

    print(f"Converting {model_id} for vLLM compatibility...")
    hf_token = os.getenv("HF_TOKEN")
    convert_esm2_checkpoint(model_id, output_dir, hf_token)
    return str(output_path)


if __name__ == "__main__":
    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)")

    # Ensure we have a converted checkpoint (strips 'esm.' prefix from weight keys)
    converted_model_path = ensure_converted_model(MODEL_ID, CONVERTED_MODEL_DIR)

    # Load ESM2 as a pooling/embedding model from converted checkpoint
    print(f"\nLoading model from: {converted_model_path}")
    model = LLM(
        model=converted_model_path,
        tokenizer=converted_model_path,
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
