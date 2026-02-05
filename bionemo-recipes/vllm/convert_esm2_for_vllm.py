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

"""Convert ESM2 checkpoint for vLLM compatibility.

ESM2 checkpoints are saved from NVEsmForMaskedLM which wraps NVEsmModel inside
``self.esm``, so weight keys have an ``esm.`` prefix (e.g. ``esm.encoder.*``).

vLLM loads NVEsmModel directly via AutoModel, which has no ``esm.`` prefix.
vLLM's TransformersForEmbedding wrapper then adds ``model.`` via its generic
mapper. So we need checkpoint keys without the ``esm.`` prefix so the mapper
produces ``model.embeddings.*`` etc., which matches the wrapper's module tree.

This script strips the ``esm.`` prefix and drops ``lm_head.*`` keys (not needed
for embedding/pooling).

Usage:
    python convert_esm2_for_vllm.py [--model MODEL_ID] [--output OUTPUT_DIR]

Example:
    python convert_esm2_for_vllm.py --model facebook/esm2_t12_35M_UR50D --output ./esm2_vllm

Then use with vLLM:
    LLM(model="./esm2_vllm", runner="pooling")
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file, save_file


def convert_esm2_checkpoint(model_id: str, output_dir: str, hf_token: str | None = None):
    """Convert ESM2 checkpoint by stripping the ``esm.`` prefix from weight names.

    The checkpoint is saved from NVEsmForMaskedLM (``self.esm = NVEsmModel(...)``),
    so keys are ``esm.embeddings.*``, ``esm.encoder.*``, ``lm_head.*``.

    After conversion, keys become ``embeddings.*``, ``encoder.*`` -- matching
    what ``NVEsmModel.state_dict()`` produces. vLLM's generic mapper then adds
    the ``model.`` prefix to get ``model.embeddings.*``, which matches the
    ``TransformersForEmbedding`` wrapper structure.

    Args:
        model_id: HuggingFace model ID (e.g., "facebook/esm2_t12_35M_UR50D")
        output_dir: Directory to save the converted model
        hf_token: Optional HuggingFace token for authentication
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Converting {model_id} for vLLM compatibility...")
    print(f"Output directory: {output_path}")

    # Load config to get vocab sizes (needed for un-padding embeddings)
    config_path = hf_hub_download(model_id, "config.json", token=hf_token)
    with open(config_path) as f:
        config = json.load(f)
    vocab_size = config.get("vocab_size")
    padded_vocab_size = config.get("padded_vocab_size")
    print(f"\nvocab_size={vocab_size}, padded_vocab_size={padded_vocab_size}")

    # List all files in the repo
    repo_files = list_repo_files(model_id, token=hf_token)
    print(f"Files in repository: {len(repo_files)}")

    # Download and convert safetensors weights
    safetensor_files = [f for f in repo_files if f.endswith(".safetensors")]

    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {model_id}")

    print(f"\nProcessing {len(safetensor_files)} safetensors file(s)...")

    for sf_file in safetensor_files:
        print(f"\n  Converting: {sf_file}")

        # Download the file
        local_path = hf_hub_download(model_id, sf_file, token=hf_token)

        # Load weights
        weights = load_file(local_path)

        # Transform weight names for vLLM compatibility
        # ESM2 checkpoints have: esm.embeddings.*, esm.encoder.*, lm_head.*
        # vLLM's AutoModel.from_config() returns NVEsmModel directly (not NVEsmForMaskedLM)
        # NVEsmModel.state_dict() keys are: embeddings.*, encoder.*, pooler.*
        # So we strip 'esm.' and drop 'lm_head.*' (not needed for embedding)
        #
        # We also drop buffer tensors (e.g. inv_freq) that are computed at init
        # time and not loadable by vLLM's AutoWeightsLoader (which only iterates
        # over named_parameters, not named_buffers).
        buffers_to_skip = {"inv_freq"}

        converted_weights = {}
        skipped = []
        for name, tensor in weights.items():
            if name.startswith("esm."):
                # Strip 'esm.' prefix: esm.embeddings.* -> embeddings.*
                new_name = name[4:]

                # Skip non-parameter buffers that vLLM can't load
                leaf_name = new_name.rsplit(".", 1)[-1]
                if leaf_name in buffers_to_skip:
                    skipped.append(f"{name} (buffer)")
                    continue

                # Un-pad embedding weights: the NVEsm checkpoint stores embeddings
                # at padded_vocab_size (e.g. 64) but vLLM's VocabParallelEmbedding
                # expects the original vocab_size (e.g. 33) and handles padding itself.
                if new_name == "embeddings.word_embeddings.weight" and padded_vocab_size and vocab_size:
                    if tensor.shape[0] == padded_vocab_size and padded_vocab_size != vocab_size:
                        print(f"    Trimming embedding: {tensor.shape[0]} -> {vocab_size} (removing padding)")
                        tensor = tensor[:vocab_size]

                converted_weights[new_name] = tensor
            else:
                # lm_head.* keys are not needed for embedding/pooling -- skip
                skipped.append(name)

        # The NVEsmForMaskedLM checkpoint was saved with add_pooling_layer=False,
        # but NVEsmModel defaults to add_pooling_layer=True, so vLLM creates
        # a pooler module that expects weights. Initialize them here.
        hidden_size = config.get("hidden_size", 320)
        if "pooler.dense.weight" not in converted_weights:
            print(f"    Initializing missing pooler weights (hidden_size={hidden_size})")
            converted_weights["pooler.dense.weight"] = torch.zeros(hidden_size, hidden_size)
            torch.nn.init.xavier_uniform_(converted_weights["pooler.dense.weight"])
            converted_weights["pooler.dense.bias"] = torch.zeros(hidden_size)

        print(f"    Original:  {next(iter(weights.keys()))}")
        print(f"    Converted: {next(iter(converted_weights.keys()))}")
        print(f"    Kept: {len(converted_weights)}, Skipped: {len(skipped)}")
        if skipped:
            print(f"    Skipped keys: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

        # Save converted weights
        output_file = output_path / sf_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_file(converted_weights, str(output_file))
        print(f"    Saved to: {output_file}")

    # Copy config files and custom model code (don't modify these)
    # Include .py files for trust_remote_code=True models
    files_to_copy = [
        f
        for f in repo_files
        if f.endswith((".json", ".txt", ".py")) and not f.endswith(".safetensors") and not f.startswith(".")
    ]

    print(f"\nCopying {len(files_to_copy)} config/code files...")
    for file_to_copy in files_to_copy:
        try:
            local_path = hf_hub_download(model_id, file_to_copy, token=hf_token)
            dest_path = output_path / file_to_copy
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(local_path, dest_path)
            print(f"  Copied: {file_to_copy}")
        except Exception as e:
            print(f"  Skipped {file_to_copy}: {e}")

    # Create a marker file indicating this is a vLLM-converted model
    marker = {
        "original_model": model_id,
        "conversion": "strip_esm_prefix",
        "description": "Stripped 'esm.' prefix from weight keys for vLLM compatibility (lm_head.* dropped)",
    }
    with open(output_path / "vllm_conversion_info.json", "w") as f:
        json.dump(marker, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Conversion complete!")
    print(f"{'=' * 60}")
    print("\nUse with vLLM:")
    print(f'  model = LLM(model="{output_dir}", runner="pooling")')
    print("  outputs = model.embed(sequences)")

    return output_path


def main():
    """CLI entrypoint for ESM2 checkpoint conversion."""
    parser = argparse.ArgumentParser(description="Convert ESM2 checkpoint for vLLM compatibility")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/esm2_t12_35M_UR50D",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./esm2_vllm",
        help="Output directory for converted model",
    )
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    convert_esm2_checkpoint(args.model, args.output, hf_token)


if __name__ == "__main__":
    main()
