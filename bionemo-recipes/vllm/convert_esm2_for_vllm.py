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

This script downloads an ESM2 model from HuggingFace, adds the 'model.' prefix
to all weight names (required by vLLM's TransformersForEmbedding wrapper),
and saves the converted model locally.

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

from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file, save_file


def convert_esm2_checkpoint(model_id: str, output_dir: str, hf_token: str | None = None):
    """Convert ESM2 checkpoint by adding 'model.' prefix to weight names.

    Args:
        model_id: HuggingFace model ID (e.g., "facebook/esm2_t12_35M_UR50D")
        output_dir: Directory to save the converted model
        hf_token: Optional HuggingFace token for authentication
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Converting {model_id} for vLLM compatibility...")
    print(f"Output directory: {output_path}")

    # List all files in the repo
    repo_files = list_repo_files(model_id, token=hf_token)
    print(f"\nFiles in repository: {len(repo_files)}")

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

        # Add 'model.' prefix to all weight names
        converted_weights = {}
        for name, tensor in weights.items():
            new_name = f"model.{name}"
            converted_weights[new_name] = tensor

        print(f"    Original: {next(iter(weights.keys()))}")
        print(f"    Converted: {next(iter(converted_weights.keys()))}")
        print(f"    Total weights: {len(converted_weights)}")

        # Save converted weights
        output_file = output_path / sf_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_file(converted_weights, str(output_file))
        print(f"    Saved to: {output_file}")

    # Copy config files (don't modify these)
    config_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
    ]

    print("\nCopying config files...")
    for config_file in config_files:
        if config_file in repo_files:
            try:
                local_path = hf_hub_download(model_id, config_file, token=hf_token)
                dest_path = output_path / config_file
                shutil.copy(local_path, dest_path)
                print(f"  Copied: {config_file}")
            except Exception as e:
                print(f"  Skipped {config_file}: {e}")

    # Create a marker file indicating this is a vLLM-converted model
    marker = {
        "original_model": model_id,
        "conversion": "vllm_weight_prefix",
        "description": "Weights prefixed with 'model.' for vLLM TransformersForEmbedding compatibility",
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
