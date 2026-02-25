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

"""Convert a TransformerEngine Llama-3 checkpoint to vLLM-ready HuggingFace format.

The llama3_native_te recipe produces NVLlamaForCausalLM checkpoints (TE format).
This script converts those checkpoints to standard LlamaForCausalLM so they can
be served by vLLM, SGLang, or loaded by plain transformers without
trust_remote_code.

Usage with an existing TE checkpoint (from training):

    python export_llama3.py --te-checkpoint /path/to/recipe/final_model

Demo mode (no training needed -- creates a TE checkpoint from HuggingFace):

    python export_llama3.py
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


LLAMA3_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "llama3"
sys.path.insert(0, str(LLAMA3_MODEL_DIR))

from convert import convert_llama_hf_to_te, convert_llama_te_to_hf  # noqa: E402
from modeling_llama_te import AUTO_MAP, NVLlamaForCausalLM  # noqa: E402


HF_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
HF_CHECKPOINT_DIR = Path(__file__).resolve().parent / "llama3_hf_roundtrip_checkpoint"


def create_te_checkpoint(output_dir: Path) -> Path:
    """Create a TE checkpoint by converting a pretrained HF model.

    Follows the same workflow as bionemo-recipes/models/llama3/export.py:
    convert weights, patch config.json with auto_map, and copy the modeling
    file so the checkpoint is self-contained and loadable with
    ``trust_remote_code=True``.
    """
    print(f"  Loading pretrained model: {HF_MODEL_ID}")
    model_hf = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID)

    print("  Converting HF -> TE")
    model_te = convert_llama_hf_to_te(model_hf)
    del model_hf

    te_dir = output_dir / "te_checkpoint"
    te_dir.mkdir(parents=True, exist_ok=True)
    model_te.save_pretrained(te_dir)
    del model_te

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    tokenizer.save_pretrained(te_dir)

    config_path = te_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    config["auto_map"] = AUTO_MAP
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    shutil.copy(LLAMA3_MODEL_DIR / "modeling_llama_te.py", te_dir / "modeling_llama_te.py")

    print(f"  TE checkpoint saved to {te_dir}")
    return te_dir


def convert_te_to_vllm(te_checkpoint: Path, output_dir: Path) -> Path:
    """Convert a TE checkpoint to standard HF format for vLLM serving."""
    print(f"  Loading TE checkpoint: {te_checkpoint}")
    model_te = NVLlamaForCausalLM.from_pretrained(te_checkpoint)

    # convert_llama_te_to_hf creates the target LlamaForCausalLM on the meta
    # device in float32.  The TE checkpoint may store weights in bfloat16
    # (typical for training), so we cast to float32 to match.
    model_te = model_te.float()

    print("  Converting TE -> HF")
    model_hf = convert_llama_te_to_hf(model_te)
    del model_te

    output_dir.mkdir(parents=True, exist_ok=True)
    model_hf.save_pretrained(output_dir)
    del model_hf

    print(f"  vLLM-ready checkpoint saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--te-checkpoint",
        type=Path,
        default=None,
        help="Path to an NVLlamaForCausalLM checkpoint (e.g. from llama3_native_te recipe). "
        "If omitted, a TE checkpoint is created from HuggingFace for demo purposes.",
    )
    parser.add_argument(
        "--output", type=Path, default=HF_CHECKPOINT_DIR, help="Output directory for the vLLM checkpoint."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=HF_MODEL_ID,
        help="Tokenizer to bundle with the checkpoint (HF model ID or local path).",
    )
    args = parser.parse_args()

    # Phase 1: Obtain a TE checkpoint
    if args.te_checkpoint is not None:
        te_path = args.te_checkpoint
        print(f"[1/2] Using existing TE checkpoint: {te_path}")
    else:
        print("[1/2] No TE checkpoint provided -- creating one from HuggingFace (demo mode)")
        te_path = create_te_checkpoint(args.output.parent)

    # Phase 2: Convert TE -> HF for vLLM
    print("[2/2] Converting TE checkpoint to vLLM-ready HF format")
    convert_te_to_vllm(te_path, args.output)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.save_pretrained(args.output)

    print(f"\nDone. Serve with: vllm serve {args.output}")
