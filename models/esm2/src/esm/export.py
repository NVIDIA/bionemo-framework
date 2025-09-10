# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gc
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from esm.convert import convert_esm_hf_to_te, convert_esm_te_to_hf
from esm.modeling_esm_te import NVEsmForMaskedLM


def export_hf_checkpoint(tag: str, export_path: Path):
    """Export a Hugging Face checkpoint to a Transformer Engine checkpoint.

    Args:
        tag: The tag of the checkpoint to export.
        export_path: The parent path to export the checkpoint to.
    """
    model_hf_masked_lm = AutoModelForMaskedLM.from_pretrained(f"facebook/{tag}")
    model_hf = AutoModel.from_pretrained(f"facebook/{tag}")
    model_hf_masked_lm.esm.pooler = model_hf.pooler
    model_te = convert_esm_hf_to_te(model_hf_masked_lm)
    model_te.save_pretrained(export_path / tag)

    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{tag}")
    tokenizer.save_pretrained(export_path / tag)

    # Patch the config
    with open(export_path / tag / "config.json", "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "esm_nv.NVEsmConfig",
        "AutoModel": "esm_nv.NVEsmModel",
        "AutoModelForMaskedLM": "esm_nv.NVEsmForMaskedLM",
    }

    with open(export_path / tag / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    shutil.copy("src/esm/modeling_esm_te.py", export_path / tag / "esm_nv.py")
    shutil.copy("model_readme.md", export_path / tag / "README.md")
    shutil.copy("LICENSE", export_path / tag / "LICENSE")

    del model_hf, model_te
    gc.collect()
    torch.cuda.empty_cache()

    # Smoke test that the model can be loaded.
    model_te = AutoModelForMaskedLM.from_pretrained(
        export_path / tag,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    del model_te
    gc.collect()
    torch.cuda.empty_cache()


def export_te_checkpoint(te_checkpoint_path: str, output_path: str):
    """Export a Transformer Engine checkpoint back to the original HuggingFace Facebook ESM-2 format.

    This function converts from the NVIDIA Transformer Engine (TE) format back to the
    weight format compatible with the original facebook/esm2_* series of checkpoints.
    The TE model is also a HuggingFace model (you can load it with AutoModel.from_pretrained),
    but this conversion ensures compatibility with the original Facebook ESM-2 model format hosted on Hugging Face.

    Args:
        te_checkpoint_path (str): Path to the TE checkpoint
        output_path (str): Output path for the converted Facebook ESM-2 format model
    """
    if not Path(te_checkpoint_path).exists():
        raise FileNotFoundError(f"TE checkpoint {te_checkpoint_path} not found")

    print(f"Converting {te_checkpoint_path} from TE format back to original HuggingFace Facebook ESM-2 format")

    # Load the TE model and convert to HF format
    model_te = NVEsmForMaskedLM.from_pretrained(te_checkpoint_path)
    model_hf = convert_esm_te_to_hf(model_te)
    model_hf.save_pretrained(output_path)

    tokenizer_config_path = Path(te_checkpoint_path) / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        shutil.copy(tokenizer_config_path, Path(output_path) / "tokenizer_config.json")

    vocab_path = Path(te_checkpoint_path) / "vocab.txt"
    if vocab_path.exists():
        shutil.copy(vocab_path, Path(output_path) / "vocab.txt")

    special_tokens_path = Path(te_checkpoint_path) / "special_tokens_map.json"
    if special_tokens_path.exists():
        shutil.copy(special_tokens_path, Path(output_path) / "special_tokens_map.json")

    model_hf = AutoModelForMaskedLM.from_pretrained(
        output_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    del model_hf
    gc.collect()
    torch.cuda.empty_cache()
