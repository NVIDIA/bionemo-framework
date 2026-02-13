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

"""Export ESMC checkpoint to HuggingFace-compatible format with TransformerEngine layers.

This script:
1. Loads the EvolutionaryScale ESMC-300M pretrained weights
2. Converts them to TransformerEngine format
3. Saves the converted model for use with HuggingFace's `AutoModel.from_pretrained()`
"""

import json
import shutil
from pathlib import Path

import convert
from modeling_esmc_te import AUTO_MAP, NVEsmcConfig


def export_esmc_checkpoint(export_path: Path):
    """Export the ESMC-300M model to a TE checkpoint.

    Args:
        export_path: Directory to save the exported checkpoint.
    """
    from esm.pretrained import ESMC_300M_202412

    # Load reference model on CPU to save GPU memory
    ref_model = ESMC_300M_202412(device="cpu", use_flash_attn=False)
    ref_state_dict = ref_model.state_dict()
    del ref_model

    # Create config matching ESMC-300M architecture
    config = NVEsmcConfig(
        vocab_size=64,
        hidden_size=960,
        num_hidden_layers=30,
        num_attention_heads=15,
        intermediate_size=2560,
    )

    # Convert and save
    model_te = convert.convert_esmc_to_te(ref_state_dict, config)
    model_te.to("cpu")
    model_te.save_pretrained(export_path)

    # Patch the config with auto_map
    with open(export_path / "config.json") as f:
        config_json = json.load(f)

    config_json["auto_map"] = AUTO_MAP

    with open(export_path / "config.json", "w") as f:
        json.dump(config_json, f, indent=2, sort_keys=True)

    # Copy modeling file for standalone loading
    shutil.copy(Path(__file__).parent / "modeling_esmc_te.py", export_path / "modeling_esmc_te.py")

    # Save tokenizer
    from esm.tokenization import EsmSequenceTokenizer

    tokenizer = EsmSequenceTokenizer()
    tokenizer.save_pretrained(export_path)


if __name__ == "__main__":
    export_esmc_checkpoint(Path("checkpoint_export"))
