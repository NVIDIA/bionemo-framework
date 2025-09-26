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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import shutil
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

from amplify.state_dict_convert import convert_amplify_hf_to_te


def export_hf_checkpoint(tag: str, export_path: Path):
    """Export a Hugging Face checkpoint to a Transformer Engine checkpoint.

    Args:
        tag: The tag of the checkpoint to export.
        export_path: The parent path to export the checkpoint to.
    """
    model_hf = AutoModel.from_pretrained(f"chandar-lab/{tag}", trust_remote_code=True)
    model_te = convert_amplify_hf_to_te(model_hf)
    model_te.save_pretrained(export_path / tag)

    tokenizer = AutoTokenizer.from_pretrained(f"chandar-lab/{tag}")
    tokenizer.save_pretrained(export_path / tag)

    # Patch the config
    with open(export_path / tag / "config.json", "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "amplify_te.AMPLIFYConfig",
        "AutoModel": "amplify_te.AMPLIFY",
        "AutoModelForMaskedLM": "amplify_te.AMPLIFYForMaskedLM",
    }

    with open(export_path / tag / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    shutil.copy("src/amplify/amplify_te.py", export_path / tag / "amplify_te.py")
    shutil.copy("model_readme.md", export_path / tag / "README.md")
    shutil.copy("LICENSE", export_path / tag / "LICENSE")
