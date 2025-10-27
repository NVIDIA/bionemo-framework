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


import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_checkpoint(checkpoint_path: str, map_location: str = "cpu"):
    """Load checkpoint from either PyTorch .ckpt file or safetensors directory."""
    if Path(checkpoint_path).is_dir():
        safetensors_files = list(Path(checkpoint_path).glob("*.safetensors"))

        if not safetensors_files or len(safetensors_files) != 1:
            raise FileNotFoundError(f"Expected single .safetensors files in {checkpoint_path}")

        safetensors_path = safetensors_files[0]
        config_path = Path(checkpoint_path) / "config.json"
        state_dict = load_file(str(safetensors_path))
        with open(config_path, "r") as f:
            hparams = json.load(f)
        return {"state_dict": state_dict, "hyper_parameters": hparams}
    else:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def download_checkpoint(repo_id: str, local_dir: str = None) -> str:  # noqa: RUF013
    """Download checkpoint from huggingface hub.

    Args:
        repo_id: HuggingFace Hub repository ID (e.g., 'username/model-name')
        local_dir: Local directory to download the checkpoint to

    Returns:
        str: Local path to the downloaded checkpoint
    """
    # check that huggingface_hub is installed
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is not installed. Please install it with `pip install huggingface_hub`")

    return snapshot_download(repo_id=repo_id, cache_dir=None, local_dir=local_dir, resume_download=True)
