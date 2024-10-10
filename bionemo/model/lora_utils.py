# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import os
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Sequence

import torch
from nemo.utils.model_utils import inject_model_parallel_rank


def _load_and_megatronize_state_dict(device: torch.device | str, path: Path | str) -> Dict[str, torch.Tensor]:
    model_weights = inject_model_parallel_rank(str(path))
    state_dict = torch.load(model_weights, map_location=device)
    return state_dict


def _strip_task_layers(
    state_dict: Dict[str, torch.Tensor],
    task_layer_parts: Sequence[str],
) -> Dict[str, torch.Tensor]:
    if len(task_layer_parts) == 0:
        raise ValueError("Must have at least one task layer part to use to check layers for discarding.")

    def is_discard_layer(full_layer_name: str) -> bool:
        for part in task_layer_parts:
            if part in full_layer_name:
                return True
        return False

    return {k: v for k, v in state_dict.items() if not is_discard_layer(k)}


def extract_and_strip_fine_tuned_esm2_lora(
    input_nemo_fi: Path | str,
    output_modified_ckpt_state_dict_fi: Path | str,
    task_layer_parts: Sequence[str],
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> None:
    with tempfile.TemporaryDirectory() as tdir:
        if verbose:
            print(f"Loading nemo checkpoint from: {str(input_nemo_fi)}")

        with tarfile.open(str(input_nemo_fi), "r:") as tar:
            if verbose:
                print(f"{tdir=}")

            tar.extractall(path=tdir)

        if verbose:
            print(os.listdir(tdir))

        ckpt_original = Path(tdir) / "model_weights.ckpt"

        state_dict = _load_and_megatronize_state_dict(
            device=device,
            path=ckpt_original,
        )

        stripped_state_dict = _strip_task_layers(state_dict, task_layer_parts)

        torch.save(
            {"state_dict": stripped_state_dict},
            str(output_modified_ckpt_state_dict_fi),
        )
        if verbose:
            print(f"Saved stripped state dict ckpt to: {str(output_modified_ckpt_state_dict_fi)}")
