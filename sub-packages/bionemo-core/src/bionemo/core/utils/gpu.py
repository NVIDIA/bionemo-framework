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


import subprocess
from functools import partial
from typing import List, Optional, Sequence

import torch


__all__: Sequence[str] = (
    "get_gpu_names",
    "check_supports",
    "check_current_gpu_supports_ampere",
)


def get_gpu_names() -> Optional[List[str]]:
    """Get the GPU name using nvidia-smi."""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("No GPU detected.")
        return None

    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, universal_newlines=True
        )
    except Exception as error:
        raise ValueError("Could not use nvidia-smi to determine GPU name!") from error

    gpu_names = output.strip().split("\n")
    if len(gpu_names) == 0:
        raise ValueError("No GPUs found but nvidia-smi is installed!!")

    return gpu_names


def check_supports(supported_gpus: Sequence[str], gpu_name: str) -> bool:
    """True if the GPU name supports Ampere. False otherwise.

    Use :func:`get_gpu_name` for the input.
    """
    return any(supported_gpu in gpu_name for supported_gpu in supported_gpus)


def check_current_gpu_supports_ampere() -> bool:
    """True if the current GPU supports the Ampere architecture. False otherwise."""
    gpu_names = get_gpu_names()
    if gpu_names is None:
        return False
    check_ampere = partial(
        check_supports,
        (
            "A100",
            "A6000",
            "RTX 30",
            "RTX 40",
            "A30",
            "A40",
        ),
    )
    return any(map(check_ampere, gpu_names))
