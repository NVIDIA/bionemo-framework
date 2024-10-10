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
from typing import Callable

import torch

from bionemo.data.protein.openfold.helpers import map_tree_leaves


TORCH_SEED_MODULUS = 0xFFFF_FFFF_FFFF_FFFF + 1


def enable_tf32() -> None:
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def disable_tf32() -> None:
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def is_autocast_fp16_enabled() -> bool:
    return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16


def map_tensor_tree(fn: Callable, tree: dict) -> dict:
    """Maps tensor tree using given function."""
    return map_tree_leaves(fn=fn, tree=tree, leaf_type=torch.Tensor)
