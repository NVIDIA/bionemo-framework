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

import torch
import transformer_engine.pytorch as te


def te_linear_nd(module: te.Linear, x: torch.Tensor) -> torch.Tensor:
    """Apply a te.Linear module to an N-dimensional tensor (N >= 2).

    te.Linear is validated for 2D (B, D) and 3D (B, S, D) inputs.
    For 4D+ inputs (e.g. pair representations with shape B, N, N, D),
    we flatten leading dimensions to 2D, apply the linear, and reshape back.

    Args:
        module: A transformer_engine.pytorch.Linear module.
        x: Input tensor of shape (*leading_dims, in_features).

    Returns:
        Tensor of shape (*leading_dims, out_features).
    """
    if x.ndim <= 3:
        return module(x)
    leading = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    x = module(x)
    return x.reshape(*leading, -1)


def te_layernorm_nd(module: te.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    """Apply a te.LayerNorm module to an N-dimensional tensor (N >= 2).

    Args:
        module: A transformer_engine.pytorch.LayerNorm module.
        x: Input tensor of shape (*leading_dims, normalized_shape).

    Returns:
        Tensor of same shape as input.
    """
    if x.ndim <= 3:
        return module(x)
    leading = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    x = module(x)
    return x.reshape(*leading, -1)


def te_layernorm_linear_nd(module: te.LayerNormLinear, x: torch.Tensor) -> torch.Tensor:
    """Apply a te.LayerNormLinear module to an N-dimensional tensor (N >= 2).

    Args:
        module: A transformer_engine.pytorch.LayerNormLinear module.
        x: Input tensor of shape (*leading_dims, in_features).

    Returns:
        Tensor of shape (*leading_dims, out_features).
    """
    if x.ndim <= 3:
        return module(x)
    leading = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    x = module(x)
    return x.reshape(*leading, -1)
