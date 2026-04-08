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


def tri_mul_bmm(a: torch.Tensor, b: torch.Tensor, k_dim: int, mode: str = "off") -> torch.Tensor:
    """Batched GEMM equivalent of triangular multiplication einsum.

    Replaces:
      k_dim=2: torch.einsum("bikd,bjkd->bijd", a, b)
      k_dim=1: torch.einsum("bkid,bkjd->bijd", a, b)

    Args:
        a: Tensor of shape (B, N, N, D).
        b: Tensor of shape (B, N, N, D).
        k_dim: Spatial dimension to contract over (1 or 2).
        mode: Precision mode.
            "off": FP32 bmm (caller upcasts via .float(), default).
            "bf16": BF16 bmm (skip .float() upcast).

    Returns:
        Tensor of shape (B, N, N, D).
    """
    B, N1, N2, D = a.shape
    # Move D to dim 1: (B, D, N, N), then merge B*D for batched mm
    a = a.permute(0, 3, 1, 2).contiguous().reshape(B * D, N1, N2)
    b = b.permute(0, 3, 1, 2).contiguous().reshape(B * D, N1, N2)

    if k_dim == 2:
        # "bikd,bjkd->bijd": a is (batch, i, k), b is (batch, j, k)
        # result = a @ b^T = (batch, i, j)
        out = torch.bmm(a, b.transpose(1, 2))
    elif k_dim == 1:
        # "bkid,bkjd->bijd": a is (batch, k, i), b is (batch, k, j)
        # result = a^T @ b = (batch, i, j)
        out = torch.bmm(a.transpose(1, 2), b)
    else:
        raise ValueError(f"k_dim must be 1 or 2, got {k_dim}")

    # Reshape back: (B*D, N, N) -> (B, D, N, N) -> (B, N, N, D)
    return out.reshape(B, D, N1, N2).permute(0, 2, 3, 1)


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
