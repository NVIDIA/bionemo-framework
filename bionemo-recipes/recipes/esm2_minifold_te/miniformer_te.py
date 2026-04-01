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
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from torch import Tensor

from minifold_utils import init
from te_utils import te_layernorm_nd, te_linear_nd


class TransitionUpdateTE(nn.Module):
    """TE version of TransitionUpdate: two-layer MLP with residual connection.

    Replaces raw nn.Parameter + F.linear with te.LayerNorm + te.Linear modules.
    """

    def __init__(self, dim: int = 128, hidden: int = 512, params_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.norm = te.LayerNorm(dim, eps=1e-5, params_dtype=params_dtype)
        self.fc1 = te.Linear(dim, hidden, params_dtype=params_dtype)
        self.fc2 = te.Linear(hidden, dim, params_dtype=params_dtype)

        # Match original initialization
        init.bias_init_one_(self.norm.weight)
        init.bias_init_zero_(self.norm.bias)
        init.he_normal_init_(self.fc1.weight)
        init.bias_init_zero_(self.fc1.bias)
        init.final_init_(self.fc2.weight)
        init.bias_init_zero_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, N, D).

        Returns:
            Output tensor of shape (B, N, N, D).
        """
        x = te_layernorm_nd(self.norm, x)
        x = te_linear_nd(self.fc1, x)
        x = F.relu(x)
        x = te_linear_nd(self.fc2, x)
        return x


class TriangularUpdateTE(nn.Module):
    """TE version of TriangularUpdate.

    Replaces raw nn.Parameter + F.linear/F.layer_norm with te.LayerNorm + te.Linear.
    The einsum triangular multiplication operations remain in FP32.
    """

    def __init__(self, dim: int = 128, params_dtype: torch.dtype = torch.float32):
        super().__init__()

        # Input gating: LayerNorm + two parallel linears (projection and gate)
        self.input_norm = te.LayerNorm(dim, eps=1e-5, params_dtype=params_dtype)
        self.pi = te.Linear(dim, dim, params_dtype=params_dtype)  # input projection
        self.gi = te.Linear(dim, dim, params_dtype=params_dtype)  # input gate (sigmoid)

        # Output gating: LayerNorm + two parallel linears
        self.output_norm = te.LayerNorm(dim // 2, eps=1e-5, params_dtype=params_dtype)
        self.po = te.Linear(dim // 2, dim, params_dtype=params_dtype)  # output projection
        self.go = te.Linear(dim // 2, dim, params_dtype=params_dtype)  # output gate (sigmoid)

        # Match original initialization
        init.bias_init_one_(self.input_norm.weight)
        init.bias_init_zero_(self.input_norm.bias)

        init.lecun_normal_init_(self.pi.weight)
        init.bias_init_zero_(self.pi.bias)
        init.gating_init_(self.gi.weight)
        init.bias_init_one_(self.gi.bias)

        init.bias_init_one_(self.output_norm.weight)
        init.bias_init_zero_(self.output_norm.bias)

        init.final_init_(self.po.weight)
        init.bias_init_zero_(self.po.bias)
        init.gating_init_(self.go.weight)
        init.bias_init_one_(self.go.bias)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, N, D).
            mask: Mask tensor of shape (B, N, N).

        Returns:
            Output tensor of shape (B, N, N, D).
        """
        # Input gating: D -> D
        x = te_layernorm_nd(self.input_norm, x)
        x = te_linear_nd(self.pi, x) * te_linear_nd(self.gi, x).sigmoid()

        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Triangular multiplication (MUST stay in FP32)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
        with torch.autocast(device.type, enabled=False):
            a1, b1, a2, b2 = torch.chunk(x.float(), 4, dim=-1)
            x1 = torch.einsum("bikd,bjkd->bijd", a1, b1)
            x2 = torch.einsum("bkid,bkjd->bijd", a2, b2)
            x = torch.cat([x1, x2], dim=-1).to(mask.dtype if mask.is_floating_point() else torch.float32)

        # Output gating: D/2 -> D
        x = te_layernorm_nd(self.output_norm, x)
        x = te_linear_nd(self.po, x) * te_linear_nd(self.go, x).sigmoid()

        return x


class BlockTE(nn.Module):
    """TE version of a MiniFormer block: TriangularUpdate + TransitionUpdate."""

    def __init__(self, dim: int = 128, params_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.triangular = TriangularUpdateTE(dim, params_dtype=params_dtype)
        self.transition = TransitionUpdateTE(dim, dim * 4, params_dtype=params_dtype)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, N, D).
            mask: Mask tensor of shape (B, N, N).

        Returns:
            Output tensor of shape (B, N, N, D).
        """
        x = x + self.triangular(x, mask)
        x = x + self.transition(x)
        return x


class MiniFormerTE(nn.Module):
    """TE version of the MiniFormer module."""

    def __init__(self, dim: int = 128, blocks: int = 48, params_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.blocks = nn.ModuleList([BlockTE(dim, params_dtype=params_dtype) for _ in range(blocks)])

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, N, D).
            mask: Mask tensor of shape (B, N, N).

        Returns:
            Output tensor of shape (B, N, N, D).
        """
        for block in self.blocks:
            x = block(x, mask)
        return x
