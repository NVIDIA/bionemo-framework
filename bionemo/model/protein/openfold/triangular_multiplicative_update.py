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


# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import bionemo.model.protein.openfold.inductor as inductor
from bionemo.model.protein.openfold.layer_norm import LayerNorm
from bionemo.model.protein.openfold.linear import Linear
from bionemo.model.protein.openfold.utils.torch_utils import is_autocast_fp16_enabled


class TriangleMultiplicativeUpdate(nn.Module):
    """Triangle Multiplicative Update module.

    Supplementary '1.6.5 Triangular multiplicative update': Algorithms 11 and 12.

    Args:
        c_z: Pair or template representation dimension (channels).
        c_hidden: Hidden dimension (channels).
        tmu_type: "outgoing" or "incoming"

    """

    def __init__(
        self,
        c_z: int,
        c_hidden: int,
        tmu_type: Literal["outgoing", "incoming"],
    ) -> None:
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        if tmu_type == "outgoing":
            self._is_outgoing = True
        elif tmu_type == "incoming":
            self._is_outgoing = False
        else:
            raise ValueError(f"Invalid TMU Type '{tmu_type}', must be one of 'outgoing' or 'incoming'")

        self.linear_a_p = Linear(c_z, c_hidden, bias=True, init="default")
        self.linear_a_g = Linear(c_z, c_hidden, bias=True, init="gating")
        self.linear_b_p = Linear(c_z, c_hidden, bias=True, init="default")
        self.linear_b_g = Linear(c_z, c_hidden, bias=True, init="gating")
        self.linear_g = Linear(c_z, c_z, bias=True, init="gating")
        self.linear_z = Linear(c_hidden, c_z, bias=True, init="final")
        self.layer_norm_in = LayerNorm(c_z)
        self.layer_norm_out = LayerNorm(c_hidden)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Triangle Multiplicative Update forward pass.

        Args:
            z: [batch, N_res, N_res, c_z] pair representation
            mask: [batch, N_res, N_res] pair mask

        Returns:
            z_update: [batch, N_res, N_res, c_z] pair representation update

        """
        z = self.layer_norm_in(z)
        # z: [batch, N_res, N_res, c_z]

        mask = mask.unsqueeze(-1)
        # mask: [batch, N_res, N_res, 1]

        a, b = _compute_projections(
            z,
            mask,
            self.linear_a_g.weight,
            self.linear_a_g.bias,
            self.linear_a_p.weight,
            self.linear_a_p.bias,
            self.linear_b_g.weight,
            self.linear_b_g.bias,
            self.linear_b_p.weight,
            self.linear_b_p.bias,
        )

        if is_autocast_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
            x = self._combine_projections(a, b)
        # x: [batch, N_res, N_res, c_hidden]

        del a, b

        x = self.layer_norm_out(x)
        # x: [batch, N_res, N_res, c_hidden]

        x = _compute_output(
            x,
            z,
            self.linear_z.weight,
            self.linear_z.bias,
            self.linear_g.weight,
            self.linear_g.bias,
        )
        # x: [batch, N_res, N_res, c_z]

        return x

    def _combine_projections(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        if self._is_outgoing:
            a = a.movedim(-1, -3)
            b = b.swapdims(-1, -3)
        else:
            a = a.swapdims(-1, -3)
            b = b.movedim(-1, -3)

        p = torch.matmul(a, b)

        return p.movedim(-3, -1)


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """Triangle Multiplication Outgoing module.

    Supplementary '1.6.5 Triangular multiplicative update':
    Algorithm 11 Triangular multiplicative update using "outgoing" edges.

    Args:
        c_z: Pair or template representation dimension (channels).
        c_hidden: Hidden dimension (channels).

    """

    def __init__(
        self,
        c_z: int,
        c_hidden: int,
    ) -> None:
        super(TriangleMultiplicationOutgoing, self).__init__(
            c_z=c_z,
            c_hidden=c_hidden,
            tmu_type="outgoing",
        )


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """Triangle Multiplication Incoming module.

    Supplementary '1.6.5 Triangular multiplicative update':
    Algorithm 12 Triangular multiplicative update using "incoming" edges.

    Args:
        c_z: Pair or template representation dimension (channels).
        c_hidden: Hidden dimension (channels).

    """

    def __init__(
        self,
        c_z: int,
        c_hidden: int,
    ) -> None:
        super(TriangleMultiplicationIncoming, self).__init__(
            c_z=c_z,
            c_hidden=c_hidden,
            tmu_type="incoming",
        )


def _compute_projections_eager(
    z: torch.Tensor,
    mask: torch.Tensor,
    w_a_g: torch.Tensor,
    b_a_g: torch.Tensor,
    w_a_p: torch.Tensor,
    b_a_p: torch.Tensor,
    w_b_g: torch.Tensor,
    b_b_g: torch.Tensor,
    w_b_p: torch.Tensor,
    b_b_p: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    a = F.linear(z, w_a_g, b_a_g)
    a = torch.sigmoid(a) * mask
    a = a * F.linear(z, w_a_p, b_a_p)
    b = F.linear(z, w_b_g, b_b_g)
    b = torch.sigmoid(b) * mask
    b = b * F.linear(z, w_b_p, b_b_p)
    return a, b


_compute_projections_jit = torch.compile(_compute_projections_eager)


def _compute_projections(
    z: torch.Tensor,
    mask: torch.Tensor,
    w_a_g: torch.Tensor,
    b_a_g: torch.Tensor,
    w_a_p: torch.Tensor,
    b_a_p: torch.Tensor,
    w_b_g: torch.Tensor,
    b_b_g: torch.Tensor,
    w_b_p: torch.Tensor,
    b_b_p: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: [optim-hub]: re-check conditions on dap
    if inductor.is_enabled_on_hopper():  # and dap.size() >= 2:
        compute_projections_fn = _compute_projections_jit
    elif inductor.is_enabled_on_hopper_and_autograd_off():
        compute_projections_fn = _compute_projections_jit
    elif inductor.is_enabled_on_ampere():
        compute_projections_fn = _compute_projections_jit
    else:
        compute_projections_fn = _compute_projections_eager
    return compute_projections_fn(z, mask, w_a_g, b_a_g, w_a_p, b_a_p, w_b_g, b_b_g, w_b_p, b_b_p)


def _compute_output_eager(
    x: torch.Tensor,
    z: torch.Tensor,
    w_z: torch.Tensor,
    b_z: torch.Tensor,
    w_g: torch.Tensor,
    b_g: torch.Tensor,
) -> torch.Tensor:
    x = F.linear(x, w_z, b_z)
    g = torch.sigmoid(F.linear(z, w_g, b_g))
    x = x * g
    return x


_compute_output_jit = torch.compile(_compute_output_eager)


def _compute_output(
    x: torch.Tensor,
    z: torch.Tensor,
    w_z: torch.Tensor,
    b_z: torch.Tensor,
    w_g: torch.Tensor,
    b_g: torch.Tensor,
) -> torch.Tensor:
    # TODO: [optim-hub]: re-check conditions on dap
    if inductor.is_enabled_on_hopper():  # and dap.size() in {2, 8}:
        compute_output_fn = _compute_output_jit
    elif inductor.is_enabled_on_hopper_and_autograd_off():
        compute_output_fn = _compute_output_jit
    elif inductor.is_enabled_on_ampere():  # and dap.size() <= 1:
        compute_output_fn = _compute_output_jit
    elif inductor.is_enabled_on_ampere_and_autograd_off():
        compute_output_fn = _compute_output_jit
    else:
        compute_output_fn = _compute_output_eager
    return compute_output_fn(x, z, w_z, b_z, w_g, b_g)
