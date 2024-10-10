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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import bionemo.model.protein.openfold.inductor as inductor
from bionemo.model.protein.openfold.linear import Linear
from bionemo.model.protein.openfold.utils.rigid_utils import Rigid
from bionemo.model.protein.openfold.utils.torch_utils import is_autocast_fp16_enabled


class InvariantPointAttention(nn.Module):
    """Invariant Point Attention (IPA) module.

    Supplementary '1.8.2 Invariant point attention (IPA)': Algorithm 22.

    Args:
        c_s: Single representation dimension (channels).
        c_z: Pair representation dimension (channels).
        c_hidden: Hidden dimension (channels).
        num_heads: Number of attention heads.
        num_qk_points: Number of query/key points.
        num_v_points: Number of value points.
        inf: Safe infinity value.
        eps: Epsilon to prevent division by zero.

    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        num_heads: int,
        num_qk_points: int,
        num_v_points: int,
        inf: float,
        eps: float,
    ) -> None:
        super(InvariantPointAttention, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.inf = inf
        self.eps = eps
        # These linear layers differ from their specifications in the supplement.
        # There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias
        # and use the default Lecun initialization.
        hc = c_hidden * num_heads
        self.linear_q = Linear(c_s, hc, bias=True, init="default")
        self.linear_kv = Linear(c_s, 2 * hc, bias=True, init="default")

        hpq = num_heads * num_qk_points * 3
        self.linear_q_points = Linear(c_s, hpq, bias=True, init="default")

        hpkv = num_heads * (num_qk_points + num_v_points) * 3
        self.linear_kv_points = Linear(c_s, hpkv, bias=True, init="default")

        self.linear_b = Linear(c_z, num_heads, bias=True, init="default")

        self.head_weights = nn.Parameter(torch.zeros((num_heads)))
        ipa_point_weights_init_(self.head_weights.data)

        concat_out_dim = num_heads * (c_z + c_hidden + num_v_points * 4)
        self.linear_out = Linear(concat_out_dim, c_s, bias=True, init="final")

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Rigid,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Invariant Point Attention (IPA) forward pass.

        Args:
            s: [batch, N_res, c_s] single representation
            z: [batch, N_res, N_res, c_z] pair representation
            r: [batch, N_res] rigids transformation
            mask: [batch, N_res] sequence mask

        Returns:
            s_update: [batch, N_res, c_s] single representation update

        """
        #######################################
        # Generate scalar and point activations
        #######################################
        q, b, kv, q_pts, kv_pts = _forward_linears_on_inputs_eager(
            s,
            z,
            self.linear_q.weight,
            self.linear_q.bias,
            self.linear_b.weight,
            self.linear_b.bias,
            self.linear_kv.weight,
            self.linear_kv.bias,
            self.linear_q_points.weight,
            self.linear_q_points.bias,
            self.linear_kv_points.weight,
            self.linear_kv_points.bias,
        )
        # q: [batch, N_res, num_heads * c_hidden]
        # b: [batch, N_res, N_res, num_heads]
        # kv: [batch, N_res, num_heads * 2 * c_hidden]
        # q_pts: [batch, N_res, num_heads * num_qk_points * 3]
        # kv_pts: [batch, N_res, num_heads * (num_qk_points + num_v_points) * 3]

        q = q.view(q.shape[:-1] + (self.num_heads, self.c_hidden))
        # q: [batch, N_res, num_heads, c_hidden]

        kv = kv.view(kv.shape[:-1] + (self.num_heads, -1))
        # kv: [batch, N_res, num_heads, 2 * c_hidden]

        k, v = torch.split(kv, self.c_hidden, dim=-1)
        # k: [batch, N_res, num_heads, c_hidden]
        # v: [batch, N_res, num_heads, c_hidden]

        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r.unsqueeze(-1).apply(q_pts)
        # q_pts: [batch, N_res, num_heads * num_qk_points, 3]

        q_pts = q_pts.view(q_pts.shape[:-2] + (self.num_heads, self.num_qk_points, 3))
        # q_pts: [batch, N_res, num_heads, num_qk_points, 3]

        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)
        # kv_pts: [batch, N_res, num_heads * (num_qk_points + num_v_points), 3]

        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.num_heads, self.num_qk_points + self.num_v_points, 3))
        # kv_pts: [batch, N_res, num_heads, (num_qk_points + num_v_points), 3]

        k_pts, v_pts = torch.split(kv_pts, (self.num_qk_points, self.num_v_points), dim=-2)
        # k_pts: [batch, N_res, num_heads, num_qk_points, 3]
        # v_pts: [batch, N_res, num_heads, num_v_points, 3]

        ##########################
        # Compute attention scores
        ##########################

        if is_autocast_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                a = torch.matmul(
                    q.float().movedim(-3, -2),  # q: [batch, num_heads, N_res, c_hidden]
                    k.float().movedim(-3, -1),  # k: [batch, num_heads, c_hidden, N_res]
                )
        else:
            a = torch.matmul(
                q.movedim(-3, -2),  # q: [batch, num_heads, N_res, c_hidden]
                k.movedim(-3, -1),  # k: [batch, num_heads, c_hidden, N_res]
            )
        # a: [batch, num_heads, N_res, N_res]

        if inductor.is_enabled():
            forward_a_fn = _forward_a_jit
        else:
            forward_a_fn = _forward_a_eager
        a = forward_a_fn(
            a,
            b,
            q_pts,
            k_pts,
            mask,
            self.c_hidden,
            self.num_heads,
            self.head_weights,
            self.num_qk_points,
            self.inf,
        )
        # a: [batch, num_heads, N_res, N_res]

        ################
        # Compute output
        ################

        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)
        # o: [batch, N_res, num_heads, c_hidden]

        o = o.reshape(o.shape[:-2] + (self.num_heads * self.c_hidden,))
        # o: [batch, N_res, num_heads * c_hidden]

        o_pt = torch.sum(
            (a.unsqueeze(-3).unsqueeze(-1) * v_pts.swapdims(-4, -3).movedim(-1, -3).unsqueeze(-3)),
            dim=-2,
        )
        # o_pt: [batch, num_heads, 3, N_res, num_v_points]

        o_pt = o_pt.movedim(-3, -1).swapdims(-3, -4)
        # o_pt: [batch, N_res, num_heads, num_v_points, 3]

        o_pt = r.unsqueeze(-1).unsqueeze(-2).invert_apply_jit(o_pt)
        # o_pt: [batch, N_res, num_heads, num_v_points, 3]

        if inductor.is_enabled():
            forward_o_pt_norm_fn = _forward_o_pt_norm_jit
        else:
            forward_o_pt_norm_fn = _forward_o_pt_norm_eager
        o_pt_norm = forward_o_pt_norm_fn(o_pt, self.eps)
        # o_pt_norm: [batch, N_res, num_heads, num_v_points]

        o_pt_norm = o_pt_norm.reshape(o_pt_norm.shape[:-2] + (self.num_heads * self.num_v_points,))
        # o_pt_norm: [batch, N_res, num_heads * num_v_points]

        o_pt = o_pt.reshape(o_pt.shape[:-3] + (self.num_heads * self.num_v_points, 3))
        # o_pt: [batch, N_res, num_heads * num_v_points, 3]

        o_pair = torch.matmul(a.transpose(-2, -3), z.to(dtype=a.dtype))
        # o_pair: [batch, N_res, num_heads, c_z]

        o_pair = o_pair.reshape(o_pair.shape[:-2] + (self.num_heads * self.c_z,))
        # o_pair: [batch, N_res, num_heads * c_z]

        o_cat = (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair)
        o_cat = torch.cat(o_cat, dim=-1)
        # o_cat: [batch, N_res, num_heads * (c_hidden + num_v_points * 4 + c_z)]

        s_update = self.linear_out(o_cat.to(dtype=z.dtype))
        # s_update: [batch, N_res, c_s]

        return s_update


def ipa_point_weights_init_(weights_data: torch.Tensor) -> None:
    softplus_inverse_1 = 0.541324854612918
    weights_data.fill_(softplus_inverse_1)


def _forward_linears_on_inputs_eager(
    s: torch.Tensor,
    z: torch.Tensor,
    w_q: torch.Tensor,
    b_q: torch.Tensor,
    w_b: torch.Tensor,
    b_b: torch.Tensor,
    w_kv: torch.Tensor,
    b_kv: torch.Tensor,
    w_q_points: torch.Tensor,
    b_q_points: torch.Tensor,
    w_kv_points: torch.Tensor,
    b_kv_points: torch.Tensor,
) -> torch.Tensor:
    q = F.linear(s, w_q, b_q)
    b = F.linear(z, w_b, b_b)
    kv = F.linear(s, w_kv, b_kv)
    q_pts = F.linear(s, w_q_points, b_q_points)
    kv_pts = F.linear(s, w_kv_points, b_kv_points)
    return q, b, kv, q_pts, kv_pts


_forward_linears_on_inputs_jit = torch.compile(_forward_linears_on_inputs_eager)


def _forward_a_eager(
    a: torch.Tensor,
    b: torch.Tensor,
    q_pts: torch.Tensor,
    k_pts: torch.Tensor,
    mask: torch.Tensor,
    c_hidden: int,
    num_heads: int,
    head_weights: torch.Tensor,
    num_qk_points: int,
    inf: float,
) -> torch.Tensor:
    # a: [batch, num_heads, N_res, N_res]
    # b: [batch, N_res, N_res, num_heads]
    # q_pts: [batch, N_res, num_heads, num_qk_points, 3]
    # k_pts: [batch, N_res, num_heads, num_qk_points, 3]
    # mask: [batch, N_res]

    a = a * math.sqrt(1.0 / (3 * c_hidden))
    a = a + (math.sqrt(1.0 / 3) * b.movedim(-1, -3))
    # a: [batch, num_heads, N_res, N_res]

    pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)  # outer subtraction
    pt_att = pt_att**2
    # pt_att: [batch, N_res, N_res, num_heads, num_qk_points, 3]

    pt_att = sum(torch.unbind(pt_att, dim=-1))
    # pt_att: [batch, N_res, N_res, num_heads, num_qk_points]

    head_weights = F.softplus(head_weights)
    head_weights = head_weights.view((1,) * (pt_att.ndim - 2) + (num_heads, 1))
    head_weights = head_weights * math.sqrt(1.0 / (3 * (num_qk_points * 9.0 / 2)))
    # head_weights: [1, 1, 1, num_heads, 1]

    pt_att = pt_att * head_weights
    # pt_att: [batch, N_res, N_res, num_heads, num_qk_points]

    pt_att = -0.5 * torch.sum(pt_att, dim=-1)
    # pt_att: [batch, N_res, N_res, num_heads]

    square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # outer product
    square_mask = (square_mask - 1.0) * inf
    # square_mask: [batch, N_res, N_res]

    pt_att = pt_att.movedim(-1, -3)
    # square_mask: [batch, num_heads, N_res, N_res]

    a = a + pt_att
    # a: [batch, num_heads, N_res, N_res]

    a = a + square_mask.unsqueeze(-3)
    # a: [batch, num_heads, N_res, N_res]

    a = torch.softmax(a, dim=-1)
    # a: [batch, num_heads, N_res, N_res]

    return a


_forward_a_jit = torch.compile(_forward_a_eager)


def _forward_o_pt_norm_eager(o_pt: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(torch.sum(o_pt**2, dim=-1) + eps)


_forward_o_pt_norm_jit = torch.compile(_forward_o_pt_norm_eager)
