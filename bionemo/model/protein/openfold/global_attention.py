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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import bionemo.model.protein.openfold.inductor as inductor
from bionemo.data.protein.openfold.helpers import slice_generator
from bionemo.model.protein.openfold.linear import Linear


class GlobalAttention(nn.Module):
    """Global Attention module.

    Args:
        c_e: Extra MSA representation dimension (channels).
        c_hidden: Per-head hidden dimension (channels).
        num_heads: Number of attention heads.
        inf: Safe infinity value.
        eps: Epsilon to prevent division by zero.
        chunk_size: Optional chunk size for a batch-like dimension.

    """

    def __init__(
        self,
        c_e: int,
        c_hidden: int,
        num_heads: int,
        inf: float,
        eps: float,
        chunk_size: Optional[int],
    ) -> None:
        super(GlobalAttention, self).__init__()
        self.c_e = c_e
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.inf = inf
        self.eps = eps
        self.chunk_size = chunk_size
        self.linear_q = Linear(c_e, c_hidden * num_heads, bias=False, init="glorot")
        self.linear_k = Linear(c_e, c_hidden, bias=False, init="glorot")
        self.linear_v = Linear(c_e, c_hidden, bias=False, init="glorot")
        self.linear_g = Linear(c_e, c_hidden * num_heads, bias=True, init="gating")
        self.linear_o = Linear(c_hidden * num_heads, c_e, bias=True, init="final")

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        add_transposed_output_to: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Global Attention forward pass.

        Args:
            m: [batch, N_res, N_extra_seq, c_e] transposed extra MSA representation
            mask: [batch, N_res, N_extra_seq] transposed extra MSA mask
            add_transposed_output_to:
                Optional tensor to which transposed output will be added elementwisely.

        Returns:
            m: [batch, N_extra_seq, N_res, c_e] updated extra MSA representation

        """
        # TODO: this might be the only not guarded mlperf optimisation [optim-hub]
        if self.chunk_size is None:
            return self._forward(
                m=m,
                mask=mask,
                add_transposed_output_to=add_transposed_output_to,
            )
        else:
            return self._forward_chunked(
                m=m,
                mask=mask,
                chunk_size=self.chunk_size,
                add_transposed_output_to=add_transposed_output_to,
            )

    def _forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        add_transposed_output_to: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # torch.cuda..range_push("global_attention")
        q = _mul_sum_x2_add_div(m, mask, self.eps)
        # q: [batch, N_res, c_e]

        q = self.linear_q(q)
        # q: [batch, N_res, num_heads * c_hidden]

        q = q * math.sqrt(1 / self.c_hidden)
        # q: [batch, N_res, num_heads * c_hidden]

        q = q.view(q.shape[:-1] + (self.num_heads, self.c_hidden))
        # q: [batch, N_res, num_heads, c_hidden]

        k = self.linear_k(m)
        # k: [batch, N_res, N_extra_seq, c_hidden]

        v = self.linear_v(m)
        # v: [batch, N_res, N_extra_seq, c_hidden]

        a = torch.matmul(q, k.transpose(-1, -2))
        # a: [batch, N_res, num_heads, N_extra_seq]

        a = _add_softmax(a, mask, self.inf)
        # a: [batch, N_res, num_heads, N_extra_seq]

        o = torch.matmul(a, v)
        # o: [batch, N_res, num_heads, c_hidden]

        g = _linear(m, self.linear_g.weight, self.linear_g.bias)
        # g: [batch, N_res, N_extra_seq, num_heads * c_hidden]

        o = _sigmoid_mul(g, o, self.num_heads, self.c_hidden)
        o = o.reshape(o.shape[:-2] + (self.num_heads * self.c_hidden,))
        # o: [batch, N_res, N_extra_seq, num_heads * c_hidden]

        if add_transposed_output_to is None:
            m = self.linear_o(o)
        else:
            m = _linear_transpose_add(
                o,
                self.linear_o.weight,
                self.linear_o.bias,
                add_transposed_output_to,
            )
        # m: [batch, N_extra_seq, N_res, c_e]

        return m

    def _forward_chunked(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
        add_transposed_output_to: Optional[torch.Tensor],
    ) -> torch.Tensor:
        output_chunks = []
        subbatch_size = m.size(1)
        for left, right in slice_generator(0, subbatch_size, chunk_size):
            m_chunk = m[:, left:right]
            mask_chunk = mask[:, left:right]
            output_chunk = self._forward(
                m=m_chunk,
                mask=mask_chunk,
                add_transposed_output_to=None,
            )
            output_chunks.append(output_chunk)
        out = torch.cat(output_chunks, dim=1)
        if add_transposed_output_to is None:
            return out
        else:
            return add_transposed_output_to + out.transpose(-2, -3)


def _mul_sum_x2_add_div_eager(
    m: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    q_num = torch.sum(m * mask.unsqueeze(-1), dim=-2)
    q_den = torch.sum(mask, dim=-1).add(eps).unsqueeze(-1)
    q = q_num / q_den
    return q


_mul_sum_x2_add_div_jit = torch.compile(_mul_sum_x2_add_div_eager)


def _mul_sum_x2_add_div(
    m: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if inductor.is_enabled():
        mul_sum_x2_add_div_fn = _mul_sum_x2_add_div_jit
    else:
        mul_sum_x2_add_div_fn = _mul_sum_x2_add_div_eager
    return mul_sum_x2_add_div_fn(m, mask, eps)


def _add_softmax_eager(
    a: torch.Tensor,
    mask: torch.Tensor,
    inf: float,
) -> torch.Tensor:
    bias = ((mask - 1.0) * inf).unsqueeze(-2)
    a = a + bias
    a = torch.softmax(a, dim=-1)
    return a


_add_softmax_jit = torch.compile(_add_softmax_eager)


def _add_softmax(
    a: torch.Tensor,
    mask: torch.Tensor,
    inf: float,
) -> torch.Tensor:
    if inductor.is_enabled():
        add_softmax_fn = _add_softmax_jit
    else:
        add_softmax_fn = _add_softmax_eager
    return add_softmax_fn(a, mask, inf)


def _linear_eager(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, w, b)


_linear_jit = torch.compile(_linear_eager)


def _linear(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    # TODO: [optim-hub] re-check condition: can we safely ignore dap / do we take into account hopper
    if inductor.is_enabled_on_hopper():  # and dap.size() == 2:
        linear_fn = _linear_jit
    elif inductor.is_enabled_on_ampere_and_autograd_off():
        linear_fn = _linear_jit
    else:
        linear_fn = _linear_eager
    return linear_fn(x, w, b)


def _sigmoid_mul_eager(
    g: torch.Tensor,
    o: torch.Tensor,
    num_heads: int,
    c_hidden: int,
) -> torch.Tensor:
    g = torch.sigmoid(g)
    g = g.view(g.shape[:-1] + (num_heads, c_hidden))
    o = o.unsqueeze(-3) * g
    return o


_sigmoid_mul_jit = torch.compile(_sigmoid_mul_eager)


def _sigmoid_mul(
    g: torch.Tensor,
    o: torch.Tensor,
    num_heads: int,
    c_hidden: int,
) -> torch.Tensor:
    if inductor.is_enabled():
        sigmoid_mul_fn = _sigmoid_mul_jit
    else:
        sigmoid_mul_fn = _sigmoid_mul_eager
    return sigmoid_mul_fn(g, o, num_heads, c_hidden)


def _linear_transpose_add_eager(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    return out + F.linear(x, w, b).transpose(-2, -3)


_linear_transpose_add_jit = torch.compile(_linear_transpose_add_eager)


def _linear_transpose_add(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if inductor.is_enabled_on_hopper_and_autograd_off():
        linear_transpose_add_fn = _linear_transpose_add_jit
    elif inductor.is_enabled_on_ampere():
        linear_transpose_add_fn = _linear_transpose_add_jit
    else:
        linear_transpose_add_fn = _linear_transpose_add_eager
    return linear_transpose_add_fn(x, w, b, out)
