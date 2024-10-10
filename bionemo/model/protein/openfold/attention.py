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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import bionemo.model.protein.openfold.triton.mha as mha
from bionemo.data.protein.openfold.helpers import slice_generator
from bionemo.model.protein.openfold import inductor
from bionemo.model.protein.openfold.linear import Linear, gating_init_
from bionemo.model.protein.openfold.triton import AttnBiasJIT, AttnNoBiasJIT, AttnTri, CanSchTriMHA


class Attention(nn.Module):
    """Multi-Head Attention module.

    Args:
        c_q: Input dimension of query data tensor (channels).
        c_k: Input dimension of key data tensor (channels).
        c_v: Input dimension of value data tensor (channels).
        c_hidden: Hidden dimension (per-head).
        num_heads: Number of attention heads.
        gating: Whether the output should be gated using query data tensor.
        inf: Safe infinity value.
        chunk_size: Optional chunk size for a batch-like dimension.
            Supplementary '1.11.8 Reducing the memory consumption': Inference.

    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        num_heads: int,
        gating: bool,
        inf: float,
        chunk_size: Optional[int],
    ) -> None:
        super(Attention, self).__init__()
        assert c_k == c_v
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.gating = gating
        self.inf = inf
        self.chunk_size = chunk_size
        self.linear_q = Linear(c_q, c_hidden * num_heads, bias=False, init="glorot")
        self.linear_k = Linear(c_k, c_hidden * num_heads, bias=False, init="glorot")
        self.linear_v = Linear(c_v, c_hidden * num_heads, bias=False, init="glorot")
        self.linear_o = Linear(c_hidden * num_heads, c_q, bias=True, init="final")
        if gating:
            self.linear_g = Linear(c_q, c_hidden * num_heads, bias=True, init="gating")

    def forward(
        self,
        input_q: torch.Tensor,
        input_k: torch.Tensor,
        input_v: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Attention forward pass.

        Args:
            input_q: [*, Q, c_q] query data
            input_k: [*, K, c_k] key data
            input_v: [*, V, c_v] value data
            mask: Logit mask tensor broadcastable to [*, num_heads, Q, K]
            bias: Optional logit bias tensor broadcastable to [*, num_heads, Q, K]

        Returns:
            output: [*, Q, c_q] tensor

        """
        query, key, value = self._prep_qkv(input_q, input_k, input_v)
        # query: [*, num_heads, Q, c_hidden]
        # key:   [*, num_heads, K, c_hidden]
        # value: [*, num_heads, V, c_hidden]

        output = self._attention_forward(query, key, value, mask, bias)
        # output: [*, num_heads, Q, c_hidden]
        del query, key, value

        output = output.transpose(-2, -3)
        # output: [*, Q, num_heads, c_hidden]

        output = self._wrap_up(output, input_q)
        # output: [*, Q, c_q]

        return output

    def _prep_qkv(
        self,
        input_q: torch.Tensor,
        input_k: torch.Tensor,
        input_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_q: [*, Q, c_q]
        # input_k: [*, K, c_k]
        # input_v: [*, V, c_v]

        q = self.linear_q(input_q)
        k = self.linear_k(input_k)
        v = self.linear_v(input_v)
        # q: [*, Q, num_heads * c_hidden]
        # k: [*, K, num_heads * c_hidden]
        # v: [*, V, num_heads * c_hidden]

        q = q.view(q.shape[:-1] + (self.num_heads, self.c_hidden))
        k = k.view(k.shape[:-1] + (self.num_heads, self.c_hidden))
        v = v.view(v.shape[:-1] + (self.num_heads, self.c_hidden))
        # q: [*, Q, num_heads, c_hidden]
        # k: [*, K, num_heads, c_hidden]
        # v: [*, V, num_heads, c_hidden]

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        # q: [*, num_heads, Q, c_hidden]
        # k: [*, num_heads, K, c_hidden]
        # v: [*, num_heads, V, c_hidden]

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.chunk_size is None:
            return _attention(query, key, value, mask, bias, self.inf)
        else:
            return _attention_chunked(query, key, value, mask, bias, self.inf, self.chunk_size)

    def _wrap_up(
        self,
        output: torch.Tensor,
        input_q: torch.Tensor,
    ) -> torch.Tensor:
        if self.gating:
            gate = torch.sigmoid(self.linear_g(input_q))
            # gate: [*, Q, num_heads * c_hidden]
            gate = gate.view(gate.shape[:-1] + (self.num_heads, self.c_hidden))
            # gate: [*, Q, num_heads, c_hidden]
            output = output * gate
            # output: [*, Q, num_heads, c_hidden]

        output = output.reshape(output.shape[:-2] + (self.num_heads * self.c_hidden,))
        # output: [*, Q, num_heads * c_hidden]

        output = self.linear_o(output)
        # output: [*, Q, c_q]

        return output


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    bias: Optional[torch.Tensor],
    inf: float,
) -> torch.Tensor:
    # query:  [*, num_heads, Q, c_hidden]
    # key:    [*, num_heads, K, c_hidden]
    # value:  [*, num_heads, V, c_hidden]
    # mask:   Logit mask tensor broadcastable to [*, num_heads, Q, K]
    # bias:   Optional logit bias tensor broadcastable to [*, num_heads, Q, K]
    # inf:    Safe infinity value.
    # assuming K == V

    key = torch.swapdims(key, -2, -1)
    # key: [*, num_heads, c_hidden, K]

    a = torch.matmul(query, key)
    # a: [*, num_heads, Q, K]

    a += (mask - 1.0) * inf
    # a: [*, num_heads, Q, K]

    if bias is not None:
        a += bias
    # a: [*, num_heads, Q, K]

    a = torch.softmax(a, dim=-1)
    # a: [*, num_heads, Q, K]

    a = torch.matmul(a, value)
    # a: [*, num_heads, Q, c_hidden]

    return a


def _attention_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    bias: Optional[torch.Tensor],
    inf: float,
    chunk_size: int,
) -> torch.Tensor:
    output_chunks = []
    subbatch_size = query.size(1)
    for left, right in slice_generator(0, subbatch_size, chunk_size):
        query_chunk = query[:, left:right]
        key_chunk = key[:, left:right]
        value_chunk = value[:, left:right]
        mask_chunk = mask[:, left:right] if mask.size(1) == subbatch_size else mask
        bias_chunk = bias[:, left:right] if bias is not None and bias.size(1) == subbatch_size else bias
        output_chunk = _attention(
            query=query_chunk,
            key=key_chunk,
            value=value_chunk,
            mask=mask_chunk,
            bias=bias_chunk,
            inf=inf,
        )
        output_chunks.append(output_chunk)
    return torch.cat(output_chunks, dim=1)


def _attention_gate_eager(
    output: torch.Tensor,
    gate: torch.Tensor,
    linear_g_bias: torch.Tensor,
) -> torch.Tensor:
    gate = torch.sigmoid(gate + linear_g_bias)
    output = output * gate
    return output


_attention_gate_jit = torch.compile(_attention_gate_eager)


class SelfAttentionWithGate(nn.Module):
    """Self Multi-Head Attention module with gating.

    Args:
        c_qkv: Input dimension of query|key|value data tensor (channels).
        c_hidden: Hidden dimension (per-head).
        num_heads: Number of attention heads.
        inf: Safe infinity value.
        chunk_size: Optional chunk size for a batch-like dimension.
            Supplementary '1.11.8 Reducing the memory consumption': Inference.

    """

    def __init__(
        self,
        c_qkv: int,
        c_hidden: int,
        num_heads: int,
        inf: float,
        chunk_size: Optional[int],
    ) -> None:
        super(SelfAttentionWithGate, self).__init__()
        self.c_qkv = c_qkv
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.inf = inf
        self.chunk_size = chunk_size
        self.linear_qkvg = Linear(c_qkv, 4 * c_hidden * num_heads, bias=False, init="glorot")
        gating_init_(self.linear_qkvg.weight.data[3 * c_hidden * num_heads :])
        self.qkvg_split_shape = (
            c_hidden * num_heads,
            c_hidden * num_heads,
            c_hidden * num_heads,
            c_hidden * num_heads,
        )
        self.linear_g_bias = nn.Parameter(torch.zeros(c_hidden * num_heads))
        self.linear_o = Linear(c_hidden * num_heads, c_qkv, bias=True, init="final")

    def forward(
        self,
        input_qkv: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor],
        add_transposed_output_to: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attention forward pass.

        Args:
            input_qkv: [*, QKV, c_qkv] query data (QKV == Q == K == V)
            mask: Logit mask tensor broadcastable to [*, num_heads, Q, K]
            bias: Optional logit bias tensor broadcastable to [*, num_heads, Q, K]
            add_transposed_output_to:
                Optional tensor to which transposed output will be added elementwisely.

        Returns:
            output: [*, Q, c_qkv] tensor

        """
        query, key, value, gate = self._prep_qkvg(input_qkv)
        # query: [*, num_heads, Q, c_hidden]
        # key:   [*, num_heads, K, c_hidden]
        # value: [*, num_heads, V, c_hidden]
        # gate:  [*, Q, num_heads * c_hidden]

        output = self._attention_forward(query, key, value, mask, bias)
        # output: [*, num_heads, Q, c_hidden]
        del query, key, value

        output = rearrange(output, "1 b h s d -> 1 b s (h d)")
        # output: [*, Q, num_heads * c_hidden]

        if self.training:
            output = _attention_gate_jit(output, gate, self.linear_g_bias)
        else:
            output = _attention_gate_eager(output, gate, self.linear_g_bias)
        # output: [*, Q, num_heads * c_hidden]

        if add_transposed_output_to is None:
            output = self.linear_o(output)
        else:
            output = _linear_transpose_add(
                output,
                self.linear_o.weight,
                self.linear_o.bias,
                add_transposed_output_to,
            )
        # output: [*, Q, c_qkv]

        return output

    def _prep_qkvg(
        self,
        input_qkv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_qkv: [*, QKV, c_qkv]

        qkvg = self.linear_qkvg(input_qkv)
        q, k, v, g = torch.split(qkvg, self.qkvg_split_shape, dim=-1)
        # q: [*, Q, num_heads * c_hidden]
        # k: [*, K, num_heads * c_hidden]
        # v: [*, V, num_heads * c_hidden]
        # g: [*, Q, num_heads * c_hidden]

        q = q.view(q.shape[:-1] + (self.num_heads, self.c_hidden))
        k = k.view(k.shape[:-1] + (self.num_heads, self.c_hidden))
        v = v.view(v.shape[:-1] + (self.num_heads, self.c_hidden))
        # q: [*, Q, num_heads, c_hidden]
        # k: [*, K, num_heads, c_hidden]
        # v: [*, V, num_heads, c_hidden]

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        # q: [*, num_heads, Q, c_hidden]
        # k: [*, num_heads, K, c_hidden]
        # v: [*, num_heads, V, c_hidden]

        # q = q / math.sqrt(self.c_hidden) scaling moved to _attention function

        return q, k, v, g

    def _attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.chunk_size is None:
            if mha.is_enabled() and CanSchTriMHA(
                list(query.shape),
                bias is not None,
                inf=self.inf,
                training=self.training,
            ):
                if mask is not None:
                    mask = mask.contiguous()
                if bias is not None:
                    bias = bias.contiguous()
                return AttnTri(query, key, value, mask, bias, self.inf, torch.is_grad_enabled())
            elif mha.is_enabled() and bias is not None and self.training:
                return AttnBiasJIT(query, key, value, mask, bias, self.inf)
            elif mha.is_enabled() and bias is None and self.training:
                return AttnNoBiasJIT(query, key, value, mask, self.inf)
            elif not self.training:  # set for all child modules of a nn.Module when calling self.eval/self.train
                # source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.train
                return _attention_eager(query, key, value, mask, bias, self.inf)
            else:
                return _attention_jit(query, key, value, mask, bias, self.inf)
            # elif dist.is_async_val_enabled() and dist.is_val_rank():
            #     # TODO: we met illegal memory access on eval node when enable async eval
            #     return _attention_eager(query, key, value, mask, bias, self.inf)
        else:
            return _attention_chunked(query, key, value, mask, bias, self.inf, self.chunk_size)


class CrossAttentionNoGate(nn.Module):
    """Cross Multi-Head Attention module without gating.

    Args:
        c_q: Input dimension of query data tensor (channels).
        c_kv: Input dimension of key|value data tensor (channels).
        c_hidden: Hidden dimension (per-head).
        num_heads: Number of attention heads.
        inf: Safe infinity value.
        chunk_size: Optional chunk size for a batch-like dimension.
            Supplementary '1.11.8 Reducing the memory consumption': Inference.

    """

    def __init__(
        self,
        c_q: int,
        c_kv: int,
        c_hidden: int,
        num_heads: int,
        inf: float,
        chunk_size: Optional[int],
    ) -> None:
        super(CrossAttentionNoGate, self).__init__()
        self.c_q = c_q
        self.c_kv = c_kv
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.inf = inf
        self.chunk_size = chunk_size
        self.linear_q = Linear(c_q, c_hidden * num_heads, bias=False, init="glorot")
        self.linear_kv = Linear(c_kv, 2 * c_hidden * num_heads, bias=False, init="glorot")
        self.kv_split_shape = (
            c_hidden * num_heads,
            c_hidden * num_heads,
        )
        self.linear_o = Linear(c_hidden * num_heads, c_q, bias=True, init="final")

    def forward(
        self,
        input_q: torch.Tensor,
        input_kv: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Attention forward pass.

        Args:
            input_q: [*, Q, c_q] query data
            input_kv: [*, KV, c_kv] key|value data (KV == K == V)
            mask: Logit mask tensor broadcastable to [*, num_heads, Q, KV]
            bias: Optional logit bias tensor broadcastable to [*, num_heads, Q, KV]

        Returns:
            output: [*, Q, c_q] tensor

        """
        query, key, value = self._prep_qkv(input_q, input_kv)
        # query: [*, num_heads, Q, c_hidden]
        # key:   [*, num_heads, K, c_hidden]
        # value: [*, num_heads, V, c_hidden]

        output = self._attention_forward(query, key, value, mask, bias)
        # output: [*, num_heads, Q, c_hidden]
        del query, key, value

        output = output.transpose(-2, -3)
        # output: [*, Q, num_heads, c_hidden]

        output = output.reshape(output.shape[:-2] + (self.num_heads * self.c_hidden,))
        # output: [*, Q, num_heads * c_hidden]

        output = self.linear_o(output)
        # output: [*, Q, c_q]

        return output

    def _prep_qkv(
        self,
        input_q: torch.Tensor,
        input_kv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_q: [*, Q, c_q]
        # input_kv: [*, KV, c_kv]

        q = self.linear_q(input_q)
        kv = self.linear_kv(input_kv)
        k, v = torch.split(kv, self.kv_split_shape, dim=-1)
        # q: [*, Q, num_heads * c_hidden]
        # k: [*, K, num_heads * c_hidden]
        # v: [*, V, num_heads * c_hidden]

        q = q.view(q.shape[:-1] + (self.num_heads, self.c_hidden))
        k = k.view(k.shape[:-1] + (self.num_heads, self.c_hidden))
        v = v.view(v.shape[:-1] + (self.num_heads, self.c_hidden))
        # q: [*, Q, num_heads, c_hidden]
        # k: [*, K, num_heads, c_hidden]
        # v: [*, V, num_heads, c_hidden]

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        # q: [*, num_heads, Q, c_hidden]
        # k: [*, num_heads, K, c_hidden]
        # v: [*, num_heads, V, c_hidden]

        # q = q / math.sqrt(self.c_hidden) scaling moved to _attention function

        return q, k, v

    def _attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.chunk_size is None:
            # TODO: we met illegal memory access on eval node when enable async eval
            # if dist.is_async_val_enabled() and dist.is_val_rank():
            #     return _attention_eager(query, key, value, mask, bias, self.inf)
            if not self.training:  # set for all child modules of a nn.Module when calling self.eval/self.train
                # source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.train
                return _attention_eager(query, key, value, mask, bias, self.inf)
            else:
                return _attention_jit(query, key, value, mask, bias, self.inf)
        else:
            return _attention_chunked(query, key, value, mask, bias, self.inf, self.chunk_size)


def _attention_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    bias: Optional[torch.Tensor],
    inf: float,
) -> torch.Tensor:
    # query:  [*, num_heads, Q, c_hidden]
    # key:    [*, num_heads, K, c_hidden]
    # value:  [*, num_heads, V, c_hidden]
    # mask:   Logit mask tensor broadcastable to [*, num_heads, Q, K]
    # bias:   Optional logit bias tensor broadcastable to [*, num_heads, Q, K]
    # inf:    Safe infinity value.
    # assuming K == V

    key = torch.swapdims(key, -2, -1)
    # key: [*, num_heads, c_hidden, K]

    scaling = 1.0 / math.sqrt(query.size(-1))
    a = torch.matmul(query * scaling, key)
    # a: [*, num_heads, Q, K]

    a += (mask - 1.0) * inf
    # a: [*, num_heads, Q, K]

    if bias is not None:
        a += bias
    # a: [*, num_heads, Q, K]

    a = torch.softmax(a, dim=-1)
    # a: [*, num_heads, Q, K]

    a = torch.matmul(a, value)
    # a: [*, num_heads, Q, c_hidden]

    return a


_attention_jit = torch.compile(_attention_eager)


def _attention_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    bias: Optional[torch.Tensor],
    inf: float,
    chunk_size: int,
) -> torch.Tensor:
    output_chunks = []
    subbatch_size = query.size(1)
    # TODO: we met illegal memory access on eval node when enable async eval
    # if dist.is_async_val_enabled() and dist.is_val_rank():
    #     attention_fn = _attention_eager
    # else:
    attention_fn = _attention_jit
    for left, right in slice_generator(0, subbatch_size, chunk_size):
        query_chunk = query[:, left:right]
        key_chunk = key[:, left:right]
        value_chunk = value[:, left:right]
        mask_chunk = mask[:, left:right] if mask.size(1) == subbatch_size else mask
        bias_chunk = bias[:, left:right] if bias is not None and bias.size(1) == subbatch_size else bias
        output_chunk = attention_fn(
            query=query_chunk,
            key=key_chunk,
            value=value_chunk,
            mask=mask_chunk,
            bias=bias_chunk,
            inf=inf,
        )
        output_chunks.append(output_chunk)
    return torch.cat(output_chunks, dim=1)


def _linear_transpose_add_eager(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return y + F.linear(x, w, b).transpose(-2, -3)


_linear_transpose_add_jit = torch.compile(_linear_transpose_add_eager)


def _linear_transpose_add(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    # if inductor.is_enabled_on_hopper() and dap.size() in {2, 8}:
    #     linear_transpose_add_fn = _linear_transpose_add_jit
    if inductor.is_enabled_on_hopper_and_autograd_off():
        linear_transpose_add_fn = _linear_transpose_add_jit
    elif inductor.is_enabled_on_ampere():
        linear_transpose_add_fn = _linear_transpose_add_jit
    else:
        linear_transpose_add_fn = _linear_transpose_add_eager
    return linear_transpose_add_fn(x, w, b, y)
