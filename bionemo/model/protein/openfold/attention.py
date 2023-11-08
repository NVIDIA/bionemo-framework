# Copyright 2021 DeepMind Technologies Limited
# Copyright 2022 AlQuraishi Laboratory
# Copyright 2023 NVIDIA CORPORATION
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

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from bionemo.data.protein.openfold.helpers import slice_generator
from bionemo.model.protein.openfold.linear import Linear


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
