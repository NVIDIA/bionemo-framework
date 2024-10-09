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

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import bionemo.model.protein.openfold.inductor as inductor
from bionemo.data.protein.openfold.helpers import slice_generator
from bionemo.model.protein.openfold.layer_norm import LayerNorm
from bionemo.model.protein.openfold.linear import Linear
from bionemo.model.protein.openfold.utils.torch_utils import is_autocast_fp16_enabled


class OuterProductMean(nn.Module):
    """Outer Product Mean module.

    Supplementary '1.6.4 Outer product mean': Algorithm 10.

    Args:
        c_m: MSA (or Extra MSA) representation dimension (channels).
        c_z: Pair representation dimension (channels).
        c_hidden: Hidden dimension (channels).
        eps: Epsilon to prevent division by zero.
        chunk_size: Optional chunk size for a batch-like dimension.

    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden: int,
        eps: float,
        chunk_size: Optional[int],
    ) -> None:
        super(OuterProductMean, self).__init__()
        assert eps == 1e-3
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps
        self.chunk_size = chunk_size
        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden, bias=True, init="default")
        self.linear_2 = Linear(c_m, c_hidden, bias=True, init="default")
        self.linear_out = Linear(c_hidden**2, c_z, bias=True, init="final")

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        add_output_to: torch.Tensor,
    ) -> torch.Tensor:
        """Outer Product Mean forward pass.

        Args:
            m: [batch, N_seq, N_res, c_m] MSA representation
            mask: [batch, N_seq, N_res] MSA mask
            add_output_to: pair representation to which add outer update

        Returns:
            outer: [batch, N_res, N_res, c_z] updated pair representation

        """
        if is_autocast_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(m.float(), mask, add_output_to)
        else:
            return self._forward(m, mask, add_output_to)

    def _forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        add_output_to: torch.Tensor,
    ) -> torch.Tensor:
        m = self.layer_norm(m)
        # m: [batch, N_seq, N_res, c_m]

        mask = mask.unsqueeze(-1)
        # mask: [batch, N_seq, N_res, 1]

        a, b = _forward_linear_a_b(
            m,
            self.linear_1.weight,
            self.linear_1.bias,
            self.linear_2.weight,
            self.linear_2.bias,
            mask,
        )
        # a: [batch, N_seq, N_res, c_hidden]
        # b: [batch, N_seq, N_res, c_hidden]

        if inductor.is_enabled():
            # TODO: does it work with chunked forward?
            outer = _forward_outer_jit(
                a,
                b,
                self.linear_out.weight,
                self.linear_out.bias,
                a.shape[0],  # batch
                a.shape[2],  # a_N_res
                b.shape[2],  # b_N_res
                a.shape[3],  # c_hidden
            )
        else:
            a = a.transpose(-2, -3)
            # a: [batch, N_res, N_seq, c_hidden]
            b = b.transpose(-2, -3)
            # b: [batch, N_res, N_seq, c_hidden]
            outer = self._outer_forward(a=a, b=b)
            # outer: [batch, N_res, N_res, c_z]

        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        # norm: [batch, N_res, N_res, 1]

        outer = _forward_normalize_add(norm, outer, add_output_to, self.eps)
        # outer: [batch, N_res, N_res, c_z]

        return outer

    def _outer_forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.chunk_size is None:
            return self._outer(a=a, b=b)
        else:
            return self._outer_chunked(a=a, b=b, chunk_size=self.chunk_size)

    def _outer(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        outer = torch.einsum("...bac,...dae->...bdce", a, b)
        # outer: [batch, a_N_res, b_N_res, c_hidden, c_hidden]
        outer = outer.reshape(outer.shape[:-2] + (self.c_hidden * self.c_hidden,))
        # outer: [batch, a_N_res, b_N_res, c_hidden * c_hidden]
        outer = self.linear_out(outer)
        # outer: [batch, a_N_res, b_N_res, c_z]
        return outer

    def _outer_chunked(self, a: torch.Tensor, b: torch.Tensor, chunk_size: int) -> torch.Tensor:
        outer_chunks = []
        subbatch_size = a.size(1)
        for left, right in slice_generator(0, subbatch_size, chunk_size):
            a_chunk = a[:, left:right]
            outer_chunk = self._outer(a=a_chunk, b=b)
            outer_chunks.append(outer_chunk)
        return torch.cat(outer_chunks, dim=1)


def _forward_linear_a_b_eager(
    m: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    a = F.linear(m, w1, b1) * mask
    b = F.linear(m, w2, b2) * mask
    return a, b


_forward_linear_a_b_jit = torch.compile(_forward_linear_a_b_eager)


def _forward_linear_a_b(
    m: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: [optim-hub] re-check dap conditions
    if inductor.is_enabled_on_hopper():
        forward_linear_a_b_fn = _forward_linear_a_b_jit
    elif inductor.is_enabled_on_ampere_and_autograd_off():
        forward_linear_a_b_fn = _forward_linear_a_b_jit
    else:
        forward_linear_a_b_fn = _forward_linear_a_b_eager
    return forward_linear_a_b_fn(m, w1, b1, w2, b2, mask)


def _forward_outer_eager(
    a: torch.Tensor,
    b: torch.Tensor,
    w_o: torch.Tensor,
    b_o: torch.Tensor,
    batch: int,
    a_N_res: int,
    b_N_res: int,
    c_hidden: int,
) -> torch.Tensor:
    # a: [batch, N_seq, N_res, c_hidden]
    a = a.transpose(-2, -3)
    # a: [batch, N_res, N_seq, c_hidden]
    a = a.transpose(-1, -2)
    # a: [batch, N_res, c_hidden, N_seq]
    a = a.flatten(1, 2)
    # a: [batch, N_res * c_hidden, N_seq]

    # b: [batch, N_seq, N_res, c_hidden]
    b = b.flatten(-2, -1)
    # b: [batch, N_seq, N_res * c_hidden]

    outer = torch.bmm(a, b)
    # outer: [batch, a_N_res * c_hidden, b_N_res * c_hidden]

    outer = outer.reshape((batch, a_N_res, c_hidden, b_N_res, c_hidden))
    # outer: [batch, a_N_res, c_hidden, b_N_res, c_hidden]
    outer = outer.transpose(2, 3)
    # outer: [batch, a_N_res, b_N_res, c_hidden, c_hidden]
    outer = outer.flatten(-2, -1)
    # outer: [batch, a_N_res, b_N_res, c_hidden * c_hidden]

    outer = F.linear(outer, w_o, b_o)
    # outer: [batch, a_N_res, b_N_res, c_z]

    return outer


_forward_outer_jit = torch.compile(_forward_outer_eager)


def _forward_normalize_add_eager(
    norm: torch.Tensor,
    outer: torch.Tensor,
    z: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    outer = outer / (norm + eps)
    return z + outer


_forward_normalize_add_jit = torch.compile(_forward_normalize_add_eager)


def _forward_normalize_add(
    norm: torch.Tensor,
    outer: torch.Tensor,
    z: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if inductor.is_enabled():
        forward_normalize_add_fn = _forward_normalize_add_jit
    else:
        forward_normalize_add_fn = _forward_normalize_add_eager
    return forward_normalize_add_fn(norm, outer, z, eps)
