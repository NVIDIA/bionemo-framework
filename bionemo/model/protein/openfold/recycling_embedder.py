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

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import bionemo.model.protein.openfold.inductor as inductor
from bionemo.model.protein.openfold.layer_norm import LayerNorm
from bionemo.model.protein.openfold.linear import Linear


class RecyclingEmbedder(nn.Module):
    """Recycling Embedder module.

    Supplementary '1.10 Recycling iterations'.

    Args:
        c_m: MSA representation dimension (channels).
        c_z: Pair representation dimension (channels).
        min_bin: Smallest distogram bin (Angstroms).
        max_bin: Largest distogram bin (Angstroms).
        num_bins: Number of distogram bins.
        inf: Safe infinity value.

    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        num_bins: int,
        inf: float,
    ) -> None:
        super(RecyclingEmbedder, self).__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.num_bins = num_bins
        self.inf = inf
        self.linear = Linear(self.num_bins, self.c_z, bias=True, init="default")
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        m0_prev: torch.Tensor,
        z_prev: torch.Tensor,
        x_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recycling Embedder forward pass.

        Supplementary '1.10 Recycling iterations': Algorithm 32.

        Args:
            m: [batch, N_clust, N_res, c_m]
            z: [batch, N_res, N_res, c_z]
            m0_prev: [batch, N_res, c_m]
            z_prev: [batch, N_res, N_res, c_z]
            x_prev: [batch, N_res, 3]

        Returns:
            m: [batch, N_clust, N_res, c_m]
            z: [batch, N_res, N_res, c_z]

        """
        self._initialize_buffers(dtype=x_prev.dtype, device=x_prev.device)

        # Embed pair distances of backbone atoms:
        d = self._embed_pair_distances(x_prev)

        # Embed output Evoformer representations:
        z_update = self.layer_norm_z(z_prev)
        m0_update = self.layer_norm_m(m0_prev)
        # z_update: [batch, N_res, N_res, c_z] pair representation update
        # m0_update: [batch, N_res, c_m] first row MSA representation update

        # Update MSA and pair representations:
        m = self._msa_update(m, m0_update)
        z = self._pair_update(z, z_update, d)

        return m, z

    def _embed_pair_distances(self, x_prev: torch.Tensor) -> torch.Tensor:
        # TODO: [optim-hub]: re-check dap conditions
        if inductor.is_enabled_on_hopper():  # and dap.size() >= 2:
            embed_pair_distances_fn = _embed_pair_distances_jit
        elif inductor.is_enabled_and_autograd_off():
            embed_pair_distances_fn = _embed_pair_distances_jit
        else:
            embed_pair_distances_fn = _embed_pair_distances_eager
        d = embed_pair_distances_fn(
            x_prev,
            self.lower,
            self.upper,
            self.linear.weight,
            self.linear.bias,
        )
        return d

    def _msa_update(self, m: torch.Tensor, m0_update: torch.Tensor) -> torch.Tensor:
        m = m.clone()
        m[:, 0] += m0_update
        return m

    def _pair_update(self, z: torch.Tensor, z_update: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if inductor.is_enabled():
            pair_update_fn = _pair_update_jit
        else:
            pair_update_fn = _pair_update_eager
        z = pair_update_fn(z, z_update, d)
        return z

    def _initialize_buffers(self, dtype: torch.dtype, device: torch.device) -> None:
        if not hasattr(self, "lower") or not hasattr(self, "upper"):
            bins = torch.linspace(
                start=self.min_bin,
                end=self.max_bin,
                steps=self.num_bins,
                dtype=dtype,
                device=device,
                requires_grad=False,
            )
            lower = torch.pow(bins, 2)
            upper = torch.roll(lower, shifts=-1, dims=0)
            upper[-1] = self.inf
            self.register_buffer("lower", lower, persistent=False)
            self.register_buffer("upper", upper, persistent=False)


def _embed_pair_distances_eager(
    x_prev: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    d = (x_prev.unsqueeze(-2) - x_prev.unsqueeze(-3)).pow(2).sum(dim=-1, keepdims=True)
    d = torch.logical_and(d > lower, d < upper).to(dtype=x_prev.dtype)
    d = F.linear(d, w, b)
    return d


_embed_pair_distances_jit = torch.compile(_embed_pair_distances_eager)


def _pair_update_eager(
    z: torch.Tensor,
    z_update: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    return z + z_update + delta


_pair_update_jit = torch.compile(_pair_update_eager)
