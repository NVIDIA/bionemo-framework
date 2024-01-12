# Copyright 2021 DeepMind Technologies Limited
# Copyright 2022 AlQuraishi Laboratory
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Tuple

import torch
import torch.nn as nn

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
        m0_prev: torch.Tensor,
        z_prev: torch.Tensor,
        x_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recycling Embedder forward pass.

        Supplementary '1.10 Recycling iterations': Algorithm 32.

        Args:
            m0_prev: [batch, N_res, c_m]
            z_prev: [batch, N_res, N_res, c_z]
            x_prev: [batch, N_res, 3]

        Returns:
            m0_update: [batch, N_res, c_m]
            z_update: [batch, N_res, N_res, c_z]

        """
        # Embed pair distances of backbone atoms:
        bins = torch.linspace(
            start=self.min_bin,
            end=self.max_bin,
            steps=self.num_bins,
            dtype=x_prev.dtype,
            device=x_prev.device,
            requires_grad=False,
        )
        lower = torch.pow(bins, 2)
        upper = torch.roll(lower, shifts=-1, dims=0)
        upper[-1] = self.inf
        d = (x_prev.unsqueeze(-2) - x_prev.unsqueeze(-3)).pow(2).sum(dim=-1, keepdims=True)
        d = torch.logical_and(d > lower, d < upper).to(dtype=x_prev.dtype)
        d = self.linear(d)

        # Embed output Evoformer representations:
        z_update = d + self.layer_norm_z(z_prev)
        m0_update = self.layer_norm_m(m0_prev)

        # m0_update: [batch, N_res, c_m] first row MSA representation update
        # z_update: [batch, N_res, N_res, c_z] pair representation update
        return m0_update, z_update
