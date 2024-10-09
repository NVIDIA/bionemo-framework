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
from bionemo.model.protein.openfold.linear import Linear


class AngleResnet(nn.Module):
    """Angle Resnet module.

    Supplementary '1.8 Structure module': Algorithm 20, lines 11-14.

    Args:
        c_s: Single representation dimension (channels).
        c_hidden: Hidden dimension (channels).
        num_blocks: Number of resnet blocks.
        num_angles: Number of torsion angles to generate.
        eps: Epsilon to prevent division by zero.

    """

    def __init__(
        self,
        c_s: int,
        c_hidden: int,
        num_blocks: int,
        num_angles: int,
        eps: float,
    ) -> None:
        super(AngleResnet, self).__init__()
        self.c_s = c_s
        self.c_hidden = c_hidden
        self.num_blocks = num_blocks
        self.num_angles = num_angles
        self.eps = eps
        self.linear_in = Linear(c_s, c_hidden, bias=True, init="default")
        self.linear_initial = Linear(c_s, c_hidden, bias=True, init="default")
        self.layers = nn.ModuleList([AngleResnetBlock(c_hidden=c_hidden) for _ in range(num_blocks)])
        self.linear_out = Linear(c_hidden, num_angles * 2, bias=True, init="default")

    def forward(
        self,
        s: torch.Tensor,
        s_initial: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Angle Resnet forward pass.

        Args:
            s: [batch, N_res, c_s] single representation
            s_initial: [batch, N_res, c_s] initial single representation

        Returns:
            unnormalized_angles: [batch, N_res, num_angles, 2]
            angles: [batch, N_res, num_angles, 2]

        """
        # The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.
        s_initial = self.linear_initial(torch.relu(s_initial))
        s = self.linear_in(torch.relu(s))
        s = s + s_initial
        # s: [batch, N_res, c_hidden]

        for layer in self.layers:
            s = layer(s)
        s = torch.relu(s)
        # s: [batch, N_res, c_hidden]

        s = self.linear_out(s)
        # s: [batch, N_res, num_angles * 2]

        if inductor.is_enabled():
            forward_angles_fn = _forward_angles_jit
        else:
            forward_angles_fn = _forward_angles_eager
        unnormalized_angles, angles = forward_angles_fn(s, self.num_angles, self.eps)
        # unnormalized_angles: [batch, N_res, num_angles, 2]
        # angles: [batch, N_res, num_angles, 2]

        return unnormalized_angles, angles


class AngleResnetBlock(nn.Module):
    """Angle Resnet Block module."""

    def __init__(self, c_hidden: int) -> None:
        super(AngleResnetBlock, self).__init__()
        self.linear_1 = Linear(c_hidden, c_hidden, bias=True, init="relu")
        self.linear_2 = Linear(c_hidden, c_hidden, bias=True, init="final")

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if inductor.is_enabled_on_hopper():
            forward_angle_resnet_block_fn = _forward_angle_resnet_block_jit
        elif inductor.is_enabled_on_ampere_and_autograd_on():
            forward_angle_resnet_block_fn = _forward_angle_resnet_block_jit
        else:
            forward_angle_resnet_block_fn = _forward_angle_resnet_block_eager
        return forward_angle_resnet_block_fn(
            a,
            self.linear_1.weight,
            self.linear_1.bias,
            self.linear_2.weight,
            self.linear_2.bias,
        )


def _forward_angles_eager(
    s: torch.Tensor,
    num_angles: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s = s.view(s.shape[:-1] + (num_angles, 2))
    # s: [batch, N_res, num_angles, 2]
    unnormalized_angles = s
    # unnormalized_angles: [batch, N_res, num_angles, 2]
    norm_denom = torch.sqrt(
        torch.clamp(
            torch.sum(s**2, dim=-1, keepdim=True),
            min=eps,
        )
    )
    angles = s / norm_denom
    return unnormalized_angles, angles


_forward_angles_jit = torch.compile(_forward_angles_eager)


def _forward_angle_resnet_block_eager(
    a: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    x = torch.relu(a)
    x = F.linear(x, w1, b1)
    x = torch.relu(x)
    x = F.linear(x, w2, b2)
    y = a + x
    return y


_forward_angle_resnet_block_jit = torch.compile(_forward_angle_resnet_block_eager)
