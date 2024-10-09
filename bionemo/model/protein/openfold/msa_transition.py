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

import torch
import torch.nn as nn
import torch.nn.functional as F

import bionemo.model.protein.openfold.inductor as inductor
from bionemo.model.protein.openfold.layer_norm import LayerNorm
from bionemo.model.protein.openfold.linear import Linear


class MSATransition(nn.Module):
    """MSA Transition module.

    Supplementary '1.6.3 MSA transition': Algorithm 9.

    Args:
        c_m: MSA (or Extra MSA) representation dimension (channels).
        n: `c_m` multiplier to obtain hidden dimension (channels).

    """

    def __init__(
        self,
        c_m: int,
        n: int,
    ) -> None:
        super(MSATransition, self).__init__()
        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, n * c_m, bias=True, init="relu")
        self.linear_2 = Linear(n * c_m, c_m, bias=True, init="final")

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSA Transition forward pass.

        Args:
            m: [batch, N_seq, N_res, c_m] MSA representation
            mask: [batch, N_seq, N_res] MSA mask

        Returns:
            m: [batch, N_seq, N_res, c_m] updated MSA representation

        """
        # DeepMind forgets to apply the MSA mask here.
        # todo(jxin): open fwd when dap >= 1?
        # TODO: [optim-hub] re-check conditions on dap
        if inductor.is_enabled_on_ampere():
            forward_fn = _forward_jit
        elif inductor.is_enabled_on_hopper_and_autograd_off():
            forward_fn = _forward_jit
        else:
            forward_fn = _forward_eager
        return forward_fn(
            self.layer_norm(m),
            self.linear_1.weight,
            self.linear_1.bias,
            self.linear_2.weight,
            self.linear_2.bias,
            m,
        )


def _forward_eager(
    m: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m = F.linear(m, w1, b1)
    m = torch.relu(m)
    m = F.linear(m, w2, b2)
    m = out + m
    return m


_forward_jit = torch.compile(_forward_eager)
