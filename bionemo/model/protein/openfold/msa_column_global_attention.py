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

from typing import Optional

import torch
import torch.nn as nn

from bionemo.model.protein.openfold.global_attention import GlobalAttention
from bionemo.model.protein.openfold.layer_norm import LayerNorm


class MSAColumnGlobalAttention(nn.Module):
    """MSA Column Global Attention module.

    Supplementary '1.7.2 Unclustered MSA stack':
    Algorithm 19 MSA global column-wise gated self-attention.

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
        super(MSAColumnGlobalAttention, self).__init__()
        self.layer_norm_m = LayerNorm(c_e)
        self.global_attention = GlobalAttention(
            c_e=c_e,
            c_hidden=c_hidden,
            num_heads=num_heads,
            inf=inf,
            eps=eps,
            chunk_size=chunk_size,
        )

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSA Column Global Attention forward pass.

        Args:
            m: [batch, N_extra_seq, N_res, c_e] extra MSA representation
            mask: [batch, N_extra_seq, N_res] extra MSA mask

        Returns:
            m_update: [batch, N_extra_seq, N_res, c_e] extra MSA representation update

        """
        # TODO: [optim-hub] this is probably the only non-conditioned module
        m_transposed = m.transpose(-2, -3)
        # m_transposed: [batch, N_res, N_extra_seq, c_e]

        mask = mask.transpose(-1, -2)
        # mask: [batch, N_res, N_extra_seq]

        m_transposed_normalized = self.layer_norm_m(m_transposed)
        m = self.global_attention(
            m=m_transposed_normalized,
            mask=mask,
            add_transposed_output_to=m,
        )
        # m: [batch, N_res, N_extra_seq, c_e]

        # m = m.transpose(-2, -3) # TODO: [optim-hub] this has been moved inside global attention?
        # m: [batch, N_extra_seq, N_res, c_e]

        return m
