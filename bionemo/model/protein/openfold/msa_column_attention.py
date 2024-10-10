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

from bionemo.model.protein.openfold.attention import Attention, SelfAttentionWithGate
from bionemo.model.protein.openfold.layer_norm import LayerNorm
from bionemo.model.protein.openfold.optim_hub import OptimHub


class MSAColumnAttention(nn.Module):
    """MSA Column Attention module.

    Supplementary '1.6.2 MSA column-wise gated self-attention': Algorithm 8.

    Args:
        c_m: MSA representation dimension (channels).
        c_hidden: Per-head hidden dimension (channels).
        num_heads: Number of attention heads.
        inf: Safe infinity value.
        chunk_size: Optional chunk size for a batch-like dimension.

    """

    def __init__(
        self,
        c_m: int,
        c_hidden: int,
        num_heads: int,
        inf: float,
        chunk_size: Optional[int],
    ) -> None:
        super(MSAColumnAttention, self).__init__()
        self.layer_norm_m = LayerNorm(c_m)
        if OptimHub.config("mha_fused_gemm"):  # [optim-hub]
            self.mha = SelfAttentionWithGate(
                c_qkv=c_m,
                c_hidden=c_hidden,
                num_heads=num_heads,
                inf=inf,
                chunk_size=chunk_size,
            )
        else:
            self.mha = Attention(
                c_q=c_m,
                c_k=c_m,
                c_v=c_m,
                c_hidden=c_hidden,
                num_heads=num_heads,
                gating=True,
                inf=inf,
                chunk_size=chunk_size,
            )

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSA Column Attention forward pass.

        Args:
            m: [batch, N_seq, N_res, c_m] MSA representation
            mask: [batch, N_seq, N_res] MSA mask

        Returns:
            m_update: [batch, N_seq, N_res, c_m] MSA representation update

        """
        m_transposed = m.transpose(-2, -3)
        # m: [batch, N_res, N_seq, c_m]

        mask = mask.transpose(-1, -2)
        # mask: [batch, N_res, N_seq]

        mask = mask.unsqueeze(-2).unsqueeze(-3)
        # mask: [batch, N_res, 1, 1, N_seq]

        m_transposed_normalized = self.layer_norm_m(m_transposed)
        if OptimHub.config("mha_fused_gemm"):
            m = self.mha(
                input_qkv=m_transposed_normalized,
                mask=mask,
                bias=None,
                add_transposed_output_to=m,
            )
        else:
            m = self.mha(
                input_q=m_transposed_normalized,
                input_k=m_transposed_normalized,
                input_v=m_transposed_normalized,
                mask=mask,
                bias=None,
            )
            m = m.transpose(-2, -3)
        # m: [batch, N_seq, N_res, c_m]

        return m
