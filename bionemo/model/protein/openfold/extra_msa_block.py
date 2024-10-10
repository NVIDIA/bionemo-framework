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

from bionemo.model.protein.openfold.dropout import DropoutRowwise
from bionemo.model.protein.openfold.evoformer_block_core import EvoformerBlockCore
from bionemo.model.protein.openfold.msa_column_global_attention import MSAColumnGlobalAttention
from bionemo.model.protein.openfold.msa_row_attention_with_pair_bias import MSARowAttentionWithPairBias


class ExtraMSABlock(nn.Module):
    """Extra MSA Block module.

    Supplementary '1.7.2 Unclustered MSA stack': Algorithm 18.

    Args:
        c_e: Extra MSA representation dimension (channels).
        c_z: Pair representation dimension (channels).
        c_hidden_msa_att: Hidden dimension in MSA attention.
        c_hidden_opm: Hidden dimension in outer product mean.
        c_hidden_tri_mul: Hidden dimension in multiplicative updates.
        c_hidden_tri_att: Hidden dimension in triangular attention.
        num_heads_msa: Number of heads used in MSA attention.
        num_heads_tri: Number of heads used in triangular attention.
        transition_n: Channel multiplier in transitions.
        msa_dropout: Dropout rate for MSA activations.
        pair_dropout: Dropout rate for pair activations.
        inf: Safe infinity value.
        eps: Epsilon to prevent division by zero.
        eps_opm: Epsilon to prevent division by zero in outer product mean.
        chunk_size_msa_att: Optional chunk size for a batch-like dimension
            in MSA attention.
        chunk_size_opm: Optional chunk size for a batch-like dimension
            in outer product mean.
        chunk_size_tri_att: Optional chunk size for a batch-like dimension
            in triangular attention.

    """

    def __init__(
        self,
        c_e: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_tri_mul: int,
        c_hidden_tri_att: int,
        num_heads_msa: int,
        num_heads_tri: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
        eps_opm: float,
        chunk_size_msa_att: Optional[int],
        chunk_size_opm: Optional[int],
        chunk_size_tri_att: Optional[int],
    ) -> None:
        super(ExtraMSABlock, self).__init__()
        self.msa_att_row = MSARowAttentionWithPairBias(
            c_m=c_e,
            c_z=c_z,
            c_hidden=c_hidden_msa_att,
            num_heads=num_heads_msa,
            inf=inf,
            chunk_size=chunk_size_msa_att,
        )
        self.msa_att_col = MSAColumnGlobalAttention(
            c_e=c_e,
            c_hidden=c_hidden_msa_att,
            num_heads=num_heads_msa,
            inf=inf,
            eps=eps,
            chunk_size=chunk_size_msa_att,
        )
        self.msa_dropout_rowwise = DropoutRowwise(
            p=msa_dropout,
        )
        self.core = EvoformerBlockCore(
            c_m=c_e,
            c_z=c_z,
            c_hidden_opm=c_hidden_opm,
            c_hidden_tri_mul=c_hidden_tri_mul,
            c_hidden_tri_att=c_hidden_tri_att,
            num_heads_tri=num_heads_tri,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
            eps_opm=eps_opm,
            chunk_size_opm=chunk_size_opm,
            chunk_size_tri_att=chunk_size_tri_att,
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extra MSA Block forward pass.

        Args:
            m: [batch, N_extra_seq, N_res, c_e] extra MSA representation
            z: [batch, N_res, N_res, c_z] pair representation
            msa_mask: [batch, N_extra_seq, N_res] extra MSA mask
            pair_mask: [batch, N_res, N_res] pair mask

        Returns:
            m: [batch, N_extra_seq, N_res, c_e] updated extra MSA representation
            z: [batch, N_res, N_res, c_z] updated pair representation

        """
        m = self.msa_dropout_rowwise(
            self.msa_att_row(m=m, z=z, mask=msa_mask),
            add_output_to=m,
        )
        m = self.msa_att_col(m=m, mask=msa_mask)
        m, z = self.core(
            m=m,
            z=z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
        )
        return m, z
