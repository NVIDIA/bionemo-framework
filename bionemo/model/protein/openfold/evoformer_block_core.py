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

from bionemo.model.protein.openfold.dropout import DropoutColumnwise, DropoutRowwise
from bionemo.model.protein.openfold.msa_transition import MSATransition
from bionemo.model.protein.openfold.outer_product_mean import OuterProductMean
from bionemo.model.protein.openfold.pair_transition import PairTransition
from bionemo.model.protein.openfold.triangular_attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from bionemo.model.protein.openfold.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


class EvoformerBlockCore(nn.Module):
    """Evoformer Block Core module.

    MSA Transition, Communication and Pair stack for:
    - Supplementary '1.6 Evoformer blocks': Algorithm 6
    - Supplementary '1.7.2 Unclustered MSA stack': Algorithm 18

    Args:
        c_m: MSA (or Extra MSA) representation dimension (channels).
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
        eps_opm: Epsilon to prevent division by zero in outer product mean.
        chunk_size_opm: Optional chunk size for a batch-like dimension
            in outer product mean.
        chunk_size_tri_att: Optional chunk size for a batch-like dimension
            in triangular attention.

    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_opm: int,
        c_hidden_tri_mul: int,
        c_hidden_tri_att: int,
        num_heads_tri: int,
        transition_n: int,
        pair_dropout: float,
        inf: float,
        eps_opm: float,
        chunk_size_opm: Optional[int],
        chunk_size_tri_att: Optional[int],
    ) -> None:
        super(EvoformerBlockCore, self).__init__()
        self.msa_transition = MSATransition(
            c_m=c_m,
            n=transition_n,
        )
        self.outer_product_mean = OuterProductMean(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_opm,
            eps=eps_opm,
            chunk_size=chunk_size_opm,
        )
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z=c_z,
            c_hidden=c_hidden_tri_mul,
        )
        self.tmo_dropout_rowwise = DropoutRowwise(
            p=pair_dropout,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z=c_z,
            c_hidden=c_hidden_tri_mul,
        )
        self.tmi_dropout_rowwise = DropoutRowwise(
            p=pair_dropout,
        )
        self.tri_att_start = TriangleAttentionStartingNode(
            c_z=c_z,
            c_hidden=c_hidden_tri_att,
            num_heads=num_heads_tri,
            inf=inf,
            chunk_size=chunk_size_tri_att,
        )
        self.tasn_dropout_rowwise = DropoutRowwise(
            p=pair_dropout,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            c_z=c_z,
            c_hidden=c_hidden_tri_att,
            num_heads=num_heads_tri,
            inf=inf,
            chunk_size=chunk_size_tri_att,
        )
        self.taen_dropout_columnwise = DropoutColumnwise(
            p=pair_dropout,
        )
        self.pair_transition = PairTransition(
            c_z=c_z,
            n=transition_n,
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evoformer Block Core forward pass.

        Args:
            m: [batch, N_seq, N_res, c_m] MSA (or Extra MSA) representation
            z: [batch, N_res, N_res, c_z] pair representation
            msa_mask: [batch, N_seq, N_res] MSA (or Extra MSA) mask
            pair_mask: [batch, N_res, N_res] pair mask

        Returns:
            m: [batch, N_seq, N_res, c_m] updated MSA (or Extra MSA) representation
            z: [batch, N_res, N_res, c_z] updated pair representation

        """
        m, z = self._forward(m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask)
        return m, z

    def _forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.msa_transition(m=m, mask=msa_mask)
        z = self.outer_product_mean(m=m, mask=msa_mask, add_output_to=z)
        z = self.tmo_dropout_rowwise(
            self.tri_mul_out(z=z, mask=pair_mask),
            add_output_to=z,
        )
        z = self.tmi_dropout_rowwise(
            self.tri_mul_in(z=z, mask=pair_mask),
            add_output_to=z,
        )
        z = self.tasn_dropout_rowwise(
            self.tri_att_start(z=z, mask=pair_mask),
            add_output_to=z,
        )
        z = self.taen_dropout_columnwise(
            self.tri_att_end(z=z, mask=pair_mask),
            add_output_to=z,
        )
        z = self.pair_transition(z, mask=pair_mask)
        return m, z
