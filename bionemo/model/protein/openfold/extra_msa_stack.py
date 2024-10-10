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

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as gradient_checkpointing_fn

from bionemo.model.protein.openfold.extra_msa_block import ExtraMSABlock


class ExtraMSAStack(nn.Module):
    """Extra MSA Stack module.

    Supplementary '1.7.2 Unclustered MSA stack'.

    Args:
        c_e: Extra MSA representation dimension (channels).
        c_z: Pair representation dimension (channels).
        c_hidden_msa_att: Hidden dimension in MSA attention.
        c_hidden_opm: Hidden dimension in outer product mean.
        c_hidden_tri_mul: Hidden dimension in multiplicative updates.
        c_hidden_tri_att: Hidden dimension in triangular attention.
        num_heads_msa: Number of heads used in MSA attention.
        num_heads_tri: Number of heads used in triangular attention.
        num_blocks: Number of blocks in the stack.
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
        num_blocks: int,
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
        super(ExtraMSAStack, self).__init__()
        self.blocks = nn.ModuleList(
            [
                ExtraMSABlock(
                    c_e=c_e,
                    c_z=c_z,
                    c_hidden_msa_att=c_hidden_msa_att,
                    c_hidden_opm=c_hidden_opm,
                    c_hidden_tri_mul=c_hidden_tri_mul,
                    c_hidden_tri_att=c_hidden_tri_att,
                    num_heads_msa=num_heads_msa,
                    num_heads_tri=num_heads_tri,
                    transition_n=transition_n,
                    msa_dropout=msa_dropout,
                    pair_dropout=pair_dropout,
                    inf=inf,
                    eps=eps,
                    eps_opm=eps_opm,
                    chunk_size_msa_att=chunk_size_msa_att,
                    chunk_size_opm=chunk_size_opm,
                    chunk_size_tri_att=chunk_size_tri_att,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        gradient_checkpointing: bool,
    ) -> torch.Tensor:
        """Extra MSA Stack forward pass.

        Args:
            m: [batch, N_extra_seq, N_res, c_e] extra MSA representation
            z: [batch, N_res, N_res, c_z] pair representation
            msa_mask: [batch, N_extra_seq, N_res] extra MSA mask
            pair_mask: [batch, N_res, N_res] pair mask
            gradient_checkpointing: whether to use gradient checkpointing

        Returns:
            z: [batch, N_res, N_res, c_z] updated pair representation

        """
        if gradient_checkpointing:
            assert torch.is_grad_enabled()
            z = self._forward_blocks_with_gradient_checkpointing(
                m=m,
                z=z,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
            )
        else:
            z = self._forward_blocks(
                m=m,
                z=z,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
            )
        return z

    def _forward_blocks(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            m, z = block(m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask)
        return z

    def _forward_blocks_with_gradient_checkpointing(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        blocks = [
            partial(
                block,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
            )
            for block in self.blocks
        ]
        for block in blocks:
            m, z = gradient_checkpointing_fn(block, m, z)
        return z
