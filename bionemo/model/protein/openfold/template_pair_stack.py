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

from bionemo.model.protein.openfold.layer_norm import LayerNorm
from bionemo.model.protein.openfold.template_pair_block import TemplatePairBlock


class TemplatePairStack(nn.Module):
    """Template Pair Stack module.

    Supplementary '1.7.1 Template stack': Algorithm 16.

    Args:
        c_t: Template representation dimension (channels).
        c_hidden_tri_att: Hidden dimension in triangular attention.
        c_hidden_tri_mul: Hidden dimension in multiplicative updates.
        num_blocks: Number of blocks in the stack.
        num_heads_tri: Number of heads used in triangular attention.
        pair_transition_n: Channel multiplier in pair transition.
        dropout_rate: Dropout rate for pair activations.
        inf: Safe infinity value.
        chunk_size_tri_att: Optional chunk size for a batch-like dimension
            in triangular attention.

    """

    def __init__(
        self,
        c_t: int,
        c_hidden_tri_att: int,
        c_hidden_tri_mul: int,
        num_blocks: int,
        num_heads_tri: int,
        pair_transition_n: int,
        dropout_rate: float,
        inf: float,
        chunk_size_tri_att: Optional[int],
    ) -> None:
        super(TemplatePairStack, self).__init__()
        self.blocks = nn.ModuleList(
            [
                TemplatePairBlock(
                    c_t=c_t,
                    c_hidden_tri_att=c_hidden_tri_att,
                    c_hidden_tri_mul=c_hidden_tri_mul,
                    num_heads_tri=num_heads_tri,
                    pair_transition_n=pair_transition_n,
                    dropout_rate=dropout_rate,
                    inf=inf,
                    chunk_size_tri_att=chunk_size_tri_att,
                )
                for _ in range(num_blocks)
            ]
        )
        self.layer_norm = LayerNorm(c_t)

    def forward(
        self,
        t: torch.tensor,
        mask: torch.tensor,
        gradient_checkpointing: bool,
    ) -> torch.Tensor:
        """Template Pair Stack forward pass.

        Args:
            t: [batch, N_templ, N_res, N_res, c_t] template representation
            mask: [batch, N_res, N_res] pair mask
            gradient_checkpointing: whether to use gradient checkpointing

        Returns:
            t: [batch, N_templ, N_res, N_res, c_t] updated template representation

        """
        if gradient_checkpointing:
            assert torch.is_grad_enabled()
            t = self._forward_blocks_with_gradient_checkpointing(t=t, mask=mask)
        else:
            t = self._forward_blocks(t=t, mask=mask)
        t = self.layer_norm(t)
        return t

    def _forward_blocks(
        self,
        t: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            t = block(t=t, mask=mask)
        return t

    def _forward_blocks_with_gradient_checkpointing(
        self,
        t: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        blocks = [
            partial(
                block,
                mask=mask,
            )
            for block in self.blocks
        ]
        for block in blocks:
            t = gradient_checkpointing_fn(block, t)
        return t
