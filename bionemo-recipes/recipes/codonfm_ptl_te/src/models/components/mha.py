# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import torch
from einops import rearrange
from torch import nn

from src.models.components.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb


try:
    import xformers.ops as xops

    HAVE_XFORMERS = True
except ImportError:
    xops = None
    HAVE_XFORMERS = False


class MultiHeadAttention(nn.Module):
    """Multi-Headed Self Attention module using xformers for memory-efficient attention
    and Rotary Positional Embeddings.

    Args:
        embed_dim (int): The dimension of the embedding.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate to apply to the attention scores.
        rotary_theta (float, default=10000.0): The base for the geometric progression
            used to compute the rotation angles for Rotary Positional Embeddings.
    """  # noqa: D205

    def __init__(  # noqa: D107
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rotary_theta: float = 1e4,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.rotary_emb = RotaryEmbedding(
            dim=embed_dim // num_heads,
            theta=rotary_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Performs the forward pass for Multi-Head Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            attention_mask (torch.Tensor): Mask to prevent attention to certain positions.
                It can be a padding mask or a causal mask.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # These assertions are likely due to hardware constraints or specific optimizations
        # in the version of xformers being used, which may require sequence lengths
        # to be a multiple of 8 for optimal performance with Tensor Cores.
        assert attention_mask.shape[-1] % 8 == 0, "attention_mask must be divisible by 8"

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = rearrange(q, "b q (h d) -> b q h d", h=self.num_heads)
        k = rearrange(k, "b k (h d) -> b k h d", h=self.num_heads)
        v = rearrange(v, "b v (h d) -> b v h d", h=self.num_heads)

        # Apply rotary positional embeddings to query and key.
        cos, sin = self.rotary_emb(q)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        #
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        # torch native sdpa is numerically equivalent to Memory-efficient attention from xformers. we replaced xformers with torch native sdpa to be able to build the Transformer Engine model in the same container.
        x = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attention_mask,
            dropout_p=self.dropout_rate if self.training else 0.0,
        )

        # x: (batch_size, query_seq_len, n_head, head_dim)
        x = rearrange(x, "b h q d -> b q (h d)", h=self.num_heads)

        return x
