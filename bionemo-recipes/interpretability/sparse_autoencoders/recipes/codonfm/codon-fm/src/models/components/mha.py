import torch
from torch import nn
from einops import rearrange

import xformers.ops as xops

from src.models.components.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb

class MultiHeadAttention(nn.Module):
    """
    Multi-Headed Self Attention module using xformers for memory-efficient attention
    and Rotary Positional Embeddings.

    Args:
        embed_dim (int): The dimension of the embedding.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate to apply to the attention scores.
        rotary_theta (float, default=10000.0): The base for the geometric progression
            used to compute the rotation angles for Rotary Positional Embeddings.
        is_causal (bool, default=False): Whether to apply causal masking to prevent
            attention to future positions.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rotary_theta: float = 1e4,
        is_causal: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.is_causal = is_causal
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
        attention_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Performs the forward pass for Multi-Head Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            attention_mask (torch.Tensor): Mask to prevent attention to certain positions.
                It can be a padding mask or a causal mask.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        if attention_mask is not None:
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

        # Handle attention mask
        if attention_mask is not None:
            padding_bias = attention_mask.repeat(1, 1, attention_mask.size(-1), 1)
            padding_bias = padding_bias.to(q.dtype)
            padding_bias = padding_bias.repeat(1, self.num_heads, 1, 1)
            
        if is_causal:
            attn_bias = xops.fmha.attn_bias.LowerTriangularMask()
        else:
            attn_bias = padding_bias

        x = xops.memory_efficient_attention(
            query=q,
            key=k,
            value=v,
            op=None,
            attn_bias=attn_bias,
            p=self.dropout_rate if self.training else 0.0,
        )

        # x: (batch_size, query_seq_len, n_head, head_dim)
        x = rearrange(x, "b q h d -> b q (h d)", h=self.num_heads)
        return x
