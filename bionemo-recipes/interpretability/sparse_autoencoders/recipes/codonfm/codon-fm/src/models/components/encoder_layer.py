from typing import Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import Module
from transformers.activations import ACT2FN

from .mha import MultiHeadAttention

class EncoderLayer(nn.Module):
    """
    Encoder layer module.
    This module contains a multi-head attention layer followed by a position-wise feed-forward layer.
    The architecture uses a non-standard dual LayerNorm setup: one before the main transformation
    (attention/FFN) and another one within the sub-layer, before the residual connection.

    The data flow is as follows:
    Attention block: 
        x_attn = MHA(LN(x))
        x = x + Dropout(Dense(LN(x_attn)))
    FFN block:
        x_ffn = Act(Dense_in(LN(x)))
        x = x + Dropout(Dense_out(LN(x_ffn)))
    where LN is LayerNorm.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Attention Block
        self.pre_attn_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = MultiHeadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            rotary_theta=config.rotary_theta,
        )
        # This LayerNorm is applied after attention and before the dense layer.
        self.post_attn_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.post_attn_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)

        # FFN Block
        self.pre_ffn_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.intermediate_dense = nn.Linear(
            config.hidden_size, config.intermediate_size
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        # This LayerNorm is applied within the feed-forward network.
        self.post_ffn_layer_norm = nn.LayerNorm(
            config.intermediate_size, eps=config.layer_norm_eps
        )
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ffn_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.seq_len_dim = 1

    def ffn(self, ff_input):
        # The FFN block is applied in chunks to save memory.
        # FFN: y' = y + Dropout(Dense_out(LN(Act(Dense_in(LN(y))))))
        # Apply pre-FFN layer normalization.
        ff_hidden_states = self.pre_ffn_layer_norm(ff_input)
        ff_hidden_states = self.intermediate_dense(ff_hidden_states)
        ff_hidden_states = self.intermediate_act_fn(ff_hidden_states)
        # Apply post-FFN layer normalization.
        ff_hidden_states = self.post_ffn_layer_norm(ff_hidden_states)
        ff_hidden_states = self.output_dense(ff_hidden_states)
        ff_hidden_states = self.ffn_dropout(ff_hidden_states)
        
        # Second residual connection.
        return ff_hidden_states + ff_input
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        
        # Attention block: x' = x + Dropout(Dense(LN(MHA(LN(x)))))
        # Apply pre-attention layer normalization.
        attn_input = self.pre_attn_layer_norm(hidden_states)
        attn_output = self.attention(
            attn_input,
            attention_mask=attention_mask,
            is_causal=is_causal,
        )

        # Post-attention processing.
        attn_output = self.post_attn_layer_norm(attn_output)
        attn_output = self.post_attn_dense(attn_output)
        attn_output = self.attn_dropout(attn_output)
        
        # First residual connection.
        attention_output = hidden_states + attn_output
        
        layer_output = self.ffn(attention_output)
        return layer_output


