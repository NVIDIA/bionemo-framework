import torch.nn.functional as F
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, n_out, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads)
        
        # For cross attention, you typically need separate projections
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Feedforward layers
        self.linear1 = nn.Linear(hidden_dim, hidden_dim*2)
        self.linear2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_out)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query_input, key_value_input, attention_mask):
        # query_input: (batch, n_dim) - single query vector per batch
        # key_value_input: (batch, L, n_dim) - sequence of key/value vectors
        
        query = self.query_linear(query_input)  # (batch, n_dim)
        key = self.key_linear(key_value_input)  # (batch, L, n_dim)
        value = self.value_linear(key_value_input)  # (batch, L, n_dim)
        
        # Reshape for MultiheadAttention: (seq_len, batch, embed_dim)
        query = query.unsqueeze(0)  # (1, batch, n_dim)
        key = key.transpose(0, 1)   # (L, batch, n_dim)
        value = value.transpose(0, 1)  # (L, batch, n_dim)
        
        # Create key_padding_mask: True for positions to ignore
        # attention_mask: 1 for valid positions, 0 for padding
        # key_padding_mask: True for padding positions
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        
        # Cross attention
        attn_output, _ = self.cross_attention(query, key, value, key_padding_mask=key_padding_mask)
        
        # Reshape back and add residual connection
        attn_output = attn_output.squeeze(0)  # (batch, n_dim)
        x = query_input + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feedforward block
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        # Final output
        x = self.output(x)
        
        return x