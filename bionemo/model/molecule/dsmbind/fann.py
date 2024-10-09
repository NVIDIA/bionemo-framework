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


"""Implementation of a frame averaging neural network (https://arxiv.org/abs/2110.03336) and its dependent modules."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig


# The MultiHeadAttention API implemented in PyTorch 2.0+ has a highly optimized attention kernel but it does not support 2nd derivative computation. So we build one from basic PyTorch operations.
# See details here: https://discuss.pytorch.org/t/2nd-derivative-for-attn-is-not-working/200647
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attention_softmax_in_fp32: bool = True):
        """
        Initialization of the multi-head attention module implemented with basic PyTorch operations.

        Args:
            d_model (int): expected hidden dimentions in the input.
            num_heads (int): number of attention heads. d_model must be divisible by num_heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, depth). Transpose the result such that the shape is (batch_size, num_heads, seq_length, depth).

        Args:
            x (torch.Tensor): a tensor with shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: the split tensor with shape (batch_size, num_heads, seq_length, depth).
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, h: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the multi-head attention layer.

        Args:
            h (torch.Tensor): the sequence to the encoder with shape (batch_size, seq_length, d_model).
            src_key_padding_mask (torch.Tensor): the mask for the keys per batch with shape (batch_size, seq_length).

        Returns:
            torch.Tensor: the output tensor with shape (batch_size, seq_length, d_model).
        """
        batch_size = h.size(0)

        query = self.wq(h)
        key = self.wk(h)
        value = self.wv(h)

        query = self._split_heads(query, batch_size)
        key = self._split_heads(key, batch_size)
        value = self._split_heads(value, batch_size)

        # Scaled dot-product attention
        matmul_qk = torch.matmul(
            query, key.transpose(-2, -1)
        )  # (batch_size, seq_length, d_model) * (batch_size, d_model, seq_length) -> (batch_size, seq_length, seq_length)
        scale = self.depth**0.5
        scaled_attention_logits = matmul_qk / scale

        # Apply the key padding mask
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            scaled_attention_logits = scaled_attention_logits.masked_fill(src_key_padding_mask, float("-inf"))

        # softmax is applied to the last axis (seq_len_k) so that scores add to 1
        original_dtype = scaled_attention_logits.dtype  # Store original dtype
        if (original_dtype == torch.float16 or original_dtype == torch.bfloat16) and self.attention_softmax_in_fp32:
            scaled_attention_logits = scaled_attention_logits.float()  # Convert to float32 for softmax
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        if self.attention_softmax_in_fp32 and original_dtype in (torch.float16, torch.bfloat16):
            attention_weights = attention_weights.to(original_dtype)  # Convert back to original dtype if necessary

        output = torch.matmul(attention_weights, value)  # (batch_size, num_heads, seq_length, depth)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)  # (batch_size, seq_length, d_model)

        output = self.dense(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Initialization of the standard Tranformer encode layer.

        Args:
            d_model (int): expected hidden dimentions in the input.
            num_heads (int): number of attention heads. d_model must be divisible by num_heads.
            dim_feedforward (int): the dimension of the feedforward network model.
            dropout (float): the dropout value.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer encoder layer.

        Args:
            src (torch.Tensor): the sequence to the encoder with shape (batch_size, seq_length, d_model).
            src_key_padding_mask (torch.Tensor): the mask for the src keys per batch with shape (batch_size, seq_length).

        Returns:
            torch.Tensor: the output tensor with shape (batch_size, seq_length, d_model).
        """
        # Multihead attention sublayer with skip connection and layer norm
        src2 = self.multihead_attention(src, src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward sublayer with skip connection and layer norm
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class FANN(nn.Module):
    def __init__(self, cfg_model: DictConfig):
        """
        Initialization of the frame averaging neural network.

        Args:
            cfg_model (DictConfig): model configurations.
        """
        super(FANN, self).__init__()
        self.max_residue_atoms = cfg_model.max_residue_atoms
        self.W_bind = nn.Linear(cfg_model.hidden_size, cfg_model.hidden_size)
        self.W_tgt = nn.Linear(cfg_model.aa_size, cfg_model.hidden_size)
        self.A_bind = nn.Embedding(cfg_model.vocab_size, cfg_model.hidden_size)
        self.A_tgt = nn.Embedding(cfg_model.vocab_size, cfg_model.hidden_size)

        self.linear = nn.Linear(cfg_model.hidden_size * 2 + 3, cfg_model.hidden_size)
        self.encoder = TransformerEncoderLayer(
            d_model=cfg_model.hidden_size,
            num_heads=cfg_model.num_heads,
            dim_feedforward=cfg_model.hidden_size,
            dropout=cfg_model.dropout,
        )

    def _create_frame_and_project(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Creat a set of eight frames and then project the atom coordinates accordingly. For algorithmic details, refer to https://arxiv.org/abs/2110.03336.

        Args:
            X (torch.Tensor): the atom coordinates with shape (batch_size, num_atoms, 3).
            mask (torch.Tensor): the mask for the atoms per batch with shape (batch_size, num_atoms).
        Returns:
            torch.Tensor: the coordinates projected based on eight frames with shape (batch_size*8, num_atoms, 3).
        """
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask  # [B,N,3]
        C = torch.bmm(X.transpose(1, 2), X)  # [B,3,3] (Cov)
        _, V = torch.linalg.eigh(C.detach())
        ops = torch.tensor([[i, j, k] for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]]).to(X.device)
        F_ops = ops.unsqueeze(1).unsqueeze(0) * V.unsqueeze(1)  # [1,8,1,3] x [B,1,3,3] -> [B,8,3,3]
        h = torch.einsum("boij,bpj->bopi", F_ops.transpose(2, 3), X)  # transpose is inverse [B,8,N,3]
        h = h.view(X.size(0) * 8, X.size(1), 3)
        return h

    def forward(
        self,
        binder: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass of the frame averaging neural network.

        Args:
            binder (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): the batched ligand info. The first tensor is the ligand atom coordinates. The second tensor is the encoded ligand atom embeddings. The third tensor is a mask for indicating ligand atoms.
            target (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): the batched target pocket info. The first tensor is the residue coordinates. The second tensor is a one-hot residue embedding. The third tensor represents all residue atoms in the pocket.

        Returns:
            torch.Tensor: the out atom representations, including both ligand and target atoms.
        """
        bind_X, bind_S, bind_A = binder
        tgt_X, tgt_S, tgt_A = target

        # flatten
        B, N, M = bind_S.size(0), bind_S.size(1), tgt_S.size(1)
        bind_S = (
            bind_S[:, :, None, :].expand(-1, -1, self.max_residue_atoms, -1).reshape(B, N * self.max_residue_atoms, -1)
        )
        tgt_S = (
            tgt_S[:, :, None, :].expand(-1, -1, self.max_residue_atoms, -1).reshape(B, M * self.max_residue_atoms, -1)
        )
        bind_A = bind_A.view(B, N * self.max_residue_atoms)
        bind_X = bind_X.view(B, N * self.max_residue_atoms, 3)
        tgt_A = tgt_A.view(B, M * self.max_residue_atoms)
        tgt_X = tgt_X.view(B, M * self.max_residue_atoms, 3)

        bind_mask = bind_A.clamp(max=1).float()  # the mask for ligand atoms. Zero means a padding.
        tgt_mask = tgt_A.clamp(max=1).float()  # the mask for target atoms. Zero means a padding.
        mask = torch.cat([bind_mask, tgt_mask], dim=1)

        bind_S = torch.cat([self.W_bind(bind_S), self.A_bind(bind_A)], dim=-1)  # ligand atom embeddings
        tgt_S = torch.cat([self.W_tgt(tgt_S), self.A_tgt(tgt_A)], dim=-1)  # target atom embeddings

        h_S = torch.cat([bind_S, tgt_S], dim=1)  # all atom embeddings
        X = torch.cat([bind_X, tgt_X], dim=1)  # all atom coordinates
        h_X = self._create_frame_and_project(
            X, mask
        )  # project the atom coordinates to eight frames.  (batch_size*8, (N+M)*self.max_residue_atoms, 3)
        h_S = h_S.unsqueeze(1).expand(-1, 8, -1, -1).reshape(B * 8, (N + M) * self.max_residue_atoms, -1)
        mask = mask.unsqueeze(1).expand(-1, 8, -1).reshape(B * 8, (N + M) * self.max_residue_atoms)

        h = torch.cat([h_X, h_S], dim=-1)

        h = self.linear(h)
        h = self.encoder(h, src_key_padding_mask=~mask.bool())
        h = h.view(B, 8, (N + M) * self.max_residue_atoms, -1)
        h = h.mean(dim=1)  # frame averaging
        return h.view(B, N + M, self.max_residue_atoms, -1)
