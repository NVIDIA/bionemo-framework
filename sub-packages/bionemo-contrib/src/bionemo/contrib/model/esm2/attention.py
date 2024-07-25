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


import math
from typing import Callable, Optional, Union

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor


class ESM2DotProductAttention(DotProductAttention):
    """
    ESM2-Specific core attention

    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models:
    https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
    ) -> None:
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
        )

        self.use_esm_attention = config.use_esm_attention
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        if packed_seq_params is not None:
            raise ValueError(
                "Packed sequence is not supported by DotProductAttention \
                              Please use TEDotProductAttention instead."
            )

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        # attn_mask_type is not used.
        if (np_ng := self.num_attention_heads_per_partition // self.num_query_groups_per_partition) > 1:
            key = key.repeat_interleave(np_ng, dim=2)
            value = value.repeat_interleave(np_ng, dim=2)

        # [b, np, sq, sk]
        b, np, sq, sk = query.size(1), query.size(2), query.size(0), key.size(0)

        # ESM2 Customization
        # [sq, b, np, hn]
        if self.use_esm_attention:
            query = query / math.sqrt(self.hidden_size_per_attention_head)
        # END ESM2 Customization

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use simple strides
        # to extract the queries.
        query = query.reshape(sq, b * np, -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(sk, b * np, -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (b * np, sq, sk),
            query.dtype,
            "mpu",
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),  # [b * np, sq, hn]
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(b, np, sq, sk)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        # ESM2 Customization
        if self.use_esm_attention:
            # NOTE: the slicing here is to make the attention_mask the same shape as the extended
            # attention mask in ESM2. The multiplication by -3.4028e+38 (float32 min_val) is
            # similarly motivated by ESM2's maskikng approach, which forces softmax of attention scores
            # for masked entries to be close to 0. This number is replaced with min_val of the precision
            # using min_val instead of -inf is stable in an special case where all sequence is masked
            min_val = torch.finfo(attention_scores.dtype).min

            attention_probs: Tensor = self.esm2_scale_mask_softmax(
                attention_scores.masked_fill(attention_mask[:, :, 0:1, :].to(bool), min_val)
            )
        # END ESM2 Customization
        else:
            attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        b, np, sq, hn = value.size(1), value.size(2), query.size(0), value.size(3)

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), b * np, -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(b * np, sq, -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(b, np, sq, hn)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        context = context.view(sq, b, self.hidden_size_per_partition)

        return context

    def esm2_scale_mask_softmax(
        self,
        input: Tensor,
        mask: Optional[Tensor] = None,
        scale: Optional[Union[float, int]] = None,
        mask_func: Optional[Callable] = None,
    ) -> Tensor:
        """Scale Mask Softmax function.

        Args:
            input: Tensor of shape (Batch, NP, SK, SQ). The input may or may not have already
                had a mask applied to it.
            mask: If a mask is to be applied, it will go here.
            scale: A scale factor that will be applied before the softmax.
            mask_func: An optional function to apply to the mask. If None, it is assumed that
                the input already had the mask applied to it.

        Returns:
            probs: Tensor of normalized probabilities after the softmax has been applied,
                of shape (Batch, NP, SK, SQ).
        """
        if self.attn_mask_type.name != "padding":
            raise ValueError(
                f"self.attn_mask_type: {self.attn_mask_type} is not 'padding'. Only 'padding' type is supported currently."
            )

        original_dtype = input.dtype  # Store original dtype
        if (original_dtype == torch.float16 or original_dtype == torch.bfloat16) and self.attention_softmax_in_fp32:
            input = input.float()  # Convert to float32 for softmax

        if scale is not None:
            input = input * scale  # Apply scaling

        if mask is not None and mask_func is not None:
            input = mask_func(input, mask)  # Apply mask function if provided

        probs = torch.nn.functional.softmax(input, dim=-1)  # Apply softmax

        if self.attention_softmax_in_fp32 and original_dtype in (torch.float16, torch.bfloat16):
            probs = probs.to(original_dtype)  # Convert back to original dtype if necessary

        return probs
