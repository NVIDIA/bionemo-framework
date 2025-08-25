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

# coding=utf-8
# Copyright 2024 FAESM team. All rights reserved.
# Copyright 2025 NVIDIA Corporation. All rights reserved.
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

# Adapted from https://huggingface.co/fredzzp/esm2_t36_3B_UR50D/blob/main/modeling_faesm.py

"""Flash Attention ESM2 model implementation for Hugging Face Hub."""

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from transformers.models.esm.modeling_esm import (
    EsmAttention,
    EsmContactPredictionHead,
    EsmEmbeddings,
    EsmEncoder,
    EsmForMaskedLM,
    EsmIntermediate,
    EsmLayer,
    EsmLMHead,
    EsmModel,
    EsmOutput,
    EsmPooler,
    EsmPreTrainedModel,
    EsmSelfAttention,
    EsmSelfOutput,
)


logger = logging.getLogger(__name__)

flash_attn_installed = False
# # Flash Attention check
# flash_attn_installed = True and not (os.getenv("DISABLE_FA", "").lower() == "true")
if TYPE_CHECKING:
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input
    from flash_attn.ops.triton.rotary import apply_rotary

#     print("✅ Flash Attention detected - using optimized implementation")
# except ImportError:
#     flash_attn_installed = False
#     print(
#         """
#         ⚠️ Flash Attention not available - using PyTorch SDPA fallback.
#         For optimal performance, install Flash Attention:
#         pip install flash-attn --no-build-isolation
#         """
#     )


# ============================================================================
# Flash Attention Utilities (consolidated from fa_utils.py)
# ============================================================================


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, cos, sin, cu_seqlens, max_seqlen):
        q, k = qkv[:, 0], qkv[:, 1]
        apply_rotary(q, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, inplace=True)
        apply_rotary(k, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, inplace=True)
        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        max_seqlen = ctx.max_seqlen
        cos, sin, cu_seqlens = ctx.saved_tensors
        dq, dk = dqkv[:, 0], dqkv[:, 1]
        apply_rotary(dq, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, inplace=True, conjugate=True)
        apply_rotary(dk, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, inplace=True, conjugate=True)
        return dqkv, None, None, None, None


def apply_rotary_emb_qkv_(qkv, cos, sin, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
    """Apply rotary embedding *inplace* to the first rotary_dim of Q and K."""
    return ApplyRotaryEmbQKV_.apply(qkv, cos, sin, cu_seqlens, max_seqlen)


class RotaryEmbedding(torch.nn.Module):
    """The rotary position embeddings from RoFormer."""

    def __init__(self, dim: int, base=10000.0, pos_idx_in_fp32=True, device=None, persistent=True):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=persistent)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq

            freqs = torch.outer(t, inv_freq)
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(self, qkv: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int, *args, **kwargs) -> torch.Tensor:
        """Apply rotary embedding *inplace*."""
        self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        return apply_rotary_emb_qkv_(
            qkv, self._cos_cached, self._sin_cached, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )


@torch.compiler.disable
def unpad(input, padding_mask):
    """
    Arguments:
        input: (batch, seqlen, ...)
        padding_mask: (batch, seqlen), bool type, True means to keep, False means to remove
    Return:
        output: (total_nnz, ...), where total_nnz = number of tokens in selected in padding_mask
        indices: (total_nnz,), the indices of tokens in the original input
        cu_seqlens: (batch + 1,), the cumulative sequence lengths, used to index into output
        max_seqlen: int, the maximum sequence length in the batch
        output_pad_fn: function, to pad the output back to the original shape
    """
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    output = input.flatten(0, 1)[indices]

    def output_pad_fn(output):
        return pad_input(output, indices, batch=input.shape[0], seqlen=input.shape[1])

    return output, cu_seqlens, max_seqlen, indices, output_pad_fn


# ============================================================================
# Flash Attention ESM Model Implementation
# ============================================================================


class FAEsmSelfAttention(EsmSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.config = config
        if flash_attn_installed:
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

    def forward(self, **kwargs):
        if flash_attn_installed:
            return self.fa_forward(**kwargs)
        else:
            return self.sdpa_forward(**kwargs)

    def sdpa_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        hidden_shape = (hidden_states.shape[0], -1, self.num_attention_heads, self.attention_head_size)

        query_layer = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        query_layer = query_layer * self.attention_head_size**-0.5

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            raise NotImplementedError

        if head_mask is not None:
            raise NotImplementedError

        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()

        context_layer = F.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attn_mask=attention_mask, scale=1.0
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = (*context_layer.size()[:-2], self.all_head_size)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)
        if self.is_decoder:
            outputs = (*outputs, past_key_value)
        return outputs

    def fa_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens,
        max_seqlen,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        assert cu_seqlens is not None, "cu_seqlens must be provided for FlashAttention"
        assert max_seqlen is not None, "max_seqlen must be provided for FlashAttention"

        q = self.query(hidden_states) * self.attention_head_size**-0.5
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q, k, v = (rearrange(x, "n (h d) -> n h d", h=self.num_attention_heads) for x in (q, k, v))
        qkv = torch.stack((q, k, v), dim=1)  # (n, 3, h, d)
        qkv = self.rotary_embeddings(qkv=qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        out = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, softmax_scale=1.0)
        out = rearrange(out, "n h d -> n (h d)")
        outputs = (out,)
        return outputs


class FAEsmAttention(EsmAttention):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.self = FAEsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        hidden_states_ln = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            hidden_states=hidden_states_ln,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output, *self_outputs[1:])
        return outputs


class FAEsmLayer(EsmLayer):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = FAEsmAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = FAEsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
                    " with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output, *outputs)

        if self.is_decoder:
            outputs = (*outputs, present_key_value)
        return outputs


class FAEsmEncoder(EsmEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.layer = nn.ModuleList([FAEsmLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states=hidden_states,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states=hidden_states,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = (*next_decoder_cache, layer_outputs[-1])
            if output_attentions:
                all_self_attentions = (*all_self_attentions, layer_outputs[1])
                if self.config.add_cross_attention:
                    all_cross_attentions = (*all_cross_attentions, layer_outputs[2])

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class FAEsmModel(EsmModel):
    def __init__(self, config, add_pooling_layer=True):
        EsmPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = EsmEmbeddings(config)
        self.encoder = FAEsmEncoder(config)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads, bias=True
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = encoder_attention_mask

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        # Automatically use Flash Attention if available, otherwise use SDPA
        use_fa = flash_attn_installed

        if use_fa:
            embedding_output, cu_seqlens, max_seqlen, _, output_pad_fn = unpad(embedding_output, attention_mask)
        else:
            cu_seqlens = None
            max_seqlen = None

            def output_pad_fn(x):
                return x

        encoder_outputs = self.encoder(
            embedding_output,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = output_pad_fn(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output, *encoder_outputs[1:])

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class FAEsmForMaskedLM(EsmForMaskedLM):
    """Flash Attention ESM For Masked Language Modeling."""

    def __init__(self, config, dropout=0.1):
        config.hidden_dropout_prob = dropout
        EsmPreTrainedModel.__init__(self, config)
        self.esm = FAEsmModel(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_id)

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(prediction_scores.device)
            loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
