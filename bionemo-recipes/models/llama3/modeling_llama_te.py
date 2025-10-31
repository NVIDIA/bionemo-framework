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

from collections import OrderedDict
from typing import Unpack

import torch
import torch.nn as nn
import transformer_engine.pytorch
import transformers
from transformer_engine.pytorch.attention import InferenceParams
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformers import LlamaConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.utils.generic import TransformersKwargs


class NVLlamaConfig(LlamaConfig):
    """NVLlama configuration."""

    attn_input_format: str = "bshd"
    self_attn_mask_type: str = "padding_causal"


class NVLlamaPreTrainedModel(PreTrainedModel):
    """Base class for NVLlama models."""

    config: NVLlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ("TransformerLayer",)
    _skip_keys_device_placement = ("past_key_values",)


class NVLlamaModel(NVLlamaPreTrainedModel):
    """Llama3 model implemented in Transformer Engine."""

    def __init__(self, config: LlamaConfig):
        """Initialize the NVLlama model."""
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=config.dtype)
        self.layers = nn.ModuleList(
            [
                transformer_engine.pytorch.TransformerLayer(
                    hidden_size=config.hidden_size,
                    ffn_hidden_size=config.intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    bias=False,
                    layernorm_epsilon=config.rms_norm_eps,
                    hidden_dropout=0,
                    attention_dropout=0,
                    fuse_qkv_params=True,
                    qkv_weight_interleaved=True,
                    normalization="RMSNorm",
                    activation="swiglu",
                    attn_input_format=config.attn_input_format,
                    self_attn_mask_type=config.self_attn_mask_type,
                    num_gqa_groups=config.num_key_value_heads,
                    layer_number=layer_idx + 1,
                    params_dtype=config.dtype,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = transformer_engine.pytorch.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype)

        # We use TE's RotaryPositionEmbedding, but we ensure that we use the same inv_freq as the original
        # LlamaRotaryEmbedding.
        self.rotary_emb = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)
        self.rotary_emb.inv_freq = LlamaRotaryEmbedding(config=config).inv_freq

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: InferenceParams | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """Forward pass for the NVLlama model.

        Args:
            input_ids (torch.Tensor): The input ids.
            attention_mask (torch.Tensor): The attention mask.
            position_ids (torch.Tensor): The position ids.
            past_key_values (tuple[tuple[torch.Tensor, ...], ...]): The past key values.
            inputs_embeds (torch.Tensor): The inputs embeds.
            use_cache (bool): Whether to use cache.
            **kwargs: Additional keyword arguments.

        Returns:
            BaseModelOutputWithPast: The output of the model.
        """
        all_hidden_states = []
        output_hidden_states = kwargs.get("output_hidden_states", False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        if self.config.attn_input_format == "bshd":
            if past_key_values is not None:
                max_seq_len = past_key_values.max_sequence_length
            else:
                max_seq_len = hidden_states.shape[1]
            te_rope_emb = self.rotary_emb(max_seq_len=max_seq_len)
        elif self.config.attn_input_format == "thd":
            te_rope_emb = self.rotary_emb(max_seq_len=kwargs["cu_seq_lens_q"][-1])

        has_thd_input = [
            x is not None
            for x in [
                kwargs.get("cu_seq_lens_q", None),
                kwargs.get("cu_seq_lens_k", None),
                kwargs.get("max_length_q", None),
                kwargs.get("max_length_k", None),
            ]
        ]

        if isinstance(past_key_values, InferenceParams):
            # lengths = attention_mask.sum(dim=1) if attention_mask is not None else torch.tensor([0])
            lengths = input_ids.ne(0).sum(dim=1) if input_ids is not None else torch.tensor([0])
            past_key_values.pre_step(OrderedDict(zip(list(range(len(lengths))), lengths.tolist())))

        if self.config.attn_input_format == "thd":
            if not all(has_thd_input):
                raise ValueError(
                    "cu_seq_lens_q, cu_seq_lens_k, max_length_q, and max_length_k must be provided when using THD inputs."
                )
            assert hidden_states.dim() == 3 and hidden_states.size(0) == 1, (
                "THD expects embeddings shaped [1, total_tokens, hidden_size]."
            )
            hidden_states = hidden_states.squeeze(0)
            attention_mask = None

        elif self.config.attn_input_format == "bshd" and any(has_thd_input):
            raise ValueError(
                "cu_seq_lens_q, cu_seq_lens_k, max_length_q, and max_length_k are not allowed when using BSHD inputs."
            )

        # Construct the appropriate attention mask.
        if attention_mask is not None and self.config.self_attn_mask_type == "padding_causal":
            attention_mask = ~attention_mask.to(bool)[:, None, None, :]

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=te_rope_emb,
                inference_params=past_key_values,
                cu_seqlens_q=kwargs.get("cu_seq_lens_q", None),
                cu_seqlens_kv=kwargs.get("cu_seq_lens_k", None),
                max_seqlen_q=kwargs.get("max_length_q", None),
                max_seqlen_kv=kwargs.get("max_length_k", None),
            )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )


class NVLlamaForCausalLM(NVLlamaPreTrainedModel, transformers.GenerationMixin):
    """Llama3 model with causal language head."""

    _tied_weights_keys = ("lm_head.weight",)

    def __init__(self, config):
        """Initialize the NVLlamaForCausalLM model."""
        super().__init__(config)
        self.model = NVLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = transformer_engine.pytorch.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            params_dtype=config.dtype,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, ...], ...] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        only_keep_last_logits: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """Forward pass for the NVLlamaForCausalLM model.

        Args:
            input_ids (torch.Tensor): The input ids.
            attention_mask (torch.Tensor): The attention mask.
            position_ids (torch.Tensor): The position ids.
            past_key_values (tuple[tuple[torch.Tensor, ...], ...]): The past key values.
            inputs_embeds (torch.Tensor): The inputs embeds.
            labels (torch.Tensor): The labels.
            use_cache (bool): Whether to use cache.
            cache_position (torch.Tensor): The cache position.
            only_keep_last_logits (bool): Whether to keep only the last logits, as a workaround for the fact that TE
                doesn't support left-side padding with `padding_causal` attention masks.
            **kwargs: Additional keyword arguments.

        Returns:
            CausalLMOutputWithPast: The output of the model.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # TE doesn't support left-side padding with `padding_causal` attention masks, and InferenceParams doesn't
        # support arbitrary attention masks (and the attention backend for arbitrary masks is the slower, unfused
        # backend). To keep the inference interface consistent with HF's `GenerationMixin.generate` interface, we use a
        # `only_keep_last_logits` flag to indicate that we should pick out and return only the last token's hidden state
        # during pre-fill. This allows generation to work with right-side padding. Note, make sure that you decode the
        # tokens with `skip_special_tokens=True` when using this flag, otherwise padding tokens will interrupt the
        # generated text.
        if (
            only_keep_last_logits
            and attention_mask is not None  # Padded inputs
            and hidden_states.shape[1] > 1  # We're in pre-fill mode
        ):
            seqlens = attention_mask.sum(dim=1)  # shape: [batch]
            # For each batch idx, select hidden_states[idx, seqlens[idx]-1, :]
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            selected_hidden_states = hidden_states[batch_indices, seqlens - 1, :]  # shape: [batch, hidden_dim]
            hidden_states = selected_hidden_states.unsqueeze(1)  # shape: [batch, 1, hidden_dim]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NVLlamaForSequenceClassification(  # noqa: D101
    transformers.modeling_layers.GenericForSequenceClassification, NVLlamaPreTrainedModel
): ...


class NVLlamaForQuestionAnswering(transformers.modeling_layers.GenericForQuestionAnswering, NVLlamaPreTrainedModel):
    """Llama3 model with question answering head."""

    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class NVLlamaForTokenClassification(  # noqa: D101
    transformers.modeling_layers.GenericForTokenClassification, NVLlamaPreTrainedModel
): ...
