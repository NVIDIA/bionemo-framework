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
import math
from typing import Unpack

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch
import transformers
from transformer_engine.pytorch.attention import InferenceParams
from transformer_engine.pytorch.attention.inference import PagedKVCacheManager
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformers import LlamaConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.utils.generic import TransformersKwargs


AUTO_MAP = {
    "AutoConfig": "modeling_llama_te.NVLlamaConfig",
    "AutoModel": "modeling_llama_te.NVLlamaModel",
    "AutoModelForCausalLM": "modeling_llama_te.NVLlamaForCausalLM",
    "AutoModelForSequenceClassification": "modeling_llama_te.NVLlamaForSequenceClassification",
    "AutoModelForQuestionAnswering": "modeling_llama_te.NVLlamaForQuestionAnswering",
    "AutoModelForTokenClassification": "modeling_llama_te.NVLlamaForTokenClassification",
}


class NVLlamaConfig(LlamaConfig):
    """NVLlama configuration."""

    attn_input_format: str = "thd"
    self_attn_mask_type: str = "padding_causal"
    use_moe: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 1
    moe_capacity_factor: float = 1.25
    moe_min_capacity: int = 4
    moe_drop_tokens: bool = True
    moe_aux_loss_coef: float = 0.01


class NVLlamaPreTrainedModel(PreTrainedModel):
    """Base class for NVLlama models."""

    config_class = NVLlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ("TransformerLayer", "NVLlamaMoETransformerLayer")
    _skip_keys_device_placement = ("past_key_values",)

    def init_empty_weights(self):
        """Handles moving the model from the meta device to the cuda device and initializing the weights."""
        # For TE layers, calling `reset_parameters` is sufficient to move them to the cuda device and apply the weight
        # initialization we passed them during module creation.
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        # The esm.embeddings layer is the only non-TE layer in this model we need to deal with. We use
        # `model._init_weights` rather than `reset_parameters` to ensure we honor the original config standard
        # deviation.
        self.model.embed_tokens.to_empty(device="cuda")
        self.model.embed_tokens.apply(self._init_weights)

        self.model.rotary_emb.inv_freq = LlamaRotaryEmbedding(config=self.model.config).inv_freq.to("cuda")

        # Meta-device init seems to break weight tying, so we re-tie the weights here.
        self.tie_weights()

    def _init_weights(self, module):
        """Initialize module weights.

        We only use this method for standard pytorch modules, TE modules handle their own weight initialization through
        `init_method` parameters and the `reset_parameters` method.
        """
        if module.__module__.startswith("transformer_engine.pytorch"):
            # Notably, we need to avoid calling this method for TE modules, since the default _init_weights will assume
            # any class with `LayerNorm` in the name should have weights initialized to 1.0; breaking `LayerNormLinear`
            # and `LayerNormMLP` modules that use `weight` for the linear layer and `layer_norm_weight` for the layer
            # norm.
            return

        super()._init_weights(module)


def _te_device() -> str:
    return "meta" if torch.get_default_device() == torch.device("meta") else "cuda"


class NVLlamaMoEFeedForward(nn.Module):
    """MoE feed-forward network using Transformer Engine grouped GEMMs."""

    def __init__(self, config: NVLlamaConfig):
        super().__init__()
        if config.moe_top_k < 1:
            raise ValueError("moe_top_k must be >= 1.")

        self.num_experts = config.moe_num_experts
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.intermediate_size
        self.top_k = config.moe_top_k
        self.capacity_factor = config.moe_capacity_factor
        self.min_capacity = config.moe_min_capacity
        self.drop_tokens = config.moe_drop_tokens

        self.router = nn.Linear(
            config.hidden_size,
            config.moe_num_experts,
            bias=False,
            dtype=config.dtype,
            device=_te_device(),
        )

        def _init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        self.fc1_gate = transformer_engine.pytorch.GroupedLinear(
            config.hidden_size,
            config.intermediate_size,
            num_gemms=config.moe_num_experts,
            bias=False,
            params_dtype=config.dtype,
            device=_te_device(),
            init_method=_init_method,
        )
        self.fc1_up = transformer_engine.pytorch.GroupedLinear(
            config.hidden_size,
            config.intermediate_size,
            num_gemms=config.moe_num_experts,
            bias=False,
            params_dtype=config.dtype,
            device=_te_device(),
            init_method=_init_method,
        )
        self.fc2 = transformer_engine.pytorch.GroupedLinear(
            config.intermediate_size,
            config.hidden_size,
            num_gemms=config.moe_num_experts,
            bias=False,
            params_dtype=config.dtype,
            device=_te_device(),
            init_method=_init_method,
        )

    def _compute_capacity(self, num_tokens: int) -> int:
        base_capacity = math.ceil(self.capacity_factor * num_tokens * self.top_k / self.num_experts)
        return max(self.min_capacity, base_capacity)

    def _select_expert_tokens(
        self,
        flat_expert_idx: torch.Tensor,
        flat_probs: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        if not self.drop_tokens:
            return torch.ones_like(flat_expert_idx, dtype=torch.bool)

        selected = torch.zeros_like(flat_expert_idx, dtype=torch.bool)
        for expert_id in range(self.num_experts):
            expert_mask = flat_expert_idx == expert_id
            if not torch.any(expert_mask):
                continue
            expert_positions = torch.nonzero(expert_mask, as_tuple=False).squeeze(-1)
            if expert_positions.numel() <= capacity:
                selected[expert_positions] = True
                continue
            expert_probs = flat_probs[expert_positions]
            top_positions = torch.topk(expert_probs, k=capacity, sorted=False).indices
            selected[expert_positions[top_positions]] = True
        return selected

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))

        logits = self.router(hidden_states)
        router_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        routing_map = torch.zeros(
            hidden_states.size(0),
            self.num_experts,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        routing_map.scatter_(1, topk_indices, 1)

        importance = router_probs.sum(dim=0)
        load = routing_map.to(router_probs.dtype).sum(dim=0)
        aux_loss = self.num_experts * torch.sum(importance * load) / (
            hidden_states.size(0) * self.top_k
        )

        token_ids = torch.arange(hidden_states.size(0), device=hidden_states.device)
        flat_token_idx = token_ids[:, None].expand(-1, self.top_k).reshape(-1)
        flat_expert_idx = topk_indices.reshape(-1)
        flat_probs = topk_probs.to(hidden_states.dtype).reshape(-1)

        capacity = self._compute_capacity(hidden_states.size(0))
        selected_mask = self._select_expert_tokens(flat_expert_idx, flat_probs, capacity)

        selected_token_idx = flat_token_idx[selected_mask]
        selected_expert_idx = flat_expert_idx[selected_mask]
        selected_probs = flat_probs[selected_mask]

        if selected_token_idx.numel() == 0:
            output = hidden_states.new_zeros((hidden_states.size(0), self.hidden_size))
            if len(original_shape) == 3:
                output = output.view(original_shape)
            return output, aux_loss

        sort_order = torch.argsort(selected_expert_idx, stable=True)
        selected_token_idx = selected_token_idx[sort_order]
        selected_expert_idx = selected_expert_idx[sort_order]
        selected_probs = selected_probs[sort_order]

        permuted = hidden_states.index_select(0, selected_token_idx)
        m_splits = torch.bincount(selected_expert_idx, minlength=self.num_experts).tolist()
        gate_out = self.fc1_gate(permuted, m_splits)
        up_out = self.fc1_up(permuted, m_splits)
        moe_out = F.silu(gate_out) * up_out
        moe_out = self.fc2(moe_out, m_splits)

        output = hidden_states.new_zeros((hidden_states.size(0), self.hidden_size))
        output.index_add_(0, selected_token_idx, moe_out * selected_probs.unsqueeze(-1))

        if len(original_shape) == 3:
            output = output.view(original_shape)
        return output, aux_loss


class NVLlamaMoETransformerLayer(nn.Module):
    """Transformer block with TE attention and MoE MLP."""

    def __init__(self, config: NVLlamaConfig, layer_number: int):
        super().__init__()
        self.attn_input_format = config.attn_input_format
        self.self_attn_mask_type = config.self_attn_mask_type

        self.attn_norm = transformer_engine.pytorch.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            device=_te_device(),
        )
        self.self_attn = transformer_engine.pytorch.MultiheadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_gqa_groups=config.num_key_value_heads,
            attention_dropout=0.0,
            layernorm_epsilon=config.rms_norm_eps,
            bias=False,
            attn_mask_type=config.self_attn_mask_type,
            qkv_format=config.attn_input_format,
            qkv_weight_interleaved=True,
            layer_number=layer_number,
            params_dtype=config.dtype,
            device=_te_device(),
        )
        self.mlp_norm = transformer_engine.pytorch.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            device=_te_device(),
        )
        self.mlp = NVLlamaMoEFeedForward(config)
        self.last_moe_aux_loss: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        inference_params: InferenceParams | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_kv: torch.Tensor | None = None,
        cu_seqlens_q_padded: torch.Tensor | None = None,
        cu_seqlens_kv_padded: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_kv: int | None = None,
        pad_between_seqs: bool | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        attn_out = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            attn_mask_type=self.self_attn_mask_type,
            rotary_pos_emb=rotary_pos_emb,
            inference_params=inference_params,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            pad_between_seqs=pad_between_seqs,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        mlp_out, aux_loss = self.mlp(hidden_states)
        self.last_moe_aux_loss = aux_loss
        hidden_states = residual + mlp_out
        return hidden_states


class NVLlamaModel(NVLlamaPreTrainedModel):
    """Llama3 model implemented in Transformer Engine."""

    def __init__(self, config: LlamaConfig):
        """Initialize the NVLlama model."""
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=config.dtype)

        def _init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        if config.use_moe:
            self.layers = nn.ModuleList(
                [NVLlamaMoETransformerLayer(config, layer_idx + 1) for layer_idx in range(config.num_hidden_layers)]
            )
        else:
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
                        device=_te_device(),
                        init_method=_init_method,
                        output_layer_init_method=_init_method,
                    )
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )
        self.norm = transformer_engine.pytorch.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            device=_te_device(),
        )

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
        moe_aux_loss: torch.Tensor | None = None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # TE-specific input handling.
        has_thd_input = [x in kwargs for x in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]]
        should_pack_inputs = not any(has_thd_input) and self.config.attn_input_format == "thd"

        if should_pack_inputs:
            # Left-side padding is not supported in TE layers, so to make huggingface-style generation work with TE we
            # dynamically convert to THD-style inputs in our forward pass, and then convert back to BSHD for the output.
            # This lets the entire transformer stack run in THD mode. This might be slower for BSHD + padding with fused
            # attention backend, but it should be faster for the flash attention backend.
            assert attention_mask is not None, "Attention mask is required when packing BSHD inputs."
            batch_size = hidden_states.size(0)
            hidden_states, indices, cu_seqlens, max_seqlen, _ = _unpad_input(hidden_states, attention_mask)
            kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_k"] = cu_seqlens
            kwargs["max_length_q"] = kwargs["max_length_k"] = max_seqlen

        if self.config.attn_input_format == "thd" and hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            # For THD, the embedding output is a 3-dimensional tensor with shape [1, total_tokens, hidden_size], but TE
            # expects a 2-dimensional tensor with shape [total_tokens, hidden_size].
            hidden_states = hidden_states.squeeze(0)

        if self.config.attn_input_format == "bshd" and attention_mask is not None and attention_mask.dim() == 2:
            # If we're using padded BSHD inputs, we need to convert the 2-dimensional mask to a 4-dimensional mask in
            # the expected boolean format for TE.
            attention_mask = attention_mask[:, None, None, :] < -1

        if isinstance(past_key_values, InferenceParams):  # InferenceParams is TE's way of managing kv-caching.
            # In generation mode, we set the length to 1 for each batch index. Otherwise, we use the attention mask to
            # compute the lengths of each sequence in the batch.
            lengths = (
                attention_mask.sum(dim=1).tolist()
                if attention_mask.shape == input_ids.shape
                else [1] * input_ids.shape[0]
            )
            past_key_values.pre_step(OrderedDict(zip(list(range(len(lengths))), lengths)))

        # Ensure that rotary embeddings are computed with at a higher precision
        with torch.autocast(device_type="cuda", enabled=False):
            te_rope_emb = self.rotary_emb(max_seq_len=self.config.max_position_embeddings)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None if self.config.attn_input_format == "thd" else attention_mask,
                rotary_pos_emb=te_rope_emb,
                inference_params=past_key_values,
                cu_seqlens_q=kwargs.get("cu_seq_lens_q", None),
                cu_seqlens_kv=kwargs.get("cu_seq_lens_k", None),
                cu_seqlens_q_padded=kwargs.get("cu_seq_lens_q_padded", None),
                cu_seqlens_kv_padded=kwargs.get("cu_seq_lens_k_padded", None),
                max_seqlen_q=kwargs.get("max_length_q", None),
                max_seqlen_kv=kwargs.get("max_length_k", None),
                pad_between_seqs=kwargs.get("pad_between_seqs", None),
            )
            if self.config.use_moe and isinstance(decoder_layer, NVLlamaMoETransformerLayer):
                if decoder_layer.last_moe_aux_loss is not None:
                    moe_aux_loss = (
                        decoder_layer.last_moe_aux_loss
                        if moe_aux_loss is None
                        else moe_aux_loss + decoder_layer.last_moe_aux_loss
                    )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer. Note that these will be in THD format; we could possibly pad
        # these with the same _pad_input call as below if we wanted them returned in BSHD format.
        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        if should_pack_inputs:
            # If we've converted BSHD to THD for our TE layers, we need to convert back to BSHD for the output.
            hidden_states = _pad_input(hidden_states, indices, batch_size, max_seqlen)

        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )
        if moe_aux_loss is not None:
            outputs.moe_aux_loss = moe_aux_loss
        return outputs


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
            device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
            init_method=lambda x: torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range),
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
        logits_to_keep: int | torch.Tensor = 0,
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
            logits_to_keep (int | torch.Tensor): Whether to keep only the last logits to reduce the memory footprint of
                the model during generation.
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
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        with transformer_engine.pytorch.fp8_autocast(enabled=False):
            if hidden_states.ndim == 3:
                logits = self.lm_head(hidden_states[:, slice_indices, :])
            else:  # With THD inputs, batch and sequence dimensions are collapsed in the first dimension.
                logits = self.lm_head(hidden_states[slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            moe_aux_loss = getattr(outputs, "moe_aux_loss", None)
            if moe_aux_loss is not None:
                loss = loss + self.config.moe_aux_loss_coef * moe_aux_loss

        lm_outputs = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        if hasattr(outputs, "moe_aux_loss"):
            lm_outputs.moe_aux_loss = outputs.moe_aux_loss
        return lm_outputs


class NVLlamaForSequenceClassification(  # noqa: D101
    transformers.modeling_layers.GenericForSequenceClassification, NVLlamaPreTrainedModel
): ...


class NVLlamaForQuestionAnswering(transformers.modeling_layers.GenericForQuestionAnswering, NVLlamaPreTrainedModel):
    """Llama3 model with question answering head."""

    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class NVLlamaForTokenClassification(  # noqa: D101
    transformers.modeling_layers.GenericForTokenClassification, NVLlamaPreTrainedModel
): ...


torch._dynamo.config.capture_scalar_outputs = True


@torch.compile
def _pad_input(hidden_states, indices, batch, seqlen):
    """Convert a THD tensor to a BSHD equivalent tensor.

    Adapted from huggingface/transformers/modeling_flash_attention_utils.py

    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.

    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[1:]
    output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.view(batch, seqlen, *dim)


@torch.compile
def _unpad_input(hidden_states, attention_mask, unused_mask=None):
    """Convert a BSHD tensor to a THD equivalent tensor.

    Adapted from huggingface/transformers/modeling_flash_attention_utils.py

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.

    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    batch_size = hidden_states.size(0)
    seq_length = hidden_states.size(1)

    if attention_mask.shape[1] != seq_length:  # Likely in generation mode with kv-caching
        return (
            hidden_states.squeeze(1),  # hidden_states
            torch.arange(batch_size, dtype=torch.int64, device=hidden_states.device),  # indices
            torch.arange(batch_size + 1, dtype=torch.int32, device=hidden_states.device),  # cu_seqlens
            1,  # max_seqlen
            1,  # seqused
        )

    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    return (
        hidden_states.reshape(-1, *hidden_states.shape[2:])[indices],
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


class HFInferenceParams(InferenceParams):
    """Extension of the InferenceParams class to support beam search."""

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder the cache based on the beam indices."""
        if isinstance(self.cache_manager, PagedKVCacheManager):
            raise NotImplementedError("Beam search is not supported for paged cache manager.")
        for layer_number, (key_cache, value_cache) in self.cache_manager.cache.items():
            updated_key_cache = key_cache.index_select(0, beam_idx)
            updated_value_cache = value_cache.index_select(0, beam_idx)
            self.cache_manager.cache[layer_number] = (updated_key_cache, updated_value_cache)
