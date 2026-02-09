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
from typing import ClassVar, Unpack

import torch
import torch.nn as nn
import transformer_engine.pytorch
from transformer_engine.pytorch.attention import InferenceParams
from transformer_engine.pytorch.attention.inference import PagedKVCacheManager
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformers import MixtralConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.utils.generic import TransformersKwargs


AUTO_MAP = {
    "AutoConfig": "modeling_mixtral_te.NVMixtralConfig",
    "AutoModel": "modeling_mixtral_te.NVMixtralModel",
    "AutoModelForCausalLM": "modeling_mixtral_te.NVMixtralForCausalLM",
}


class NVMixtralConfig(MixtralConfig):
    """NVMixtral configuration."""

    attn_input_format: str = "thd"
    self_attn_mask_type: str = "padding_causal"


class NVMixtralPreTrainedModel(PreTrainedModel):
    """Base class for NVMixtral models."""

    config_class = NVMixtralConfig
    base_model_prefix = "model"
    _no_split_modules = ("NVMixtralDecoderLayer",)
    _skip_keys_device_placement = ("past_key_values",)
    _do_not_quantize = ("lm_head", "model.layers.*.mlp.gate")  # Flag for testing that these layers are not quantized.

    def init_empty_weights(self):
        """Handles moving the model from the meta device to the cuda device and initializing the weights."""
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.model.embed_tokens.to_empty(device="cuda")
        self.model.embed_tokens.apply(self._init_weights)

        self.model.rotary_emb.inv_freq = LlamaRotaryEmbedding(config=self.model.config).inv_freq.to("cuda")

        self.tie_weights()

    def _init_weights(self, module):
        """Initialize module weights.

        We only use this method for standard pytorch modules, TE modules handle their own weight initialization through
        `init_method` parameters and the `reset_parameters` method.
        """
        if module.__module__.startswith("transformer_engine.pytorch"):
            return

        super()._init_weights(module)

    def state_dict(self, *args, **kwargs):
        """Override state_dict to filter out TransformerEngine's _extra_state keys."""
        state_dict = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state_dict.items() if not k.endswith("_extra_state")}


class NVMixtralSparseMoeBlock(nn.Module):
    """Mixture of Experts block using TransformerEngine GroupedLinear."""

    def __init__(self, config: MixtralConfig):
        """Initialize the sparse MoE block."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        device = "meta" if torch.get_default_device() == torch.device("meta") else "cuda"

        def _init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        with transformer_engine.pytorch.quantized_model_init(enabled=False):
            self.gate = transformer_engine.pytorch.Linear(
                self.hidden_size,
                self.num_experts,
                bias=False,
                device=device,
                params_dtype=config.dtype,
                init_method=_init_method,
            )

        # Expert FFNs using TE GroupedLinear for efficient parallel processing
        self.experts_gate_up = transformer_engine.pytorch.GroupedLinear(
            num_gemms=self.num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size,
            bias=False,
            params_dtype=config.dtype,
            device=device,
            init_method=_init_method,
        )
        self.experts_down = transformer_engine.pytorch.GroupedLinear(
            num_gemms=self.num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias=False,
            params_dtype=config.dtype,
            device=device,
            init_method=_init_method,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MoE block.

        Args:
            hidden_states: Input tensor of shape [B, S, H] (bshd) or [T, H] (thd).

        Returns:
            Output tensor of the same shape as the input.
        """
        original_shape = hidden_states.shape
        # Flatten to [N, H] for routing
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, self.hidden_size)

        # Router: compute expert assignments
        with transformer_engine.pytorch.autocast(enabled=False):
            # Keep the router logits in bf16 during FP8 training
            router_logits = self.gate(hidden_states)  # [N, num_experts]

        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)  # [N, top_k]
        # Normalize routing weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Permute tokens by expert using TE moe_permute
        permuted_hidden, row_id_map = transformer_engine.pytorch.moe_permute(
            hidden_states, selected_experts, map_type="index"
        )

        # Compute m_splits: number of tokens per expert
        # selected_experts is [N, top_k], flatten and count per expert
        flat_experts = selected_experts.reshape(-1)
        m_splits = torch.histc(flat_experts.float(), bins=self.num_experts, min=0, max=self.num_experts - 1)
        m_splits = m_splits.int().tolist()

        # Expert computation: gate_up projection
        gate_up_output = self.experts_gate_up(permuted_hidden, m_splits=m_splits)  # [total_tokens, 2*ffn]

        # Split into gate and up, apply SwiGLU activation
        gate_output, up_output = gate_up_output.chunk(2, dim=-1)
        intermediate = torch.nn.functional.silu(gate_output) * up_output

        # Down projection
        expert_output = self.experts_down(intermediate, m_splits=m_splits)  # [total_tokens, H]

        # Unpermute and combine with routing weights
        output = transformer_engine.pytorch.moe_unpermute(
            expert_output,
            row_id_map,
            merging_probs=routing_weights.to(expert_output.dtype),
            map_type="index",
        )

        # Reshape back to original shape
        output = output.reshape(original_shape)
        return output


class NVMixtralDecoderLayer(nn.Module):
    """Mixtral decoder layer using TE attention and MoE MLP."""

    def __init__(self, config: MixtralConfig, layer_idx: int):
        """Initialize the decoder layer."""
        super().__init__()
        self.hidden_size = config.hidden_size

        device = "meta" if torch.get_default_device() == torch.device("meta") else "cuda"

        def _init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        self.self_attention = transformer_engine.pytorch.MultiheadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_gqa_groups=config.num_key_value_heads,
            bias=False,
            layernorm_epsilon=config.rms_norm_eps,
            attention_dropout=0,
            fuse_qkv_params=True,
            qkv_weight_interleaved=True,
            normalization="RMSNorm",
            input_layernorm=True,
            qkv_format=config.attn_input_format,
            attn_mask_type=config.self_attn_mask_type,
            layer_number=layer_idx + 1,
            params_dtype=config.dtype,
            device=device,
            init_method=_init_method,
            output_layer_init_method=_init_method,
        )

        self.post_attention_layernorm = transformer_engine.pytorch.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            device=device,
        )

        self.mlp = NVMixtralSparseMoeBlock(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        inference_params: InferenceParams | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for the decoder layer."""
        # Self attention with fused input layernorm
        attn_output = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            inference_params=inference_params,
            cu_seqlens_q=kwargs.get("cu_seqlens_q", None),
            cu_seqlens_kv=kwargs.get("cu_seqlens_kv", None),
            cu_seqlens_q_padded=kwargs.get("cu_seqlens_q_padded", None),
            cu_seqlens_kv_padded=kwargs.get("cu_seqlens_kv_padded", None),
            max_seqlen_q=kwargs.get("max_seqlen_q", None),
            max_seqlen_kv=kwargs.get("max_seqlen_kv", None),
            pad_between_seqs=kwargs.get("pad_between_seqs", None),
        )

        # Residual connection
        hidden_states = hidden_states + attn_output

        # Post-attention layernorm + MoE MLP + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class NVMixtralModel(NVMixtralPreTrainedModel):
    """Mixtral model implemented in Transformer Engine."""

    def __init__(self, config: MixtralConfig):
        """Initialize the NVMixtral model."""
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=config.dtype)

        self.layers = nn.ModuleList(
            [NVMixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = transformer_engine.pytorch.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
        )

        self.rotary_emb = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)
        self.rotary_emb.inv_freq = LlamaRotaryEmbedding(config=config).inv_freq

        self.gradient_checkpointing = False

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
        """Forward pass for the NVMixtral model."""
        all_hidden_states = []
        output_hidden_states = kwargs.get("output_hidden_states", False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # TE-specific input handling
        has_thd_input = [x in kwargs for x in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]]
        should_pack_inputs = not any(has_thd_input) and self.config.attn_input_format == "thd"

        if should_pack_inputs:
            assert attention_mask is not None, "Attention mask is required when packing BSHD inputs."
            batch_size = hidden_states.size(0)
            padded_seq_len = input_ids.size(1)
            hidden_states, indices, cu_seqlens, max_seqlen, _ = _unpad_input(hidden_states, attention_mask)
            kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_k"] = cu_seqlens
            kwargs["max_length_q"] = kwargs["max_length_k"] = max_seqlen

        if self.config.attn_input_format == "thd" and hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)

        if self.config.attn_input_format == "bshd" and attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :] < -1

        if isinstance(past_key_values, InferenceParams):
            lengths = (
                attention_mask.sum(dim=1).tolist()
                if attention_mask.shape == input_ids.shape
                else [1] * input_ids.shape[0]
            )
            past_key_values.pre_step(OrderedDict(zip(list(range(len(lengths))), lengths)))

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

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        if should_pack_inputs:
            hidden_states = _pad_input(hidden_states, indices, batch_size, padded_seq_len)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )


class NVMixtralForCausalLM(NVMixtralPreTrainedModel, __import__("transformers").GenerationMixin):
    """Mixtral model with causal language head."""

    _tied_weights_keys: ClassVar[list[str]] = []

    def __init__(self, config):
        """Initialize the NVMixtralForCausalLM model."""
        super().__init__(config)
        self.model = NVMixtralModel(config)
        self.vocab_size = config.vocab_size

        with transformer_engine.pytorch.quantized_model_init(enabled=False):
            self.lm_head = transformer_engine.pytorch.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                params_dtype=config.dtype,
                device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
                init_method=lambda x: torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range),
            )

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, ...], ...] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        shift_labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """Forward pass for the NVMixtralForCausalLM model."""
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
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        with transformer_engine.pytorch.autocast(enabled=False):
            if hidden_states.ndim == 3:
                logits = self.lm_head(hidden_states[:, slice_indices, :])
            else:
                logits = self.lm_head(hidden_states[slice_indices, :])

        loss = None
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, shift_labels=shift_labels, vocab_size=self.config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


torch._dynamo.config.capture_scalar_outputs = True


@torch.compile
def _pad_input(hidden_states, indices, batch, seqlen):
    """Convert a THD tensor to a BSHD equivalent tensor.

    Adapted from huggingface/transformers/modeling_flash_attention_utils.py
    """
    dim = hidden_states.shape[1:]
    output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.view(batch, seqlen, *dim)


@torch.compile
def _unpad_input(hidden_states, attention_mask, unused_mask=None):
    """Convert a BSHD tensor to a THD equivalent tensor.

    Adapted from huggingface/transformers/modeling_flash_attention_utils.py
    """
    batch_size = hidden_states.size(0)
    seq_length = hidden_states.size(1)

    if attention_mask.shape[1] != seq_length:
        return (
            hidden_states.squeeze(1),
            torch.arange(batch_size, dtype=torch.int64, device=hidden_states.device),
            torch.arange(batch_size + 1, dtype=torch.int32, device=hidden_states.device),
            1,
            1,
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
