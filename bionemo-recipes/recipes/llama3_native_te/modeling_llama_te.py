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


class NVLlamaPreTrainedModel(PreTrainedModel):
    """Base class for NVLlama models."""

    config_class = NVLlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ("TransformerLayer",)
    _skip_keys_device_placement = ("past_key_values",)

    def _init_weights(self, module):
        """Initialize module weights.

        This method ensures that models with randomly-initialized weights get the correct initial value distribution,
        which can be critical for training stability. We also call this method directly when using meta-device init, as
        the `to_empty` method does not initialize the weights. While the base Transformers model has a similar method,
        we need to extend it to handle TE-specific modules.

        Args:
            module (nn.Module): The module to initialize the weights for.
        """
        super()._init_weights(module)

        # Copied from transformers.modeling_utils.PreTrainedModel._init_weights
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        if isinstance(
            module, (nn.Linear, transformer_engine.pytorch.Linear, transformer_engine.pytorch.LayerNormLinear)
        ):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, transformer_engine.pytorch.LayerNorm):
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, transformer_engine.pytorch.RMSNorm):
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
        if isinstance(module, transformer_engine.pytorch.LayerNormLinear):
            module.layer_norm_weight.data.fill_(1.0)
            if module.layer_norm_bias is not None:
                module.layer_norm_bias.data.zero_()
        if isinstance(module, transformer_engine.pytorch.LayerNormMLP):
            module.layer_norm_weight.data.fill_(1.0)
            if hasattr(module, "fc1_weight") and module.fc1_weight is not None:
                module.fc1_weight.data.normal_(mean=0.0, std=std)
            if hasattr(module, "fc2_weight") and module.fc2_weight is not None:
                module.fc2_weight.data.normal_(mean=0.0, std=std)
            if hasattr(module, "fc1_bias") and module.fc1_bias is not None and module.fc1_bias.numel() > 0:
                module.fc1_bias.data.zero_()
            if hasattr(module, "fc2_bias") and module.fc2_bias is not None and module.fc2_bias.numel() > 0:
                module.fc2_bias.data.zero_()
        if isinstance(module, RotaryPositionEmbedding) and hasattr(module, "inv_freq"):
            module.inv_freq = LlamaRotaryEmbedding(config=self.config).inv_freq.to(module.inv_freq.device)


class NVLlamaModel(NVLlamaPreTrainedModel):
    """Llama3 model implemented in Transformer Engine."""

    def __init__(self, config: LlamaConfig):
        """Initialize the NVLlama model."""
        super().__init__(config)
        self.config = config
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
                    self_attn_mask_type="padding_causal",
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
        fp8_enabled: bool = True,
        fp8_recipe=None,
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
            fp8_enabled (bool): Whether to enable FP8 for middle layers. First and last layers always use bf16.
            fp8_recipe: The FP8 recipe to use for quantization.
            **kwargs: Additional keyword arguments.

        Returns:
            BaseModelOutputWithPast: The output of the model.
        """
        all_hidden_states = []
        output_hidden_states = kwargs.get("output_hidden_states", False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Embedding layer - ALWAYS in bf16 for stability
        if inputs_embeds is None:
            with transformer_engine.pytorch.fp8_autocast(enabled=False):
                inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        has_thd_input = [x in kwargs for x in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]]
        should_pack_inputs = not any(has_thd_input) and self.config.attn_input_format == "thd"

        # This might be slower for BSHD + padding with fused attention backend. But it should be faster for the flash
        # attention backend.
        self_attn_mask_type = "padding_causal"
        if should_pack_inputs:
            # Left-side padding is not supported in TE layers, so to make generation work with TE we dynamically convert
            # to THD-style inputs in our forward pass, and then convert back to BSHD for the output. This lets the
            # entire transformer stack run in THD mode.
            assert attention_mask is not None, "Attention mask is required when packing BSHD inputs."
            batch_size = hidden_states.size(0)
            hidden_states, indices, cu_seqlens, max_seqlen, _ = _unpad_input(hidden_states, attention_mask)
            cu_seq_lens_q = cu_seq_lens_k = cu_seqlens
            max_length_q = max_length_k = max_seqlen

        elif self.config.attn_input_format == "thd":
            # Here, we're providing THD-style inputs, so we can just grab the kwargs.
            assert hidden_states.dim() == 3 and hidden_states.size(0) == 1, (
                "THD expects embeddings shaped [1, total_tokens, hidden_size]."
            )
            hidden_states = hidden_states.squeeze(0)
            cu_seq_lens_q = kwargs["cu_seq_lens_q"]
            cu_seq_lens_k = kwargs["cu_seq_lens_k"]
            max_length_q = kwargs["max_length_q"]
            max_length_k = kwargs["max_length_k"]

        else:
            if attention_mask is not None:
                attention_mask = attention_mask[:, None, None, :] < -1
            else:
                self_attn_mask_type = "causal"
            cu_seq_lens_q = cu_seq_lens_k = None
            max_length_q = max_length_k = hidden_states.size(1)

        # If we're using kv-caching, we can't trust the max_length_q value as the true max length for rotary
        # embeddings, since this will be 1 in generation. Instead we can take the max sequence length from the past
        # key values object.
        te_rope_emb = self.rotary_emb(
            max_seq_len=max_length_q if past_key_values is None else past_key_values.max_ctx_len
        )

        if isinstance(past_key_values, InferenceParams):
            # In generation mode, we set the length to 1 for each batch index. Otherwise, we use the attention mask to
            # compute the lengths of each sequence in the batch.
            lengths = (
                attention_mask.sum(dim=1).tolist()
                if attention_mask.shape == input_ids.shape
                else [1] * input_ids.shape[0]
            )
            past_key_values.pre_step(OrderedDict(zip(list(range(len(lengths))), lengths)))

        # Process transformer layers with selective FP8
        # Keep first and last layers in bf16, use FP8 for middle layers
        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            decoder_layer = self.layers[layer_idx]

            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            # First layer (0) and last layer stay in bf16 for numerical stability
            # Middle layers use FP8 for performance
            use_fp8_for_layer = fp8_enabled and (layer_idx != 0) and (layer_idx != num_layers - 1)

            with transformer_engine.pytorch.fp8_autocast(enabled=use_fp8_for_layer, fp8_recipe=fp8_recipe):
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=None if self.config.attn_input_format == "thd" else attention_mask,
                    rotary_pos_emb=te_rope_emb,
                    self_attn_mask_type=self_attn_mask_type,
                    inference_params=past_key_values,
                    cu_seqlens_q=cu_seq_lens_q,
                    cu_seqlens_kv=cu_seq_lens_k,
                    max_seqlen_q=max_length_q,
                    max_seqlen_kv=max_length_k,
                )

        # Final norm - ALWAYS in bf16 for stability
        with transformer_engine.pytorch.fp8_autocast(enabled=False):
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer. Note that these will be in THD format; we could possibly pad
        # these with the same _pad_input call as below if we wanted them returned in BSHD format.
        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        if should_pack_inputs:
            # If we've converted BSHD to THD for our TE layers, we need to convert back to BSHD for the output.
            hidden_states = _pad_input(hidden_states, indices, batch_size, max_length_q)

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
        logits_to_keep: int | torch.Tensor = 0,
        fp8_enabled: bool = True,
        fp8_recipe=None,
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
            fp8_enabled (bool): Whether to enable FP8 for middle layers. First and last layers always use bf16.
            fp8_recipe: The FP8 recipe to use for quantization.
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
            fp8_enabled=fp8_enabled,
            fp8_recipe=fp8_recipe,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        # LM head - ALWAYS in bf16 for numerical stability
        with transformer_engine.pytorch.fp8_autocast(enabled=False):
            if hidden_states.ndim == 3:
                logits = self.lm_head(hidden_states[:, slice_indices, :])
            else:  # With THD inputs, batch and sequence dimensions are collapsed in the first dimension.
                logits = self.lm_head(hidden_states[slice_indices, :])

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
