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

import math
from collections import OrderedDict
from contextlib import nullcontext
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
    """NVLlama configuration.

    Additional attributes:
        attn_input_format: Input format for attention ("thd" or "bshd").
        self_attn_mask_type: Attention mask type for self-attention.
        embedding_init_std: Standard deviation for embedding initialization.
            If None, uses initializer_range (typically 0.02).
            Set to 1.0 for Spike-No-More paper approach.
        use_megatron_scaled_init: Whether to use Megatron's scaled initialization
            for residual output layers (attention proj, MLP fc2).
            Scaled init uses std / sqrt(2 * num_layers) for these layers.
        fp8_first_last_bf16: When True, keeps first and last transformer layers
            in bf16 for FP8 numerical stability. The lm_head is always kept in bf16.
    """

    attn_input_format: str = "thd"
    self_attn_mask_type: str = "padding_causal"
    embedding_init_std: float | None = None  # None means use initializer_range
    use_megatron_scaled_init: bool = False  # Use scaled init for proj/fc2 (std/sqrt(2*n))
    fp8_first_last_bf16: bool = False  # Keep first/last transformer layers in bf16 for FP8 stability


class NVLlamaPreTrainedModel(PreTrainedModel):
    """Base class for NVLlama models."""

    config_class = NVLlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ("TransformerLayer",)
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

        # TE's reset_parameters() doesn't use output_layer_init_method for proj/fc2.
        # If use_megatron_scaled_init is enabled, we need to manually apply scaled init.
        use_scaled_init = getattr(self.config, "use_megatron_scaled_init", False)
        if use_scaled_init:
            std = getattr(self.config, "initializer_range", 0.02)
            num_layers = getattr(self.config, "num_hidden_layers", 32)
            output_std = std / math.sqrt(2.0 * num_layers)

            # Apply scaled init to attention proj and MLP fc2 in each TransformerLayer
            for layer in self.model.layers:
                # Attention output projection
                if hasattr(layer, "self_attention") and hasattr(layer.self_attention, "proj"):
                    proj = layer.self_attention.proj
                    if hasattr(proj, "weight") and proj.weight is not None:
                        proj.weight.data.normal_(mean=0.0, std=output_std)

                # MLP fc2 (output layer)
                if hasattr(layer, "layernorm_mlp") and hasattr(layer.layernorm_mlp, "fc2_weight"):
                    layer.layernorm_mlp.fc2_weight.data.normal_(mean=0.0, std=output_std)

        # Meta-device init seems to break weight tying, so we re-tie the weights here.
        self.tie_weights()

    def _init_weights(self, module):
        """Initialize module weights.

        This method ensures that models with randomly-initialized weights get the correct initial value distribution,
        which can be critical for training stability. We also call this method when using meta-device init via
        `init_empty_weights()`.

        Initialization strategy:
        - Embeddings: config.embedding_init_std if set, else initializer_range (0.02)
          - For Spike-No-More: set embedding_init_std=1.0 in config
        - QKV, fc1, LM head: regular init = initializer_range (0.02)
        - Attention proj, MLP fc2: depends on config.use_megatron_scaled_init:
          - If False (default): regular init = initializer_range (0.02)
          - If True: scaled init = initializer_range / sqrt(2 * num_layers)

        Note: Megatron uses scaled init for proj and fc2, but TE's fused TransformerLayer
        was designed with uniform initialization. If you see loss increasing after initial
        decrease, try setting use_megatron_scaled_init=False.

        Args:
            module (nn.Module): The module to initialize the weights for.
        """
        # Get base std from config (typically 0.02)
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        # Check if we should use Megatron's scaled init for residual output layers
        use_scaled_init = getattr(self.config, "use_megatron_scaled_init", False)
        if use_scaled_init:
            num_layers = getattr(self.config, "num_hidden_layers", 32)
            output_std = std / math.sqrt(2.0 * num_layers)
        else:
            output_std = std  # Use regular std for all layers (TE's default)

        # Embedding init std: default to regular std (0.02), use 1.0 only if explicitly set
        # For Spike-No-More paper approach, set config.embedding_init_std = 1.0
        embedding_init_std = getattr(self.config, "embedding_init_std", None)
        embedding_std = embedding_init_std if embedding_init_std is not None else std

        # Embeddings
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=embedding_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        # Regular Linear layers: LM head and other Linear use regular std (0.02)
        # Note: In Megatron, LM head (output_layer) uses init_method, NOT output_layer_init_method
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

        # MLP: fc1 uses regular std, fc2 uses output_std (scaled if use_megatron_scaled_init=True)
        if isinstance(module, transformer_engine.pytorch.LayerNormMLP):
            module.layer_norm_weight.data.fill_(1.0)
            if hasattr(module, "fc1_weight") and module.fc1_weight is not None:
                module.fc1_weight.data.normal_(mean=0.0, std=std)
            if hasattr(module, "fc2_weight") and module.fc2_weight is not None:
                module.fc2_weight.data.normal_(mean=0.0, std=output_std)
            if hasattr(module, "fc1_bias") and module.fc1_bias is not None and module.fc1_bias.numel() > 0:
                module.fc1_bias.data.zero_()
            if hasattr(module, "fc2_bias") and module.fc2_bias is not None and module.fc2_bias.numel() > 0:
                module.fc2_bias.data.zero_()

        # TE TransformerLayer: attention output projection uses output_std
        if isinstance(module, transformer_engine.pytorch.TransformerLayer):
            if hasattr(module, "self_attention") and hasattr(module.self_attention, "proj"):
                proj = module.self_attention.proj
                if hasattr(proj, "weight") and proj.weight is not None:
                    proj.weight.data.normal_(mean=0.0, std=output_std)
                if hasattr(proj, "bias") and proj.bias is not None:
                    proj.bias.data.zero_()

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

        # Spike-No-More embedding init is handled in _init_weights using config.embedding_init_std
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=config.dtype)

        # Regular init method for QKV, fc1, LM head (use standard initializer_range)
        def _init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        # Output init method for attention proj, MLP fc2 (use scaled init if enabled)
        use_scaled_init = getattr(config, "use_megatron_scaled_init", False)
        if use_scaled_init:
            output_std = config.initializer_range / math.sqrt(2.0 * config.num_hidden_layers)
        else:
            output_std = config.initializer_range

        def _output_init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=output_std)

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
                    device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
                    init_method=_init_method,
                    output_layer_init_method=_output_init_method,  # Use scaled init if use_megatron_scaled_init=True
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = transformer_engine.pytorch.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
        )

        # We use TE's RotaryPositionEmbedding, but we ensure that we use the same inv_freq as the original
        # LlamaRotaryEmbedding.
        self.rotary_emb = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)
        hf_rope = LlamaRotaryEmbedding(config=config)
        self.rotary_emb.inv_freq = hf_rope.inv_freq

        # DEBUG: Print RoPE configuration
        import logging

        _logger = logging.getLogger(__name__)
        _logger.info("=" * 80)
        _logger.info("[ROPE DEBUG] RoPE Configuration:")
        _logger.info(f"[ROPE DEBUG]   rope_theta (config): {getattr(config, 'rope_theta', 'NOT SET')}")
        _logger.info(f"[ROPE DEBUG]   rope_scaling (config): {getattr(config, 'rope_scaling', 'NOT SET')}")
        _logger.info(f"[ROPE DEBUG]   rope_parameters (config): {getattr(config, 'rope_parameters', 'NOT SET')}")
        _logger.info(f"[ROPE DEBUG]   inv_freq shape: {self.rotary_emb.inv_freq.shape}")
        _logger.info(f"[ROPE DEBUG]   inv_freq device: {self.rotary_emb.inv_freq.device}")
        # Skip printing values if on meta device
        if self.rotary_emb.inv_freq.device.type != "meta":
            _logger.info(f"[ROPE DEBUG]   inv_freq first 5: {self.rotary_emb.inv_freq[:5].tolist()}")
        _logger.info("=" * 80)

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

        num_layers = self.config.num_hidden_layers
        for layer_idx, decoder_layer in enumerate(self.layers[:num_layers]):
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            # Optionally keep first and last layers in bf16 for FP8 numerical stability
            use_bf16_for_layer = getattr(self.config, "fp8_first_last_bf16", False) and (
                layer_idx == 0 or layer_idx == num_layers - 1
            )

            # If fp8_first_last_bf16 is enabled, disable FP8 for first/last layers
            # This nested fp8_autocast will override the outer one from training script
            with transformer_engine.pytorch.fp8_autocast(enabled=False) if use_bf16_for_layer else nullcontext():
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

        # add hidden states from the last decoder layer. Note that these will be in THD format; we could possibly pad
        # these with the same _pad_input call as below if we wanted them returned in BSHD format.
        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        if should_pack_inputs:
            # If we've converted BSHD to THD for our TE layers, we need to convert back to BSHD for the output.
            hidden_states = _pad_input(hidden_states, indices, batch_size, max_seqlen)

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
