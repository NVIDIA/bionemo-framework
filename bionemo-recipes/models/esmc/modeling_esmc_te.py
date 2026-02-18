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

"""TransformerEngine-optimized ESMC (EvolutionaryScale ESM-Cambrian) model.

This module provides HuggingFace-compatible ESMC model classes using NVIDIA's TransformerEngine
for optimized attention and MLP computation. The model is an encoder-only protein language model
with bidirectional attention, RoPE, SwiGLU activation, and full d_model QK LayerNorm.

Unlike models that use TE's TransformerLayer (which applies per-head QK norm), this implementation
uses lower-level TE components (LayerNormLinear, DotProductAttention, LayerNormMLP) to apply QK
LayerNorm across the full hidden dimension, exactly matching the reference ESMC model.

Reference: EvolutionaryScale's ESMC-300M (esm PyPI package).
"""

from typing import ClassVar, Literal, Optional, TypedDict, Unpack

import torch
import transformer_engine.pytorch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding, apply_rotary_pos_emb
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.modeling_utils import PreTrainedModel


AUTO_MAP = {
    "AutoConfig": "modeling_esmc_te.NVEsmcConfig",
    "AutoModel": "modeling_esmc_te.NVEsmcModel",
    "AutoModelForMaskedLM": "modeling_esmc_te.NVEsmcForMaskedLM",
}


class TransformersKwargs(TypedDict):
    """Transformers v4 does not export a TransformersKwargs class, so we define our own."""

    cu_seq_lens_q: Optional[torch.Tensor]
    cu_seq_lens_k: Optional[torch.Tensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]
    pad_between_seqs: Optional[int]
    cu_seqlens_q_padded: Optional[torch.Tensor]
    cu_seqlens_k_padded: Optional[torch.Tensor]


class NVEsmcConfig(PretrainedConfig):
    """Configuration for the NVEsmc TransformerEngine model."""

    model_type: str = "nv_esmc"

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 960,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 15,
        intermediate_size: int = 2560,
        layer_norm_eps: float = 1e-5,
        position_embedding_type: str = "rotary",
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        # TE-specific options
        attn_input_format: Literal["bshd", "thd"] = "bshd",
        self_attn_mask_type: str = "padding",
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize NVEsmcConfig.

        Args:
            vocab_size: Vocabulary size (padded to 64 from real vocab of 33).
            hidden_size: Dimension of hidden representations.
            num_hidden_layers: Number of transformer layers.
            num_attention_heads: Number of attention heads.
            intermediate_size: FFN intermediate dimension (SwiGLU corrected).
            layer_norm_eps: Layer normalization epsilon.
            position_embedding_type: Type of position embedding (only "rotary" supported).
            initializer_range: Standard deviation for weight initialization.
            pad_token_id: Padding token ID.
            attn_input_format: Attention input format for TE ("bshd" or "thd").
            self_attn_mask_type: Attention mask type ("padding" for bidirectional).
            tie_word_embeddings: Whether to tie input/output embeddings.
            **kwargs: Additional config options.
        """
        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.initializer_range = initializer_range
        self.attn_input_format = attn_input_format
        self.self_attn_mask_type = self_attn_mask_type


class EsmcTransformerBlock(nn.Module):
    """Custom ESMC transformer block using lower-level TE components.

    This block implements full d_model QK LayerNorm (matching the reference ESMC model)
    by using individual TE components instead of TE's TransformerLayer which only supports
    per-head QK norm.

    Architecture:
        1. LayerNormLinear: pre-attention LayerNorm + QKV projection
        2. LayerNorm(d_model): full-dimension Q normalization
        3. LayerNorm(d_model): full-dimension K normalization
        4. RoPE application
        5. DotProductAttention: flash/fused attention
        6. Linear: output projection (residue scaling absorbed in weights)
        7. LayerNormMLP: pre-FFN LayerNorm + SwiGLU MLP (residue scaling absorbed in fc2)
    """

    def __init__(self, config: NVEsmcConfig, layer_idx: int):
        """Initialize EsmcTransformerBlock."""
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads
        device = "meta" if torch.get_default_device() == torch.device("meta") else "cuda"

        def _init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        # Pre-attention LayerNorm + QKV projection (fused)
        self.layernorm_qkv = transformer_engine.pytorch.LayerNormLinear(
            hidden_size,
            3 * hidden_size,
            bias=False,
            eps=config.layer_norm_eps,
            params_dtype=config.torch_dtype,
            device=device,
            init_method=_init_method,
        )

        # Full d_model QK LayerNorm (matching reference model exactly)
        self.q_norm = transformer_engine.pytorch.LayerNorm(
            hidden_size,
            eps=config.layer_norm_eps,
            params_dtype=config.torch_dtype,
            device=device,
        )
        self.k_norm = transformer_engine.pytorch.LayerNorm(
            hidden_size,
            eps=config.layer_norm_eps,
            params_dtype=config.torch_dtype,
            device=device,
        )

        # Attention computation (flash/fused attention backends)
        self.core_attention = transformer_engine.pytorch.DotProductAttention(
            num_attention_heads=num_heads,
            kv_channels=head_dim,
            num_gqa_groups=num_heads,
            attention_dropout=0,
            qkv_format=config.attn_input_format,
            attn_mask_type=config.self_attn_mask_type,
            layer_number=layer_idx + 1,
        )

        # Output projection
        self.proj = transformer_engine.pytorch.Linear(
            hidden_size,
            hidden_size,
            bias=False,
            params_dtype=config.torch_dtype,
            device=device,
            init_method=_init_method,
        )

        # FFN: pre-LayerNorm + SwiGLU MLP (fused)
        self.layernorm_mlp = transformer_engine.pytorch.LayerNormMLP(
            hidden_size,
            config.intermediate_size,
            bias=False,
            eps=config.layer_norm_eps,
            activation="swiglu",
            params_dtype=config.torch_dtype,
            device=device,
            init_method=_init_method,
            output_layer_init_method=_init_method,
        )

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_input_format = config.attn_input_format

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        cu_seqlens_q_padded: Optional[torch.Tensor] = None,
        cu_seqlens_kv_padded: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        pad_between_seqs: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward pass for a single transformer block.

        Args:
            hidden_states: Input tensor [B, S, D] (BSHD) or [T, D] (THD).
            attention_mask: Attention mask for BSHD format.
            rotary_pos_emb: Precomputed rotary position embeddings.
            cu_seqlens_q: Cumulative sequence lengths for queries (THD format).
            cu_seqlens_kv: Cumulative sequence lengths for keys/values (THD format).
            cu_seqlens_q_padded: Padded cumulative sequence lengths for queries.
            cu_seqlens_kv_padded: Padded cumulative sequence lengths for keys/values.
            max_seqlen_q: Maximum query sequence length (THD format).
            max_seqlen_kv: Maximum key/value sequence length (THD format).
            pad_between_seqs: Whether there is padding between sequences.

        Returns:
            Output tensor with same shape as input.
        """
        residual = hidden_states

        # Pre-attention LayerNorm + QKV projection
        qkv = self.layernorm_qkv(hidden_states)  # [*, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)  # each [*, D]

        # Full d_model QK LayerNorm (matching reference model)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape to multi-head format: [B, S, H, d_head] or [T, H, d_head]
        head_shape = (*q.shape[:-1], self.num_heads, self.head_dim)
        q = q.view(head_shape)
        k = k.view(head_shape)
        v = v.view(head_shape)

        # Apply RoPE
        if rotary_pos_emb is not None:
            tensor_format = "thd" if self.attn_input_format == "thd" else "bshd"
            q = apply_rotary_pos_emb(
                q,
                rotary_pos_emb,
                tensor_format=tensor_format,
                cu_seqlens=cu_seqlens_q if tensor_format == "thd" else None,
            )
            k = apply_rotary_pos_emb(
                k,
                rotary_pos_emb,
                tensor_format=tensor_format,
                cu_seqlens=cu_seqlens_kv if tensor_format == "thd" else None,
            )

        # Attention
        attn_output = self.core_attention(
            q,
            k,
            v,
            attention_mask=attention_mask,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            pad_between_seqs=pad_between_seqs,
        )  # [B, S, D] or [T, D] (DotProductAttention folds heads internally)

        # Output projection
        attn_output = self.proj(attn_output)

        # Residual connection (residue scaling absorbed in proj weights)
        hidden_states = residual + attn_output

        # FFN with pre-LayerNorm and residual (residue scaling absorbed in fc2 weights)
        residual = hidden_states
        hidden_states = residual + self.layernorm_mlp(hidden_states)

        return hidden_states


class NVEsmcPreTrainedModel(PreTrainedModel):
    """Base class for NVEsmc models."""

    config_class = NVEsmcConfig
    base_model_prefix = "esmc"
    _no_split_modules = ("EsmcTransformerBlock",)
    _tied_weights_keys: ClassVar[dict[str, str]] = {}

    def init_empty_weights(self):
        """Move model from meta device to CUDA and initialize weights."""
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.esmc.embed_tokens.to_empty(device="cuda")
        self.esmc.embed_tokens.apply(self._init_weights)

        # Meta-device init breaks weight tying, so re-tie.
        self.tie_weights()

    def _init_weights(self, module):
        """Initialize weights for standard pytorch modules.

        TE modules handle their own initialization through `init_method` and `reset_parameters`.
        """
        if module.__module__.startswith("transformer_engine.pytorch"):
            return

        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def state_dict(self, *args, **kwargs):
        """Override to filter out TE's _extra_state keys."""
        state_dict = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state_dict.items() if not k.endswith("_extra_state")}


class NVEsmcModel(NVEsmcPreTrainedModel):
    """ESMC encoder model with TransformerEngine layers."""

    def __init__(self, config: NVEsmcConfig):
        """Initialize the NVEsmc model."""
        super().__init__(config)
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id, dtype=config.torch_dtype
        )

        self.layers = nn.ModuleList(
            [EsmcTransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Final LayerNorm (no bias in reference, but TE LayerNorm always has bias; set to zeros)
        self.norm = transformer_engine.pytorch.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            params_dtype=config.torch_dtype,
            device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
        )

        self.rotary_emb = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        """Forward pass for the NVEsmc encoder model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments (THD params, output_hidden_states, etc.).

        Returns:
            BaseModelOutput with last_hidden_state and optional hidden_states.
        """
        all_hidden_states: tuple[torch.Tensor, ...] = ()
        output_hidden_states = kwargs.get("output_hidden_states", False)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Handle THD format conversion
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
            # Convert 2D HF mask (1=attend, 0=pad) to 4D TE mask (True=masked, False=attend)
            attention_mask = attention_mask[:, None, None, :] == 0

        with torch.autocast(device_type="cuda", enabled=False):
            te_rope_emb = self.rotary_emb(max_seq_len=4096)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            hidden_states = layer(
                hidden_states,
                attention_mask=None if self.config.attn_input_format == "thd" else attention_mask,
                rotary_pos_emb=te_rope_emb,
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

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states if all_hidden_states else None,
        )


class NVEsmcForMaskedLM(NVEsmcPreTrainedModel):
    """ESMC model with masked language modeling head."""

    _tied_weights_keys: ClassVar[dict[str, str]] = {}
    _do_not_quantize: ClassVar[list[str]] = ["sequence_head.dense", "sequence_head.decoder"]

    def __init__(self, config: NVEsmcConfig):
        """Initialize NVEsmcForMaskedLM."""
        super().__init__(config)
        self.esmc = NVEsmcModel(config)
        self.sequence_head = NVEsmcLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        """Get the output embeddings."""
        return self.sequence_head.decoder

    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings."""
        self.sequence_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MaskedLMOutput:
        """Forward pass with masked language modeling loss.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            inputs_embeds: Pre-computed input embeddings.
            labels: Labels for masked token prediction.
            **kwargs: Additional keyword arguments.

        Returns:
            MaskedLMOutput with loss, logits, and optional hidden states.
        """
        outputs = self.esmc(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        sequence_output = outputs.last_hidden_state

        with transformer_engine.pytorch.autocast(enabled=False):
            prediction_scores = self.sequence_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.to(prediction_scores.device).view(-1),
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
        )


class NVEsmcLMHead(nn.Module):
    """ESMC language modeling head: Linear -> GELU -> LayerNorm -> Linear.

    This matches the EvolutionaryScale `RegressionHead(d_model, output_dim)` architecture.
    """

    def __init__(self, config: NVEsmcConfig):
        """Initialize NVEsmcLMHead."""
        super().__init__()
        with transformer_engine.pytorch.quantized_model_init(enabled=False):
            self.dense = transformer_engine.pytorch.Linear(
                config.hidden_size,
                config.hidden_size,
                bias=True,
                params_dtype=config.torch_dtype,
                device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
                init_method=lambda x: torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range),
            )

            self.decoder = transformer_engine.pytorch.LayerNormLinear(
                config.hidden_size,
                config.vocab_size,
                bias=True,
                eps=config.layer_norm_eps,
                params_dtype=config.torch_dtype,
                device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
                init_method=lambda x: torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range),
            )

    def forward(self, features):
        """Forward pass: Dense -> GELU -> LayerNormLinear."""
        with transformer_engine.pytorch.autocast(enabled=False):
            x = self.dense(features)
            x = torch.nn.functional.gelu(x)
            x = self.decoder(x)
        return x


# ===================== Utility Functions for THD Packing =====================

torch._dynamo.config.capture_scalar_outputs = True


@torch.compile
def _pad_input(hidden_states, indices, batch, seqlen):
    """Convert a THD tensor to BSHD format."""
    dim = hidden_states.shape[1:]
    output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.view(batch, seqlen, *dim)


@torch.compile
def _unpad_input(hidden_states, attention_mask, unused_mask=None):
    """Convert a BSHD tensor to THD format."""
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
