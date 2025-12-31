# coding=utf-8
# noqa: license-check
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 NVIDIA CORPORATION. All rights reserved.
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


"""TransformerEngine-optimized ESM model.

Adapted from `modeling_esm.py` in huggingface/transformers.
"""

from typing import Literal, Optional, Unpack

# TODO: put import guard around transformer_engine here, with an informative error message around
# installation and the nvidia docker container.
import torch
import transformer_engine.pytorch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    TokenClassifierOutput,
)
from transformers.models.esm.configuration_esm import EsmConfig
from transformers.models.esm.modeling_esm import EsmPooler, EsmPreTrainedModel
from transformers.utils import logging
from transformers.utils.generic import TransformersKwargs


logger = logging.get_logger(__name__)

# Dictionary that gets inserted into config.json to map Auto** classes to our TE-optimized model classes defined below.
# These should be prefixed with esm_nv., since we name the file esm_nv.py in our exported checkpoints.
AUTO_MAP = {
    "AutoConfig": "esm_nv.NVEsmConfig",
    "AutoModel": "esm_nv.NVEsmModel",
    "AutoModelForMaskedLM": "esm_nv.NVEsmForMaskedLM",
    "AutoModelForTokenClassification": "esm_nv.NVEsmForTokenClassification",
}


class NVEsmConfig(EsmConfig):
    """NVEsmConfig is a configuration for the NVEsm model."""

    model_type: str = "nv_esm"

    def __init__(
        self,
        qkv_weight_interleaved: bool = True,
        encoder_activation: str = "gelu",
        attn_input_format: Literal["bshd", "thd"] = "bshd",
        fuse_qkv_params: bool = True,
        micro_batch_size: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        padded_vocab_size: Optional[int] = 64,
        attn_mask_type: str = "padding",
        **kwargs,
    ):
        """Initialize the NVEsmConfig with additional TE-related config options.

        Args:
            qkv_weight_interleaved: Whether to interleave the qkv weights. If set to `False`, the
                QKV weight is interpreted as a concatenation of query, key, and value weights along
                the `0th` dimension. The default interpretation is that the individual `q`, `k`, and
                `v` weights for each attention head are interleaved. This parameter is set to `False`
                when using :attr:`fuse_qkv_params=False`.
            encoder_activation: The activation function to use in the encoder.
            attn_input_format: The input format to use for the attention. This controls
                whether the dimensions of the intermediate hidden states is 'batch first'
                ('bshd') or 'sequence first' ('sbhd'). `s` stands for the sequence length,
                `b` batch size, `h` the number of heads, `d` head size. Note that these
                formats are very closely related to the `qkv_format` in the
                `MultiHeadAttention` and `DotProductAttention` modules.
            fuse_qkv_params: Whether to fuse the qkv parameters. If set to `True`,
                `TransformerLayer` module exposes a single fused parameter for query-key-value.
                This enables optimizations such as QKV fusion without concatentations/splits and
                also enables the argument `fuse_wgrad_accumulation`.
            micro_batch_size: The micro batch size to use for the attention. This is needed for
                JIT Warmup, a technique where jit fused functions are warmed up before training to
                ensure same kernels are used for forward propogation and activation recompute phase.
            max_seq_length: The maximum sequence length to use for the attention. This is needed for
                JIT Warmup, a technique where jit fused functions are warmed up before training to
                ensure same kernels are used for forward propogation and activation recompute phase.
            padded_vocab_size: The padded vocabulary size to support FP8. If not provided, defaults
                to vocab_size. Must be greater than or equal to vocab_size.
            attn_mask_type: The type of attention mask to use.
            **kwargs: Additional config options to pass to EsmConfig.
        """
        super().__init__(**kwargs)
        # Additional TE-related config options.
        self.qkv_weight_interleaved = qkv_weight_interleaved
        self.encoder_activation = encoder_activation
        self.attn_input_format = attn_input_format
        self.fuse_qkv_params = fuse_qkv_params
        self.micro_batch_size = micro_batch_size
        self.max_seq_length = max_seq_length
        self.attn_mask_type = attn_mask_type

        # Set padded_vocab_size with default fallback to vocab_size
        self.padded_vocab_size = padded_vocab_size if padded_vocab_size is not None else self.vocab_size

        # Ensure padded_vocab_size is at least as large as vocab_size
        if self.padded_vocab_size is not None and self.vocab_size is not None:
            assert self.padded_vocab_size >= self.vocab_size, (
                f"padded_vocab_size ({self.padded_vocab_size}) must be greater than or equal to vocab_size ({self.vocab_size})"
            )


class NVEsmEncoder(nn.Module):
    """NVEsmEncoder is a TransformerEngine-optimized ESM encoder."""

    def __init__(self, config: NVEsmConfig):
        """Initialize a NVEsmEncoder.

        Args:
            config (NVEsmConfig): The configuration of the model.
        """
        super().__init__()
        self.config = config

        def _init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        self.layers = nn.ModuleList(
            [
                transformer_engine.pytorch.TransformerLayer(
                    hidden_size=config.hidden_size,
                    ffn_hidden_size=config.intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    layernorm_epsilon=config.layer_norm_eps,
                    hidden_dropout=config.hidden_dropout_prob,
                    attention_dropout=config.attention_probs_dropout_prob,
                    qkv_weight_interleaved=config.qkv_weight_interleaved,
                    layer_number=i + 1,
                    layer_type="encoder",
                    self_attn_mask_type=config.attn_mask_type,
                    activation=config.encoder_activation,
                    attn_input_format=config.attn_input_format,
                    seq_length=config.max_seq_length,
                    micro_batch_size=config.micro_batch_size,
                    num_gqa_groups=config.num_attention_heads,
                    fuse_qkv_params=config.fuse_qkv_params,
                    params_dtype=config.dtype,
                    window_size=(-1, -1),
                    device=str(torch.get_default_device()),
                    init_method=_init_method,
                    output_layer_init_method=_init_method,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.emb_layer_norm_after = transformer_engine.pytorch.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            params_dtype=config.dtype,
            device=str(torch.get_default_device()),
        )
        if config.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        """Forward pass of the NVEsmEncoder.

        Args:
            hidden_states (torch.Tensor): The hidden states.
            attention_mask (torch.Tensor): The attention mask.
            **kwargs: Additional arguments, see TransformersKwargs for more details.
        """
        all_hidden_states: tuple[torch.Tensor, ...] = ()
        has_thd_input = [
            x is not None
            for x in [
                kwargs.get("cu_seq_lens_q", None),
                kwargs.get("cu_seq_lens_k", None),
                kwargs.get("max_length_q", None),
                kwargs.get("max_length_k", None),
            ]
        ]

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

        # Ensure that rotary embeddings are computed with at a higher precision outside the torch autocast context.
        with torch.autocast(device_type="cuda", enabled=False):
            if self.config.position_embedding_type == "rotary":
                if self.config.attn_input_format == "bshd":
                    te_rope_emb = self.rotary_embeddings(max_seq_len=hidden_states.shape[1])
                elif self.config.attn_input_format == "thd":
                    te_rope_emb = self.rotary_embeddings(
                        max_seq_len=kwargs["cu_seq_lens_q_padded"][-1]
                        if "cu_seq_lens_q_padded" in kwargs
                        else kwargs["cu_seq_lens_q"][-1]
                    )
            te_rope_emb = te_rope_emb.to(hidden_states.device, non_blocking=True)

        for layer_module in self.layers:
            if kwargs.get("output_hidden_states", False):
                all_hidden_states = (*all_hidden_states, hidden_states)

            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                rotary_pos_emb=te_rope_emb,
                cu_seqlens_q=kwargs.get("cu_seq_lens_q", None),
                cu_seqlens_kv=kwargs.get("cu_seq_lens_k", None),
                cu_seqlens_q_padded=kwargs.get("cu_seq_lens_q_padded", None),
                cu_seqlens_kv_padded=kwargs.get("cu_seq_lens_k_padded", None),
                max_seqlen_q=kwargs.get("max_length_q", None),
                max_seqlen_kv=kwargs.get("max_length_k", None),
                pad_between_seqs=kwargs.get("pad_between_seqs", None),
            )

        hidden_states = self.emb_layer_norm_after(hidden_states)

        if kwargs.get("output_hidden_states", False):
            all_hidden_states = (*all_hidden_states, hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states if all_hidden_states else None,
        )


class NVEsmPreTrainedModel(EsmPreTrainedModel):
    """An abstract class to handle weights initialization and pretrained model loading."""

    config_class = NVEsmConfig
    base_model_prefix = "esm"
    supports_gradient_checkpointing = False
    accepts_loss_kwargs = False
    _no_split_modules = (
        "TransformerLayer",
        "EsmEmbeddings",
    )

    def _init_weights(self, module: nn.Module):
        """Initialize model weights.

        This method ensures that models with randomly-initialized weights get the correct initial value distribution,
        which can be critical for training stability. We also call this method directly when using meta-device init, as
        the `to_empty` method does not initialize the weights. While the base Transformers model has a similar method,
        we need to extend it to handle TE-specific modules.

        Args:
            module (nn.Module): The module to initialize the weights for.
        """
        super()._init_weights(module)
        if isinstance(
            module,
            (
                transformer_engine.pytorch.module.base.TransformerEngineBaseModule,
                transformer_engine.pytorch.LayerNorm,
                transformer_engine.pytorch.RMSNorm,
            ),
        ):
            module.reset_parameters(defer_init=str(torch.get_default_device()) == "meta")

        if isinstance(module, RotaryPositionEmbedding) and hasattr(module, "inv_freq"):
            # When we initialize the model with `to_empty`, the `inv_freq` attribute is not initialized, so we need to
            # re-initialize it here with the correct values.
            module.inv_freq = RotaryPositionEmbedding(
                self.config.hidden_size // self.config.num_attention_heads
            ).inv_freq.to(module.inv_freq.device)

    @classmethod
    def get_init_context(cls, is_quantized: bool, _is_ds_init_called: bool):
        """Override the default get_init_context method to allow for fp8 model initialization."""
        return []


class NVEsmModel(NVEsmPreTrainedModel):
    """The ESM Encoder-only protein language model.

    This model uses NVDIA's TransformerEngine to optimize attention layer training and inference.
    """

    def __init__(self, config: NVEsmConfig, add_pooling_layer: bool = True):
        """Initialize a NVEsmModel.

        Args:
            config (NVEsmConfig): The configuration of the model.
            add_pooling_layer (bool): Whether to add a pooling layer.
        """
        super().__init__(config)
        self.config = config

        # Ensure pad_token_id is set properly, defaulting to 0 if not specified
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = 0
        self.embeddings = NVEsmEmbeddings(config)
        self.encoder = NVEsmEncoder(config)
        self.pooler = EsmPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """Get the input embeddings of the model."""
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: torch.Tensor):
        """Set the input embeddings of the model.

        Args:
            value (torch.Tensor): The input embeddings.
        """
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        """Forward pass of the NVEsmModel.

        Args:
            input_ids (torch.Tensor): The input ids.
            attention_mask (torch.Tensor): The attention mask.
            position_ids (torch.Tensor): The position ids.
            inputs_embeds (torch.Tensor): The input embeddings.
            **kwargs: Additional arguments, see TransformersKwargs for more details.

        Returns:
            BaseModelOutputWithPooling: The output of the model.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # TE expects a boolean attention mask, where 1s are masked and 0s are not masked
        extended_attention_mask = extended_attention_mask < -1

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            **kwargs,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class NVEsmForMaskedLM(NVEsmPreTrainedModel):
    """NVEsmForMaskedLM is a TransformerEngine-optimized ESM model for masked language modeling."""

    _tied_weights_keys = ("lm_head.decoder.weight",)

    def __init__(self, config: NVEsmConfig):
        """Initialize a NVEsmForMaskedLM.

        Args:
            config (NVEsmConfig): The configuration of the model.
        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `EsmForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.esm = NVEsmModel(config, add_pooling_layer=False)
        self.lm_head = NVEsmLMHead(config)

        self.init_weights()
        self.post_init()

    def get_output_embeddings(self):
        """Get the output embeddings of the model."""
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings of the model."""
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MaskedLMOutput:
        """Forward pass of the NVEsmForMaskedLM.

        Args:
            input_ids (torch.LongTensor): The input ids.
            attention_mask (torch.Tensor): The attention mask.
            position_ids (torch.LongTensor): The position ids.
            inputs_embeds (torch.FloatTensor): The input embeddings.
            labels (torch.LongTensor): The labels.
            **kwargs: Additional arguments, see TransformersKwargs for more details.

        Returns:
            MaskedLMOutput: The output of the model.
        """
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        # Truncate logits back to original vocab_size if padding was used
        if self.config.padded_vocab_size != self.config.vocab_size:
            prediction_scores = prediction_scores[..., : self.config.vocab_size]

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


class NVEsmLMHead(nn.Module):
    """ESM Head for masked language modeling using TransformerEngine."""

    def __init__(self, config: NVEsmConfig):
        """Initialize a NVEsmLMHead.

        Args:
            config (NVEsmConfig): The configuration of the model.
        """
        super().__init__()
        self.dense = transformer_engine.pytorch.Linear(
            config.hidden_size,
            config.hidden_size,
            params_dtype=config.dtype,
            device=str(torch.get_default_device()),
            init_method=lambda x: torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range),
        )

        self.decoder = transformer_engine.pytorch.LayerNormLinear(
            config.hidden_size,
            config.padded_vocab_size if config.padded_vocab_size is not None else config.vocab_size,
            bias=True,
            eps=config.layer_norm_eps,
            params_dtype=config.dtype,
            device=str(torch.get_default_device()),
            init_method=lambda x: torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range),
        )

    def forward(self, features, **kwargs):
        """Forward pass of the NVEsmLMHead.

        Args:
            features (torch.Tensor): The features.
            **kwargs: Additional arguments.
        """
        x = self.dense(features)
        x = torch.nn.functional.gelu(x)
        x = self.decoder(x)
        return x


class NVEsmEmbeddings(nn.Module):
    """Modified version of EsmEmbeddings to support THD inputs."""

    def __init__(self, config):
        """Initialize a NVEsmEmbeddings."""
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            dtype=config.dtype,
        )

        self.layer_norm = (
            transformer_engine.pytorch.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_eps,
                params_dtype=config.dtype,
                device=str(torch.get_default_device()),
            )
            if config.emb_layer_norm_before
            else None
        )

        if config.position_embedding_type != "rotary":
            raise ValueError(
                "The TE-accelerated ESM-2 model only supports rotary position embeddings, received "
                f"{config.position_embedding_type}"
            )

        self.padding_idx = config.pad_token_id
        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        """Forward pass of the NVEsmEmbeddings."""
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # Note that if we want to support ESM-1 (not 1b!) in future then we need to support an
        # embedding_scale factor here.
        embeddings = inputs_embeds

        if (
            kwargs.get("cu_seq_lens_q") is not None
            and kwargs.get("cu_seq_lens_k") is not None
            and kwargs.get("max_length_q") is not None
            and kwargs.get("max_length_k") is not None
        ):
            using_thd = True
            attention_mask = None
        else:
            using_thd = False

        # Matt: ESM has the option to handle masking in MLM in a slightly unusual way. If the token_dropout
        # flag is False then it is handled in the same was as BERT/RoBERTa. If it is set to True, however,
        # masked tokens are treated as if they were selected for input dropout and zeroed out.
        # This "mask-dropout" is compensated for when masked tokens are not present, by scaling embeddings by
        # a factor of (fraction of unmasked tokens during training) / (fraction of unmasked tokens in sample).
        # This is analogous to the way that dropout layers scale down outputs during evaluation when not
        # actually dropping out values (or, equivalently, scale up their un-dropped outputs in training).
        if self.token_dropout and input_ids is not None:
            embeddings = embeddings.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8  # Hardcoded as the ratio used in all ESM model training runs

            if not using_thd:
                # BSHD token dropout correction
                src_lengths = attention_mask.sum(-1) if attention_mask is not None else input_ids.shape[1]
                n_masked_per_seq = (input_ids == self.mask_token_id).sum(-1).float()
                mask_ratio_observed = n_masked_per_seq / src_lengths
                scale_factor = (1 - mask_ratio_train) / (1 - mask_ratio_observed)
                embeddings = (embeddings * scale_factor[:, None, None]).to(embeddings.dtype)

            else:
                src_lengths = torch.diff(kwargs["cu_seq_lens_q"])
                # We need to find the number of masked tokens in each sequence in the padded batch.
                is_masked = (input_ids == self.mask_token_id).squeeze(0)
                n_masked_per_seq = torch.nested.nested_tensor_from_jagged(
                    is_masked, offsets=kwargs["cu_seq_lens_q"]
                ).sum(1)
                mask_ratio_observed = n_masked_per_seq.float() / src_lengths
                scale_factor = (1 - mask_ratio_train) / (1 - mask_ratio_observed)
                reshaped_scale_factor = torch.repeat_interleave(scale_factor, src_lengths, dim=0)
                embeddings = (embeddings * reshaped_scale_factor.unsqueeze(-1)).to(embeddings.dtype)

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)

        if attention_mask is not None:
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)

        return embeddings


class NVEsmForTokenClassification(NVEsmPreTrainedModel):
    """Adds a token classification head to the model.

    Adapted from EsmForTokenClassification in Hugging Face Transformers `modeling_esm.py`.
    """

    def __init__(self, config):
        """Initialize NVEsmForTokenClassification."""
        super().__init__(config)
        self.num_labels = config.num_labels

        self.esm = NVEsmModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = transformer_engine.pytorch.Linear(
            config.hidden_size,
            config.num_labels,
            params_dtype=config.dtype,
            device=str(torch.get_default_device()),
        )

        self.init_weights()
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TokenClassifierOutput:
        """Forward pass for the token classification head.

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
