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

from typing import Optional, Tuple, Union

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
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.esm.configuration_esm import EsmConfig
from transformers.models.esm.modeling_esm import EsmEmbeddings, EsmPooler
from transformers.utils import logging


logger = logging.get_logger(__name__)


class NVEsmConfig(EsmConfig):
    """NVEsmConfig is a configuration for the NVEsm model."""

    model_type: str = "nv_esm"

    def __init__(
        self,
        qkv_weight_interleaved: bool = True,
        encoder_activation: str = "gelu",
        attn_input_format: str = "bshd",
        fuse_qkv_params: bool = True,
        micro_batch_size: Optional[int] = None,
        max_seq_length: Optional[int] = None,
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


class NVEsmEncoder(nn.Module):
    """NVEsmEncoder is a TransformerEngine-optimized ESM encoder."""

    def __init__(self, config: NVEsmConfig):
        """Initialize a NVEsmEncoder.

        Args:
            config (NVEsmConfig): The configuration of the model.
        """
        super().__init__()
        self.config = config
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
                    self_attn_mask_type="padding",
                    activation=config.encoder_activation,
                    attn_input_format=config.attn_input_format,
                    seq_length=config.max_seq_length,
                    micro_batch_size=config.micro_batch_size,
                    num_gqa_groups=config.num_attention_heads,
                    fuse_qkv_params=config.fuse_qkv_params,
                    params_dtype=config.torch_dtype,
                    window_size=(-1, -1),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.emb_layer_norm_after = transformer_engine.pytorch.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)
            self.te_rope_emb = self.rotary_embeddings(max_seq_len=config.max_position_embeddings).cuda()
        else:
            self.te_rope_emb = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ):
        """Forward pass of the NVEsmEncoder.

        Args:
            hidden_states (torch.Tensor): The hidden states.
            attention_mask (torch.Tensor): The attention mask.
            output_hidden_states (bool): Whether to output the hidden states.
        """
        all_hidden_states = () if output_hidden_states else None

        for layer_module in self.layers:
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                rotary_pos_emb=self.te_rope_emb,
            )

        hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class NVEsmPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and pretrained model loading."""

    config_class = NVEsmConfig
    base_model_prefix = "esm"
    supports_gradient_checkpointing = False
    _no_split_modules = (
        "TransformerLayer",
        "EsmEmbeddings",
    )

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module: nn.Module):
        """Initialize the weights.

        Args:
            module (nn.Module): The module to initialize the weights for.
        """
        if isinstance(
            module, (nn.Linear, transformer_engine.pytorch.Linear, transformer_engine.pytorch.LayerNormLinear)
        ):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, (nn.LayerNorm, transformer_engine.pytorch.LayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, transformer_engine.pytorch.LayerNormLinear):
            module.layer_norm_weight.data.fill_(1.0)
            if module.layer_norm_bias is not None:
                module.layer_norm_bias.data.zero_()


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

        self.embeddings = EsmEmbeddings(config)
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
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """Forward pass of the NVEsmModel.

        Args:
            input_ids (torch.Tensor): The input ids.
            attention_mask (torch.Tensor): The attention mask.
            position_ids (torch.Tensor): The position ids.
            head_mask (torch.Tensor): The head mask.
            inputs_embeds (torch.Tensor): The input embeddings.
            output_hidden_states (bool): Whether to output the hidden states.

        Returns:
            BaseModelOutputWithPooling: The output of the model.
        """
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention if the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input.
            This mask is used in the cross-attention if the model is configured as a decoder. Mask
            values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            Note that this mask is inverted when it is passed to TransformerEngine, which expects a
            boolean mask where 1s are masked and 0s are not masked.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

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

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_hidden_states=output_hidden_states,
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
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        """Forward pass of the NVEsmForMaskedLM.

        Args:
            input_ids (torch.LongTensor): The input ids.
            attention_mask (torch.Tensor): The attention mask.
            position_ids (torch.LongTensor): The position ids.
            inputs_embeds (torch.FloatTensor): The input embeddings.
            labels (torch.LongTensor): The labels.
            output_hidden_states (bool): Whether to output the hidden states.

        Returns:
            MaskedLMOutput: The output of the model.
        """
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, *optional*, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            labels = labels.to(prediction_scores.device)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
        )

    def predict_contacts(self, tokens: torch.Tensor, attention_mask: torch.Tensor):
        """Predict the contacts of the model.

        Args:
            tokens (torch.Tensor): The tokens.
            attention_mask (torch.Tensor): The attention mask.

        Returns:
            torch.Tensor: The predicted contacts.
        """
        return self.esm.predict_contacts(tokens, attention_mask=attention_mask)


class NVEsmLMHead(nn.Module):
    """ESM Head for masked language modeling using TransformerEngine."""

    def __init__(self, config: NVEsmConfig):
        """Initialize a NVEsmLMHead.

        Args:
            config (NVEsmConfig): The configuration of the model.
        """
        super().__init__()
        self.dense = transformer_engine.pytorch.Linear(config.hidden_size, config.hidden_size)

        self.decoder = transformer_engine.pytorch.LayerNormLinear(
            config.hidden_size,
            config.vocab_size,
            bias=True,
            eps=config.layer_norm_eps,
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
