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

import torch
from accelerate import init_empty_weights
from nemo.lightning import io
from torch import nn

from esm.modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM


mapping = {
    "esm.encoder.layer.*.attention.output.dense.weight": "esm.encoder.layers.*.self_attention.proj.weight",
    "esm.encoder.layer.*.attention.output.dense.bias": "esm.encoder.layers.*.self_attention.proj.bias",
    "esm.encoder.layer.*.attention.LayerNorm.weight": "esm.encoder.layers.*.self_attention.layernorm_qkv.layer_norm_weight",
    "esm.encoder.layer.*.attention.LayerNorm.bias": "esm.encoder.layers.*.self_attention.layernorm_qkv.layer_norm_bias",
    "esm.encoder.layer.*.intermediate.dense.weight": "esm.encoder.layers.*.layernorm_mlp.fc1_weight",
    "esm.encoder.layer.*.intermediate.dense.bias": "esm.encoder.layers.*.layernorm_mlp.fc1_bias",
    "esm.encoder.layer.*.output.dense.weight": "esm.encoder.layers.*.layernorm_mlp.fc2_weight",
    "esm.encoder.layer.*.output.dense.bias": "esm.encoder.layers.*.layernorm_mlp.fc2_bias",
    "esm.encoder.layer.*.LayerNorm.weight": "esm.encoder.layers.*.layernorm_mlp.layer_norm_weight",
    "esm.encoder.layer.*.LayerNorm.bias": "esm.encoder.layers.*.layernorm_mlp.layer_norm_bias",
    "esm.encoder.emb_layer_norm_after.weight": "esm.encoder.emb_layer_norm_after.weight",
    "esm.encoder.emb_layer_norm_after.bias": "esm.encoder.emb_layer_norm_after.bias",
    "lm_head.dense.weight": "lm_head.dense.weight",
    "lm_head.dense.bias": "lm_head.dense.bias",
    "lm_head.layer_norm.weight": "lm_head.decoder.layer_norm_weight",
    "lm_head.layer_norm.bias": "lm_head.decoder.layer_norm_bias",
}


def convert_esm_hf_to_te(model_hf: nn.Module, **config_kwargs) -> nn.Module:
    """Convert a Hugging Face model to a Transformer Engine model.

    Args:
        model_hf (nn.Module): The Hugging Face model.
        **config_kwargs: Additional configuration kwargs to be passed to NVEsmConfig.

    Returns:
        nn.Module: The Transformer Engine model.
    """
    # TODO (peter): this is super similar method to the AMPLIFY one, maybe we can abstract or keep simlar naming? models/amplify/src/amplify/state_dict_convert.py:convert_amplify_hf_to_te
    te_config = NVEsmConfig(**model_hf.config.to_dict(), **config_kwargs)
    with init_empty_weights():
        model_te = NVEsmForMaskedLM(te_config)

    output_model = io.apply_transforms(
        model_hf,
        model_te,
        mapping,
        [_pack_qkv_weight, _pack_qkv_bias, _pad_embeddings, _pad_decoder_weights, _pad_bias],
        state_dict_ignored_entries=["lm_head.decoder.weight"],
    )

    output_model.tie_weights()

    return output_model


@io.state_transform(
    source_key=(
        "esm.encoder.layer.*.attention.self.query.weight",
        "esm.encoder.layer.*.attention.self.key.weight",
        "esm.encoder.layer.*.attention.self.value.weight",
    ),
    target_key="esm.encoder.layers.*.self_attention.layernorm_qkv.weight",
)
def _pack_qkv_weight(ctx: io.TransformCTX, query, key, value):
    """Pad the embedding layer to the new input dimension."""
    concat_weights = torch.cat((query, key, value), dim=0)
    input_shape = concat_weights.size()
    np = ctx.target.config.num_attention_heads
    # transpose weights
    # [sequence length, batch size, num_splits_model_parallel * attention head size * #attention heads]
    # --> [sequence length, batch size, attention head size * num_splits_model_parallel * #attention heads]
    concat_weights = concat_weights.view(3, np, -1, query.size()[-1])
    concat_weights = concat_weights.transpose(0, 1).contiguous()
    concat_weights = concat_weights.view(*input_shape)
    return concat_weights


@io.state_transform(
    source_key=(
        "esm.encoder.layer.*.attention.self.query.bias",
        "esm.encoder.layer.*.attention.self.key.bias",
        "esm.encoder.layer.*.attention.self.value.bias",
    ),
    target_key="esm.encoder.layers.*.self_attention.layernorm_qkv.bias",
)
def _pack_qkv_bias(ctx: io.TransformCTX, query, key, value):
    """Pad the embedding layer to the new input dimension."""
    concat_biases = torch.cat((query, key, value), dim=0)
    input_shape = concat_biases.size()
    np = ctx.target.config.num_attention_heads
    # transpose biases
    # [num_splits_model_parallel * attention head size * #attention heads]
    # --> [attention head size * num_splits_model_parallel * #attention heads]
    concat_biases = concat_biases.view(3, np, -1)
    concat_biases = concat_biases.transpose(0, 1).contiguous()
    concat_biases = concat_biases.view(*input_shape)
    return concat_biases


def _pad_weights(ctx: io.TransformCTX, source_embed):
    """Pad the embedding layer to the new input dimension."""
    target_embedding_dimension = ctx.target.config.padded_vocab_size
    hf_embedding_dimension = source_embed.size(0)
    num_padding_rows = target_embedding_dimension - hf_embedding_dimension
    padding_rows = torch.zeros(
        num_padding_rows, source_embed.size(1), dtype=source_embed.dtype, device=source_embed.device
    )
    return torch.cat((source_embed, padding_rows), dim=0)


_pad_embeddings = io.state_transform(
    source_key="esm.embeddings.word_embeddings.weight",
    target_key="esm.embeddings.word_embeddings.weight",
)(_pad_weights)

_pad_decoder_weights = io.state_transform(
    source_key="lm_head.decoder.weight",
    target_key="lm_head.decoder.weight",
)(_pad_weights)


@io.state_transform(
    source_key="lm_head.bias",
    target_key="lm_head.decoder.bias",
)
def _pad_bias(ctx: io.TransformCTX, source_bias):
    """Pad the embedding layer to the new input dimension."""
    target_embedding_dimension = ctx.target.config.padded_vocab_size
    hf_embedding_dimension = source_bias.size(0)
    output_bias = torch.finfo(source_bias.dtype).min * torch.ones(
        target_embedding_dimension, dtype=source_bias.dtype, device=source_bias.device
    )
    output_bias[:hf_embedding_dimension] = source_bias
    return output_bias
