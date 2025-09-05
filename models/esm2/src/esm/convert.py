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
from transformers import EsmConfig, EsmForMaskedLM

from esm.modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM


mapping = {
    "esm.embeddings.word_embeddings.weight": "esm.embeddings.word_embeddings.weight",
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
    "lm_head.bias": "lm_head.decoder.bias",
    "lm_head.decoder.weight": "lm_head.decoder.weight",
    "lm_head.dense.weight": "lm_head.dense.weight",
    "lm_head.dense.bias": "lm_head.dense.bias",
    "lm_head.layer_norm.weight": "lm_head.decoder.layer_norm_weight",
    "lm_head.layer_norm.bias": "lm_head.decoder.layer_norm_bias",
}

# Reverse mapping from TE to HF format by reversing the original mapping
reverse_mapping = {v: k for k, v in mapping.items()}


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
        [_pack_qkv_weight, _pack_qkv_bias],
        state_dict_ignored_entries=["lm_head.decoder.weight"],
    )

    output_model.tie_weights()

    return output_model


def convert_esm_te_to_hf(model_te: nn.Module, **config_kwargs) -> nn.Module:
    """Convert a Transformer Engine model back to the original HuggingFace Facebook ESM-2 format.

    This function converts from the NVIDIA Transformer Engine (TE) format back to the
    weight format compatible with the original facebook/esm2_* series of checkpoints.
    The TE model is also a HuggingFace model, but this conversion ensures compatibility
    with the original Facebook ESM-2 model architecture and weight format hosted on Hugging Face.

    Args:
        model_te (nn.Module): The Transformer Engine model.
        **config_kwargs: Additional configuration kwargs to be passed to EsmConfig.

    Returns:
        nn.Module: The Hugging Face model in original Facebook ESM-2 format hosted on Hugging Face.
    """
    # Convert TE config to HF config
    hf_config_dict = model_te.config.to_dict()

    # Remove TE-specific config options
    te_specific_keys = [
        "qkv_weight_interleaved",
        "encoder_activation",
        "attn_input_format",
        "fuse_qkv_params",
        "micro_batch_size",
        "max_seq_length",
        "model_type",
    ]
    for key in te_specific_keys:
        hf_config_dict.pop(key, None)

    hf_config = EsmConfig(**hf_config_dict, **config_kwargs)

    with init_empty_weights():
        model_hf = EsmForMaskedLM(hf_config)

        # Remove contact_head since it's not present in TE models
        if hasattr(model_hf.esm, "contact_head"):
            delattr(model_hf.esm, "contact_head")

    output_model = io.apply_transforms(
        model_te,
        model_hf,
        reverse_mapping,
        [_unpack_qkv_weight, _unpack_qkv_bias],
        state_dict_ignored_entries=[
            "lm_head.decoder.weight",
            "esm.contact_head.regression.weight",
            "esm.contact_head.regression.bias",
        ],
    )

    output_model.tie_weights()

    # Note: contact_head parameters are not preserved in TE models
    # They are lost during HF -> TE conversion and cannot be recovered
    # The converted model will not have the original contact_head weights

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


@io.state_transform(
    source_key="esm.encoder.layers.*.self_attention.layernorm_qkv.weight",
    target_key=(
        "esm.encoder.layer.*.attention.self.query.weight",
        "esm.encoder.layer.*.attention.self.key.weight",
        "esm.encoder.layer.*.attention.self.value.weight",
    ),
)
def _unpack_qkv_weight(ctx: io.TransformCTX, qkv_weight):
    """Unpack the fused QKV weight into separate query, key, and value weights."""
    np = ctx.source.config.num_attention_heads

    # Reverse the packing transformation
    # First, reshape to separate the interleaved Q, K, V
    # [attention head size * num_splits_model_parallel * #attention heads]
    # --> [num_splits_model_parallel * attention head size * #attention heads]
    qkv_weight = qkv_weight.view(np, 3, -1, qkv_weight.size()[-1])  # Output:[num_heads, 3, head_dim, vocab_size]
    qkv_weight = qkv_weight.transpose(0, 1).contiguous()  # Output:[3, num_heads, head_dim, vocab_size]

    # Split into Q, K, V directly from the transposed tensor
    # qkv_weight shape: [3, num_heads, head_dim, input_dim]
    query = qkv_weight[0]  # [num_heads, head_dim, input_dim]
    key = qkv_weight[1]  # [num_heads, head_dim, input_dim]
    value = qkv_weight[2]  # [num_heads, head_dim, input_dim]

    # Reshape to match HF format: [total_head_dim, input_dim]
    query = query.view(-1, query.size()[-1])  # [num_heads * head_dim, input_dim]
    key = key.view(-1, key.size()[-1])  # [num_heads * head_dim, input_dim]
    value = value.view(-1, value.size()[-1])  # [num_heads * head_dim, input_dim]

    return query, key, value


@io.state_transform(
    source_key="esm.encoder.layers.*.self_attention.layernorm_qkv.bias",
    target_key=(
        "esm.encoder.layer.*.attention.self.query.bias",
        "esm.encoder.layer.*.attention.self.key.bias",
        "esm.encoder.layer.*.attention.self.value.bias",
    ),
)
def _unpack_qkv_bias(ctx: io.TransformCTX, qkv_bias):
    """Unpack the fused QKV bias into separate query, key, and value biases."""
    np = ctx.source.config.num_attention_heads

    # Reverse the packing transformation
    # First, reshape to separate the interleaved Q, K, V
    # [num_splits_model_parallel * attention head size * #attention heads]
    # --> [attention head size * num_splits_model_parallel * #attention heads]
    qkv_bias = qkv_bias.view(np, 3, -1)
    qkv_bias = qkv_bias.transpose(0, 1).contiguous()

    # Split into Q, K, V directly from the transposed tensor
    # qkv_bias shape: [3, num_heads, head_dim]
    query = qkv_bias[0]  # [num_heads, head_dim]
    key = qkv_bias[1]  # [num_heads, head_dim]
    value = qkv_bias[2]  # [num_heads, head_dim]

    # Reshape to match HF format: [total_head_dim]
    query = query.view(-1)  # [num_heads * head_dim]
    key = key.view(-1)  # [num_heads * head_dim]
    value = value.view(-1)  # [num_heads * head_dim]

    return query, key, value
