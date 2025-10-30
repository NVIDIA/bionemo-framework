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
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

import state
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM


mapping = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    "model.layers.*.input_layernorm.weight": "model.layers.*.self_attention.layernorm_qkv.layer_norm_weight",
    "model.layers.*.self_attn.o_proj.weight": "model.layers.*.self_attention.proj.weight",
    "model.layers.*.post_attention_layernorm.weight": "model.layers.*.layernorm_mlp.layer_norm_weight",
    "model.layers.*.mlp.down_proj.weight": "model.layers.*.layernorm_mlp.fc2_weight",
    "model.norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}

# Reverse mapping from TE to HF format by reversing the original mapping
reverse_mapping = {v: k for k, v in mapping.items()}


def convert_llama_hf_to_te(model_hf: nn.Module, **config_kwargs) -> nn.Module:
    """Convert a Hugging Face model to a Transformer Engine model.

    Args:
        model_hf (nn.Module): The Hugging Face model.
        **config_kwargs: Additional configuration kwargs to be passed to NVLlamaConfig.

    Returns:
        nn.Module: The Transformer Engine model.
    """
    te_config = NVLlamaConfig(**model_hf.config.to_dict(), **config_kwargs)
    with torch.device("meta"):
        model_te = NVLlamaForCausalLM(te_config)

    output_model = state.apply_transforms(
        model_hf,
        model_te,
        mapping,
        [
            state.state_transform(
                source_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                target_key="model.layers.*.self_attention.layernorm_qkv.weight",
                fn=state.TransformFns.merge_qkv,
            ),
            state.state_transform(
                source_key=(
                    "model.layers.*.mlp.gate_proj.weight",
                    "model.layers.*.mlp.up_proj.weight",
                ),
                target_key="model.layers.*.layernorm_mlp.fc1_weight",
                fn=state.TransformFns.merge_fc1,
            ),
        ],
    )

    output_model.model.rotary_emb.inv_freq = model_hf.model.rotary_emb.inv_freq.clone()
    output_model.tie_weights()

    return output_model


def convert_llama_te_to_hf(model_te: nn.Module, **config_kwargs) -> nn.Module:
    """Convert a Hugging Face model to a Transformer Engine model.

    Args:
        model_te (nn.Module): The Hugging Face model.
        **config_kwargs: Additional configuration kwargs to be passed to NVLlamaConfig.

    Returns:
        nn.Module: The Transformer Engine model.
    """
    hf_config = LlamaConfig(**model_te.config.to_dict(), **config_kwargs)

    with torch.device("meta"):
        model_hf = LlamaForCausalLM(hf_config)

    output_model = state.apply_transforms(
        model_te,
        model_hf,
        reverse_mapping,
        [
            state.state_transform(
                source_key="model.layers.*.self_attention.layernorm_qkv.weight",
                target_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                fn=state.TransformFns.split_qkv,
            ),
            state.state_transform(
                source_key="model.layers.*.layernorm_mlp.fc1_weight",
                target_key=(
                    "model.layers.*.mlp.gate_proj.weight",
                    "model.layers.*.mlp.up_proj.weight",
                ),
                fn=state.TransformFns.split_fc1,
            ),
        ],
    )

    output_model.model.rotary_emb.inv_freq = model_te.model.rotary_emb.inv_freq.clone()
    output_model.tie_weights()

    return output_model
