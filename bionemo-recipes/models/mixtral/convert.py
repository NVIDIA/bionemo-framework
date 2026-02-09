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
from transformers import MixtralConfig, MixtralForCausalLM

import state
from modeling_mixtral_te import NVMixtralConfig, NVMixtralForCausalLM


mapping = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    "model.layers.*.input_layernorm.weight": "model.layers.*.self_attention.layernorm_qkv.layer_norm_weight",
    "model.layers.*.self_attn.o_proj.weight": "model.layers.*.self_attention.proj.weight",
    "model.layers.*.post_attention_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
    "model.layers.*.mlp.gate.weight": "model.layers.*.mlp.gate.weight",
    "model.norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}

reverse_mapping = {v: k for k, v in mapping.items()}


def _split_experts_gate_up(gate_up_proj: torch.Tensor):
    """Split a stacked expert gate_up tensor into per-expert tensors.

    Args:
        gate_up_proj: Tensor of shape [num_experts, 2*ffn, hidden].

    Returns:
        Tuple of per-expert tensors, each of shape [2*ffn, hidden].
    """
    return tuple(gate_up_proj[i] for i in range(gate_up_proj.shape[0]))


def _split_experts_down(down_proj: torch.Tensor):
    """Split a stacked expert down_proj tensor into per-expert tensors.

    Args:
        down_proj: Tensor of shape [num_experts, hidden, ffn].

    Returns:
        Tuple of per-expert tensors, each of shape [hidden, ffn].
    """
    return tuple(down_proj[i] for i in range(down_proj.shape[0]))


def _make_merge_experts_fn(num_experts: int):
    """Create a merge function with the correct number of named parameters.

    The state.py transform system maps function parameter names to source keys, so we need a function
    with exactly `num_experts` named parameters (weight0, weight1, ...).
    """
    param_names = [f"weight{i}" for i in range(num_experts)]
    code = f"def merge_experts({', '.join(param_names)}):\n    return torch.stack([{', '.join(param_names)}])"
    local_ns = {"torch": torch}
    exec(code, local_ns)
    return local_ns["merge_experts"]


def convert_mixtral_hf_to_te(model_hf: MixtralForCausalLM, **config_kwargs) -> NVMixtralForCausalLM:
    """Convert a Hugging Face Mixtral model to a Transformer Engine model.

    Args:
        model_hf: The Hugging Face Mixtral model.
        **config_kwargs: Additional configuration kwargs to be passed to NVMixtralConfig.

    Returns:
        The Transformer Engine Mixtral model.
    """
    te_config = NVMixtralConfig(**model_hf.config.to_dict(), **config_kwargs)
    with torch.device("meta"):
        model_te = NVMixtralForCausalLM(te_config)

    num_experts = model_hf.config.num_local_experts

    # Build expert weight target keys for gate_up and down projections
    gate_up_target_keys = tuple(f"model.layers.*.mlp.experts_gate_up.weight{i}" for i in range(num_experts))
    down_target_keys = tuple(f"model.layers.*.mlp.experts_down.weight{i}" for i in range(num_experts))

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
                source_key="model.layers.*.mlp.experts.gate_up_proj",
                target_key=gate_up_target_keys,
                fn=_split_experts_gate_up,
            ),
            state.state_transform(
                source_key="model.layers.*.mlp.experts.down_proj",
                target_key=down_target_keys,
                fn=_split_experts_down,
            ),
        ],
    )

    output_model.model.rotary_emb.inv_freq = model_hf.model.rotary_emb.inv_freq.clone()

    return output_model


def convert_mixtral_te_to_hf(model_te: NVMixtralForCausalLM, **config_kwargs) -> MixtralForCausalLM:
    """Convert a Transformer Engine Mixtral model to a Hugging Face model.

    Args:
        model_te: The Transformer Engine Mixtral model.
        **config_kwargs: Additional configuration kwargs to be passed to MixtralConfig.

    Returns:
        The Hugging Face Mixtral model.
    """
    te_config_dict = model_te.config.to_dict()
    valid_keys = set(MixtralConfig.__init__.__code__.co_varnames)
    filtered_config = {k: v for k, v in te_config_dict.items() if k in valid_keys}
    hf_config = MixtralConfig(**filtered_config, **config_kwargs)

    with torch.device("meta"):
        model_hf = MixtralForCausalLM(hf_config)

    num_experts = hf_config.num_local_experts

    gate_up_source_keys = tuple(f"model.layers.*.mlp.experts_gate_up.weight{i}" for i in range(num_experts))
    down_source_keys = tuple(f"model.layers.*.mlp.experts_down.weight{i}" for i in range(num_experts))

    merge_fn = _make_merge_experts_fn(num_experts)

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
                source_key=gate_up_source_keys,
                target_key="model.layers.*.mlp.experts.gate_up_proj",
                fn=merge_fn,
            ),
            state.state_transform(
                source_key=down_source_keys,
                target_key="model.layers.*.mlp.experts.down_proj",
                fn=merge_fn,
            ),
        ],
    )

    output_model.model.rotary_emb.inv_freq = model_te.model.rotary_emb.inv_freq.clone()
    output_model.tie_weights()

    return output_model
