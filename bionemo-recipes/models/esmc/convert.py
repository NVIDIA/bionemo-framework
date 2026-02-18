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

"""Weight conversion between EvolutionaryScale ESMC and NVEsmc (TransformerEngine) formats.

The ESMC reference model uses:
- QKV as a Sequential(LayerNorm, Linear) producing [Q||K||V] concatenated
- QK LayerNorm over full d_model dimension (960), bias=False
- Residue scaling: divides attn output and FFN output by sqrt(n_layers/36)
- FFN as Sequential(LayerNorm, Linear, SwiGLU, Linear)

NVEsmc TE model uses:
- LayerNormLinear for QKV with [Q||K||V] concatenated weights (no interleaving)
- Full d_model QK LayerNorm via separate TE LayerNorm modules (exact match)
- DotProductAttention for flash/fused attention
- Residue scaling absorbed into output projection and fc2 weights
- LayerNormMLP for fused FFN
"""

import math

import torch

from modeling_esmc_te import NVEsmcConfig, NVEsmcForMaskedLM


def convert_esmc_to_te(ref_state_dict: dict[str, torch.Tensor], config: NVEsmcConfig) -> NVEsmcForMaskedLM:
    """Convert EvolutionaryScale ESMC weights to NVEsmc (TransformerEngine) format.

    This performs:
    1. Key remapping from ESMC ref format to TE format
    2. QK norm weight direct copy (both use full d_model LayerNorm)
    3. Residue scaling absorption into output projection and fc2 weights

    Args:
        ref_state_dict: State dict from the EvolutionaryScale ESMC model (.pth file).
        config: NVEsmcConfig for the target TE model.

    Returns:
        NVEsmcForMaskedLM with converted weights.
    """
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    scale_factor = math.sqrt(num_layers / 36)

    te_state_dict = {}

    # Embedding
    te_state_dict["esmc.embed_tokens.weight"] = ref_state_dict["embed.weight"]

    for layer_idx in range(num_layers):
        ref_prefix = f"transformer.blocks.{layer_idx}"
        te_prefix = f"esmc.layers.{layer_idx}"

        # Attention LayerNorm (pre-QKV)
        te_state_dict[f"{te_prefix}.layernorm_qkv.layer_norm_weight"] = ref_state_dict[
            f"{ref_prefix}.attn.layernorm_qkv.0.weight"
        ]
        te_state_dict[f"{te_prefix}.layernorm_qkv.layer_norm_bias"] = ref_state_dict[
            f"{ref_prefix}.attn.layernorm_qkv.0.bias"
        ]

        # QKV weight: direct copy (stored as [Q||K||V] concatenated, no interleaving)
        te_state_dict[f"{te_prefix}.layernorm_qkv.weight"] = ref_state_dict[
            f"{ref_prefix}.attn.layernorm_qkv.1.weight"
        ]

        # QK norm: direct copy (both use full d_model LayerNorm)
        # Reference has bias=False, TE LayerNorm always has bias -> set to zeros
        te_state_dict[f"{te_prefix}.q_norm.weight"] = ref_state_dict[f"{ref_prefix}.attn.q_ln.weight"]
        te_state_dict[f"{te_prefix}.q_norm.bias"] = torch.zeros(
            hidden_size, dtype=ref_state_dict[f"{ref_prefix}.attn.q_ln.weight"].dtype
        )
        te_state_dict[f"{te_prefix}.k_norm.weight"] = ref_state_dict[f"{ref_prefix}.attn.k_ln.weight"]
        te_state_dict[f"{te_prefix}.k_norm.bias"] = torch.zeros(
            hidden_size, dtype=ref_state_dict[f"{ref_prefix}.attn.k_ln.weight"].dtype
        )

        # Attention output projection: absorb residue scaling
        out_proj_weight = ref_state_dict[f"{ref_prefix}.attn.out_proj.weight"]
        te_state_dict[f"{te_prefix}.proj.weight"] = out_proj_weight / scale_factor

        # FFN LayerNorm (pre-MLP)
        te_state_dict[f"{te_prefix}.layernorm_mlp.layer_norm_weight"] = ref_state_dict[f"{ref_prefix}.ffn.0.weight"]
        te_state_dict[f"{te_prefix}.layernorm_mlp.layer_norm_bias"] = ref_state_dict[f"{ref_prefix}.ffn.0.bias"]

        # FFN fc1 (gate + up proj concatenated for SwiGLU)
        te_state_dict[f"{te_prefix}.layernorm_mlp.fc1_weight"] = ref_state_dict[f"{ref_prefix}.ffn.1.weight"]

        # FFN fc2 (down proj): absorb residue scaling
        fc2_weight = ref_state_dict[f"{ref_prefix}.ffn.3.weight"]
        te_state_dict[f"{te_prefix}.layernorm_mlp.fc2_weight"] = fc2_weight / scale_factor

    # Final LayerNorm
    te_state_dict["esmc.norm.weight"] = ref_state_dict["transformer.norm.weight"]
    # ESMC final norm has bias=False, but TE LayerNorm always has bias. Set to zeros.
    te_state_dict["esmc.norm.bias"] = torch.zeros(hidden_size, dtype=ref_state_dict["transformer.norm.weight"].dtype)

    # Sequence head (RegressionHead): Linear -> GELU -> LayerNorm -> Linear
    # ref: sequence_head.0 = Linear(960, 960)
    te_state_dict["sequence_head.dense.weight"] = ref_state_dict["sequence_head.0.weight"]
    te_state_dict["sequence_head.dense.bias"] = ref_state_dict["sequence_head.0.bias"]
    # ref: sequence_head.2 = LayerNorm(960), sequence_head.3 = Linear(960, 64)
    # TE LayerNormLinear fuses both
    te_state_dict["sequence_head.decoder.layer_norm_weight"] = ref_state_dict["sequence_head.2.weight"]
    te_state_dict["sequence_head.decoder.layer_norm_bias"] = ref_state_dict["sequence_head.2.bias"]
    te_state_dict["sequence_head.decoder.weight"] = ref_state_dict["sequence_head.3.weight"]
    te_state_dict["sequence_head.decoder.bias"] = ref_state_dict["sequence_head.3.bias"]

    # Build the TE model and load state dict
    with torch.device("meta"):
        model_te = NVEsmcForMaskedLM(config)

    target_state = model_te.state_dict()

    # Directly load the pre-transformed state dict
    for key in list(target_state.keys()):
        if key.endswith("_extra_state"):
            continue
        if key in te_state_dict:
            target_state[key] = te_state_dict[key]

    # Load into model
    model_te.load_state_dict(target_state, strict=False, assign=True)
    model_te.tie_weights()

    return model_te


def convert_esmc_te_to_ref(model_te: NVEsmcForMaskedLM) -> dict[str, torch.Tensor]:
    """Convert NVEsmc (TransformerEngine) weights back to EvolutionaryScale ESMC format.

    This reverses the transformations from convert_esmc_to_te:
    1. QK norm weight direct copy (both use full d_model)
    2. Residue scaling removal from projection weights

    Args:
        model_te: NVEsmcForMaskedLM model with TE weights.

    Returns:
        State dict in EvolutionaryScale ESMC format.
    """
    config = model_te.config
    num_layers = config.num_hidden_layers
    scale_factor = math.sqrt(num_layers / 36)

    te_sd = model_te.state_dict()
    ref_state_dict = {}

    # Embedding
    ref_state_dict["embed.weight"] = te_sd["esmc.embed_tokens.weight"]

    for layer_idx in range(num_layers):
        te_prefix = f"esmc.layers.{layer_idx}"
        ref_prefix = f"transformer.blocks.{layer_idx}"

        # Attention LayerNorm
        ref_state_dict[f"{ref_prefix}.attn.layernorm_qkv.0.weight"] = te_sd[
            f"{te_prefix}.layernorm_qkv.layer_norm_weight"
        ]
        ref_state_dict[f"{ref_prefix}.attn.layernorm_qkv.0.bias"] = te_sd[f"{te_prefix}.layernorm_qkv.layer_norm_bias"]

        # QKV weight: direct copy (no deinterleaving needed)
        ref_state_dict[f"{ref_prefix}.attn.layernorm_qkv.1.weight"] = te_sd[f"{te_prefix}.layernorm_qkv.weight"]

        # QK norm: direct copy (both use full d_model LayerNorm)
        ref_state_dict[f"{ref_prefix}.attn.q_ln.weight"] = te_sd[f"{te_prefix}.q_norm.weight"]
        ref_state_dict[f"{ref_prefix}.attn.k_ln.weight"] = te_sd[f"{te_prefix}.k_norm.weight"]

        # Attention output projection: reverse scaling
        ref_state_dict[f"{ref_prefix}.attn.out_proj.weight"] = te_sd[f"{te_prefix}.proj.weight"] * scale_factor

        # FFN LayerNorm
        ref_state_dict[f"{ref_prefix}.ffn.0.weight"] = te_sd[f"{te_prefix}.layernorm_mlp.layer_norm_weight"]
        ref_state_dict[f"{ref_prefix}.ffn.0.bias"] = te_sd[f"{te_prefix}.layernorm_mlp.layer_norm_bias"]

        # FFN fc1
        ref_state_dict[f"{ref_prefix}.ffn.1.weight"] = te_sd[f"{te_prefix}.layernorm_mlp.fc1_weight"]

        # FFN fc2: reverse scaling
        ref_state_dict[f"{ref_prefix}.ffn.3.weight"] = te_sd[f"{te_prefix}.layernorm_mlp.fc2_weight"] * scale_factor

    # Final LayerNorm (no bias in ref)
    ref_state_dict["transformer.norm.weight"] = te_sd["esmc.norm.weight"]

    # Sequence head
    ref_state_dict["sequence_head.0.weight"] = te_sd["sequence_head.dense.weight"]
    ref_state_dict["sequence_head.0.bias"] = te_sd["sequence_head.dense.bias"]
    ref_state_dict["sequence_head.2.weight"] = te_sd["sequence_head.decoder.layer_norm_weight"]
    ref_state_dict["sequence_head.2.bias"] = te_sd["sequence_head.decoder.layer_norm_bias"]
    ref_state_dict["sequence_head.3.weight"] = te_sd["sequence_head.decoder.weight"]
    ref_state_dict["sequence_head.3.bias"] = te_sd["sequence_head.decoder.bias"]

    return ref_state_dict
