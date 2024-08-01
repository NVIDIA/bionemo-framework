# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Sequence


__all__: Sequence[str] = ("nemo1_to_nemo2_biobert_key_mapping",)


def nemo1_to_nemo2_biobert_key_mapping(
    old_key: str,
    new_model_prefix: str = "module",
    old_model_prefix: str = "model",
    te_mapping: bool = False,
) -> str:
    """This function is used to map the keys from the old nemo BERT models to the new BioBERT models

    Args:
        old_key (str): old key we want to map to the expected new key name.
        new_model_prefix (str, optional): The new key for the base weights.
            If you point this at the core megatron model set it to "".
            For the regular nemo2 lightning module following standards, set it to "module".
            Defaults to "module".
        old_model_prefix (str, optional): The previous saved weight prefix. Defaults to "model" which was the standard in nemo1.

    Returns:
        str: New key name
    """  # noqa: D415
    # add the . to the end of the input prefixes if they are not the empty string,
    #  unless the user has already done so.
    if old_model_prefix != "":
        old_model_prefix = f"{old_model_prefix.rstrip('.')}."
    if new_model_prefix != "":
        new_model_prefix = f"{new_model_prefix.rstrip('.')}."

    # This function is used to map the keys from the old nemo BERT models to the new BioBERT models
    base_rename = old_key.replace(f"{old_model_prefix}language_model.", f"{new_model_prefix}")
    base_rename = base_rename.replace(f"{old_model_prefix}", f"{new_model_prefix}")
    if "dense_h_to_4h" in base_rename:
        return base_rename.replace("dense_h_to_4h", "linear_fc1")
    if "dense_4h_to_h" in base_rename:
        return base_rename.replace("dense_4h_to_h", "linear_fc2")
    if "query_key_value" in base_rename:
        return base_rename.replace("query_key_value", "linear_qkv")
    if "self_attention.dense" in base_rename:
        #  This is definitely the linear_proj and not the qkv. The linear_proj shapes are 256x256
        #   which match dense but not query_key_value
        # (Pdb) new_state_dict['encoder.layers.4.self_attention.linear_proj.weight'].shape
        #  torch.Size([256, 256])
        # (Pdb) new_state_dict['encoder.layers.4.self_attention.linear_qkv.weight'].shape
        # torch.Size([768, 256])
        # (Pdb) new_state_dict['encoder.layers.4.self_attention.linear_qkv.bias'].shape
        # torch.Size([768])
        return base_rename.replace("self_attention.dense", "self_attention.linear_proj")
    if "lm_head.bias" in base_rename:
        return base_rename.replace("lm_head.bias", "output_layer.bias")
    if "lm_head.weight" in base_rename:
        return base_rename.replace("lm_head.weight", "output_layer.weight")
    if "lm_head.layernorm" in base_rename:
        return base_rename.replace("lm_head.layernorm", "lm_head.layer_norm")

    if "post_attention_layernorm" in base_rename:
        base_rename = base_rename.replace("post_attention_layernorm", "pre_mlp_layernorm")

    # Handle the transformer engine spec's differences in layer naming and where things like layernorm are stored.
    #  TE moves layernorm from  an object that's part of the main attention layer to being an internal component of
    #  the linear layers, probably for efficiency/fusion of some sort.
    if te_mapping:
        if ".input_layernorm.weight" in base_rename:
            return base_rename.replace(".input_layernorm.weight", ".self_attention.linear_qkv.layer_norm_weight")
        if ".input_layernorm.bias" in base_rename:
            return base_rename.replace(".input_layernorm.bias", ".self_attention.linear_qkv.layer_norm_bias")
        if ".pre_mlp_layernorm.bias" in base_rename:
            return base_rename.replace(".pre_mlp_layernorm.bias", ".mlp.linear_fc1.layer_norm_bias")
        if ".pre_mlp_layernorm.weight" in base_rename:
            return base_rename.replace(".pre_mlp_layernorm.weight", ".mlp.linear_fc1.layer_norm_weight")
    return base_rename
