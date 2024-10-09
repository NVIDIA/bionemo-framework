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


from pathlib import Path
from typing import Dict, Union

import torch
from nemo.utils import logging

from bionemo.model.protein.openfold.openfold_model import AlphaFold
from bionemo.model.protein.openfold.optim_hub import OptimHub


StateDict = Dict[str, torch.Tensor]


def remap_layers_names(
    checkpoint_state_dict: StateDict,
) -> StateDict:
    init_state_dict = {}
    for key, tensor in checkpoint_state_dict.items():
        # these layer names are converted to fit public OpenFold checkpoint as available on 07/27/2023
        key = key.replace("template_pointwise_att.", "template_pointwise_attention.")
        key = key.replace("evoformer.", "evoformer_stack.")
        key = key.replace("aux_heads.", "auxiliary_heads.")
        key = key.replace("._msa_att.", ".")
        key = key.replace(".transition.layers.0.", ".transition.")
        assert isinstance(tensor, torch.Tensor)
        init_state_dict[key] = tensor
    return init_state_dict


def remap_mlperf_layers(
    alphafold: AlphaFold,
    checkpoint_state_dict: StateDict,
) -> StateDict:
    alphafold_state_dict_keys = set(alphafold.state_dict().keys())
    init_state_dict = {}

    for key in alphafold_state_dict_keys:
        # Rename gate bias:
        if ".mha.linear_g_bias" in key:
            init_key = key.replace("linear_g_bias", "linear_g.bias")
            init_param = checkpoint_state_dict[init_key]
            init_state_dict[key] = init_param
        # Fuse MHA linear weights:
        elif ".mha.linear_qkvg.weight" in key:
            init_key_q = key.replace(".linear_qkvg.", ".linear_q.")
            init_key_k = key.replace(".linear_qkvg.", ".linear_k.")
            init_key_v = key.replace(".linear_qkvg.", ".linear_v.")
            init_key_g = key.replace(".linear_qkvg.", ".linear_g.")
            init_param_q = checkpoint_state_dict[init_key_q]
            init_param_k = checkpoint_state_dict[init_key_k]
            init_param_v = checkpoint_state_dict[init_key_v]
            init_param_g = checkpoint_state_dict[init_key_g]

            init_param = torch.cat(
                [
                    init_param_q,
                    init_param_k,
                    init_param_v,
                    init_param_g,
                ],
                dim=0,
            )

            init_state_dict[key] = init_param
        elif ".mha.linear_kv.weight" in key:
            init_key_k = key.replace(".linear_kv.", ".linear_k.")
            init_key_v = key.replace(".linear_kv.", ".linear_v.")
            init_param_k = checkpoint_state_dict[init_key_k]
            init_param_v = checkpoint_state_dict[init_key_v]
            init_param = torch.cat(
                [
                    init_param_k,
                    init_param_v,
                ],
                dim=0,
            )
            init_state_dict[key] = init_param
        # Identity:
        else:
            init_state_dict[key] = checkpoint_state_dict[key]
    return init_state_dict


def load_pt_checkpoint(model: AlphaFold, checkpoint_path: Union[str, Path]):
    try:
        checkpoint_state_dict = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_state_dict)
        return
    except RuntimeError:
        logging.warning(
            f"Failed to directly load {checkpoint_path} checkpoint. Trying again with layer names remapping."
        )

    # if direct checkpoint loading fails, remap names...
    checkpoint_state_dict = remap_layers_names(checkpoint_state_dict)
    # ... check if any layers been fused ...
    if OptimHub.config("mha_fused_gemm"):
        checkpoint_state_dict = remap_mlperf_layers(model, checkpoint_state_dict)

    # Finally, re-try again
    model.load_state_dict(checkpoint_state_dict, strict=True)
