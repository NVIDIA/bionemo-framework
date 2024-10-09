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


#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch import nn
from torch.nn import functional as F

from bionemo.model.protein.equidock.model.graph_norm import GraphNorm


def get_non_lin(type, negative_slope):
    if type == "swish":
        return nn.SiLU()
    else:
        assert type == "lkyrelu"
        return nn.LeakyReLU(negative_slope=negative_slope)


def get_layer_norm(layer_norm_type, dim):
    if layer_norm_type == "BN":
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == "LN":
        return nn.LayerNorm(dim)
    else:
        return nn.Identity()


def get_final_h_layer_norm(layer_norm_type, dim):
    if layer_norm_type == "BN":
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == "LN":
        return nn.LayerNorm(dim)
    elif layer_norm_type == "GN":
        return GraphNorm(dim)
    else:
        assert layer_norm_type == "0"
        return nn.Identity()


def apply_final_h_layer_norm(g, h, node_type, norm_type, norm_layer):
    if norm_type == "GN":
        return norm_layer(g, h, node_type)
    return norm_layer(h)


def compute_cross_attention(queries, keys, values, mask, cross_msgs):
    """Compute cross attention.
    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    Args:
      queries: NxD float tensor --> queries
      keys: MxD float tensor --> keys
      values: Mxd
      mask: NxM a boolean or boolean-like (eg 0,1 integer) attention mask
      cross_msgs: bool. If false, we bypass attention and just return a zero vector in the shape of queries (TODO shouldn't this be value shaped?)
    Returns:
      attention_x: Nxd float tensor.

    See https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html for notes on how to enable the different
        scaled dot product attention speed-ups if you don't want the default selection to be used (chosen based on inputs).
    """
    if not cross_msgs:
        return queries * 0.0

    # The original equidock paper does not apply dimension-based scaling, and is not causal. They also do not apply attention dropout
    #  during training.
    return F.scaled_dot_product_attention(
        queries, keys, values, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=1.0
    )


def get_mask(ligand_batch_num_nodes, receptor_batch_num_nodes, device):
    rows = sum(ligand_batch_num_nodes)
    cols = sum(receptor_batch_num_nodes)
    # mask is expected to be a boolean matrix in F.scaled_dot_product_attention
    mask = torch.zeros(rows, cols, dtype=torch.bool, device=device)
    partial_l = 0
    partial_r = 0
    for l_n, r_n in zip(ligand_batch_num_nodes, receptor_batch_num_nodes):
        mask[partial_l : partial_l + l_n, partial_r : partial_r + r_n] = 1
        partial_l = partial_l + l_n
        partial_r = partial_r + r_n
    return mask
