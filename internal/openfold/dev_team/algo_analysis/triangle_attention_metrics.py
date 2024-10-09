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


from typing import Tuple

from internal.openfold.dev_team.algo_analysis.algo_metrics import (  # noqa
    AlgoMetrics,
    layer_norm_metrics,
    linear_metrics,
    linear_no_bias_metrics,
    multiply_elem_wise,
    sigmoid_metrics,
    soft_max_metrics,
)
from internal.openfold.dev_team.algo_analysis.row_and_column_attention_metrics_helpers import (
    create_context_vector_metrics,
    create_mha_qkv_metrics,
    create_row_or_column_attention_scores,
)


def triangle_attention_metrics(
    z_shape: Tuple,
    c_hidden_tri_att: int = 32,
    num_heads_tri: int = 8,
) -> Tuple[Tuple, AlgoMetrics]:
    """
    Algorithm 13 and 14 in the supplementary material of the alphafold 2 paper.

    starting: attend on second index of z, for fixed first index of z
        --> similar to row attention

    ending: attend of first index of z, for fixed second index of z
        --> similar to column attention

    """
    num_r = z_shape[0]

    algo_metrics_list = []

    # input projection
    # (1)
    zeta_shape, zeta_algo_metrics = layer_norm_metrics(z_shape)
    algo_metrics_list += [zeta_algo_metrics]

    # (2)
    (q_shape, k_shape, v_shape), mha_qkv_metrics = create_mha_qkv_metrics(
        input_shape=zeta_shape, c_hidden_att=c_hidden_tri_att, num_heads=num_heads_tri
    )
    algo_metrics_list += [mha_qkv_metrics]

    # (3)
    pair_bias_shape, pair_bias_metrics = linear_no_bias_metrics(
        output_shape=(zeta_shape[0], zeta_shape[1], num_heads_tri),
        input_shape=zeta_shape,
        count_input_memory=False,
    )
    algo_metrics_list += [pair_bias_metrics]

    # (4a)
    gamma_shape, gamma_algo_metrics = linear_metrics(
        output_shape=(num_r, num_r, num_heads_tri * c_hidden_tri_att),
        input_shape=zeta_shape,
        count_input_memory=False,
    )
    algo_metrics_list += [gamma_algo_metrics]

    # (4b)
    gate_shape, gate_algo_metrics = sigmoid_metrics(gamma_shape)
    algo_metrics_list += [gate_algo_metrics]

    # (5a)
    score_shape, score_metrics = create_row_or_column_attention_scores(
        q_shape,
        k_shape,
        q_attending_axis=1,
        q_embedding_axis=-1,
        num_heads=num_heads_tri,
        b_shape=pair_bias_shape,
        is_row=True,
    )
    algo_metrics_list += [score_metrics]

    # (5b)
    attention_weight_shape, soft_max_algo_metrics = soft_max_metrics(
        score_shape,
        softmax_dim=2,
        count_input_memory=False,
    )
    algo_metrics_list += [soft_max_algo_metrics]

    # (6a)
    context_vec_shape, context_vector_metrics = create_context_vector_metrics(v_shape, attention_weight_shape)
    algo_metrics_list += [context_vector_metrics]

    # (6b)
    o_shape, o_algo_metrics = multiply_elem_wise(gate_shape, context_vec_shape)
    algo_metrics_list += [o_algo_metrics]

    # since heads extend the embedding axis, the concat is a no-op
    rho_shape = o_shape

    # (7)
    z_tilde_shape, linear_algo_metrics = linear_metrics(
        output_shape=z_shape,
        input_shape=rho_shape,
        count_input_memory=True,
    )
    algo_metrics_list += [linear_algo_metrics]

    algo_metrics_out = AlgoMetrics(
        function_name="triangle_attention_metrics",
        number_of_adds_fwd=sum([x.number_of_adds_fwd for x in algo_metrics_list]),
        number_of_mults_fwd=sum([x.number_of_mults_fwd for x in algo_metrics_list]),
        number_of_nonlinear_ops=sum([x.number_of_nonlinear_ops for x in algo_metrics_list]),
        memory_footprint_num_els=sum([x.memory_footprint_num_els for x in algo_metrics_list]),
        number_of_params=sum([x.number_of_params for x in algo_metrics_list]),
    )

    return z_tilde_shape, algo_metrics_out
