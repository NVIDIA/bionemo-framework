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


# BR Todo
#
from typing import Tuple

from internal.openfold.dev_team.algo_analysis.algo_metrics import (  # noqa
    AlgoMetrics,
    layer_norm_metrics,
    linear_metrics,
    linear_no_bias_metrics,
    multiply_elem_wise,
    sigmoid_metrics,
    soft_max_metrics,
    sum_algo_metrics_list,
)
from internal.openfold.dev_team.algo_analysis.row_and_column_attention_metrics_helpers import (
    create_context_vector_metrics,
    create_mha_qkv_metrics,
    create_row_or_column_attention_scores,
)


def msa_row_attention_with_pair_bias_metrics(
    m_shape: Tuple,
    z_shape: Tuple,
    c_hidden_msa_att: int = 32,
    num_heads_msa: int = 8,
) -> Tuple[Tuple, AlgoMetrics]:
    """
    Algorithm 7 in the supplementary material of the alphafold 2 paper.
    """
    num_s = m_shape[0]
    num_r = m_shape[1]

    algo_metrics_list = []

    # input projection
    # (1)
    mu_shape, mu_algo_metrics = layer_norm_metrics(m_shape)
    algo_metrics_list += [mu_algo_metrics]

    # (2)
    (q_shape, k_shape, v_shape), mha_qkv_metrics = create_mha_qkv_metrics(
        input_shape=mu_shape, c_hidden_att=c_hidden_msa_att, num_heads=num_heads_msa
    )
    algo_metrics_list += [mha_qkv_metrics]

    # (3a)
    zeta_shape, zeta_algo_metrics = layer_norm_metrics(z_shape)
    algo_metrics_list += [zeta_algo_metrics]

    # (3b)
    pair_bias_shape, pair_bias_metrics = linear_no_bias_metrics(
        output_shape=(zeta_shape[0], zeta_shape[1], num_heads_msa, 1),
        input_shape=zeta_shape,
    )
    algo_metrics_list += [pair_bias_metrics]

    # (4a)
    nu_shape, nu_algo_metrics = linear_metrics(
        output_shape=(num_s, num_r, c_hidden_msa_att * num_heads_msa),
        input_shape=mu_shape,
    )
    algo_metrics_list += [nu_algo_metrics]

    # (4b)
    gate_shape, gate_algo_metrics = sigmoid_metrics(nu_shape)
    algo_metrics_list += [gate_algo_metrics]

    # (5a)
    score_shape, score_metrics = create_row_or_column_attention_scores(
        q_shape,
        k_shape,
        q_attending_axis=1,
        q_embedding_axis=-1,
        num_heads=num_heads_msa,
        b_shape=pair_bias_shape,
        is_row=True,
    )
    algo_metrics_list += [score_metrics]

    # (5b)
    attention_weight_shape, soft_max_algo_metrics = soft_max_metrics(score_shape)
    algo_metrics_list += [soft_max_algo_metrics]

    # (6a)
    context_vec_shape, context_vector_metrics = create_context_vector_metrics(v_shape, attention_weight_shape)
    algo_metrics_list += [context_vector_metrics]

    # BR TODO
    # (6b)
    o_shape, o_algo_metrics = multiply_elem_wise(gate_shape, context_vec_shape)
    algo_metrics_list += [o_algo_metrics]

    # (7)
    m_tilde_shape, linear_algo_metrics = linear_metrics(
        output_shape=m_shape,
        input_shape=o_shape,
    )
    algo_metrics_list += [linear_algo_metrics]

    algo_metrics_out = sum_algo_metrics_list(
        function_name="msa_row_attention_with_pair_bias_metrics",
        algo_metrics_list=algo_metrics_list,
    )
    return m_tilde_shape, algo_metrics_out
