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
    sum_algo_metrics_list,
)
from internal.openfold.dev_team.algo_analysis.row_and_column_attention_metrics_helpers import (
    create_column_global_attention_scores,
    create_context_vector_metrics_column_global,
)
from internal.openfold.dev_team.algo_analysis.tensor_shape_helpers import num_elements, num_elements_excluding


def msa_column_global_attention_metrics(
    m_shape: Tuple[int],
    c: int = 8,
    num_heads_msa: int = 8,
):
    """
    Algorithm 19 in the supplementary material of the alphafold 2 paper.
    m_shape is N_s, N_r, c
    """

    algo_metrics_list = []

    # (1)
    mu_shape, mu_algo_metrics = layer_norm_metrics(m_shape)
    algo_metrics_list += [mu_algo_metrics]

    # (2)a
    q_shape, q_algo_metrics = linear_no_bias_metrics(
        output_shape=(mu_shape[0], mu_shape[1], c * num_heads_msa),
        input_shape=mu_shape,
    )
    algo_metrics_list += [q_algo_metrics]

    # 2(b)
    k_shape, k_algo_metrics = linear_no_bias_metrics(
        output_shape=(mu_shape[0], mu_shape[1], c), input_shape=mu_shape, count_input_memory=False
    )
    algo_metrics_list += [k_algo_metrics]

    v_shape, v_algo_metrics = linear_no_bias_metrics(
        output_shape=(mu_shape[0], mu_shape[1], c), input_shape=mu_shape, count_input_memory=False
    )
    algo_metrics_list += [v_algo_metrics]

    # 3 average q over sequences
    q_h_shape = (q_shape[1], q_shape[2])
    q_h_metric = AlgoMetrics(
        function_name="mean_q_metric",
        number_of_adds_fwd=num_elements_excluding(q_shape, [0]) * (q_shape[0] - 1),
        number_of_mults_fwd=num_elements(q_h_shape),
        number_of_params=0,
        memory_footprint_num_els=num_elements(q_shape),
    )
    algo_metrics_list += [q_h_metric]

    # 4(a)
    nu_shape, nu_algo_metrics = linear_metrics(
        output_shape=q_shape,
        input_shape=mu_shape,
        count_input_memory=False,
    )
    algo_metrics_list += [nu_algo_metrics]

    assert nu_shape == q_shape

    # 4(b)
    g_shape, sigmoid_algo_metrics = sigmoid_metrics(nu_shape)
    algo_metrics_list += [sigmoid_algo_metrics]
    assert g_shape == nu_shape

    # BR TODO, context vector shaps shouldn't match  gate shape

    # 5(a) compute scores
    #
    # score_shape = (q_shape[0], q_shape[1], q_shape[2])
    # score_metric_list = AlgoMetrics(
    #     function_name="smaller_score_array",
    #     number_of_adds_fwd=num_elements_excluding(q_shape, [2]) * (q_shape[2] - 1),
    #     number_of_mults_fwd=num_elements_excluding(q_shape, [2]) * (q_shape[2] + 1),
    #     number_of_params=0,
    #     memory_footprint_num_els=int(num_elements(q_h_shape) + num_elements(k_shape)),
    # )
    score_shape, score_metrics = create_column_global_attention_scores(
        q_h_shape,
        k_shape,
    )
    algo_metrics_list += [score_metrics]

    # 5(b) attention weights
    a_shape, a_metric_list = soft_max_metrics(score_shape, 1)
    algo_metrics_list += [a_metric_list]
    assert a_shape == score_shape

    # (6a)
    context_vec_shape, context_vector_metrics = create_context_vector_metrics_column_global(
        v_shape,
        a_shape,
    )
    algo_metrics_list += [context_vector_metrics]
    context_vec_shape = q_shape

    # (6b)
    o_shape, o_algo_metrics = multiply_elem_wise(g_shape, context_vec_shape)
    algo_metrics_list += [o_algo_metrics]

    # (7)
    m_tilde_shape, linear_algo_metrics = linear_metrics(
        output_shape=m_shape,
        input_shape=(o_shape[0], o_shape[1], o_shape[2]),
    )
    algo_metrics_list += [linear_algo_metrics]

    algo_metrics_out = sum_algo_metrics_list(
        function_name="msa_column_global_attention_metrics",
        algo_metrics_list=algo_metrics_list,
    )
    return m_tilde_shape, algo_metrics_out
