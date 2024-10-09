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


from copy import deepcopy
from typing import Tuple

from internal.openfold.dev_team.algo_analysis.algo_metrics import (
    AlgoMetrics,
    linear_no_bias_metrics,
    sum_algo_metrics_list,
)
from internal.openfold.dev_team.algo_analysis.tensor_shape_helpers import (
    num_elements,
    num_elements_excluding,
)


def create_mha_qkv_metrics(input_shape: Tuple, c_hidden_att=32, num_heads=8, input_embedding_axis=-1):
    assert input_embedding_axis == -1

    q_shape = (input_shape[0], input_shape[1], c_hidden_att * num_heads)
    k_shape = q_shape
    v_shape = q_shape

    metrics_for_each_linear_operation = []
    for i, output_shape in enumerate([q_shape, k_shape, v_shape]):
        x_output_shape, x_metrics = linear_no_bias_metrics(
            input_shape=input_shape,
            output_shape=output_shape,
            count_input_memory=(i == 0),
        )
        assert output_shape == x_output_shape
        metrics_for_each_linear_operation
        metrics_for_each_linear_operation += [x_metrics]

    mha_qkv_metrics = sum_algo_metrics_list(
        function_name="create_mha_qkv_metrics",
        algo_metrics_list=metrics_for_each_linear_operation,
    )

    return (q_shape, k_shape, v_shape), mha_qkv_metrics


def create_row_or_column_attention_scores(
    q_shape: Tuple,
    k_shape: Tuple,
    q_attending_axis: int = 1,
    q_embedding_axis: int = -1,
    num_heads: int = 4,
    b_shape: Tuple = None,
    is_row: bool = True,
):
    """
    Following the openfold code, assume the embedding dimension is exended
    with the number of heads.

    for row-attention-with-pair-bias
        q_shape = (num_sequences, num_residues, c * num_heads)

    for column-attention
        q_shape = (num_sequences, num_residues, c * num_heads)

    for triangle-attention
        q_shape = (num_residues, num_residues, c * num_heads)

    """

    # sanity checks
    assert q_shape == k_shape
    if is_row:
        assert b_shape[0] == q_shape[q_attending_axis]  # the bias if for pairs on the attending axis
        assert b_shape[1] == q_shape[q_attending_axis]

    c = int(q_shape[q_embedding_axis] / num_heads + 10**-6)  # embedding dimension

    number_of_hidden_vector_in_q = num_elements_excluding(q_shape, excluded_axes=[q_embedding_axis])
    num_els_of_output = number_of_hidden_vector_in_q * num_heads
    if is_row:
        num_els_of_output *= q_shape[1]
    else:
        num_els_of_output *= q_shape[0]

    # Compute metrics resulting from computing 1 element of output,
    # which pertains to 1 head.  There is a non-linear operation for the sqrt.
    #
    num_mults_fwd_for_one_el_of_output = c + 1
    num_of_adds_fwd_for_one_el_of_output = (c - 1) + int(is_row)
    num_nonlinear_ops_for_one_el_of_output = 1

    # The memory footprint is the count of numbers in the q and k tensors
    #   don't neet to store the b's to compute deriv's
    memory_footprint_num_els = num_elements(q_shape) + num_elements(k_shape)

    score_metrics = AlgoMetrics(
        function_name="create_row_or_column_attention_scores",
        number_of_mults_fwd=num_els_of_output * num_mults_fwd_for_one_el_of_output,
        number_of_adds_fwd=num_els_of_output * num_of_adds_fwd_for_one_el_of_output,
        number_of_nonlinear_ops=num_els_of_output * num_nonlinear_ops_for_one_el_of_output,
        number_of_params=0,
        memory_footprint_num_els=memory_footprint_num_els,
    )
    # determine output shape
    score_shape_list = [q_shape[q_attending_axis], q_shape[q_attending_axis]]
    if is_row:
        score_shape = tuple([q_shape[0]] + score_shape_list + [num_heads])
    else:
        score_shape = tuple(score_shape_list + [q_shape[1]] + [num_heads])

    return score_shape, score_metrics


def create_column_global_attention_scores(
    mean_q_shape: Tuple,
    k_shape: Tuple,
    mean_q_embedding_axis: int = -1,
    k_embedding_axis: int = -1,
):
    """
    Following the openfold code, assume the embedding dimension is exended
    with the number of heads.


        mean_q_shape = (num_residues, c_hidden_msa_att * num_heads)

        k_shape = (num_sequences, num_residues, c_hidden_msa_att)

    """

    c = k_shape[k_embedding_axis]
    num_heads = int(mean_q_shape[mean_q_embedding_axis] / c + 10**-6)

    score_shape = (k_shape[0], k_shape[1], num_heads)
    num_els_of_output = num_elements(score_shape)

    score_metrics = AlgoMetrics(
        function_name="create_column_global_attention_scores",
        number_of_mults_fwd=num_els_of_output * (c + 1),
        number_of_adds_fwd=num_els_of_output * (c - 1),
        number_of_nonlinear_ops=1,
        number_of_params=0,
        memory_footprint_num_els=int(num_elements(mean_q_shape) + num_elements(k_shape)),
    )

    return score_shape, score_metrics


def create_context_vector_metrics(
    v_shape: Tuple,
    attention_weight_shape: Tuple,
    v_attending_dim: int = 1,
):
    """
    For row-attention-with-pair-bias or column-attention

        v_shape = (N_s, N_r, c * N_h)

    For triangle-attention
        v_shape = (N_r, N_r, c_tri_att * N_H_tri)

    """

    num_mults_one_output_component = v_shape[v_attending_dim]
    num_adds_one_output_component = v_shape[v_attending_dim] - 1
    context_vector_shape = deepcopy(v_shape)

    context_vector_metrics = AlgoMetrics(
        function_name="create_context_vector_metrics",
        number_of_adds_fwd=num_elements(context_vector_shape) * num_mults_one_output_component,
        number_of_mults_fwd=num_elements(context_vector_shape) * num_adds_one_output_component,
        number_of_nonlinear_ops=0,
        number_of_params=0,
        memory_footprint_num_els=num_elements(v_shape) + num_elements(attention_weight_shape),
    )

    return context_vector_shape, context_vector_metrics


def create_context_vector_metrics_column_global(
    v_shape: Tuple,
    attention_weight_shape: Tuple,
    v_embedding_axis: int = -1,
    attention_head_axis: int = -1,
):
    """
    For column global

        v_shape = (N_s, N_r, c)
        a_shapte = (N_s, N_r, N_h)

    """
    context_vector_shape = (v_shape[1], v_shape[v_embedding_axis] * attention_weight_shape[attention_head_axis])

    num_mults_one_output_component = v_shape[0]
    num_adds_one_output_component = v_shape[0] - 1

    context_vector_metrics = AlgoMetrics(
        function_name="create_context_vector_metrics",
        number_of_mults_fwd=num_elements(context_vector_shape) * num_mults_one_output_component,
        number_of_adds_fwd=num_elements(context_vector_shape) * num_adds_one_output_component,
        number_of_nonlinear_ops=0,
        number_of_params=0,
        memory_footprint_num_els=num_elements(v_shape) + num_elements(attention_weight_shape),
    )
    return context_vector_shape, context_vector_metrics
