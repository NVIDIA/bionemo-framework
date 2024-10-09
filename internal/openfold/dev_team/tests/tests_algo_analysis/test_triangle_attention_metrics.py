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


from internal.openfold.dev_team.algo_analysis.algo_metrics import (  # noqa
    layer_norm_metrics,
    linear_metrics,
    linear_no_bias_metrics,
    multiply_elem_wise,
    sigmoid_metrics,
    soft_max_metrics,
    num_elements,
)
from internal.openfold.dev_team.algo_analysis.row_and_column_attention_metrics_helpers import (  # noqa
    create_context_vector_metrics,
    create_mha_qkv_metrics,
    create_row_or_column_attention_scores,
)


from internal.openfold.dev_team.algo_analysis.triangle_attention_metrics import (
    triangle_attention_metrics,
)


def test_010_layer_norm_metrics_for_triangle_attention():
    # ############################
    # EvoformerStack parameter set
    # ############################
    num_residues = 256
    c_z = 128  # pair_embedding_dim

    z_shape = (num_residues, num_residues, c_z)

    # (1) first step is layer norm
    zeta_shape, zeta_algo_metrics = layer_norm_metrics(z_shape)

    mults_fwd_in_gsheet = 25_427_968
    params_in_gsheet = 2 * c_z
    assert zeta_shape == z_shape
    assert zeta_algo_metrics.number_of_mults_fwd == mults_fwd_in_gsheet
    assert zeta_algo_metrics.number_of_params == params_in_gsheet


def test_020_create_mha_qkv_metrics_for_triangle_attention():
    # ############################
    # EvoformerStack parameter set
    # ############################
    num_residues = 256
    num_heads_tri = 4
    c_hidden_tri_att = 32
    c_z = 128  # pair_embedding_dim

    z_shape = (num_residues, num_residues, c_z)

    # (1) first step is layer norm
    (q_shape, k_shape, v_shape), qkv_metrics = create_mha_qkv_metrics(
        z_shape,
        c_hidden_att=c_hidden_tri_att,
        num_heads=num_heads_tri,
    )

    mults_fwd_in_gsheet = 3_221_225_472
    adds_fwd_in_gsheet = 3_196_059_648
    params_in_gsheet = 49_152
    memory_footprint_in_bytes_in_gsheet = 4 * num_residues * num_residues * c_z
    assert q_shape == (num_residues, num_residues, num_heads_tri * c_hidden_tri_att)
    assert qkv_metrics.number_of_mults_fwd == mults_fwd_in_gsheet
    assert qkv_metrics.number_of_adds_fwd == adds_fwd_in_gsheet
    assert qkv_metrics.number_of_params == params_in_gsheet
    assert qkv_metrics.memory_footprint_in_bytes == memory_footprint_in_bytes_in_gsheet


def test_030_linear_no_bias_metrics_for_triangle_attention():
    # ############################
    # EvoformerStack parameter set
    # ############################
    num_residues = 256
    num_heads_tri = 4
    c_z = 128  # pair_embedding_dim

    zeta_shape = (num_residues, num_residues, c_z)

    # (1) first step is layer norm
    pair_bias_shape, pair_bias_metrics = linear_no_bias_metrics(
        output_shape=(zeta_shape[0], zeta_shape[1], num_heads_tri),
        input_shape=zeta_shape,
    )

    mults_fwd_in_gsheet = 33_554_432
    assert pair_bias_shape == (num_residues, num_residues, num_heads_tri)
    assert pair_bias_metrics.number_of_mults_fwd == mults_fwd_in_gsheet


def test_050_create_attention_scores():
    # ############################
    # EvoformerStack parameter set
    # ############################
    num_residues = 256
    num_heads_tri = 4
    c_hidden_tri_att = 32

    q_shape = (num_residues, num_residues, c_hidden_tri_att * num_heads_tri)

    # (1) first step is layer norm
    score_shape, score_metrics = create_row_or_column_attention_scores(
        q_shape=q_shape,
        k_shape=q_shape,
        q_attending_axis=1,
        q_embedding_axis=-1,
        num_heads=num_heads_tri,
        b_shape=(num_residues, num_residues, num_heads_tri),
        is_row=True,
    )

    simple_number_of_mults_fwd = num_residues * num_residues * num_residues * num_heads_tri * (c_hidden_tri_att + 1)
    simple_number_of_adds_fwd = num_residues * num_residues * num_residues * num_heads_tri * (c_hidden_tri_att)
    gsheet_number_of_mults_fwd = 2_281_701_376
    assert score_shape == (num_residues, num_residues, num_residues, num_heads_tri)
    assert abs(gsheet_number_of_mults_fwd - simple_number_of_mults_fwd) / simple_number_of_mults_fwd < 0.10
    assert simple_number_of_mults_fwd == score_metrics.number_of_mults_fwd
    assert simple_number_of_adds_fwd == score_metrics.number_of_adds_fwd


def test_055_create_attention_weights():
    # ############################
    # EvoformerStack parameter set
    # ############################
    num_residues = 256
    num_heads_tri = 4
    score_shape = (num_residues, num_residues, num_residues, num_heads_tri)
    attention_weight_shape, soft_max_algo_metrics = soft_max_metrics(
        score_shape, softmax_dim=2, count_input_memory=False
    )

    gsheet_number_of_parmas = 0
    gsheet_number_of_adds = num_heads_tri * num_residues * num_residues * (num_residues - 1)
    # gsheet_number_of_mults = 3 * num_heads_tri * num_residues * num_residues * num_residues

    simple_number_of_mults = num_heads_tri * num_residues * num_residues * num_residues
    simple_number_of_nonlinear_ops = num_heads_tri * num_residues * num_residues * num_residues
    simple_memory_footprint_num_els = 0

    assert score_shape == attention_weight_shape
    assert gsheet_number_of_parmas == soft_max_algo_metrics.number_of_params
    assert gsheet_number_of_adds == soft_max_algo_metrics.number_of_adds_fwd
    assert simple_number_of_mults == soft_max_algo_metrics.number_of_mults_fwd
    assert simple_number_of_nonlinear_ops == soft_max_algo_metrics.number_of_nonlinear_ops
    assert simple_memory_footprint_num_els == soft_max_algo_metrics.memory_footprint_num_els


def test_060_create_context_vectors():
    # ############################
    # EvoformerStack parameter set
    # ############################
    num_residues = 256
    num_heads_tri = 4
    c_hidden_tri_att = 32

    v_shape = (num_residues, num_residues, num_heads_tri, c_hidden_tri_att)
    attention_weight_shape = (num_residues, num_residues, num_residues, num_heads_tri)

    # (1) first step is layer norm
    context_vec_shape, context_vector_metrics = create_context_vector_metrics(v_shape, attention_weight_shape)

    mults_fwd_in_gsheet = 2_147_483_648
    adds_fwd_in_gsheet = 2_139_095_040
    assert context_vec_shape == v_shape

    assert abs(context_vector_metrics.number_of_mults_fwd - mults_fwd_in_gsheet) / mults_fwd_in_gsheet < 0.10
    assert abs(context_vector_metrics.number_of_adds_fwd - adds_fwd_in_gsheet) / adds_fwd_in_gsheet < 0.10


def test_070_linear():
    # ############################
    # EvoformerStack parameter set
    # ############################
    num_residues = 256
    num_heads_tri = 4
    c_hidden_tri_att = 32
    c_z = 128

    z_shape = (num_residues, num_residues, c_z)
    o_shape = (num_residues, num_residues, num_heads_tri, c_hidden_tri_att)

    # (1) first step is layer norm
    z_tilde_shape, linear_algo_metrics = linear_metrics(
        output_shape=z_shape,
        input_shape=(o_shape[0], o_shape[1], num_heads_tri * c_hidden_tri_att),
    )

    mults_fwd_simple = num_residues * num_residues * c_z * num_heads_tri * c_hidden_tri_att

    mults_fwd_in_gsheet = 1_073_741_824
    adds_fwd_in_gsheet = 1_073_741_824
    assert z_shape == z_tilde_shape

    assert abs(linear_algo_metrics.number_of_mults_fwd - mults_fwd_simple) / mults_fwd_simple < 0.10
    assert abs(linear_algo_metrics.number_of_mults_fwd - mults_fwd_in_gsheet) / mults_fwd_in_gsheet < 0.10
    assert abs(linear_algo_metrics.number_of_adds_fwd - adds_fwd_in_gsheet) / adds_fwd_in_gsheet < 0.10


def test_100_triangle_attention_metrics_evoformer():
    # ############################
    # EvoformerStack parameter set
    # ############################
    num_residues = 256

    c_z = 128  # pair_embedding_dim
    c_hidden_tri_att = 32
    num_heads_tri = 4

    z_shape = (num_residues, num_residues, c_z)

    tri_att_starting_shape, tri_att_starting_metrics = triangle_attention_metrics(  # noqa
        z_shape,
        c_hidden_tri_att=c_hidden_tri_att,
        num_heads_tri=num_heads_tri,
    )

    params_in_gsheet = 82_944
    mults_fwd_in_gsheet = 10_074_980_352  # 10 billion
    adds_fwd_in_gsheet = 9_763_815_424
    memory_footprint_in_bytes_in_gsheet = 536_870_912

    assert z_shape == tri_att_starting_shape
    assert params_in_gsheet == tri_att_starting_metrics.number_of_params
    assert memory_footprint_in_bytes_in_gsheet == tri_att_starting_metrics.memory_footprint_in_bytes
    assert abs(tri_att_starting_metrics.number_of_mults_fwd - mults_fwd_in_gsheet) / mults_fwd_in_gsheet < 0.03
    assert abs(tri_att_starting_metrics.number_of_adds_fwd - adds_fwd_in_gsheet) / adds_fwd_in_gsheet < 0.02
