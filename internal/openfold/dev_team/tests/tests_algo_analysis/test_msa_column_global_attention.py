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
)
from internal.openfold.dev_team.algo_analysis.row_and_column_attention_metrics_helpers import (  # noqa
    create_context_vector_metrics_column_global,
    create_column_global_attention_scores,
)


def test_020():
    """Algo 19, step 2, q component"""

    # ExtraMsaStack parameter set
    num_sequences = 1024  # model.max_extra_msa
    num_residues = 256  # model.train_sequence_crop_size

    c_e = 64  # model.extra_msa_stack_config.c_e
    c_hidden_msa_att = 8  # model.extra_msa_stack_config.c_hidden_msa_att
    num_heads_msa = 8  # model.extra_msa_stack_config.num_heads_msa

    # Second step in msa-column-global-attention is
    mu_shape = (num_sequences, num_residues, c_e)
    q_shape, q_algo_metrics = linear_no_bias_metrics(
        output_shape=(mu_shape[0], mu_shape[1], c_hidden_msa_att * num_heads_msa),
        input_shape=mu_shape,
    )

    gsheet_number_of_mults_fwd = num_sequences * num_residues * num_heads_msa * c_hidden_msa_att * c_e
    gsheet_number_of_adds_fwd = num_sequences * num_residues * num_heads_msa * c_hidden_msa_att * (c_e - 1)
    gsheet_number_of_params = num_heads_msa * c_hidden_msa_att * c_e
    assert (mu_shape[0], mu_shape[1], c_hidden_msa_att * num_heads_msa) == q_shape
    assert gsheet_number_of_params == q_algo_metrics.number_of_params
    assert gsheet_number_of_mults_fwd == q_algo_metrics.number_of_mults_fwd
    assert gsheet_number_of_adds_fwd == q_algo_metrics.number_of_adds_fwd


def test_021():
    """Algo 19, step 2, v component"""

    # ExtraMsaStack parameter set
    num_sequences = 1024  # model.max_extra_msa
    num_residues = 256  # model.train_sequence_crop_size

    c_e = 64  # model.extra_msa_stack_config.c_e
    c_hidden_msa_att = 8  # model.extra_msa_stack_config.c_hidden_msa_att

    # Second step in msa-column-global-attention is
    mu_shape = (num_sequences, num_residues, c_e)
    v_shape, v_algo_metrics = linear_no_bias_metrics(
        output_shape=(mu_shape[0], mu_shape[1], c_hidden_msa_att), input_shape=mu_shape, count_input_memory=False
    )

    gsheet_number_of_mults_fwd = num_sequences * num_residues * c_hidden_msa_att * c_e
    gsheet_number_of_adds_fwd = num_sequences * num_residues * c_hidden_msa_att * (c_e - 1)
    gsheet_number_of_params = c_hidden_msa_att * c_e
    assert (mu_shape[0], mu_shape[1], c_hidden_msa_att) == v_shape
    assert gsheet_number_of_params == v_algo_metrics.number_of_params
    assert gsheet_number_of_mults_fwd == v_algo_metrics.number_of_mults_fwd
    assert gsheet_number_of_adds_fwd == v_algo_metrics.number_of_adds_fwd


def test_051():
    """Algo 19, step 5, compute score"""

    # ExtraMsaStack parameter set
    num_sequences = 1024  # model.max_extra_msa
    num_residues = 256  # model.train_sequence_crop_size

    c_hidden_msa_att = 8  # model.extra_msa_stack_config.c_hidden_msa_att
    num_heads_msa = 8  # model.extra_msa_stack_config.num_heads_msa

    mean_q_shape = (num_residues, c_hidden_msa_att * num_heads_msa)
    k_shape = (num_sequences, num_residues, c_hidden_msa_att)

    score_shape, score_metrics = create_column_global_attention_scores(
        mean_q_shape,
        k_shape,
    )

    gsheet_number_of_mults_fwd = num_residues * num_sequences * num_heads_msa * (c_hidden_msa_att + 1)
    gsheet_number_of_adds_fwd = num_residues * num_sequences * num_heads_msa * (c_hidden_msa_att - 1)
    gsheet_number_of_params = 0
    assert (num_sequences, num_residues, num_heads_msa) == score_shape
    assert gsheet_number_of_params == score_metrics.number_of_params
    assert gsheet_number_of_mults_fwd == score_metrics.number_of_mults_fwd
    assert gsheet_number_of_adds_fwd == score_metrics.number_of_adds_fwd


def test_060_context_vector():
    """Algo 19, step 6, compute context vectors"""

    # ExtraMsaStack parameter set
    num_sequences = 1024  # model.max_extra_msa
    num_residues = 256  # model.train_sequence_crop_size

    c_hidden_msa_att = 8  # model.extra_msa_stack_config.c_hidden_msa_att
    num_heads_msa = 8  # model.extra_msa_stack_config.num_heads_msa

    v_shape = (num_sequences, num_residues, c_hidden_msa_att)
    a_shape = (num_sequences, num_residues, num_heads_msa)

    context_vec_shape, context_vector_metrics = create_context_vector_metrics_column_global(v_shape, a_shape)

    gsheet_number_of_mults_fwd = num_residues * num_heads_msa * c_hidden_msa_att * num_sequences
    gsheet_number_of_adds_fwd = num_residues * num_heads_msa * c_hidden_msa_att * (num_sequences - 1)
    gsheet_number_of_params = 0
    assert (num_residues, num_heads_msa * c_hidden_msa_att) == context_vec_shape
    assert gsheet_number_of_params == context_vector_metrics.number_of_params
    assert gsheet_number_of_mults_fwd == context_vector_metrics.number_of_mults_fwd
    assert gsheet_number_of_adds_fwd == context_vector_metrics.number_of_adds_fwd
