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

from internal.openfold.dev_team.algo_analysis.algo_metrics import sum_algo_metrics_list
from internal.openfold.dev_team.algo_analysis.msa_column_global_attention_metrics import (
    msa_column_global_attention_metrics,
)
from internal.openfold.dev_team.algo_analysis.msa_row_attention_with_pair_bias_metrics import (
    msa_row_attention_with_pair_bias_metrics,
)
from internal.openfold.dev_team.algo_analysis.outer_product_mean_metrics import outer_product_mean_metrics
from internal.openfold.dev_team.algo_analysis.transition import (
    msa_transition_metrics,
    pair_transition_metrics,
)
from internal.openfold.dev_team.algo_analysis.triangle_attention_metrics import (
    triangle_attention_metrics,
)
from internal.openfold.dev_team.algo_analysis.triangle_metrics import triangle_multplication


def extra_msa_stack_metrics(
    e_shape: Tuple,
    z_shape: Tuple,
    c_hidden_msa_att: int = 8,
    c_hidden_tri_att: int = 32,
    c_hidden_tri_mul: int = 128,
    c_hidden_opm: int = 32,
    num_heads_msa: int = 8,
    num_extra_msa_blocks: int = 4,
    num_heads_tri: int = 4,
    transition_n: int = 4,
):
    """
    Algorithm 18 in the supplementary material of the alphafold 2 paper.
    """

    e_shape[-1]
    c_z = z_shape[-1]
    algo_metrics_list_over_blocks = []

    for i_block in range(num_extra_msa_blocks):
        algo_metrics_list_this_block = []

        # #######################################
        # msa block
        # #######################################
        msa_row_shape, msa_row_algo_metrics = msa_row_attention_with_pair_bias_metrics(  # noqa
            e_shape,
            z_shape,
            c_hidden_msa_att=c_hidden_msa_att,
            num_heads_msa=num_heads_msa,
        )
        algo_metrics_list_this_block += [msa_row_algo_metrics]

        msa_column_global_shape, msa_column_global_attention_metrics_out = msa_column_global_attention_metrics(
            m_shape=msa_row_shape,
            c=c_hidden_msa_att,
            num_heads_msa=num_heads_msa,
        )
        algo_metrics_list_this_block += [msa_column_global_attention_metrics_out]

        msa_transition_shape, msa_transition_metrics_out = msa_transition_metrics(
            in_shape=msa_column_global_shape,
            n=transition_n,
        )
        algo_metrics_list_this_block += [msa_transition_metrics_out]

        # #######################################
        # communication
        # #######################################
        z_outer_product_mean_shape, outer_product_mean_metrics_out = outer_product_mean_metrics(
            m_shape=msa_transition_shape,
            c_hidden_opm=c_hidden_opm,
            c_z=c_z,
        )
        algo_metrics_list_this_block += [outer_product_mean_metrics_out]

        # #########################################
        # pair stack
        # ########################################
        z_tri_mult_og_shape, tri_mult_og_metrics_out = triangle_multplication(
            z_shape=z_outer_product_mean_shape,
            c_hidden_tri_mul=c_hidden_tri_mul,
        )
        algo_metrics_list_this_block += [tri_mult_og_metrics_out]

        z_tri_mult_ic_shape, tri_mult_ic_metrics_out = triangle_multplication(
            z_shape=z_tri_mult_og_shape,
            c_hidden_tri_mul=c_hidden_tri_mul,
        )
        algo_metrics_list_this_block += [tri_mult_ic_metrics_out]

        tri_att_starting_out, tri_att_starting_metrics = triangle_attention_metrics(  # noqa
            z_shape=z_tri_mult_ic_shape,
            c_hidden_tri_att=c_hidden_tri_att,
            num_heads_tri=num_heads_tri,
        )
        algo_metrics_list_this_block += [tri_att_starting_metrics]

        tri_att_ending_out, tri_att_ending_metrics = triangle_attention_metrics(  # noqa
            tri_att_starting_out,
            c_hidden_tri_att=c_hidden_tri_att,
            num_heads_tri=num_heads_tri,
        )
        algo_metrics_list_this_block += [tri_att_ending_metrics]

        pair_transition_shape, pair_transition_metrics_out = pair_transition_metrics(
            in_shape=tri_att_ending_out, n=transition_n
        )
        algo_metrics_list_this_block += [pair_transition_metrics_out]

        # ##############################################
        # summary
        # ##############################################
        z_shape = pair_transition_shape
        extra_msa_stack_metrics_this_block = sum_algo_metrics_list(
            function_name=f"extra_msa_block_{i_block}", algo_metrics_list=algo_metrics_list_this_block
        )

        algo_metrics_list_over_blocks += [extra_msa_stack_metrics_this_block]

    extra_msa_stack_metrics = sum_algo_metrics_list(
        function_name="extra_msa_stack", algo_metrics_list=algo_metrics_list_over_blocks
    )
    return z_shape, extra_msa_stack_metrics
