# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# PB todo
from typing import Tuple

from internal.openfold.dev_team.algo_analysis.algo_metrics import (
    AlgoMetrics,
    flatten_outer_product_mean,
    layer_norm_metrics,
    linear_metrics,
)


def outer_product_mean_metrics(
    m_shape: Tuple[int],
    c_hidden_opm: int = 32,
    c_z: int = 128,
):
    # Algorithm 10 in the 2021 alphafold paper supplementary materials
    # m_shape is N_s * N_r * c_m
    # mu_shape, mu_algo_metrics = layer_norm_metrics(m_shape)
    # linear
    # flatten outer product mean
    # linear
    algo_metrics_list = []
    _, layer_norm_count = layer_norm_metrics(m_shape)
    algo_metrics_list += [layer_norm_count]

    a_shape, first_linear_count = linear_metrics(
        output_shape=(m_shape[0], m_shape[1], c_hidden_opm), input_shape=m_shape
    )

    algo_metrics_list += [first_linear_count]
    _, second_linear_count = linear_metrics(
        output_shape=(m_shape[0], m_shape[1], c_hidden_opm), input_shape=m_shape, count_input_memory=False
    )

    algo_metrics_list += [second_linear_count]

    o_shape, outer_product_count = flatten_outer_product_mean(a_shape)
    algo_metrics_list += [outer_product_count]

    z_shape, second_linear_count = linear_metrics(output_shape=(o_shape[0], o_shape[1], c_z), input_shape=o_shape)
    algo_metrics_list += [second_linear_count]

    outer_product_mean_metrics_out = AlgoMetrics(
        function_name="outer_product_mean_metrics",
        number_of_adds_fwd=sum([x.number_of_adds_fwd for x in algo_metrics_list]),
        number_of_mults_fwd=sum([x.number_of_mults_fwd for x in algo_metrics_list]),
        number_of_nonlinear_ops=sum([x.number_of_nonlinear_ops for x in algo_metrics_list]),
        memory_footprint_num_els=sum([x.memory_footprint_num_els for x in algo_metrics_list]),
        number_of_params=sum([x.number_of_params for x in algo_metrics_list]),
    )
    return z_shape, outer_product_mean_metrics_out
