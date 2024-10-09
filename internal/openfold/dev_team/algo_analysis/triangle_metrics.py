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

from internal.openfold.dev_team.algo_analysis.algo_metrics import (
    AlgoMetrics,
    layer_norm_metrics,
    linear_metrics,
    multiply_elem_wise,
    sigmoid_metrics,
    sum_algo_metrics_list,
)
from internal.openfold.dev_team.algo_analysis.tensor_shape_helpers import num_elements


def triangle_multplication(z_shape: Tuple[int], c_hidden_tri_mul: int = 128):
    """
    Algorithms 11 and 12 in the supplementary material of the alphafold 2 paper.
    z_shape is (N_r, N_r, c_z)
    """

    algo_metrics_list = []

    # (1) LayerNorm
    zeta_shape, layer_norm_count = layer_norm_metrics(z_shape)
    algo_metrics_list += [layer_norm_count]

    # (2)  computing a and b
    zeta_1_shape, first_linear_count = linear_metrics(  # noqa
        output_shape=(z_shape[0], z_shape[1], 2 * c_hidden_tri_mul),
        input_shape=zeta_shape,
    )
    algo_metrics_list += [first_linear_count]

    zeta_2_shape, second_linear_count = linear_metrics(  # noqa
        output_shape=(z_shape[0], z_shape[1], 2 * c_hidden_tri_mul), input_shape=zeta_shape, count_input_memory=False
    )
    algo_metrics_list += [second_linear_count]

    gate_1_shape, sigmoid_counts = sigmoid_metrics(zeta_1_shape)
    algo_metrics_list += [sigmoid_counts]

    a_b_shape, elem_wise_counts = multiply_elem_wise(gate_1_shape, zeta_2_shape)
    algo_metrics_list += [elem_wise_counts]
    a_shape = (a_b_shape[0], a_b_shape[1], c_hidden_tri_mul)

    # (3) gate
    psi_shape, psi_linear_count = linear_metrics(  # noqa
        output_shape=zeta_shape,
        input_shape=zeta_shape,
        count_input_memory=False,
    )
    algo_metrics_list += [psi_linear_count]

    gate_shape, psi_sigmoid_counts = sigmoid_metrics(psi_shape)
    algo_metrics_list += [psi_sigmoid_counts]

    # (4) kappa = /sum_k a_ki circle-dot b_kj
    kappa_shape = psi_shape
    kappa_counts = AlgoMetrics(
        function_name="sum_of_elementwise_mult",
        number_of_adds_fwd=num_elements(a_shape) * (a_shape[0] - 1),
        number_of_mults_fwd=num_elements(a_shape) * a_shape[0],
        number_of_nonlinear_ops=0,
        memory_footprint_num_els=int(num_elements(a_shape) + num_elements(a_shape)),
        number_of_params=0,
    )

    algo_metrics_list += [kappa_counts]
    lambda_shape, layer_norm_count = layer_norm_metrics(kappa_shape)
    algo_metrics_list += [layer_norm_count]

    pi_shape, third_linear_count = linear_metrics(  # noqa
        output_shape=z_shape,
        input_shape=lambda_shape,
    )
    algo_metrics_list += [third_linear_count]

    final_shape, final_counts = multiply_elem_wise(gate_shape, pi_shape)
    algo_metrics_list += [final_counts]

    triangle_multiplication_out = sum_algo_metrics_list(
        function_name="triangle_metrics",
        algo_metrics_list=algo_metrics_list,
    )

    return final_shape, triangle_multiplication_out
