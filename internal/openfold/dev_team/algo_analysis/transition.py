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


# PB todo
from typing import Tuple

from internal.openfold.dev_team.algo_analysis.algo_metrics import (
    AlgoMetrics,
    layer_norm_metrics,
    linear_metrics,
)


def msa_transition_metrics(in_shape: Tuple[int], n=4):
    # Algorithm 9 in the paper:
    return transition_base(in_shape, n, function_name="msa_transition_metrics")


def pair_transition_metrics(in_shape: Tuple[int], n=4):
    # Algorithm 15 in paper
    return transition_base(in_shape, n, function_name="pair_transition_metrics")


def transition_base(m_shape: Tuple[int], n: int = 4, function_name=None):
    algo_metrics_list = []
    _, layer_norm_count = layer_norm_metrics(m_shape)
    algo_metrics_list += [layer_norm_count]

    a_shape, linear_counts = linear_metrics(output_shape=(m_shape[0], m_shape[1], n * m_shape[2]), input_shape=m_shape)
    algo_metrics_list += [linear_counts]

    # assume relu - no calculations
    m_final_shape, linear2_counts = linear_metrics(output_shape=m_shape, input_shape=a_shape)
    # assume relu - no calculations
    algo_metrics_list += [linear2_counts]
    msa_transition_metrics_mean_metrics_out = AlgoMetrics(
        function_name=function_name if function_name else "transition_base",
        number_of_adds_fwd=sum([x.number_of_adds_fwd for x in algo_metrics_list]),
        number_of_mults_fwd=sum([x.number_of_mults_fwd for x in algo_metrics_list]),
        number_of_nonlinear_ops=sum([x.number_of_nonlinear_ops for x in algo_metrics_list]),
        memory_footprint_num_els=sum([x.memory_footprint_num_els for x in algo_metrics_list]),
        number_of_params=sum([x.number_of_params for x in algo_metrics_list]),
    )
    return m_final_shape, msa_transition_metrics_mean_metrics_out
