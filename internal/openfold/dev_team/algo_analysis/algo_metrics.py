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
from dataclasses import dataclass
from typing import Tuple

from internal.openfold.dev_team.algo_analysis.tensor_shape_helpers import num_elements, num_elements_excluding  # noqa


@dataclass
class AlgoMetrics:
    function_name: str = None
    number_of_mults_fwd: int = 0
    number_of_adds_fwd: int = 0
    number_of_nonlinear_ops: int = 0
    memory_footprint_num_els: int = 0
    memory_data_type: str = "fp32"
    number_of_params: int = 0
    param_data_type: str = "fp32"
    other: dict = None

    def __repr__(self):
        message = f"""
            function_name: \t\t{self.function_name}
            number_of_mults_fwd:\t{self.number_of_mults_fwd:,}
            number_of_adds_fwd: \t{self.number_of_adds_fwd:,}
            number_of_flops_fwd:\t{self.number_of_flops_fwd:,}
            number_of_nonlinear_ops:\t{self.number_of_nonlinear_ops:,}
            memory_footprint_num_els:\t{self.memory_footprint_num_els:,}
            memory_footprint_in_bytes:\t{self.memory_footprint_in_bytes:,}
            memory_data_type:\t\t{self.memory_data_type}
            param_data_type:\t\t{self.param_data_type}
            number_of_params:\t\t{self.number_of_params}
        """
        return message

    @property
    def number_of_flops_fwd(self):
        return self.number_of_adds_fwd + self.number_of_mults_fwd

    @property
    def memory_footprint_in_bytes(self):
        return self.memory_footprint_num_els * self.bytes_per_data(self.memory_data_type)

    def bytes_per_data(self, data_type: str):
        if data_type == "fp32":
            return 4
        else:
            raise NotImplementedError


def sum_algo_metrics_list(function_name, algo_metrics_list):
    return AlgoMetrics(
        function_name=function_name,
        number_of_adds_fwd=sum([x.number_of_adds_fwd for x in algo_metrics_list]),
        number_of_mults_fwd=sum([x.number_of_mults_fwd for x in algo_metrics_list]),
        number_of_nonlinear_ops=sum([x.number_of_nonlinear_ops for x in algo_metrics_list]),
        number_of_params=sum([x.number_of_params for x in algo_metrics_list]),
        memory_footprint_num_els=sum([x.memory_footprint_num_els for x in algo_metrics_list]),
    )


def layer_norm_metrics(input_shape, agg_dims=(-1,)):
    """Metric for the layer norm. So far, we assume that the aggretation
    dimesion is -1.
    """
    if agg_dims == (-1,):
        num_mults_for_1_input_subtensor = 3 * input_shape[-1] + 4
        num_adds_for_1_input_subtensor = 4 * input_shape[-1]

        k = num_elements_excluding(input_shape, agg_dims)

        layer_norm_algo_metrics = AlgoMetrics(
            function_name="layer_norm_metrics",
            number_of_mults_fwd=(k * num_mults_for_1_input_subtensor),
            number_of_adds_fwd=(k * num_adds_for_1_input_subtensor),
            memory_footprint_num_els=num_elements(input_shape),
            number_of_params=(2 * input_shape[-1]),
        )
    else:
        raise NotImplementedError

    return input_shape, layer_norm_algo_metrics


def linear_metrics(
    output_shape: Tuple[int],
    input_shape: Tuple[int],
    output_hidden_dim: int = -1,
    input_hidden_dim: int = -1,
    with_bias=True,
    count_input_memory: bool = True,
):
    """
    Example: Algo 07, msa_row_attention_with_pair_bias_metrics

    number_of_params = (c_out) * (c_m + 1)
    adds: (N_h)(N_s)(N_r) (3) (c_out) (c_m)
    mults: (N_h)(N_s)(N_r) (3) (c_out) (c_m)
    """
    assert output_hidden_dim == -1 and input_hidden_dim == -1

    c_out = output_shape[output_hidden_dim]
    c_in = input_shape[input_hidden_dim]

    number_of_mults_fwd_1_right_vector = c_out * c_in
    number_of_adds_fwd_1_right_vector = c_out * (c_in - 1) + c_out * with_bias
    number_of_params = c_out * c_in + c_out * with_bias

    k = num_elements_excluding(output_shape, excluded_axes=[output_hidden_dim])

    linear_metrics = AlgoMetrics(
        function_name="linear_metrics",
        number_of_mults_fwd=(k * number_of_mults_fwd_1_right_vector),
        number_of_adds_fwd=(k * number_of_adds_fwd_1_right_vector),
        number_of_params=number_of_params,
        memory_footprint_num_els=num_elements(input_shape) if count_input_memory else 0,
    )

    return output_shape, linear_metrics


def linear_no_bias_metrics(
    output_shape: Tuple[int],
    input_shape: Tuple[int],
    output_hidden_dim: int = -1,
    input_hidden_dim: int = -1,
    count_input_memory: bool = True,
):
    return linear_metrics(
        output_shape=output_shape,
        input_shape=input_shape,
        output_hidden_dim=output_hidden_dim,
        input_hidden_dim=input_hidden_dim,
        with_bias=False,
        count_input_memory=count_input_memory,
    )


def sigmoid_metrics(input_shape: Tuple):
    sigmoid_shape = deepcopy(input_shape)
    sigmoid_metrics = AlgoMetrics(
        function_name="sigmoid_metrics",
        number_of_adds_fwd=num_elements(input_shape),
        number_of_mults_fwd=num_elements(input_shape),
        number_of_nonlinear_ops=num_elements(input_shape),
        number_of_params=0,
        memory_footprint_num_els=0,
    )

    return sigmoid_shape, sigmoid_metrics


def soft_max_metrics(input_shape: Tuple, softmax_dim: int = 0, count_input_memory: bool = True):
    soft_max_shape = deepcopy(input_shape)
    # mults = [input_shape[i] for i in range(len(input_shape)) if i != (softmax_dim % len(input_shape))]
    # mults += [input_shape[softmax_dim] - 1]
    number_of_mults_fwd = num_elements(input_shape)

    # compute norm just once for the aggregation axis
    number_of_adds_fwd = num_elements_excluding(input_shape, excluded_axes=[softmax_dim]) * (
        input_shape[softmax_dim] - 1
    )

    soft_max_metrics = AlgoMetrics(
        function_name="soft_max_metrics",
        number_of_adds_fwd=number_of_adds_fwd,
        number_of_mults_fwd=number_of_mults_fwd,
        number_of_nonlinear_ops=num_elements(input_shape),
        number_of_params=0,
        memory_footprint_num_els=int(count_input_memory) * num_elements(input_shape),
    )

    return soft_max_shape, soft_max_metrics


def multiply_elem_wise(
    left_shape: Tuple,
    right_shape: Tuple,
    count_input_memory: bool = True,
):
    assert left_shape == right_shape
    output_shape = deepcopy(left_shape)
    multiple_elem_wise_metrics = AlgoMetrics(
        function_name="multiply_elem_wise",
        number_of_adds_fwd=0,
        number_of_mults_fwd=num_elements(left_shape),
        number_of_nonlinear_ops=0,
        memory_footprint_num_els=(num_elements(left_shape) + num_elements(right_shape)) if count_input_memory else 0,
        number_of_params=0,
    )

    return output_shape, multiple_elem_wise_metrics


def num_elements_other_components(x: Tuple[int], x_hidden_dim: int = -1):
    """Get the number of elements that are not in the hidden dimension."""
    return num_elements_excluding(x, excluded_axes=[x_hidden_dim])


def flatten_outer_product_mean(element_shape: Tuple[int], element_hidden_dim=-1):
    """
    See SOL analysis report A.4
    element_size is of size (N_s, N_r, c)
    """
    assert element_hidden_dim == -1
    c = element_shape[element_hidden_dim]

    outer_product_metrics = AlgoMetrics(
        number_of_adds_fwd=element_shape[1] ** 2 * (element_shape[0] - 1) * c**2,
        number_of_mults_fwd=element_shape[1] ** 2 * (element_shape[0] + 1) * c**2,
        number_of_nonlinear_ops=0,
        number_of_params=0,
        memory_footprint_num_els=(2 * num_elements(element_shape)),
    )
    output_size = (element_shape[1], element_shape[1], c**2)
    return output_size, outer_product_metrics
