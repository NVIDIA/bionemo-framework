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


from internal.openfold.dev_team.algo_analysis.algo_metrics import (
    flatten_outer_product_mean,
    layer_norm_metrics,
    linear_metrics,
    linear_no_bias_metrics,
    multiply_elem_wise,
)


# EvoFormerStack parameter set
N_heads = 8
N_sequences = 124
N_residues = 256
c_m = 256
c = 32
c_z = 128


def test_layer_norm():
    input_shape = (N_sequences, N_residues, c_m)
    output_shape, output_count = layer_norm_metrics(input_shape)
    mults_fwd_in_gsheet = 24_506_368
    adds_fwd_in_gsheet = 32_505_856
    memory_footprint_in_bytes_in_gsheet = 32_505_856
    params_in_gsheet = 512
    assert output_shape == input_shape
    assert output_count.number_of_mults_fwd == mults_fwd_in_gsheet
    assert output_count.number_of_adds_fwd == adds_fwd_in_gsheet
    assert output_count.memory_footprint_in_bytes == memory_footprint_in_bytes_in_gsheet
    assert output_count.number_of_params == params_in_gsheet


def test_linear_no_bias():
    input_shape = (N_sequences, N_residues, c_m)
    output_shape = (N_sequences, N_residues, 3 * c * N_heads)
    linear_shape, linear_count = linear_no_bias_metrics(output_shape, input_shape)
    mults_fwd_in_gsheet = 6_241_124_352
    adds_fwd_in_gsheet = 6_216_744_960
    memory_footprint_in_bytes_in_gsheet = 32_505_856
    params_in_gsheet = 196_608
    assert output_shape == linear_shape
    assert linear_count.number_of_mults_fwd == mults_fwd_in_gsheet
    assert linear_count.number_of_adds_fwd == adds_fwd_in_gsheet
    assert linear_count.memory_footprint_in_bytes == memory_footprint_in_bytes_in_gsheet
    assert linear_count.number_of_params == params_in_gsheet


def test_linear():
    input_shape = (N_sequences, N_residues, c_m)
    output_shape = (N_sequences, N_residues, c * N_heads)
    linear_shape, linear_count = linear_metrics(output_shape, input_shape, count_input_memory=False)
    mults_fwd_in_gsheet = 2_080_374_784
    adds_fwd_in_gsheet = 2_080_374_784
    memory_footprint_in_bytes_in_gsheet = 0
    params_in_gsheet = 65_792

    assert output_shape == linear_shape
    assert linear_count.number_of_mults_fwd == mults_fwd_in_gsheet
    assert linear_count.number_of_adds_fwd == adds_fwd_in_gsheet
    assert linear_count.memory_footprint_in_bytes == memory_footprint_in_bytes_in_gsheet
    assert linear_count.number_of_params == params_in_gsheet


def test_flatten_outer_product_mean():
    element_shape = (N_sequences, N_residues, c)
    outer_shape, outer_count = flatten_outer_product_mean(element_shape)
    mults_fwd_in_gsheet = 8_388_608_000
    adds_fwd_in_gsheet = 8_254_390_272
    memory_footprint_in_bytes_in_gsheet = 8_126_464
    params_in_gsheet = 0
    assert outer_shape == (N_residues, N_residues, c**2)
    assert outer_count.number_of_mults_fwd == mults_fwd_in_gsheet
    assert outer_count.number_of_adds_fwd == adds_fwd_in_gsheet
    assert outer_count.memory_footprint_in_bytes == memory_footprint_in_bytes_in_gsheet
    assert outer_count.number_of_params == params_in_gsheet


def test_multiply_elem_wise():
    element_shape = (N_residues, N_residues, c_z)
    elem_wise_shape, elem_wise_count = multiply_elem_wise(element_shape, element_shape)
    mults_fwd_in_gsheet = 8_388_608
    adds_fwd_in_gsheet = 0
    memory_footprint_in_bytes_in_gsheet = 67_108_864
    params_in_gsheet = 0
    assert elem_wise_shape == (N_residues, N_residues, c_z)
    assert elem_wise_count.number_of_mults_fwd == mults_fwd_in_gsheet
    assert elem_wise_count.number_of_adds_fwd == adds_fwd_in_gsheet
    assert elem_wise_count.memory_footprint_in_bytes == memory_footprint_in_bytes_in_gsheet
    assert elem_wise_count.number_of_params == params_in_gsheet
