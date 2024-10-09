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


from internal.openfold.dev_team.algo_analysis.outer_product_mean_metrics import outer_product_mean_metrics


# EvoformerStack parameter set
N_sequences = 124
N_residues = 256
c_m = 256
c_hidden_opm = 32
c_z = 128


def test_outer_product_mean_metrics():
    m_shape = (N_sequences, N_residues, c_m)
    z_shape, outer_product_mean_metrics_out = outer_product_mean_metrics(m_shape, c_hidden_opm, c_z)
    mults_fwd_in_gsheet = 17_523_142_656
    adds_fwd_in_gsheet = 17_396_924_416
    memory_footprint_in_bytes_in_gsheet = 341_573_632
    params_in_gsheet = 148_160
    assert z_shape == (256, 256, 128)
    assert outer_product_mean_metrics_out.number_of_mults_fwd == mults_fwd_in_gsheet
    assert outer_product_mean_metrics_out.number_of_adds_fwd == adds_fwd_in_gsheet
    assert outer_product_mean_metrics_out.memory_footprint_in_bytes == memory_footprint_in_bytes_in_gsheet
    assert outer_product_mean_metrics_out.number_of_params == params_in_gsheet


test_outer_product_mean_metrics()
