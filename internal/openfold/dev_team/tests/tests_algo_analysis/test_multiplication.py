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


from internal.openfold.dev_team.algo_analysis.triangle_metrics import triangle_multplication


# EvoformerStack parameter set
N_sequences = 124
N_residues = 256
c = 128
c_z = 128


def test_triangle_metrics():
    z_shape = (N_residues, N_residues, c_z)
    z_shape, triangle_multplication_metrics_out = triangle_multplication(z_shape, c)
    mults_fwd_in_gsheet = 8_674_344_960
    adds_fwd_in_gsheet = 8_648_654_848
    memory_footprint_in_bytes_in_gsheet = 402_653_184
    params_in_gsheet = 99_328
    assert z_shape == (256, 256, 128)
    assert abs(triangle_multplication_metrics_out.number_of_params - params_in_gsheet) / params_in_gsheet < 0.05
    assert (
        abs(triangle_multplication_metrics_out.number_of_mults_fwd - mults_fwd_in_gsheet) / mults_fwd_in_gsheet < 0.05
    )
    assert abs(triangle_multplication_metrics_out.number_of_adds_fwd - adds_fwd_in_gsheet) / adds_fwd_in_gsheet < 0.05
    assert (
        abs(triangle_multplication_metrics_out.memory_footprint_in_bytes - memory_footprint_in_bytes_in_gsheet)
        / memory_footprint_in_bytes_in_gsheet
        < 0.05
    )
