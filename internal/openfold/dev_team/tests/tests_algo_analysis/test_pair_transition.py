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


from internal.openfold.dev_team.algo_analysis.transition import (
    pair_transition_metrics,
)


def test_pair_transition():
    c_z = 128
    N_residues = 256
    input_shape = (N_residues, N_residues, c_z)
    transition_n = 4
    pair_transition_shape, pair_transition_metrics_out = pair_transition_metrics(in_shape=input_shape, n=transition_n)

    gsheet_number_of_mults_fwd = 8_615_362_560
    gsheet_number_of_adds_fwd = 8_623_489_024
    gsheet_number_of_params = 131_968
    memory_footprint_in_bytes_in_gsheet = 201_326_592
    assert input_shape == pair_transition_shape
    assert gsheet_number_of_params == pair_transition_metrics_out.number_of_params
    assert gsheet_number_of_mults_fwd == pair_transition_metrics_out.number_of_mults_fwd
    assert gsheet_number_of_adds_fwd == pair_transition_metrics_out.number_of_adds_fwd
    assert memory_footprint_in_bytes_in_gsheet == pair_transition_metrics_out.memory_footprint_in_bytes
