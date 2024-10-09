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


from internal.openfold.dev_team.algo_analysis.tensor_shape_helpers import num_elements_excluding


def test_000():
    # EvoformerStack parameter set
    num_residues = 256
    c_z = 128  # residude-residue pair_embedding_dim

    z_shape = (num_residues, num_residues, c_z)
    out = num_elements_excluding(z_shape, excluded_axes=[-1])

    # assert on results
    expected_num = num_residues * num_residues
    assert expected_num == out
