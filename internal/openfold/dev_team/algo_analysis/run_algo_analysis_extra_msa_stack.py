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


#
#   title: run_algo_analysis_extra_msa_stack.py
#   description:
#
#   notes:


# %%


import seaborn as sns

from internal.openfold.dev_team.algo_analysis.extra_msa_stack_metrics import (
    extra_msa_stack_metrics,
)


sns.set()
# %% Set parameters
#   -ultimate these parameters can be ready directly from conf yaml file
#

# ExtraMsaStack parameter set
num_residues = 256  # model.train_sequence_crop_size
num_sequences = 1024  # model.max_extra_msa
c_e = 64  # model.extra_msa_stack_config.c_e
c_z = 128  # model.extra_msa_stack_config.c_z
c_hidden_msa_att = 8
c_hidden_tri_att = 32
c_hidden_tri_mul = 128
c_hidden_opm = 32
num_heads_msa = 8
num_heads_tri = 4
transition_n = 4
num_extra_msa_blocks = 4


e_shape = (num_sequences, num_residues, c_e)
z_shape = (num_residues, num_residues, c_z)

# %%
#

extra_msa_stack_shape, extra_msa_stack_metrics = extra_msa_stack_metrics(
    e_shape,
    z_shape,
    c_hidden_msa_att=c_hidden_msa_att,
    c_hidden_tri_att=c_hidden_tri_att,
    c_hidden_tri_mul=c_hidden_tri_mul,
    c_hidden_opm=c_hidden_opm,
    num_heads_msa=num_heads_msa,
    num_extra_msa_blocks=num_extra_msa_blocks,
    num_heads_tri=num_heads_tri,
    transition_n=transition_n,
)


# %%
print(extra_msa_stack_metrics)

print("run_algo_analysis_extra_msa_stack.py, end")
