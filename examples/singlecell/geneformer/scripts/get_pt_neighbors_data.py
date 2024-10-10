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


import subprocess
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix


data_path = "examples/tests/test_data/singlecell/cellxgene_2023-12-15_small/input_data/test"

dataset_names = [
    "assay__10x_3_v2/sex__male/development_stage__45-year-old_human_stage/self_reported_ethnicity__unknown/tissue_general__small_intestine/dataset_id__ee195b7d-184d-4dfa-9b1c-51a7e601ac11/sidx_19480503_2689_0.h5ad",
    "assay__10x_3_v3/sex__male/development_stage__42-year-old_human_stage/self_reported_ethnicity__European/tissue_general__brain/dataset_id__00476f9f-ebc1-4b72-b541-32f912ce36ea/sidx_29791758_10099_1.h5ad",
]
datasets = []


for data_ind, f_name in enumerate(dataset_names):
    f_path = Path(data_path, f_name)
    data = sc.read_h5ad(f_path)
    datasets.append(data)


def add_pt_neighbors(adata):
    consecutive = np.all(np.diff([int(x) for x in datasets[1].obs.index]) == 1)
    assert consecutive

    first, last = int(adata.obs.index[0]), adata.n_obs
    offset = last if first == 0 else first
    k_neighbors = [0, 5, 10]
    neighbors = np.zeros((adata.n_obs, adata.n_obs), dtype=int)

    # Assign neighbors for each sample
    for obs_index in adata.obs.index:
        i = int(obs_index) % offset  # to avoid out of bound errors as some adata starts from indices > 0
        # Assign the next k_neighbors indices as neighbors
        k_i = k_neighbors[i % 3]
        for j in range(1, k_i + 1):
            neighbor_index = (i + j) % adata.n_obs
            neighbors[i, neighbor_index] = 1
    adata.obsp["next_cell_ids"] = csr_matrix(neighbors)


for ind in range(2):
    f_name, adata = dataset_names[ind], datasets[ind]
    add_pt_neighbors(adata)
    fname = f"{f_name.split('.')[0]}_pseudotime_added.h5ad"
    path = Path(data_path, fname)
    print(path)
    adata.write_h5ad(
        path,
    )

process_path = "examples/tests/test_data/singlecell/cellxgene_2023-12-15_small/processed_data/pseudotime/test"
command = [
    "python",
    "/workspace/bionemo/bionemo/data/singlecell/sc_memmap.py",
    "--save-path",
    process_path,
    "--next-cell",
    "--data-path",
    data_path,
]
# create the metadata, in this combined memmaps two datasets have temporal information and two do not, we are testing if the
# loockup_neighbor works as expected for these two cases
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
