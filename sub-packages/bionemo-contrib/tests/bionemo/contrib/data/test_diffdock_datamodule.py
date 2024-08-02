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

import glob
import multiprocessing as mp
import sys
import torch

import lightning

from bionemo.contrib.data.molecule.diffdock.datamodule import Split


def test_ScoreModelWDS_init(get_diffdock_score_model_heterodata,
                            create_ScoreModelWDS):
    (_, _, names_subset_train, names_subset_val,
     names_subset_test) = get_diffdock_score_model_heterodata
    data_module, prefix_dir_tars_wds = create_ScoreModelWDS
    assert data_module._sizes[Split.train] == len(names_subset_train),\
        f"Wrong train set size: expected {len(names_subset_train)}"\
        f"but got {data_module._sizes[Split.train]}"
    assert data_module._sizes[Split.val] == len(names_subset_val),\
        f"Wrong val set size: expected {len(names_subset_val)}"\
        f"but got {data_module._sizes[Split.val]}"
    assert data_module._sizes[Split.test] == len(names_subset_test),\
        f"Wrong test set size: expected {len(names_subset_test)} "\
        f"but got {data_module._sizes[Split.test]}"
    assert data_module._dirs_tars_wds[Split.train] ==\
        f"{prefix_dir_tars_wds}train",\
        f"Wrong tar files directory: expected {prefix_dir_tars_wds}train "\
        f"but got {data_module._dirs_tars_wds[Split.train]}"
    assert data_module._dirs_tars_wds[Split.val] ==\
        f"{prefix_dir_tars_wds}val",\
        f"Wrong tar files directory: expected {prefix_dir_tars_wds}val "\
        f"but got {data_module._dirs_tars_wds[Split.val]}"
    assert data_module._dirs_tars_wds[Split.test] ==\
        f"{prefix_dir_tars_wds}test",\
        f"Wrong tar files directory: expected {prefix_dir_tars_wds}test "\
        f"but got {data_module._dirs_tars_wds[Split.test]}"


def test_ScoreModelWDS_prepare_data(create_ScoreModelWDS):
    data_module, _ = create_ScoreModelWDS
    # LightningDataModule.prepare_data() is supposed to be called from the main
    # process in a Lightning-managed multi-process context so we can call it in
    # a single process
    data_module.prepare_data()
    files_tars_train = glob.glob(
        f"{data_module._dirs_tars_wds[Split.train]}/"\
        f"{data_module._prefix_tars_wds}-*.tar")
    assert len(files_tars_train) >= data_module._n_tars_wds,\
        f"Wrong num of train tar files: expected {data_module._n_tars_wds}"\
        f"got {len(files_tars_train)}"
    files_tars_val = glob.glob(
        f"{data_module._dirs_tars_wds[Split.val]}/"\
        f"{data_module._prefix_tars_wds}-*.tar")
    assert len(files_tars_val) >= data_module._n_tars_wds,\
        f"Wrong num of val tar files: expected {data_module._n_tars_wds}"\
        f"got {len(files_tars_val)}"
    files_tars_test = glob.glob(
        f"{data_module._dirs_tars_wds[Split.test]}/"\
        f"{data_module._prefix_tars_wds}-*.tar")
    assert len(files_tars_test) >= data_module._n_tars_wds,\
        f"Wrong num of test tar files: expected {data_module._n_tars_wds}"\
        f"got {len(files_tars_test)}"



def test_ScoreModelWDS_setup_dataset(create_ScoreModelWDS, create_another_ScoreModelWDS):
    data_modules= [create_ScoreModelWDS[0], create_another_ScoreModelWDS[0]]
    lists_complex_name = []
    lists_pos_ligand = []
    stage = "fit"
    for m in data_modules:
        m.prepare_data()
        m.setup(stage)
        lightning.seed_everything(2823828)
        names = []
        pos_ligand = []
        for sample in m._dataset_train:
            names.append(sample.name)
            pos_ligand.append(sample["ligand"].pos)
        lists_complex_name.append(names)
        lists_pos_ligand.append(pos_ligand)

    assert len(lists_complex_name[0]) > 0, "No names in dataset"
    assert lists_complex_name[0] == lists_complex_name[1],\
        f"Inconsistent sample name from data module instances: "\
        f"{lists_complex_name[0]} \n\nvs.\n\n"\
        f"{lists_complex_name[1]}"

    assert len(lists_pos_ligand[0]) > 0, "No ligand position found in dataset"
    assert len(lists_pos_ligand[0]) == len(lists_pos_ligand[1]),\
        "Inconsistent number of ligand position from data module instances: "\
        f"{len(lists_pos_ligand[0])} \n\nvs.\n\n"\
        f"{len(lists_pos_ligand[1])}"
    for i in range(len(lists_pos_ligand[0])):
        pos_0 = lists_pos_ligand[0][i]
        pos_1 = lists_pos_ligand[1][i]
        torch.testing.assert_close(pos_0, pos_1,
                                   msg=lambda m :
                                   f"Inconsistent ligand position in the "
                                   f"{i}'th sample/batch between two data "
                                   f"module instances:\n\n{m}")
