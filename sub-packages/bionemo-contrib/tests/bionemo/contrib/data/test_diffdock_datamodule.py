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
from numpy import isin
import pytest
import torch

import lightning
from torch_geometric.data import Batch, HeteroData

from bionemo.contrib.data.molecule.diffdock.datamodule import Split


@pytest.mark.parametrize("split", [s for s in Split])
def test_datamodule_init(split, get_diffdock_heterodata, create_datamodule):
    name_split = str(split).split('.')[1]
    (_, _, names, model) = get_diffdock_heterodata
    data_module, prefix_dir_tars_wds = create_datamodule
    assert data_module._sizes[split] == len(names[split]),\
        f"Wrong {split}-set size for {model} model: "\
        f"expected {len(names[split])} "\
        f"but got {data_module._sizes[split]}"
    assert data_module._dirs_tars_wds[split] ==\
        f"{prefix_dir_tars_wds}{name_split}",\
        f"Wrong tar files directory for {model} model: "\
        f"expected {prefix_dir_tars_wds}{split} "\
        f"but got {data_module._dirs_tars_wds[split]}"


@pytest.mark.parametrize("split", [s for s in Split])
def test_datamodule_prepare_data(split, create_datamodule):
    data_module, _ = create_datamodule
    # LightningDataModule.prepare_data() is supposed to be called from the main
    # process in a Lightning-managed multi-process context so we can call it in
    # a single process
    data_module.prepare_data()
    files_tars = sorted(glob.glob(
        f"{data_module._dirs_tars_wds[split]}/"\
        f"{data_module._prefix_tars_wds}-*.tar"))
    assert len(files_tars) >= data_module._n_tars_wds,\
        f"Wrong num of {split}-set tar files: "\
        f"expected {data_module._n_tars_wds} "\
        f"got {len(files_tars)}"


@pytest.mark.parametrize("split", [s for s in Split])
def test_datamodule_setup_dataset(split, create_datamodule, create_another_datamodule):
    data_modules= [create_datamodule[0], create_another_datamodule[0]]
    lists_complex_name = []
    lists_pos_ligand = []
    for m in data_modules:
        m.prepare_data()
        # run through all the possible stages first to setup all the correps.
        # dataset objects
        m.setup("fit")
        m.setup("test")
        lightning.seed_everything(2823828)
        names = []
        pos_ligand = []
        for sample in m._dataset[split]:
            assert isinstance(sample, HeteroData),\
                "Sample yield from dataset is not PyG HeteroData"
            names.append(sample.name)
            pos_ligand.append(sample["ligand"].pos)
        lists_complex_name.append(names)
        lists_pos_ligand.append(pos_ligand)

    assert len(lists_complex_name[0]) > 0,\
        "No names in {split} dataset"
    assert lists_complex_name[0] == lists_complex_name[1],\
        f"Inconsistent sample name in {split}-set from data module instances: "\
        f"{lists_complex_name[0]} \n\nvs.\n\n"\
        f"{lists_complex_name[1]}"

    assert len(lists_pos_ligand[0]) > 0, "No ligand position found in dataset"
    assert len(lists_pos_ligand[0]) == len(lists_pos_ligand[1]),\
        f"Inconsistent number of ligand position in {split}-set from data "\
        f"module instances: {len(lists_pos_ligand[0])} \n\nvs.\n\n"\
        f"{len(lists_pos_ligand[1])}"
    for i in range(len(lists_pos_ligand[0])):
        pos_0 = lists_pos_ligand[0][i]
        pos_1 = lists_pos_ligand[1][i]
        torch.testing.assert_close(pos_0, pos_1,
                                   msg=lambda m :
                                   f"Inconsistent ligand position in the "
                                   f"{i}'th sample/batch of {split}-set "
                                   f"between two data module instances:\n\n{m}")


@pytest.mark.parametrize("split", [s for s in Split])
def test_datamodule_setup_dataloader(split, create_datamodule, create_another_datamodule):
    data_modules= [create_datamodule[0], create_another_datamodule[0]]
    lists_complex_name = []
    lists_pos_ligand = []
    for m in data_modules:
        m.prepare_data()
        # run through all the possible stages first to setup all the correps.
        # dataset objects
        m.setup("fit")
        m.setup("test")
        lightning.seed_everything(2823828)
        names = []
        pos_ligand = []
        loader = None
        if split == Split.train:
            loader = m.train_dataloader()
        elif split == Split.val:
            loader = m.val_dataloader()
        elif split == Split.test:
            loader = m.test_dataloader()
        else:
            raise RuntimeError(f"Test for split {split} not implemented")
        assert loader is not None, "dataloader not instantated"
        for samples in loader:
            # PyG's HeteroDataBatch is Batch inherited from HeteroData
            assert isinstance(samples, Batch),\
                f"Sample object is not PyG Batch"
            assert isinstance(samples, HeteroData),\
                f"Sample object is not PyG HeteroData"
            names.append(samples.name)
            pos_ligand.append(samples["ligand"].pos)
        lists_complex_name.append(names)
        lists_pos_ligand.append(pos_ligand)

    assert len(lists_complex_name[0]) > 0,\
        "No names in {split} dataloader"
    assert lists_complex_name[0] == lists_complex_name[1],\
        f"Inconsistent sample name in {split}-set from data module instances: "\
        f"{lists_complex_name[0]} \n\nvs.\n\n"\
        f"{lists_complex_name[1]}"

    assert len(lists_pos_ligand[0]) > 0,\
        "No ligand position found in dataloader"
    assert len(lists_pos_ligand[0]) == len(lists_pos_ligand[1]),\
        f"Inconsistent number of ligand position in {split}-set from data "\
        f"module instances: {len(lists_pos_ligand[0])} \n\nvs.\n\n"\
        f"{len(lists_pos_ligand[1])}"
    for i in range(len(lists_pos_ligand[0])):
        pos_0 = lists_pos_ligand[0][i]
        pos_1 = lists_pos_ligand[1][i]
        torch.testing.assert_close(pos_0, pos_1,
                                   msg=lambda m :
                                   f"Inconsistent ligand position in the "
                                   f"{i}'th sample/batch of {split}-set "
                                   f"between two data module instances:\n\n{m}")
