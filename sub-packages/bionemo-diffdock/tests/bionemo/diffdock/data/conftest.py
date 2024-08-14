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

from pathlib import Path
from enum import Enum, auto
from functools import partial
import random

import lightning as L
import pytest
import torch
from torch_geometric.loader.dataloader import Collater
import webdataset as wds
from webdataset.filters import batched, shuffle

from bionemo.core.data.datamodule import PickledDataWDS, Split
from bionemo.diffdock.utils.data import SelectPoseAndLabelData, SizeAwareBatching, estimate_size
from bionemo.diffdock.utils.diffusion import GenerateNoise, t_to_sigma


@pytest.fixture(scope="module")
def get_path(request):
    path_test = Path(request.module.__file__).resolve()
    dir_test = path_test.parents[0]
    dir_data = path_test.parents[6] / "test_data" / \
        "bionemo-diffdock" / "data" / "pyg_heterodata_pickled"
    return str(dir_test), str(dir_data)


class DiffDockModel(Enum):
    score = auto()
    confidence = auto()


@pytest.fixture(scope="module", params=list(DiffDockModel))
def get_diffdock_heterodata(get_path, request):
    _, dir_data = get_path
    model = request.param
    name_model = str(model).split(".")[-1]
    dir_heterodata = f"{dir_data}/{name_model}_model"
    suffix_heterodata = "heterodata.pyd"
    names = {
        Split.train: [
            "6t88",
            "6vs3",
            "6wtn",
            "6yqv",
            "7amc",
            "7bmi",
            "7cuo",
            "7d5c",
            "7din",
            "7fha",
            "7jnb",
            "7k0v",
            "7kb1",
            "7km8",
            "7l7c",
            "7lcu",
            "7msr",
            "7my1",
            "7n6f",
            "7np6",
        ],
        Split.val: ["7nr6", "7oeo", "7oli", "7oso", "7p5t", "7q5i", "7qhl", "7rh3", "7rzl", "7sgv"],
        Split.test: ["7sne", "7t2i", "7tbu", "7tsf", "7umv", "7up3", "7uq3", "7wpw", "7xek", "7xij"],
    }
    return (dir_heterodata, suffix_heterodata, names, model)


def _create_datamodule_score_model_impl(tmp_path_factory, dir_heterodata, suffix_heterodata, names):
    prefix_dir_tars_wds = tmp_path_factory.mktemp("diffdock_score_model_tars_wds").as_posix()
    tr_sigma_min, tr_sigma_max = (0.1, 19)
    rot_sigma_min, rot_sigma_max = (0.03, 1.55)
    tor_sigma_min, tor_sigma_max = (0.0314, 3.14)
    is_all_atom = False
    no_torsion = False
    sigma_t = partial(
        t_to_sigma, tr_sigma_min, tr_sigma_max, rot_sigma_min, rot_sigma_max, tor_sigma_min, tor_sigma_max
    )
    seed_rng_shfl = 822782392
    # webdataset pipeline
    pipeline_wds = {
        Split.train: [GenerateNoise(sigma_t, no_torsion, is_all_atom, copy_ref_pos=False),
                      shuffle(len(names[Split.train]),
                              rng=random.Random(seed_rng_shfl))],
        Split.val: GenerateNoise(sigma_t, no_torsion, is_all_atom, copy_ref_pos=True),
        Split.test: GenerateNoise(sigma_t, no_torsion, is_all_atom, copy_ref_pos=False),
    }
    local_batch_size = 2
    global_batch_size = 2
    size_cuda_mem = 0.85 * torch.cuda.get_device_properties("cuda:0").total_memory / 2**20
    batch_pyg = batched(local_batch_size, collation_fn=Collater(dataset=[], follow_batch=None, exclude_keys=None))
    # WebLoader pipeline
    pipelines_wdl_batch = {
        Split.train: [shuffle(40, rng=random.Random(seed_rng_shfl)),
                      SizeAwareBatching(max_total_size=size_cuda_mem,
                                        size_fn=estimate_size,
                                        no_single_sample=True)],
        Split.val: batch_pyg,
        Split.test: batch_pyg,
    }
    n_tars_wds = 4
    kwargs_wds = {
        split : {'shardshuffle' : split == Split.train,
                 'nodesplitter' : wds.split_by_node,
                 'seed' : seed_rng_shfl}
        for split in Split
        }
    kwargs_wld = {
        Split.train: {"num_workers": 2},
        Split.val: {"num_workers": 2},
        Split.test: {"num_workers": 2},
    }
    data_module = PickledDataWDS(
        dir_heterodata,
        suffix_heterodata,
        names,
        prefix_dir_tars_wds,
        global_batch_size,
        n_tars_wds=n_tars_wds,
        prefix_tars_wds="heterographs",
        pipeline_wds=pipeline_wds,
        pipeline_prebatch_wld=pipelines_wdl_batch,
        kwargs_wds=kwargs_wds,
        kwargs_wld=kwargs_wld,
    )
    return data_module, prefix_dir_tars_wds


def _create_datamodule_confidence_model_impl(tmp_path_factory, dir_heterodata, suffix_heterodata, names):
    prefix_dir_tars_wds = tmp_path_factory.mktemp("diffdock_confidence_model_tars_wds").as_posix()
    # webdataset pipeline
    rmsd_classification_cutoff = 2.0
    samples_per_complex = 7
    balance = False
    is_all_atom = True
    seed_rng_shfl = 822782392
    select_pose = SelectPoseAndLabelData(
        rmsd_classification_cutoff, samples_per_complex, balance, is_all_atom, seed=seed_rng_shfl
    )
    pipeline_wds = {
        Split.train: [select_pose, shuffle(len(names[Split.train]),
                                           rng=random.Random(seed_rng_shfl))],
        Split.val: select_pose,
        Split.test: select_pose,
    }
    local_batch_size = 2
    global_batch_size = 2
    batch_pyg = batched(local_batch_size, collation_fn=Collater(dataset=[], follow_batch=None, exclude_keys=None))
    # WebLoader pipeline
    pipelines_wdl_batch = {
        Split.train: [shuffle(40, rng=random.Random(seed_rng_shfl)), batch_pyg],
        Split.val: batch_pyg,
        Split.test: batch_pyg,
    }
    n_tars_wds = 4
    kwargs_wds = {
        split : {'shardshuffle' : split == Split.train,
                 'nodesplitter' : wds.split_by_node,
                 'seed' : seed_rng_shfl}
        for split in Split
        }
    kwargs_wld = {
        Split.train: {"num_workers": 2},
        Split.val: {"num_workers": 2},
        Split.test: {"num_workers": 2},
    }
    data_module = PickledDataWDS(
        dir_heterodata,
        suffix_heterodata,
        names,
        prefix_dir_tars_wds,
        global_batch_size,
        n_tars_wds=n_tars_wds,
        prefix_tars_wds="heterographs",
        pipeline_wds=pipeline_wds,
        pipeline_prebatch_wld=pipelines_wdl_batch,
        kwargs_wds=kwargs_wds,
        kwargs_wld=kwargs_wld,
    )
    return data_module, prefix_dir_tars_wds


@pytest.fixture(scope="module")
def create_datamodule(tmp_path_factory, get_diffdock_heterodata):
    dir_heterodata, suffix_heterodata, names, model = get_diffdock_heterodata
    if model == DiffDockModel.score:
        return _create_datamodule_score_model_impl(tmp_path_factory, dir_heterodata, suffix_heterodata, names)
    elif model == DiffDockModel.confidence:
        return _create_datamodule_confidence_model_impl(tmp_path_factory, dir_heterodata, suffix_heterodata, names)


@pytest.fixture(scope="module")
def create_another_datamodule(tmp_path_factory, get_diffdock_heterodata):
    dir_heterodata, suffix_heterodata, names, model = get_diffdock_heterodata
    if model == DiffDockModel.score:
        return _create_datamodule_score_model_impl(tmp_path_factory, dir_heterodata, suffix_heterodata, names)
    elif model == DiffDockModel.confidence:
        return _create_datamodule_confidence_model_impl(tmp_path_factory, dir_heterodata, suffix_heterodata, names)


class ModelTestDiffDock(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self._model = torch.nn.Linear(3, 3)
        self._samples = {split: [] for split in Split}

    def forward(self, x):
        return self._model(x["ligand"].pos)

    def training_step(self, batch):
        self._samples[Split.train].append(batch.name)
        loss = self(batch).sum()
        return loss

    def validation_step(self, batch, batch_index):
        self._samples[Split.val].append(batch.name)
        return torch.zeros(1)

    def test_step(self, batch, batch_index):
        self._samples[Split.test].append(batch.name)

    def predict_step(self, batch, batch_index):
        self._samples[Split.test].append(batch.name)
        return torch.zeros(1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer


@pytest.fixture(scope="function")
def create_trainer_and_model():
    trainer = L.Trainer(max_epochs=1, accelerator="gpu", devices=1, val_check_interval=1)
    model = ModelTestDiffDock()
    return trainer, model
