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

import os
import random
from pathlib import Path
from enum import Enum, auto
from functools import partial
from typing import Any, Iterable

import lightning as L
import pytest
import torch
from webdataset.filters import batched, shuffle

from bionemo.core.data.datamodule import WebDataModule, Split


@pytest.fixture(scope="module")
def get_path(request):
    path_test = Path(request.module.__file__).resolve()
    dir_test = path_test.parents[0]
    dir_data = path_test.parents[6] / "test_data" / "bionemo" / "core" / "data" / "webdatamodule"
    return str(dir_test), str(dir_data)


def _create_webdatamodule(dir_tars_wds):
    suffix_keys_wds = "tensor.pyd"
    local_batch_size = 2
    global_batch_size = 2
    prefix_tars_wds = "tensor"
    seed_rng_shfl = 82838392

    dirs_tars_wds = { split : dir_tars_wds for split in Split }

    n_samples = { split : 10 for split in Split }

    batch = batched(local_batch_size, collation_fn=lambda
                    list_samples : torch.vstack(list_samples))

    untuple = lambda source : (sample for (sample,) in source)

    pipeline_wds = {
        Split.train : [untuple, shuffle(n_samples[Split.train],
                                        rng=random.Random(seed_rng_shfl))],
        Split.val : untuple,
        Split.test : untuple
        }

    pipeline_prebatch_wld = {
        Split.train: [shuffle(n_samples[Split.train],
                              rng=random.Random(seed_rng_shfl)), batch],
        Split.val : batch,
        Split.test : batch
        }

    kwargs_dl = {
        split : {"num_workers": 2} for split in Split
        }

    data_module = WebDataModule(dirs_tars_wds, n_samples, suffix_keys_wds,
                                global_batch_size,
                                prefix_tars_wds=prefix_tars_wds,
                                pipeline_wds=pipeline_wds,
                                pipeline_prebatch_wld=pipeline_prebatch_wld,
                                seed_rng_shfl=seed_rng_shfl, kwargs_dl=kwargs_dl)

    return data_module, dir_tars_wds


@pytest.fixture(scope="module")
def create_webdatamodule(get_path):
    _, dir_tars_wds = get_path
    return _create_webdatamodule(dir_tars_wds)


@pytest.fixture(scope="module")
def create_another_webdatamodule(get_path):
    _, dir_tars_wds = get_path
    return _create_webdatamodule(dir_tars_wds)


class ModelTestWebDataModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self._model = torch.nn.Linear(1, 1)
        self._samples = {split: [] for split in Split}

    def forward(self, x):
        return self._model(x.float())

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
    model = ModelTestWebDataModule()
    return trainer, model
