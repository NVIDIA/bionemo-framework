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
# ruff: noqa: D101, D102, D103, D107

import pickle
import random

import lightning as L
import pytest
import torch
import webdataset as wds
from webdataset.filters import batched, shuffle

from bionemo.webdatamodule.datamodule import PickledDataWDS, Split, WebDataModule
from bionemo.webdatamodule.utils import pickles_to_tars


@pytest.fixture(scope="module")
def gen_pickle_files(tmp_path_factory):
    dir_pickles = tmp_path_factory.mktemp("pickleddatawds").as_posix()
    prefix_sample = "sample"
    suffix_sample = ["tensor.pyd", "tensor_copy.pyd"]
    n_samples_per_split = 10
    prefixes = []
    # generate the pickles for train, val, and test
    for i in range(n_samples_per_split * 3):
        prefix = f"{prefix_sample}-{i:04}"
        prefixes.append(prefix)
        t = torch.tensor(i, dtype=torch.int32)
        for suffix in suffix_sample:
            with open(f"{dir_pickles}/{prefix}.{suffix}", "wb") as fh:
                pickle.dump(t, fh)
    prefixes_pickle = {
        Split.train: prefixes[0:n_samples_per_split],
        Split.val: prefixes[n_samples_per_split : n_samples_per_split * 2],
        Split.test: prefixes[n_samples_per_split * 2 : n_samples_per_split * 3],
    }
    return (
        dir_pickles,
        prefix_sample,
        suffix_sample,
        prefixes_pickle,
        n_samples_per_split,
    )


@pytest.fixture(scope="module", params=[1, 2])
def gen_test_data(tmp_path_factory, gen_pickle_files, request):
    dir_pickles, prefix_sample, suffixes, prefixes_pickle, n_samples_per_split = gen_pickle_files
    n_suffixes = request.param
    if n_suffixes <= 1:
        suffix_sample = suffixes[0]
    else:
        suffix_sample = suffixes[0:n_suffixes]
    dir_tars_tmp = tmp_path_factory.mktemp("webdatamodule").as_posix()
    dir_tars = {split: f"{dir_tars_tmp}{str(split).split('.')[-1]}" for split in Split}
    prefix_tar = "tensor"
    n_samples = {split: n_samples_per_split for split in Split}
    # generate the tars
    pickles_to_tars(
        dir_pickles,
        prefixes_pickle[Split.train],
        suffix_sample,
        dir_tars[Split.train],
        prefix_tar,
        min_num_shards=3,
    )
    pickles_to_tars(
        dir_pickles,
        prefixes_pickle[Split.val],
        suffix_sample,
        dir_tars[Split.val],
        prefix_tar,
        min_num_shards=3,
    )
    pickles_to_tars(
        dir_pickles,
        prefixes_pickle[Split.test],
        suffix_sample,
        dir_tars[Split.test],
        prefix_tar,
        min_num_shards=3,
    )
    return (
        dir_pickles,
        dir_tars,
        prefix_sample,
        suffix_sample,
        prefix_tar,
        n_samples,
        prefixes_pickle,
    )


def _create_webdatamodule(gen_test_data, num_workers=2):
    (_, dirs_tars_wds, _, suffix_keys_wds, prefix_tars_wds, n_samples, _) = gen_test_data
    local_batch_size = 2
    seed_rng_shfl = 82838392

    batch = batched(local_batch_size, collation_fn=lambda list_samples: torch.vstack(list_samples))

    if isinstance(suffix_keys_wds, str):
        untuple = lambda source: (sample[0] for sample in source)  # noqa: E731
    elif isinstance(suffix_keys_wds, list):
        untuple = lambda source: (torch.vstack(sample) for sample in source)  # noqa: E731

    pipeline_wds = {
        Split.train: [
            untuple,
            shuffle(n_samples[Split.train], rng=random.Random(seed_rng_shfl)),
        ],
        Split.val: untuple,
        Split.test: untuple,
    }

    pipeline_prebatch_wld = {
        Split.train: [
            shuffle(n_samples[Split.train], rng=random.Random(seed_rng_shfl)),
            batch,
        ],
        Split.val: batch,
        Split.test: batch,
    }

    kwargs_wds = {
        split: {
            "shardshuffle": split == Split.train,
            "nodesplitter": wds.split_by_node,
            "seed": seed_rng_shfl,
        }
        for split in Split
    }

    kwargs_wld = {split: {"num_workers": num_workers} for split in Split}

    global_batch_size = 2
    invoke_wld = {
        split: [("with_epoch", {"nbatches": (n_samples[split] + global_batch_size - 1) // global_batch_size})]
        for split in Split
    }

    data_module = WebDataModule(
        suffix_keys_wds,
        dirs_tars_wds,
        prefix_tars_wds=prefix_tars_wds,
        pipeline_wds=pipeline_wds,
        pipeline_prebatch_wld=pipeline_prebatch_wld,
        kwargs_wds=kwargs_wds,
        kwargs_wld=kwargs_wld,
        invoke_wld=invoke_wld,
    )

    return data_module, dirs_tars_wds


@pytest.fixture(scope="module")
def create_webdatamodule(gen_test_data):
    return _create_webdatamodule(gen_test_data)


@pytest.fixture(scope="module")
def create_another_webdatamodule(gen_test_data):
    return _create_webdatamodule(gen_test_data)


@pytest.fixture(scope="module")
def create_webdatamodule_with_5_workers(gen_test_data):
    return _create_webdatamodule(gen_test_data, num_workers=5)


class ModelTestWebDataModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self._model = torch.nn.Linear(1, 1)
        self._samples = {split: [] for split in Split}

    def forward(self, x):
        return self._model(x.float())

    def training_step(self, batch):
        self._samples[Split.train].append(batch)
        loss = self(batch).sum()
        return loss

    def validation_step(self, batch, batch_index):
        self._samples[Split.val].append(batch)
        return torch.zeros(1)

    def test_step(self, batch, batch_index):
        self._samples[Split.test].append(batch)

    def predict_step(self, batch, batch_index):
        self._samples[Split.test].append(batch)
        return torch.zeros(1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer


@pytest.fixture(scope="function")
def create_trainer_and_model():
    trainer = L.Trainer(max_epochs=1, accelerator="gpu", devices=1, val_check_interval=1)
    model = ModelTestWebDataModule()
    return trainer, model


def _create_pickleddatawds(tmp_path_factory, gen_test_data):
    (
        dir_pickles,
        _,
        _,
        suffix_keys_wds,
        prefix_tars_wds,
        n_samples,
        names,
    ) = gen_test_data
    local_batch_size = 2
    seed_rng_shfl = 82838392
    n_tars_wds = 3

    prefix_dir_tars_wds = tmp_path_factory.mktemp("pickleddatawds_tars_wds").as_posix()
    dirs_tars_wds = {s: f"{prefix_dir_tars_wds}{str(s).split('.')[-1]}" for s in Split}

    batch = batched(local_batch_size, collation_fn=lambda list_samples: torch.vstack(list_samples))

    untuple = lambda source: (sample[0] for sample in source)  # noqa: E731

    pipeline_wds = {
        Split.train: [
            untuple,
            shuffle(n_samples[Split.train], rng=random.Random(seed_rng_shfl)),
        ],
        Split.val: untuple,
        Split.test: untuple,
    }

    pipeline_prebatch_wld = {
        Split.train: [
            shuffle(n_samples[Split.train], rng=random.Random(seed_rng_shfl)),
            batch,
        ],
        Split.val: batch,
        Split.test: batch,
    }

    kwargs_wds = {
        split: {
            "shardshuffle": split == Split.train,
            "nodesplitter": wds.split_by_node,
            "seed": seed_rng_shfl,
        }
        for split in Split
    }

    kwargs_wld = {split: {"num_workers": 2} for split in Split}

    global_batch_size = 2
    invoke_wld = {
        split: [("with_epoch", {"nbatches": (n_samples[split] + global_batch_size - 1) // global_batch_size})]
        for split in Split
    }

    data_module = PickledDataWDS(
        dir_pickles,
        names,
        suffix_keys_wds,
        dirs_tars_wds,
        n_tars_wds=n_tars_wds,
        prefix_tars_wds=prefix_tars_wds,
        pipeline_wds=pipeline_wds,
        pipeline_prebatch_wld=pipeline_prebatch_wld,
        kwargs_wds=kwargs_wds,
        kwargs_wld=kwargs_wld,
        invoke_wld=invoke_wld,
    )

    return data_module, dirs_tars_wds, n_tars_wds


@pytest.fixture(scope="module")
def create_pickleddatawds(tmp_path_factory, gen_test_data):
    return _create_pickleddatawds(tmp_path_factory, gen_test_data)


@pytest.fixture(scope="module")
def create_another_pickleddatawds(tmp_path_factory, gen_test_data):
    return _create_pickleddatawds(tmp_path_factory, gen_test_data)
