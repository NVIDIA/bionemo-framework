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
from enum import Enum, auto

import lightning as L
import pytest
import torch

from bionemo.webdatamodule.datamodule import Split


@pytest.mark.parametrize("split", list(Split))
def test_webdatamodule_init(split, create_webdatamodule):
    data_module, dirs_tars_wds = create_webdatamodule
    assert data_module._dirs_tars_wds[split] == f"{dirs_tars_wds[split]}", (
        f"Wrong tar files directory: "
        f"expected {dirs_tars_wds[split]} "
        f"but got {data_module._dirs_tars_wds[split]}"
    )


@pytest.mark.parametrize("split", list(Split))
def test_webdatamodule_setup_dataset(split, create_webdatamodule, create_another_webdatamodule):
    data_modules = [create_webdatamodule[0], create_another_webdatamodule[0]]
    lists_tensors = []
    for m in data_modules:
        m.prepare_data()
        # run through all the possible stages first to setup all the correps.
        # dataset objects
        m.setup("fit")
        m.setup("test")
        L.seed_everything(2823828)
        tensors = []
        for sample in m._dataset[split]:
            assert isinstance(sample, torch.Tensor), "Sample yield from dataset is not tensor"
            tensors.append(sample)
        lists_tensors.append(tensors)

    assert len(lists_tensors[0]) > 0, "No names in {split} dataset"
    torch.testing.assert_close(torch.vstack(lists_tensors[0]), torch.vstack(lists_tensors[1]))


@pytest.mark.parametrize("split", list(Split))
def test_webdatamodule_setup_dataloader(split, create_webdatamodule, create_another_webdatamodule):
    data_modules = [create_webdatamodule[0], create_another_webdatamodule[0]]
    lists_tensors = []
    for m in data_modules:
        m.prepare_data()
        # run through all the possible stages first to setup all the correps.
        # dataset objects
        m.setup("fit")
        m.setup("test")
        L.seed_everything(2823828)
        tensors = []
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
            assert isinstance(samples, torch.Tensor), "Sample object is not torch.Tensor"
            tensors.append(samples)
        lists_tensors.append(tensors)

    assert len(lists_tensors[0]) > 0, "No names in {split} dataloader"
    torch.testing.assert_close(torch.vstack(lists_tensors[0]), torch.vstack(lists_tensors[1]))


@pytest.mark.parametrize("split", list(Split))
def test_webdatamodule_throw_on_many_workers(split, create_webdatamodule_with_5_workers):
    data_module = create_webdatamodule_with_5_workers[0]
    urls = glob.glob(f"{data_module._dirs_tars_wds[split]}/" f"{data_module._prefix_tars_wds}-*.tar")
    n_tars = len(urls)
    data_module._kwargs_wld[split]["num_workers"] = n_tars + 1
    data_module.prepare_data()
    data_module.setup("fit")
    data_module.setup("test")
    loader = None
    if split == Split.train:
        loader = data_module.train_dataloader()
    elif split == Split.val:
        loader = data_module.val_dataloader()
    elif split == Split.test:
        loader = data_module.test_dataloader()
    else:
        raise RuntimeError(f"Test for split {split} not implemented")
    assert loader is not None, "dataloader not instantated"
    try:
        for _ in loader:
            pass
    except ValueError as e:
        # this is expected
        assert "have fewer shards than workers" in str(e), (
            f"'have fewer shards than workers' not found in exception " f"raised from data loading: {e}"
        )
    except Exception as e:
        raise RuntimeError(
            f"WebLoader doesn't raise ValueError with fewer " f"shards than workers but raise this instead: {e}"
        )
    else:
        raise NotImplementedError(
            "WebLoader doesn't throw error with num_workers > num_shards "
            "User should report this issue to webdataset and create "
            "less shards than workers in practice as a workaround"
        )


class Stage(Enum):
    fit = auto()
    validate = auto()
    test = auto()
    predict = auto()


@pytest.mark.parametrize("stage", list(Stage))
def test_webdatamodule_in_lightning(
    stage, create_webdatamodule, create_another_webdatamodule, create_trainer_and_model
):
    data_modules = [create_webdatamodule[0], create_another_webdatamodule[0]]
    trainer, model = create_trainer_and_model
    # get the list of samples from the loader
    L.seed_everything(2823828)
    data_modules[0].prepare_data()
    split = None
    if stage == Stage.fit:
        split = Split.train
    elif stage == Stage.validate:
        split = Split.val
    elif stage == Stage.test or stage == Stage.predict:
        split = Split.test
    else:
        raise RuntimeError(f"{stage} stage not implemented")
    name_stage = str(stage).split(".")[-1]
    data_modules[0].setup(name_stage)
    # get the list of samples from the workflow
    get_dataloader = getattr(data_modules[0], f"{str(split).split('.')[-1]}_dataloader")
    loader = get_dataloader()
    L.seed_everything(2823828)
    workflow = getattr(trainer, name_stage)
    workflow(model, data_modules[1])
    device = model._samples[split][0].device
    samples = [sample.to(device=device) for sample in loader]
    torch.testing.assert_close(torch.stack(model._samples[split], dim=0), torch.stack(samples, dim=0))


@pytest.mark.parametrize("split", list(Split))
def test_pickleddatawds_init(split, create_pickleddatawds):
    data_module, dirs_tars_wds, _ = create_pickleddatawds
    assert data_module._dirs_tars_wds[split] == dirs_tars_wds[split], (
        f"Wrong tar files directory: "
        f"expected {dirs_tars_wds[split]} "
        f"but got {data_module._dirs_tars_wds[split]}"
    )


@pytest.mark.parametrize("split", list(Split))
def test_pickleddatawds_prepare_data(split, create_pickleddatawds):
    data_module, _, n_tars_min = create_pickleddatawds
    data_module.prepare_data()
    dir_tars = f"{data_module._dirs_tars_wds[split]}"
    tars = glob.glob(f"{dir_tars}/{data_module._prefix_tars_wds}-*.tar")
    n_tars = len(tars)
    assert n_tars_min <= n_tars and n_tars <= n_tars_min + 1, (
        f"Number of tar files: {n_tars} in {dir_tars} is outside the range " f"[{n_tars_min}, {n_tars_min + 1}]"
    )


@pytest.mark.parametrize("split", list(Split))
def test_pickleddatawds_setup_dataset(split, create_pickleddatawds, create_another_pickleddatawds):
    data_modules = [create_pickleddatawds[0], create_another_pickleddatawds[0]]
    lists_tensors = []
    for m in data_modules:
        m.prepare_data()
        # run through all the possible stages first to setup all the correps.
        # dataset objects
        m.setup("fit")
        m.setup("test")
        L.seed_everything(2823828)
        tensors = []
        for sample in m._dataset[split]:
            assert isinstance(sample, torch.Tensor), "Sample yield from dataset is not tensor"
            tensors.append(sample)
        lists_tensors.append(tensors)

    assert len(lists_tensors[0]) > 0, "No names in {split} dataset"
    torch.testing.assert_close(torch.vstack(lists_tensors[0]), torch.vstack(lists_tensors[1]))


def test_pickleddatawds_sample_overlap(create_pickleddatawds):
    data_module = create_pickleddatawds[0]
    # this writes the tar files to disk
    data_module.prepare_data()
    # read the data back by setting up the dataset object and loop over it
    data_module.setup("fit")
    data_module.setup("test")
    results = {split: {sample.item() for sample in data_module._dataset[split]} for split in Split}
    overlap_train_val = results[Split.train] & results[Split.val]
    overlap_train_test = results[Split.train] & results[Split.test]
    overlap_val_test = results[Split.val] & results[Split.test]
    assert len(overlap_train_val) == 0, "Shared samples found between train and val datasets"
    assert len(overlap_train_test) == 0, "Shared samples found between train and test datasets"
    assert len(overlap_val_test) == 0, "Shared samples found between val and test datasets"
