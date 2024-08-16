# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from enum import Enum, auto

import pytest

import torch
import lightning as L

from bionemo.core.data.datamodule import Split


@pytest.mark.parametrize("split", list(Split))
def test_webdatamodule_init(split, create_webdatamodule):
    data_module, prefix_dir_tars_wds = create_webdatamodule
    assert data_module._n_samples[split] == 10, (
        f"Wrong {split}-set size: "
        f"expected 10 "
        f"but got {data_module._n_samples[split]}"
    )
    assert data_module._dirs_tars_wds[split] == f"{prefix_dir_tars_wds}", (
        f"Wrong tar files directory: "
        f"expected {prefix_dir_tars_wds} "
        f"but got {data_module._dirs_tars_wds[split]}"
    )


@pytest.mark.parametrize("split", list(Split))
def test_webdatamodule_setup_dataset(split, create_webdatamodule,
                                     create_another_webdatamodule):
    data_modules = [create_webdatamodule[0], create_another_webdatamodule[0]]
    lists_tensors = []
    for m in data_modules:
        m.prepare_data()
        # run through all the possible stages first to setup all the correps.
        # dataset objects
        m.setup("fit")
        m.setup("test")
        L.seed_everything(2823828)
        tensors= []
        for sample in m._dataset[split]:
            assert isinstance(sample, torch.Tensor),\
                "Sample yield from dataset is not tensor"
            tensors.append(sample)
        lists_tensors.append(tensors)

    assert len(lists_tensors[0]) > 0, "No names in {split} dataset"
    torch.testing.assert_close(torch.vstack(lists_tensors[0]),
                               torch.vstack(lists_tensors[1]))


@pytest.mark.parametrize("split", list(Split))
def test_webdatamodule_setup_dataloader(split, create_webdatamodule,
                                        create_another_webdatamodule):
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
            assert isinstance(samples, torch.Tensor),\
                "Sample object is not torch.Tensor"
            tensors.append(samples)
        lists_tensors.append(tensors)

    assert len(lists_tensors[0]) > 0, "No names in {split} dataloader"
    torch.testing.assert_close(torch.vstack(lists_tensors[0]),
                               torch.vstack(lists_tensors[1]))


class Stage(Enum):
    fit = auto()
    validate = auto()
    test = auto()
    predict = auto()


@pytest.mark.parametrize("stage", list(Stage))
def test_webdatamodule_in_lightning(stage, create_webdatamodule,
                                 create_another_webdatamodule,
                                 create_trainer_and_model):
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
    samples = [ sample.name for sample in loader ]
    L.seed_everything(2823828)
    workflow = getattr(trainer, name_stage)
    workflow(model, data_modules[1])
    assert model._samples[split] == samples


@pytest.mark.parametrize("split", list(Split))
def test_pickleddatawds_init(split, create_pickleddatawds):
    data_module, prefix_dir_tars_wds = create_pickleddatawds
    assert data_module._n_samples[split] == 10, (
        f"Wrong {split}-set size: "
        f"expected 10 "
        f"but got {data_module._n_samples[split]}"
    )
    name_split = str(split).split(".")[-1]
    assert data_module._dirs_tars_wds[split] == f"{prefix_dir_tars_wds}{name_split}", (
        f"Wrong tar files directory: "
        f"expected {prefix_dir_tars_wds}{name_split} "
        f"but got {data_module._dirs_tars_wds[split]}"
    )

@pytest.mark.parametrize("split", list(Split))
def test_pickleddatawds_setup_dataset(split, create_pickleddatawds,
                                      create_another_pickleddatawds):
    data_modules = [create_pickleddatawds[0], create_another_pickleddatawds[0]]
    lists_tensors = []
    for m in data_modules:
        m.prepare_data()
        # run through all the possible stages first to setup all the correps.
        # dataset objects
        m.setup("fit")
        m.setup("test")
        L.seed_everything(2823828)
        tensors= []
        for sample in m._dataset[split]:
            assert isinstance(sample, torch.Tensor),\
                "Sample yield from dataset is not tensor"
            tensors.append(sample)
        lists_tensors.append(tensors)

    assert len(lists_tensors[0]) > 0, "No names in {split} dataset"
    torch.testing.assert_close(torch.vstack(lists_tensors[0]),
                               torch.vstack(lists_tensors[1]))
