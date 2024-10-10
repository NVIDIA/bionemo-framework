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


"""
This file tests the data-related utilities for ESM2.
"""

import os

import pytest

from bionemo.data import mapped_dataset
from bionemo.data.preprocess.protein.preprocess import ESM2Preprocess
from bionemo.model.protein.esm1nv.esm1nv_model import ESM2nvModel
from bionemo.model.utils import initialize_distributed_parallel_state, setup_trainer
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import teardown_apex_megatron_cuda


@pytest.fixture(scope="function")
def cfg(config_path_for_tests):
    cfg = load_model_config(config_name="esm2nv_data_test", config_path=config_path_for_tests)
    # Ensure default ESM2 pretraining behavior
    assert cfg.model.data.train.custom_pretraining_fasta_path is None
    assert cfg.model.data.val.custom_pretraining_fasta_path is None
    assert cfg.model.data.test.custom_pretraining_fasta_path is None

    return cfg


@pytest.fixture(scope="function")
def model(cfg) -> ESM2nvModel:
    initialize_distributed_parallel_state()
    trainer = setup_trainer(cfg)
    model = ESM2nvModel(cfg.model, trainer)
    yield model
    teardown_apex_megatron_cuda()


def test_esm2nv_model_creates_train_dataset_with_expected_number_of_samples(model, cfg):
    num_train_samples = 10
    train_dataset = model.build_train_dataset(cfg.model, num_train_samples)

    sample = next(iter(train_dataset))

    assert len(train_dataset) == 10
    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample["sequence"], str)
    assert isinstance(sample["sequence_id"], str)


def test_esm2nv_model_creates_train_dataset_fails_when_num_samples_is_none(model, cfg):
    num_train_samples = None
    cfg.model.data.train.use_upsampling = True
    with pytest.raises(AssertionError):
        model.build_train_dataset(cfg.model, num_train_samples)


def test_esm2nv_model_creates_validation_dataset_with_valid_outputs_given_num_samples_is_none(model, cfg):
    num_val_samples = None
    val_dataset = model.build_val_dataset(cfg.model, num_val_samples)

    sample = next(iter(val_dataset))

    assert len(val_dataset) == 200
    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample["sequence"], str)
    assert isinstance(sample["sequence_id"], str)


def test_esm2nv_model_creates_validation_dataset_with_set_length(model, cfg):
    num_val_samples = 10
    cfg.model.data.val.use_upsampling = True
    val_dataset = model.build_val_dataset(cfg.model, num_val_samples)

    assert len(val_dataset) == num_val_samples
    sample = next(iter(val_dataset))

    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample["sequence"], str)
    assert isinstance(sample["sequence_id"], str)


def test_esm2nv_model_creates_test_dataset_with_valid_outputs(model, cfg):
    cfg.model.data.test.use_upsampling = False
    test_dataset = model.build_test_dataset(cfg.model, None)

    sample = next(iter(test_dataset))
    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample["sequence"], str)
    assert isinstance(sample["sequence_id"], str)


def test_esm2nv_model_creates_test_dataset_with_set_length(model, cfg):
    num_test_samples = 10
    cfg.model.data.test.use_upsampling = True

    test_dataset = model.build_test_dataset(cfg.model, num_test_samples)

    assert len(test_dataset) == num_test_samples

    sample = next(iter(test_dataset))
    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample["sequence"], str)
    assert isinstance(sample["sequence_id"], str)


def test_esm2nv_model_build_train_valid_test_datasets_returns_valid_datasets(model):
    train_ds, val_ds, test_ds = model.build_train_valid_test_datasets()

    assert isinstance(train_ds, mapped_dataset.Uniref90ClusterMappingDataset)
    assert isinstance(test_ds, mapped_dataset.ResamplingMappedDataset)


def test_build_train_valid_test_dataset_limits_train_batches_based_on_max_steps(model):
    model.trainer.fit_loop.epoch_loop.max_steps = 5
    model._cfg.micro_batch_size = 4
    train_ds, _, _ = model.build_train_valid_test_datasets()

    assert len(train_ds) == 20


def test_esm2nv_model_fails_if_test_use_upsampling_false_but_limit_test_batches_not_1(model):
    model.trainer.limit_test_batches = 100
    model._cfg.data.test.use_upsampling = False
    with pytest.raises(ValueError, match=r"config.model.data.test.use_upsampling is False "):
        model.build_train_valid_test_datasets()


def test_esm2nv_model_fails_if_val_use_upsampling_false_but_limit_val_batches_not_1(model):
    model.trainer.limit_val_batches = 100
    model._cfg.data.val.use_upsampling = False
    with pytest.raises(ValueError, match=r"config.model.data.val.use_upsampling"):
        model.build_train_valid_test_datasets()


def test_build_train_valid_test_dataset_limits_val_batches_uses_fraction_of_dataset(model):
    model.trainer.fit_loop.epoch_loop.max_steps = 500
    model.trainer.limit_val_batches = 0.5
    model.trainer.val_check_interval = 1
    model._cfg.data.val.use_upsampling = True
    _, val_ds, _ = model.build_train_valid_test_datasets()
    val_dataset_length = 200
    expected_len_val_ds = (
        (model.trainer.max_steps // model.trainer.val_check_interval + 1)
        * ((val_dataset_length * model.trainer.limit_val_batches) / model._cfg.global_batch_size)
        * model._cfg.global_batch_size
    )
    assert expected_len_val_ds == 50100.0
    assert len(val_ds) == expected_len_val_ds


def test_build_train_valid_test_dataset_limits_test_batches_uses_fraction_of_dataset(model):
    model.trainer.limit_test_batches = 0.5
    model._cfg.data.test.use_upsampling = True
    _, _, test_ds = model.build_train_valid_test_datasets()
    test_dataset_length = 200
    expected_len_test_ds = test_dataset_length * model.trainer.limit_test_batches
    assert expected_len_test_ds == 100
    assert len(test_ds) == expected_len_test_ds


def test_build_train_valid_test_dataset_limits_val_batches_uses_full_dataset(model):
    model.trainer.limit_val_batches = 1.0
    model.trainer.fit_loop.epoch_loop.max_steps = 10
    model.trainer.val_check_interval = 2
    model._cfg.data.val.use_upsampling = True
    _, val_ds, _ = model.build_train_valid_test_datasets()

    val_dataset_length = 200
    expected_len_val_ds = (
        (model.trainer.max_steps // model.trainer.val_check_interval + 1)
        * ((val_dataset_length * model.trainer.limit_val_batches) / model._cfg.global_batch_size)
        * model._cfg.global_batch_size
    )

    assert expected_len_val_ds == 1200.0
    assert len(val_ds) == expected_len_val_ds


def test_build_train_valid_test_dataset_limits_val_batches_int_2(model):
    model.trainer.limit_val_batches = int(2)
    model.trainer.fit_loop.epoch_loop.max_steps = 10
    model.trainer.val_check_interval = 2
    model._cfg.data.val.use_upsampling = True

    _, val_ds, _ = model.build_train_valid_test_datasets()
    expected_len_val_ds = (
        (model.trainer.max_steps // model.trainer.val_check_interval + 1)
        * model._cfg.global_batch_size
        * model.trainer.limit_val_batches
    )

    assert expected_len_val_ds == 24
    assert len(val_ds) == expected_len_val_ds


def test_build_train_valid_test_dataset_limits_val_batches_error_when_zero(model):
    model.trainer.limit_val_batches = 0
    model._cfg.data.val.use_upsampling = True
    with pytest.raises(ValueError, match=r"trainer.limit_val_batches is set to 0"):
        model.build_train_valid_test_datasets()


def test_build_train_valid_test_dataset_throws_error_if_limit_val_batches_but_no_upsampling_in_cfg(model):
    model.trainer.limit_val_batches = 0.5
    model._cfg.data.val.use_upsampling = False

    with pytest.raises(ValueError, match=r"config.model.data.val.use_upsampling"):
        model.build_train_valid_test_datasets()


def test_build_train_valid_test_dataset_limits_test_batches_uses_batch_num_specified_no_scaling(model):
    # Note: The test dataset does not really work at all in any capacity.
    model.trainer.fit_loop.epoch_loop.max_steps = 10
    model.trainer.limit_test_batches = 200
    model._cfg.data.test.use_upsampling = True
    _, _, test_ds = model.build_train_valid_test_datasets()

    assert len(test_ds) == model._cfg.global_batch_size * model.trainer.limit_test_batches
    assert len(test_ds) == 400


def test_esm2nv_preprocess_training_dataset_creates_non_empty_dirs(tmp_path, cfg):
    preprocessor = ESM2Preprocess()

    preprocessor.prepare_dataset(
        uf50_datapath=cfg.model.data.train.uf50_datapath,
        uf90_datapath=cfg.model.data.train.uf90_datapath,
        cluster_mapping_tsv=cfg.model.data.train.cluster_mapping_tsv,
        uf50_output_dir=tmp_path / "uf50_output_dir_train",
        uf90_output_dir=tmp_path / "uf90_output_dir_train",
        sort_fastas=cfg.model.data.train.sort_fastas,
        mode="train",
    )

    # Make sure the resulting directory is non empty.
    assert os.listdir(tmp_path / "uf50_output_dir_train")
    assert os.listdir(tmp_path / "uf90_output_dir_train")


def test_esm2nv_preprocess_val_dataset_creates_non_empty_dirs(tmp_path, cfg):
    preprocessor = ESM2Preprocess()
    preprocessor.prepare_dataset(
        uf50_datapath=cfg.model.data.val.uf50_datapath,
        uf50_output_dir=tmp_path / "uf50_output_dir_val",
        sort_fastas=False,
        mode="val",
    )
    assert os.listdir(tmp_path / "uf50_output_dir_val")


def test_esm2nv_preprocess_test_dataset_creates_non_empty_dirs(tmp_path, cfg):
    preprocessor = ESM2Preprocess()
    preprocessor.prepare_dataset(
        uf50_datapath=cfg.model.data.test.uf50_datapath,
        uf50_output_dir=tmp_path / "uf50_output_dir_test",
        sort_fastas=False,
        mode="test",
    )
    assert os.listdir(tmp_path / "uf50_output_dir_test")
