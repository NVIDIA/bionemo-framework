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
import shutil
from pathlib import Path

import pytest
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import CSVFieldsMemmapDataset
from omegaconf import DictConfig

from bionemo.data import mapped_dataset
from bionemo.data.preprocess.protein.preprocess import FastaPreprocess
from bionemo.model.protein.esm1nv.esm1nv_model import ESM2nvModel
from bionemo.model.utils import initialize_distributed_parallel_state, setup_trainer
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import teardown_apex_megatron_cuda


CUSTOM_PRETRAINING_TRAIN_FASTA_PATH = (
    Path(os.environ["BIONEMO_HOME"]) / "examples/tests/test_data/protein/esm2nv/example_train.fasta"
)
CUSTOM_PRETRAINING_VAL_FASTA_PATH = (
    Path(os.environ["BIONEMO_HOME"]) / "examples/tests/test_data/protein/esm2nv/example_val.fasta"
)
CUSTOM_PRETRAINING_TEST_FASTA_PATH = (
    Path(os.environ["BIONEMO_HOME"]) / "examples/tests/test_data/protein/esm2nv/example_test.fasta"
)

NUM_EXAMPLE_SEQUENCES = 100
NUM_OUTPUT_SEQUENCES = 50
NUM_CSV_FILES = 50  # must match with cfg.model.val.data.range, etc.

assert NUM_EXAMPLE_SEQUENCES >= NUM_OUTPUT_SEQUENCES and NUM_OUTPUT_SEQUENCES > 1


@pytest.fixture(scope="function")
def cfg(tmp_path: str, config_path_for_tests: str) -> DictConfig:
    """Create esm2nv config with dataset_path and custom_pretraining_fasta_path overriden.
    custom_pretraining_fasta_path will be overriden by CUSTOM_PRETRAINING_FASTA_PATH.

    Args:
        tmp_path (str): Temporary directory to override dataset_path in train, val and test. The actual output directory will be at '{temp_path}/{mode}'.
        config_path_for_tests (str): Defauly testing config path.

    Returns:
        (DictConfig): Overriden test config
    """
    cfg = load_model_config(config_name="esm2nv_data_test", config_path=config_path_for_tests)

    # avoid overwriting index files
    target_custom_pretraining_train_fasta_path = tmp_path / os.path.basename(CUSTOM_PRETRAINING_TRAIN_FASTA_PATH)
    shutil.copy(CUSTOM_PRETRAINING_TRAIN_FASTA_PATH, target_custom_pretraining_train_fasta_path)
    cfg.model.data.train.custom_pretraining_fasta_path = target_custom_pretraining_train_fasta_path

    target_custom_pretraining_val_fasta_path = tmp_path / os.path.basename(CUSTOM_PRETRAINING_VAL_FASTA_PATH)
    shutil.copy(CUSTOM_PRETRAINING_VAL_FASTA_PATH, target_custom_pretraining_val_fasta_path)
    cfg.model.data.val.custom_pretraining_fasta_path = target_custom_pretraining_val_fasta_path

    target_custom_pretraining_test_fasta_path = tmp_path / os.path.basename(CUSTOM_PRETRAINING_TEST_FASTA_PATH)
    shutil.copy(CUSTOM_PRETRAINING_TEST_FASTA_PATH, target_custom_pretraining_test_fasta_path)
    cfg.model.data.test.custom_pretraining_fasta_path = target_custom_pretraining_test_fasta_path

    cfg.model.data.train.dataset_path = tmp_path
    cfg.model.data.val.dataset_path = tmp_path
    cfg.model.data.test.dataset_path = tmp_path

    assert cfg.model.data.train.use_upsampling is True
    assert cfg.model.data.val.use_upsampling is False
    assert cfg.model.data.test.use_upsampling is True

    return cfg


@pytest.fixture(scope="function")
def model(cfg: DictConfig) -> ESM2nvModel:
    """Yield ESM2nvModel model from config.

    Args:
        cfg (ConfigDict): ESM2 config.

    Yields:
        (ESM2nvModel): ESM2 model instance.
    """
    initialize_distributed_parallel_state()
    trainer = setup_trainer(cfg)
    model = ESM2nvModel(cfg.model, trainer)
    yield model
    teardown_apex_megatron_cuda()


def preprocess_fasta_dataset(root_directory: str, cfg: DictConfig, mode: str, num_csv_files: int) -> None:
    """Preprocess dataset with FastaPreprocess.

    Args:
        root_directory (str): Root directory to preprocessed dataset. The actual output directory will be at '{root_directory}/{mode}'.
        cfg (DictConfig): Configuration in full.
        mode (str): Options are 'train', 'val' and 'test'.
        num_csv_files (int): Number of csv file(s).
    """
    assert mode in ["train", "val", "test"]

    preprocessor = FastaPreprocess(root_directory=root_directory)
    custom_pretraining_fasta_path = cfg.model.data[mode].custom_pretraining_fasta_path

    preprocessor.prepare_dataset(
        fasta_path=custom_pretraining_fasta_path,
        mode=mode,
        num_csv_files=num_csv_files,
    )


def preprocess_fasta_train_val_test_dataset(root_directory: str, cfg: DictConfig, num_csv_files: int) -> None:
    """Preprocess train, val and test datasets with FastaPreprocess.

    Args:
        root_directory (str): Root directory to preprocessed dataset. The actual output directory will be at '{root_directory}/{mode}'.
        cfg (DictConfig): Configuration in full.
        num_csv_files (int): Number of csv file(s).
    """
    for mode in ["train", "val", "test"]:
        preprocess_fasta_dataset(
            root_directory=root_directory,
            cfg=cfg,
            num_csv_files=num_csv_files,
            mode=mode,
        )


# TODO [sichu] Unknown model leakage on CI to other tests unless marked as skip.
def test_fasta_preprocess_training_dataset_creates_non_empty_dirs(tmp_path: str, cfg: DictConfig):
    """Test whether FastaPreprocess creates the correct number of csv files in train dataset.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        mode="train",
        num_csv_files=NUM_CSV_FILES,
    )
    # info: empty csv when NUM_CSV_FILES > NUM_EXAMPLE_SEQUENCES
    assert len(os.listdir(tmp_path / "train")) == NUM_CSV_FILES


def test_fasta_preprocess_val_dataset_creates_non_empty_dirs(tmp_path: str, cfg: DictConfig):
    """Test whether FastaPreprocess creates the correct number of csv files in val dataset.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        mode="val",
        num_csv_files=NUM_CSV_FILES,
    )
    # info: empty csv when NUM_CSV_FILES > NUM_EXAMPLE_SEQUENCES
    assert len(os.listdir(tmp_path / "val")) == NUM_CSV_FILES


def test_fasta_preprocess_test_dataset_creates_non_empty_dirs(tmp_path: str, cfg: DictConfig):
    """Test whether FastaPreprocess creates the correct number of csv files in test dataset.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        mode="test",
        num_csv_files=NUM_CSV_FILES,
    )
    # info: empty csv when NUM_CSV_FILES > NUM_EXAMPLE_SEQUENCES
    assert len(os.listdir(tmp_path / "test")) == NUM_CSV_FILES


def test_esm2nv_model_creates_train_dataset_with_expected_number_of_samples(
    tmp_path: str, model: ESM2nvModel, cfg: DictConfig
):
    """Test train dataset length and types of samples from ESM2nvModel when upsampling with expected number of samples.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        mode="train",
        num_csv_files=NUM_CSV_FILES,
    )

    # build training dataset in model
    train_dataset = model.build_train_dataset(
        cfg.model, num_samples=NUM_OUTPUT_SEQUENCES
    )  # restrict to subset of sequences

    sample = next(iter(train_dataset))

    assert len(train_dataset) == NUM_OUTPUT_SEQUENCES
    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample["sequence"], str)
    assert isinstance(sample["sequence_id"], str)


def test_esm2nv_model_creates_train_dataset_with_valid_outputs_given_num_samples_is_none(
    tmp_path: str, model: ESM2nvModel, cfg: DictConfig
):
    """Test train dataset length and types of samples from ESM2nvModel without expected number of samples.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        mode="train",
        num_csv_files=NUM_CSV_FILES,
    )

    # build training dataset in model
    cfg.model.data.train.use_upsampling = False
    train_dataset = model.build_train_dataset(cfg.model, num_samples=None)

    sample = next(iter(train_dataset))

    assert len(train_dataset) == NUM_EXAMPLE_SEQUENCES
    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample["sequence"], str)
    assert isinstance(sample["sequence_id"], str)


def test_esm2nv_model_creates_train_dataset_fails_when_num_samples_is_none_with_upsampling(
    tmp_path: str, model: ESM2nvModel, cfg: DictConfig
):
    """Test whether exception is raised to mandate num_samples is passed when upsampling.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    cfg.model.data.train.use_upsampling = True
    preprocess_fasta_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        mode="train",
        num_csv_files=NUM_CSV_FILES,
    )

    with pytest.raises(AssertionError):
        model.build_train_dataset(cfg.model, num_samples=None)


# TODO [sichu] model leakage starting from including test functions here onwards
@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_esm2nv_model_creates_validation_dataset_with_valid_outputs_given_num_samples_is_none(
    tmp_path: str, model: ESM2nvModel, cfg: DictConfig
):
    """Test val dataset length and types of samples from ESM2nvModel without expected number of samples.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        mode="val",
        num_csv_files=NUM_CSV_FILES,
    )

    num_val_samples = None
    val_dataset = model.build_val_dataset(cfg.model, num_val_samples)

    sample = next(iter(val_dataset))

    assert len(val_dataset) == NUM_EXAMPLE_SEQUENCES
    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample["sequence"], str)
    assert isinstance(sample["sequence_id"], str)


@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_esm2nv_model_creates_validation_dataset_with_set_length(tmp_path: str, model: ESM2nvModel, cfg: DictConfig):
    """Test val dataset length and types of samples from ESM2nvModel when upsampling with expected number of samples.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        mode="val",
        num_csv_files=NUM_CSV_FILES,
    )

    cfg.model.data.val.use_upsampling = True
    val_dataset = model.build_val_dataset(cfg.model, num_samples=NUM_OUTPUT_SEQUENCES)

    assert len(val_dataset) == NUM_OUTPUT_SEQUENCES
    sample = next(iter(val_dataset))

    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample["sequence"], str)
    assert isinstance(sample["sequence_id"], str)


@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_esm2nv_model_creates_test_dataset_with_valid_outputs(tmp_path: str, model: ESM2nvModel, cfg: DictConfig):
    """Test test dataset length and types of samples from ESM2nvModel without expected number of samples.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        mode="test",
        num_csv_files=NUM_CSV_FILES,
    )

    cfg.model.data.test.use_upsampling = False
    test_dataset = model.build_test_dataset(cfg.model, None)

    sample = next(iter(test_dataset))
    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample["sequence"], str)
    assert isinstance(sample["sequence_id"], str)


@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_esm2nv_model_creates_test_dataset_with_set_length(tmp_path: str, model: ESM2nvModel, cfg: DictConfig):
    """Test test dataset length and types of samples from ESM2nvModel when upsampling with expected number of samples.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        mode="test",
        num_csv_files=NUM_CSV_FILES,
    )
    cfg.model.data.test.use_upsampling = True

    test_dataset = model.build_test_dataset(cfg.model, num_samples=NUM_OUTPUT_SEQUENCES)

    assert len(test_dataset) == NUM_OUTPUT_SEQUENCES

    sample = next(iter(test_dataset))
    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample["sequence"], str)
    assert isinstance(sample["sequence_id"], str)


@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_esm2nv_model_build_train_valid_test_datasets_returns_valid_train_dataset(
    tmp_path: str, model: ESM2nvModel, cfg: DictConfig
):
    """Test type of train dataset from ESM2nvModel.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_train_val_test_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        num_csv_files=NUM_CSV_FILES,
    )
    train_ds, val_ds, test_ds = model.build_train_valid_test_datasets()
    assert isinstance(
        train_ds, mapped_dataset.ResamplingMappedDataset
    )  # already checked to be on manually in cfg fixture


@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_esm2nv_model_build_train_valid_test_datasets_returns_valid_val_dataset(
    tmp_path: str, model: ESM2nvModel, cfg: DictConfig
):
    """Test type of val dataset from ESM2nvModel.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_train_val_test_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        num_csv_files=NUM_CSV_FILES,
    )
    train_ds, val_ds, test_ds = model.build_train_valid_test_datasets()
    assert isinstance(
        val_ds, CSVFieldsMemmapDataset
    )  # warning: assume cfg.model.data.val.data_impl == 'csv_fields_mmap'


@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_esm2nv_model_build_train_valid_test_datasets_returns_valid_test_dataset(
    tmp_path: str, model: ESM2nvModel, cfg: DictConfig
):
    """Test type of test dataset from ESM2nvModel.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_train_val_test_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        num_csv_files=NUM_CSV_FILES,
    )
    train_ds, val_ds, test_ds = model.build_train_valid_test_datasets()
    isinstance(test_ds, mapped_dataset.MappedDataset)


@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_build_train_valid_test_dataset_limits_val_batches_uses_fraction_of_dataset(
    tmp_path: str, model: ESM2nvModel, cfg: DictConfig
):
    """Test val dataset length under resampling when limiting val batches to 0.5.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_train_val_test_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        num_csv_files=NUM_CSV_FILES,
    )

    model.trainer.fit_loop.epoch_loop.max_steps = 500
    model.trainer.limit_val_batches = 0.5
    model.trainer.val_check_interval = 1
    model._cfg.data.val.use_upsampling = True
    _, val_ds, _ = model.build_train_valid_test_datasets()
    val_dataset_length = NUM_EXAMPLE_SEQUENCES
    expected_len_val_ds = (
        (model.trainer.max_steps // model.trainer.val_check_interval + 1)
        * ((val_dataset_length * model.trainer.limit_val_batches) / model._cfg.global_batch_size)
        * model._cfg.global_batch_size
    )
    assert len(val_ds) == expected_len_val_ds


@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_build_train_valid_test_dataset_limits_test_batches_uses_fraction_of_dataset(
    tmp_path: str, model: ESM2nvModel, cfg: DictConfig
):
    """Test test dataset length under resampling when limiting test batches to 0.5.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_train_val_test_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        num_csv_files=NUM_CSV_FILES,
    )

    model.trainer.limit_test_batches = 0.5
    model._cfg.data.test.use_upsampling = True
    _, _, test_ds = model.build_train_valid_test_datasets()
    test_dataset_length = NUM_EXAMPLE_SEQUENCES
    expected_len_test_ds = test_dataset_length * model.trainer.limit_test_batches
    assert len(test_ds) == expected_len_test_ds


@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_build_train_valid_test_dataset_limits_val_batches_uses_full_dataset(
    tmp_path: str, model: ESM2nvModel, cfg: DictConfig
):
    """Test val dataset length under resampling when limiting val batches to 1.0.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_train_val_test_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        num_csv_files=NUM_CSV_FILES,
    )

    model.trainer.limit_val_batches = 1.0
    model.trainer.fit_loop.epoch_loop.max_steps = 10
    model.trainer.val_check_interval = 2
    model._cfg.data.val.use_upsampling = True
    _, val_ds, _ = model.build_train_valid_test_datasets()

    val_dataset_length = NUM_EXAMPLE_SEQUENCES
    expected_len_val_ds = (
        (model.trainer.max_steps // model.trainer.val_check_interval + 1)
        * ((val_dataset_length * model.trainer.limit_val_batches) / model._cfg.global_batch_size)
        * model._cfg.global_batch_size
    )
    assert len(val_ds) == expected_len_val_ds


@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_build_train_valid_test_dataset_limits_val_batches_int_2(tmp_path: str, model: ESM2nvModel, cfg: DictConfig):
    """Test val dataset length under resampling when limiting val batches to int(2).

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_train_val_test_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        num_csv_files=NUM_CSV_FILES,
    )

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


@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_build_train_valid_test_dataset_limits_val_batches_error_when_zero(
    tmp_path: str, model: ESM2nvModel, cfg: DictConfig
):
    """Test val dataset creation under resampling when limiting val batches to 0 to mandate raising ValueError.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_train_val_test_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        num_csv_files=NUM_CSV_FILES,
    )

    model.trainer.limit_val_batches = 0
    model._cfg.data.val.use_upsampling = True
    with pytest.raises(ValueError, match=r"trainer.limit_val_batches is set to 0"):
        model.build_train_valid_test_datasets()


@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_build_train_valid_test_dataset_throws_error_if_limit_val_batches_but_no_upsampling_in_cfg(
    tmp_path: str, model: ESM2nvModel, cfg: DictConfig
):
    """Test val dataset creation without resampling when limiting val batches to 0 to mandate raising ValueError.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_train_val_test_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        num_csv_files=NUM_CSV_FILES,
    )

    model.trainer.limit_val_batches = 0.5
    model._cfg.data.val.use_upsampling = False

    with pytest.raises(ValueError, match=r"config.model.data.val.use_upsampling"):
        model.build_train_valid_test_datasets()


@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_build_train_valid_test_dataset_limits_train_batches_based_on_max_steps(
    tmp_path: str, model: ESM2nvModel, cfg: DictConfig
):
    """Test train dataset length by overriding max_steps and micro_batch_size.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_train_val_test_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        num_csv_files=NUM_CSV_FILES,
    )

    model.trainer.fit_loop.epoch_loop.max_steps = 5
    model._cfg.micro_batch_size = 4
    train_ds, _, _ = model.build_train_valid_test_datasets()

    assert len(train_ds) == 20


@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_esm2nv_model_fails_if_test_use_upsampling_false_but_limit_test_batches_not_1(
    tmp_path: str, model: ESM2nvModel, cfg: DictConfig
):
    """Test exception raised when forcing test batches larger than provided without upsampling.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_train_val_test_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        num_csv_files=NUM_CSV_FILES,
    )

    model.trainer.limit_test_batches = 100
    model._cfg.data.test.use_upsampling = False
    with pytest.raises(ValueError, match=r"config.model.data.test.use_upsampling is False "):
        model.build_train_valid_test_datasets()


@pytest.mark.skip(reason="unknown model leakage to other tests")
def test_esm2nv_model_fails_if_val_use_upsampling_false_but_limit_val_batches_not_1(
    tmp_path: str, model: ESM2nvModel, cfg: DictConfig
):
    """Test exception raised when forcing val batches larger than provided without upsampling.

    Args:
        tmp_path (str): The actual output directory will be at '{temp_path}/{mode}'.
        model (ESM2nvModel): ESM2 model created from ESM2 config.
        cfg (ConfigDict): ESM2 config.
    """
    preprocess_fasta_train_val_test_dataset(
        root_directory=tmp_path,
        cfg=cfg,
        num_csv_files=NUM_CSV_FILES,
    )

    model.trainer.limit_val_batches = 100
    model._cfg.data.val.use_upsampling = False
    with pytest.raises(ValueError, match=r"config.model.data.val.use_upsampling"):
        model.build_train_valid_test_datasets()
