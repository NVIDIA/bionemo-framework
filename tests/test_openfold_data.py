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
import tempfile
from typing import Dict

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import bionemo.data.protein.openfold.residue_constants as rc
from bionemo.data.preprocess.protein.postprocess import OpenFoldSampleCreator
from bionemo.data.protein.openfold.datahub import (
    get_structured_paths,
)
from bionemo.data.protein.openfold.datasets import (
    FinetuningDataset,
    InitialTrainingDataset,
    PredictDataset,
    SelfDistillationDataset,
    ValidationDataset,
)
from bionemo.data.protein.openfold.features import create_sequence_features
from bionemo.data.protein.openfold.helpers import collate
from bionemo.model.protein.openfold.openfold_model import AlphaFold
from bionemo.model.protein.openfold.schema import FEATURE_SHAPES
from bionemo.model.utils import setup_trainer
from bionemo.utils.hydra import load_model_config


BIONEMO_HOME = os.getenv("BIONEMO_HOME")
EXAMPLE_CONFIG_PATH = os.path.join(BIONEMO_HOME, "examples/protein/openfold/conf")
TEST_DATA_PATH = os.path.join(BIONEMO_HOME, "examples/tests/test_data")
SAMPLE_DATA_PATH = os.path.join(TEST_DATA_PATH, "protein", "openfold")
SAMPLE_INFER_DATA_PATH = os.path.join(SAMPLE_DATA_PATH, "inference")

TRAINING_DATA_VARIANT = "processed"
TEST_SAMPLE_VARIANT = "processed_test"
SAMPLE_PDB_CHAIN_IDS = [
    # 2 chains from date range: ("2018-01-01", "2020-12-31")
    "5vf6_A",
    "5wka_D",
]
SAMPLE_CAMEO_CHAIN_IDS = [
    # 1 chain for CAMEO target non-overlapping with training date range
    "7f17_B",  # 2021-10-23
]
SAMPLE_UNICLUST30_IDS = [
    # the first 2 uniclust30 ids
    "A0A009JVE4",
    "A0A010PP53",
]

SEQUENCE_FEATURE_NAMES = {
    "aatype",
    "residue_index",
    "seq_length",
    "bert_mask",
    "atom14_atom_exists",
    "residx_atom14_to_atom37",
    "residx_atom37_to_atom14",
    "atom37_atom_exists",
}
MSA_FEATURE_NAMES = {
    "msa_mask",
    "msa_row_mask",
    "msa_feat",
    "true_msa",
    "seq_mask",
    "target_feat",
}
EXTRA_MSA_FEATURE_NAMES = {
    "extra_msa",
    "extra_msa_mask",
    "extra_msa_row_mask",
    "extra_has_deletion",
    "extra_deletion_value",
}
TEMPLATE_FEATURE_NAMES = {
    "template_aatype",
    "template_all_atom_positions",
    "template_all_atom_mask",
    "template_sum_probs",
    "template_mask",
    "template_pseudo_beta",
    "template_pseudo_beta_mask",
    "template_torsion_angles_sin_cos",
    "template_alt_torsion_angles_sin_cos",
    "template_torsion_angles_mask",
}
TARGET_FEATURE_NAMES = {
    "all_atom_positions",
    "all_atom_mask",
    "resolution",
    "is_distillation",
    "atom14_gt_exists",
    "atom14_gt_positions",
    "atom14_alt_gt_positions",
    "atom14_alt_gt_exists",
    "atom14_atom_is_ambiguous",
    "rigidgroups_gt_frames",
    "rigidgroups_gt_exists",
    "rigidgroups_group_exists",
    "rigidgroups_group_is_ambiguous",
    "rigidgroups_alt_gt_frames",
    "pseudo_beta",
    "pseudo_beta_mask",
    "backbone_rigid_tensor",
    "backbone_rigid_mask",
    "chi_angles_sin_cos",
    "chi_mask",
    "use_clamped_fape",
    "id",
}

PREDICT_FEATURE_NAMES = {
    "seq_index",
    "seq_name",
    *SEQUENCE_FEATURE_NAMES,
    *MSA_FEATURE_NAMES,
    *EXTRA_MSA_FEATURE_NAMES,
    *TEMPLATE_FEATURE_NAMES,
}

INITIAL_TRAINING_FEATURE_NAMES = {
    *TARGET_FEATURE_NAMES,
    *SEQUENCE_FEATURE_NAMES,
    *MSA_FEATURE_NAMES,
    *EXTRA_MSA_FEATURE_NAMES,
    *TEMPLATE_FEATURE_NAMES,
}


@pytest.fixture(scope="function")
def infer_cfg() -> DictConfig:
    """Setting up the general inference config object.

    Returns:
        DictConfig: Inference Config object containing path and name
    """
    return load_model_config(config_name="infer", config_path=EXAMPLE_CONFIG_PATH)


@pytest.fixture(scope="function")
def initial_training_cfg() -> DictConfig:
    """Setting up the general initial training config object.

    Returns:
        DictConfig: Initial training Config object containing path and name
    """
    cfg = load_model_config(config_name="openfold_initial_training", config_path=EXAMPLE_CONFIG_PATH)

    # switch to sample training data
    cfg.model.data.dataset_path = SAMPLE_DATA_PATH
    cfg.model.data.dataset_variant = TRAINING_DATA_VARIANT
    cfg.model.train_ds.num_workers = 1
    cfg.model.validation_ds.num_workers = 1

    return cfg


@pytest.fixture(scope="function")
def finetuning_cfg() -> DictConfig:
    """Setting up the general finetuning config object.

    Yields:
        DictConfig: Initial training Config object containing path and name
    """
    cfg = load_model_config(config_name="openfold_finetuning", config_path=EXAMPLE_CONFIG_PATH)

    # switch to sample training data
    cfg.model.data.dataset_path = SAMPLE_DATA_PATH
    cfg.model.data.dataset_variant = TRAINING_DATA_VARIANT
    cfg.model.train_ds.num_workers = 1
    cfg.model.validation_ds.num_workers = 1

    return cfg


@pytest.fixture(scope="function")
def predict_dataloader(infer_cfg: DictConfig) -> DataLoader:
    """Setup of prediction dataloader for test purposes; it contains all input features for the
    AlphaFold model, but not the ground truth coordinates.

    Args:
        cfg (DictConfig): Config file to genereate this dataset.

    Returns:
        DataLoader: datalodaer object containing PredictDataset with AF features for input sequences.
    """
    dataset_paths = get_structured_paths(infer_cfg.model.data)
    dataset = PredictDataset(
        sequences=infer_cfg.sequences,
        seq_names=infer_cfg.seq_names,
        pdb_mmcif_chains_filepath=dataset_paths.mmcif_chains,
        pdb_mmcif_dicts_dirpath=dataset_paths.mmcif_dicts,
        pdb_obsolete_filepath=dataset_paths.obsolete_filepath,
        template_hhr_filepaths=infer_cfg.model.data.template_hhr_filepaths,
        msa_a3m_filepaths=infer_cfg.model.data.msa_a3m_filepaths,
        generate_templates_if_missing=infer_cfg.model.data.generate_templates_if_missing,
        pdb70_database_path=infer_cfg.model.data.pdb70_database_path,
        cfg=infer_cfg.model,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=infer_cfg.model.micro_batch_size,
        num_workers=infer_cfg.model.data.num_workers,
        collate_fn=collate,
    )
    return dataloader


@pytest.fixture(scope="function")
def initial_training_dataloader(initial_training_cfg: DictConfig) -> DataLoader:
    """Set up initial training dataloader from AlphaFold model class.

    Args:
        initial_training_cfg (DictConfig): Config object for initial training.

    Returns:
        DataLoader: Initial training dataloader
    """
    trainer = setup_trainer(initial_training_cfg, callbacks=[])
    alphafold = AlphaFold(cfg=initial_training_cfg.model, trainer=trainer)
    return alphafold.train_dataloader()


@pytest.fixture(scope="function")
def finetuning_dataloader(finetuning_cfg: DictConfig) -> DataLoader:
    """Set up fine-tuning dataloader from AlphaFold model class.

    Args:
        finetuning_cfg (DictConfig): Config object for fine-tuning.

    Returns:
        DataLoader: Fine-tuning dataloader
    """
    trainer = setup_trainer(finetuning_cfg, callbacks=[])
    alphafold = AlphaFold(cfg=finetuning_cfg.model, trainer=trainer)
    return alphafold.train_dataloader()


def get_sequence_features(sequence: str) -> Dict[str, np.array]:
    """Helper function to get sequence features from sequence in string

    Args:
        sequence (str): Protein sequence in string.

    Returns:
        Dict: Sequence features.
    """
    domain_name = "description"  # ref: bionemo/data/protein/openfold/dataset.py (line 654)
    sequence_features = create_sequence_features(sequence, domain_name)
    return sequence_features


def get_initial_training_dataset(initial_training_cfg: DictConfig) -> InitialTrainingDataset:
    """Set up initial training dataset bypassing AlphaFold model class.

    Args:
        initial_training_cfg (DictConfig): Config object for initial training.

    Returns:
        InitialTrainingDataset: Initial training dataset
    """
    ds_paths = get_structured_paths(initial_training_cfg.model.data)
    model_cfg = initial_training_cfg.model
    ds_cfg = initial_training_cfg.model.train_ds

    initial_training_dataset = InitialTrainingDataset(
        pdb_mmcif_chains_filepath=ds_paths.mmcif_chains,
        pdb_mmcif_dicts_dirpath=ds_paths.mmcif_dicts,
        pdb_obsolete_filepath=ds_paths.obsolete_filepath,
        pdb_alignments_dirpath=ds_paths.alignments_dirpath,
        max_pdb_release_date=ds_cfg.train_max_pdb_release_date,
        alphafold_config=model_cfg,
        filter_by_alignments=model_cfg.data.filter_by_alignments,
        realign_when_required=ds_cfg.realign_when_required,
        use_only_pdb_chain_ids=model_cfg.data.use_only_pdb_chain_ids,
    )
    return initial_training_dataset


def get_self_distillation_dataset(finetuning_cfg: DictConfig) -> SelfDistillationDataset:
    """Set up self-distillation dataset in fine-tuning bypassing AlphaFold model class.

    Args:
        finetuning_cfg (DictConfig): Config object for fine-tuning.

    Returns:
        SelfDistillationDataset: Self-distillation dataset.
    """
    ds_paths = get_structured_paths(finetuning_cfg.model.data)
    model_cfg = finetuning_cfg.model
    ds_cfg = finetuning_cfg.model.train_ds

    self_distillation_dataset = SelfDistillationDataset(
        uniclust30_alignments_dirpath=ds_paths.uniclust30_alignments,
        uniclust30_targets_dirpath=ds_paths.uniclust30_targets,
        pdb_mmcif_chains_filepath=ds_paths.mmcif_chains,
        pdb_mmcif_dicts_dirpath=ds_paths.mmcif_dicts,
        pdb_obsolete_filepath=ds_paths.obsolete_filepath,
        max_pdb_release_date=ds_cfg.train_max_pdb_release_date,
        realign_when_required=ds_cfg.realign_when_required,
        alphafold_config=model_cfg,
    )
    return self_distillation_dataset


def get_finetuning_dataset(finetuning_cfg: DictConfig) -> FinetuningDataset:
    """Set up \"full\" dataset in fine-tuning bypassing AlphaFold model class.

    Args:
        finetuning_cfg (DictConfig): Config object for fine-tuning.

    Returns:
        FinetuningDataset: The combined dataset from initial training and self-distillation datasets.
    """
    initial_training_dataset = get_initial_training_dataset(finetuning_cfg)
    self_distillation_dataset = get_self_distillation_dataset(finetuning_cfg)

    finetuning_dataset = FinetuningDataset(
        initial_training_dataset=initial_training_dataset,
        self_distillation_dataset=self_distillation_dataset,
    )
    return finetuning_dataset


def get_validation_dataset(initial_training_cfg: DictConfig) -> ValidationDataset:
    """Set up validation dataset bypassing AlphaFold model class.

    Args:
        initial_training_cfg (DictConfig): Config object for initial training.

    Returns:
        ValidationDataset: Validation dataset
    """
    ds_paths = get_structured_paths(initial_training_cfg.model.data)
    model_cfg = initial_training_cfg.model
    ds_cfg = initial_training_cfg.model.validation_ds

    validation_dataset = ValidationDataset(
        pdb_mmcif_chains_filepath=ds_paths.mmcif_chains,
        pdb_mmcif_dicts_dirpath=ds_paths.mmcif_dicts,
        pdb_obsolete_filepath=ds_paths.obsolete_filepath,
        pdb_alignments_dirpath=ds_paths.alignments_dirpath,
        min_cameo_submission_date=ds_cfg.val_min_cameo_submission_date,
        max_cameo_submission_date=ds_cfg.val_max_cameo_submission_date,
        max_sequence_length=ds_cfg.val_max_sequence_length,
        alphafold_config=model_cfg,
        filter_by_alignments=model_cfg.data.filter_by_alignments,
        realign_when_required=ds_cfg.realign_when_required,
        use_only_pdb_chain_ids=model_cfg.data.use_only_pdb_chain_ids,
    )
    return validation_dataset


def raise_exception_if_feature_shape_mismatch(features: dict, cfg: DictConfig, N_res: int):
    """Raise exception if feature shape does not match to reference.

    Args:
        features (dict): Feature set/sample from AlphaFold dataset.
        cfg (DictConfig): Config object for AlphaFold setup.
        N_res (int) : Protein sequence length for this sample.

    Returns:
        None
    """
    mapping = {
        "N_res": N_res,
        "N_clust": cfg.model.max_msa_clusters,
        "N_extra_seq": cfg.model.max_extra_msa,
        "N_templ": cfg.model.max_templates,
    }
    N_recycling = cfg.model.num_recycling_iters + 1

    for k, feature in features.items():
        if k not in FEATURE_SHAPES or not torch.is_tensor(feature):
            continue

        feature_shape = []
        for size in FEATURE_SHAPES[k]:
            assert isinstance(size, (str, int))
            if isinstance(size, str):
                size = mapping[size]
            feature_shape.append(size)
        feature_shape.append(N_recycling)

        assert tuple(feature.shape) == tuple(feature_shape), f"Shape mismatch in feature {k}"


def test_sample_data_exists():
    """Test whether sample data for OpenFold unittest exists"""
    if not os.path.exists(SAMPLE_DATA_PATH):
        raise FileNotFoundError(
            "Before testing, users must download openfold sample data through download_artifacts.py."
        )


@pytest.mark.skipif(not os.path.exists(SAMPLE_DATA_PATH), reason="Test sample data not found")
def test_initial_training_dataset_shape(initial_training_cfg: DictConfig):
    """Test feature shape in initial training dataset

    Args:
        initial_training_cfg (DictConfig): Initial training config.
    """
    initial_training_dataset = get_initial_training_dataset(initial_training_cfg)

    sample = initial_training_dataset[(0, initial_training_cfg.model.seed)]
    N_res = sample["aatype"].size(0)
    raise_exception_if_feature_shape_mismatch(sample, initial_training_cfg, N_res)


@pytest.mark.skipif(not os.path.exists(SAMPLE_DATA_PATH), reason="Test sample data not found")
def test_self_distillation_dataset_shape(finetuning_cfg: DictConfig):
    """Test feature shape in self-distillation dataset

    Args:
        finetuning_cfg (DictConfig): Finetuning config.
    """
    self_distillation_dataset = get_self_distillation_dataset(finetuning_cfg)

    sample = self_distillation_dataset[(0, finetuning_cfg.model.seed)]
    N_res = sample["aatype"].size(0)
    raise_exception_if_feature_shape_mismatch(sample, finetuning_cfg, N_res)


@pytest.mark.skipif(not os.path.exists(SAMPLE_DATA_PATH), reason="Test sample data not found")
def test_create_sequence_features_aatype(infer_cfg: DictConfig):
    """Test aatype in sequence features.

    Args:
        infer_cfg (DictConfig): Inference config.
    """
    sequence = infer_cfg.sequences[0]
    sequence_features = get_sequence_features(sequence)
    onehot = sequence_features["aatype"]

    assert onehot.shape[0] == len(sequence)
    assert onehot.shape[1] == len(rc.RESTYPES_WITH_X)

    assert ((onehot == 0) | (onehot == 1)).all()
    aa_idx = onehot.argmax(axis=1)
    assert rc.aatype_to_str_sequence(aa_idx) == sequence


@pytest.mark.skipif(not os.path.exists(SAMPLE_DATA_PATH), reason="Test sample data not found")
def test_predict_dataloader_feature_names(predict_dataloader: DataLoader):
    """Test whether batch from dataloader of PredictDataset has all the features.

    Args:
        predict_dataloader (DataLoader): Data loader for PredictDataset from AlphaFold model class.
    """
    batch = next(iter(predict_dataloader))
    assert set(batch) == PREDICT_FEATURE_NAMES


@pytest.mark.skipif(not os.path.exists(SAMPLE_DATA_PATH), reason="Test sample data not found")
def test_initial_training_dataloader_feature_names(initial_training_dataloader: DataLoader):
    """Test whether batch from dataloader of InitialTrainingDataset has all the features.

    Args:
        initial_training_dataloader (DataLoader): Data loader for InitialTrainingDataset from AlphaFold model class.
    """
    batch = next(iter(initial_training_dataloader))
    assert set(batch) == INITIAL_TRAINING_FEATURE_NAMES


@pytest.mark.skipif(not os.path.exists(SAMPLE_DATA_PATH), reason="Test sample data not found")
def test_finetuning_dataloader_feature_names(finetuning_dataloader: DataLoader):
    """Test whether batch from dataloader of FinetuningDataset has all the features.

    Args:
        finetuning_dataloader (DataLoader): Data loader for FinetuningDataset from AlphaFold model class.
    """
    batch = next(iter(finetuning_dataloader))
    assert set(batch) == INITIAL_TRAINING_FEATURE_NAMES


@pytest.mark.skipif(not os.path.exists(SAMPLE_DATA_PATH), reason="Test sample data not found")
@pytest.mark.slow
def test_openfold_sample_creator_initial_training_dataset(initial_training_cfg: DictConfig):
    """Test OpenFoldSampleCreator for initial training dataset.

    Args:
        initial_training_cfg (DictConfig): Config object for initial training.
    """
    # create test case config
    initial_training_cfg.model.micro_batch_size = 1
    initial_training_cfg.model.data.realign_when_required = False
    initial_training_cfg.model.data.prepare.sample.sample_variant = TEST_SAMPLE_VARIANT
    initial_training_cfg.model.data.prepare.sample_pdb_chain_ids = SAMPLE_PDB_CHAIN_IDS
    initial_training_cfg.model.data.prepare.sample_cameo_chain_ids = []  # skip cameo validation set
    initial_training_cfg.model.data.prepare.sample_uniclust30_ids = []  # skip uniclus30

    with tempfile.TemporaryDirectory() as temp_dir:
        # create test data
        initial_training_cfg.model.data.prepare.sample.output_root_path = temp_dir

        sample_creator = OpenFoldSampleCreator(
            dataset_root_path=initial_training_cfg.model.data.dataset_path,
            **initial_training_cfg.model.data.prepare.sample,
        )
        sample_creator.prepare(
            sample_pdb_chain_ids=initial_training_cfg.model.data.prepare.sample_pdb_chain_ids,
            sample_cameo_chain_ids=initial_training_cfg.model.data.prepare.sample_cameo_chain_ids,
            sample_uniclust30_ids=initial_training_cfg.model.data.prepare.sample_uniclust30_ids,
        )

        # test sample pdb chain ids and switch to load from test data
        initial_training_cfg.model.data.dataset_path = temp_dir
        initial_training_cfg.model.data.dataset_variant = TEST_SAMPLE_VARIANT

        initial_training_dataset = get_initial_training_dataset(initial_training_cfg)
        assert len(initial_training_dataset) == len(SAMPLE_PDB_CHAIN_IDS)

        seed = initial_training_cfg.model.seed
        for sample_idx in range(len(initial_training_dataset)):
            sample = initial_training_dataset[(sample_idx, seed)]
            dataset_name, index, seed, pdb_chain_id, seqlen = sample["id"]
            assert pdb_chain_id in SAMPLE_PDB_CHAIN_IDS


@pytest.mark.skipif(not os.path.exists(SAMPLE_DATA_PATH), reason="Test sample data not found")
@pytest.mark.slow
def test_openfold_sample_creator_self_distillation_dataset(finetuning_cfg: DictConfig):
    """Test OpenFoldSampleCreator for self-distillation dataset.

    Args:
        finetuning_cfg (DictConfig): Config object for fine-tuning.
    """
    # create test case config
    finetuning_cfg.model.micro_batch_size = 1
    finetuning_cfg.model.data.realign_when_required = False
    finetuning_cfg.model.data.prepare.sample.sample_variant = TEST_SAMPLE_VARIANT
    finetuning_cfg.model.data.prepare.sample_pdb_chain_ids = []  # skip pdb chain ids
    finetuning_cfg.model.data.prepare.sample_cameo_chain_ids = []  # skip cameo validation set
    finetuning_cfg.model.data.prepare.sample_uniclust30_ids = SAMPLE_UNICLUST30_IDS

    with tempfile.TemporaryDirectory() as temp_dir:
        # create test data
        finetuning_cfg.model.data.prepare.sample.output_root_path = temp_dir
        sample_creator = OpenFoldSampleCreator(
            dataset_root_path=finetuning_cfg.model.data.dataset_path, **finetuning_cfg.model.data.prepare.sample
        )
        sample_creator.prepare(
            sample_pdb_chain_ids=finetuning_cfg.model.data.prepare.sample_pdb_chain_ids,
            sample_cameo_chain_ids=finetuning_cfg.model.data.prepare.sample_cameo_chain_ids,
            sample_uniclust30_ids=finetuning_cfg.model.data.prepare.sample_uniclust30_ids,
        )

        # test sample uniclus30 ids and # switch to load from test data
        finetuning_cfg.model.data.dataset_path = temp_dir
        finetuning_cfg.model.data.dataset_variant = TEST_SAMPLE_VARIANT
        self_distillation_dataset = get_self_distillation_dataset(finetuning_cfg)
        assert len(self_distillation_dataset) == len(SAMPLE_UNICLUST30_IDS)

        seed = finetuning_cfg.model.seed
        for sample_idx in range(len(self_distillation_dataset)):
            sample = self_distillation_dataset[(sample_idx, seed)]
            dataset_name, index, seed, pdb_chain_id, seqlen = sample["id"]
            assert pdb_chain_id in SAMPLE_UNICLUST30_IDS


@pytest.mark.skipif(not os.path.exists(SAMPLE_DATA_PATH), reason="Test sample data not found")
@pytest.mark.slow
def test_openfold_sample_creator_validation_dataset(initial_training_cfg: DictConfig):
    """Test OpenFoldSampleCreator for fine-tuning dataset.

    Args:
        initial_training_cfg (DictConfig): Config object for initial training.
    """
    # create test case config
    initial_training_cfg.model.micro_batch_size = 1
    initial_training_cfg.model.data.realign_when_required = False
    initial_training_cfg.model.data.prepare.sample.sample_variant = TEST_SAMPLE_VARIANT
    initial_training_cfg.model.data.prepare.sample_pdb_chain_ids = []  # skip pdb initial training dataset
    initial_training_cfg.model.data.prepare.sample_cameo_chain_ids = SAMPLE_CAMEO_CHAIN_IDS
    initial_training_cfg.model.data.prepare.sample_uniclust30_ids = []  # skip uniclus30

    with tempfile.TemporaryDirectory() as temp_dir:
        # create test data
        initial_training_cfg.model.data.prepare.sample.output_root_path = temp_dir

        sample_creator = OpenFoldSampleCreator(
            dataset_root_path=initial_training_cfg.model.data.dataset_path,
            **initial_training_cfg.model.data.prepare.sample,
        )
        sample_creator.prepare(
            sample_pdb_chain_ids=initial_training_cfg.model.data.prepare.sample_pdb_chain_ids,
            sample_cameo_chain_ids=initial_training_cfg.model.data.prepare.sample_cameo_chain_ids,
            sample_uniclust30_ids=initial_training_cfg.model.data.prepare.sample_uniclust30_ids,
        )

        # test sample pdb chain ids and switch to load from test data
        initial_training_cfg.model.data.dataset_path = temp_dir
        initial_training_cfg.model.data.dataset_variant = TEST_SAMPLE_VARIANT

        validation_dataset = get_validation_dataset(initial_training_cfg)
        assert len(validation_dataset) == len(SAMPLE_CAMEO_CHAIN_IDS)

        seed = initial_training_cfg.model.seed
        for sample_idx in range(len(validation_dataset)):
            sample = validation_dataset[(sample_idx, seed)]
            dataset_name, index, seed, pdb_chain_id, seqlen = sample["id"]
            assert pdb_chain_id in SAMPLE_CAMEO_CHAIN_IDS
