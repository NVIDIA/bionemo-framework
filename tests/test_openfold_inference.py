# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import tempfile
from pathlib import Path
from typing import Tuple

import pytest
import pytorch_lightning as plt
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from bionemo.data.protein.openfold.datahub import get_structured_paths
from bionemo.data.protein.openfold.datasets import PredictDataset
from bionemo.data.protein.openfold.features import create_mmcif_features
from bionemo.data.protein.openfold.helpers import collate
from bionemo.data.protein.openfold.mmcif import load_mmcif_file, parse_mmcif_string
from bionemo.model.protein.openfold.openfold_model import AlphaFold
from bionemo.model.protein.openfold.validation_metrics import compute_validation_metrics
from bionemo.model.protein.openfold.writer import PredictionFeatureWriter
from bionemo.model.utils import setup_trainer
from bionemo.utils.hydra import load_model_config


GRADIENT_CHECKPOINTING = False

BIONEMO_HOME = os.getenv("BIONEMO_HOME")
EXAMPLE_CONFIG_PATH = os.path.join(BIONEMO_HOME, "examples/protein/openfold/conf")
TEST_DATA_PATH = os.path.join(BIONEMO_HOME, "examples/tests/test_data")
SAMPLE_DATA_PATH = os.path.join(TEST_DATA_PATH, "openfold_data")

S3_DATA_PATH = "s3://bionemo-ci/test-data/openfold/openfold_vprocessed_sample/openfold_sample_data.tar.gz"

SAMPLE_INFER_DATA_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../examples/tests/test_data/openfold_data/inference"
)

PDB_DIR = os.path.join(SAMPLE_INFER_DATA_PATH, "pdb")
CIF_NAMES = ["7b4q.cif", "7dnu.cif"]
CIF_PATHS = [Path(os.path.join(PDB_DIR, cif)) for cif in CIF_NAMES]
CIF_CHAIN_IDS = ["A", "A"]

MSA_DIR = os.path.join(SAMPLE_INFER_DATA_PATH, "msas")
MSA_PATHS = [
    [
        os.path.join(MSA_DIR, "7b4q_A", "bfd_uniclust_hits.a3m"),
        os.path.join(MSA_DIR, "7b4q_A", "mgnify_hits.a3m"),
        os.path.join(MSA_DIR, "7b4q_A", "uniref90_hits.a3m"),
    ],
    [
        os.path.join(MSA_DIR, "7dnu_A", "bfd_uniclust_hits.a3m"),
        os.path.join(MSA_DIR, "7dnu_A", "mgnify_hits.a3m"),
        os.path.join(MSA_DIR, "7dnu_A", "uniref90_hits.a3m"),
    ],
]

DRYRUN_SEQUENCES = ["AAAAA", "CCCCC"]
DRYRUN_SEQ_NAMES = ["first", "second"]


@pytest.fixture(scope="function")
def infer_cfg() -> DictConfig:
    """Setting up the general inference config object.

    Returns:
        DictConfig: Inference Config object containing path and name
    """
    return load_model_config(config_name="infer", config_path=EXAMPLE_CONFIG_PATH)


def get_alphafold_model_trainer(cfg: DictConfig) -> Tuple[AlphaFold, plt.Trainer]:
    """Setting up the AF model and trainer.

    Args:
        alphafold_cfg (DictConfig): Config object for model and trainer setup.

    Returns:
        Tuple[AlphaFold, plt.Trainer]: AlphaFold model and trainer.
    """
    trainer = setup_trainer(cfg, callbacks=[])
    alphafold = AlphaFold.restore_from(restore_path=cfg.restore_from_path, override_config_path=cfg, trainer=trainer)
    return alphafold, trainer


def get_predict_dataset(cfg: DictConfig) -> PredictDataset:
    """Setup of prediction dataset for test purposes; it contains all input features for the
    AlphaFold model, but not the ground truth coordinates.

    Args:
        cfg (DictConfig): Config file to genereate this dataset.

    Returns:
        PredictDataset: dataset object containing AF features for input sequences.
    """
    dataset_paths = get_structured_paths(cfg.model.data)
    dataset = PredictDataset(
        sequences=cfg.sequences,
        seq_names=cfg.seq_names,
        pdb_mmcif_chains_filepath=dataset_paths.mmcif_chains,
        pdb_mmcif_dicts_dirpath=dataset_paths.mmcif_dicts,
        pdb_obsolete_filepath=dataset_paths.obsolete_filepath,
        template_hhr_filepaths=cfg.model.data.template_hhr_filepaths,
        msa_a3m_filepaths=cfg.model.data.msa_a3m_filepaths,
        generate_templates_if_missing=cfg.model.data.generate_templates_if_missing,
        pdb70_database_path=cfg.model.data.pdb70_database_path,
        cfg=cfg.model,
    )
    return dataset


@pytest.mark.parametrize(
    "outputs",
    [
        ["single", "msa", "pair", "sm_single"],
    ],
)
def test_openfold_prediction_pdb_writer(infer_cfg: DictConfig, outputs: str):
    """Test if OpenFold inference and output writing works with different
        output options (single, msa, pair, sm_single).

    Args:
        infer_cfg (DictConfig): Inference config specifying model options.
        outputs (str): setting determining the output format
    """
    # setup input/output samples for writer
    sequence = infer_cfg.sequences[0]
    seq_name = infer_cfg.seq_names[0]
    N_res = len(sequence)
    batch_size = infer_cfg.model.micro_batch_size
    N_seq = infer_cfg.model.max_msa_clusters

    input_dict = {"seq_name": [seq_name]}
    output_dict = {
        "single": torch.empty(batch_size, N_res, infer_cfg.model.evoformer_stack_config.c_s),
        "msa": torch.empty(batch_size, N_seq, N_res, infer_cfg.model.evoformer_stack_config.c_m),
        "pair": torch.empty(batch_size, N_res, N_res, infer_cfg.model.evoformer_stack_config.c_z),
        "sm_single": torch.empty(batch_size, N_res, infer_cfg.model.structure_module_config.c_s),
    }

    # call writer
    with tempfile.TemporaryDirectory() as temp_dir:
        callback = PredictionFeatureWriter(temp_dir, outputs)
        callback.on_predict_batch_end(
            outputs=output_dict,
            batch=input_dict,
            trainer=None,  # dummy
            pl_module=None,  # dummy
            batch_idx=None,  # dummy
        )


@pytest.mark.xfail(reason="Issue #195")
@pytest.mark.needs_gpu
def test_openfold_inference_sequence_only_dryrun(
    infer_cfg: DictConfig,
):
    """Testing if inference itself can dryrun on sequence-only input.

    Args:
        alphafold_cfg (DictConfig): Config object for the model and dataset setup.
        alphafold_model_trainer (Tuple[AlphaFold, plt.Trainer]): Model and Trainer for inference.
    """
    infer_cfg.sequences = DRYRUN_SEQUENCES
    infer_cfg.seq_names = DRYRUN_SEQ_NAMES
    infer_cfg.model.data.msa_a3m_filepaths = None
    infer_cfg.model.data.generate_templates_if_missing = False

    alphafold_model, trainer = get_alphafold_model_trainer(infer_cfg)
    dataset = get_predict_dataset(infer_cfg)
    assert len(dataset) > 0

    dl = DataLoader(
        dataset,
        batch_size=infer_cfg.model.micro_batch_size,
        num_workers=infer_cfg.model.data.num_workers,
        collate_fn=collate,
    )
    trainer.predict(alphafold_model, dl, return_predictions=False)


@pytest.mark.xfail(reason="Issue #195")
@pytest.mark.needs_gpu
def test_openfold_inference_sequence_and_msa_dryrun(
    infer_cfg: DictConfig,
):
    """Testing if inference itself can dryrun on sequence and msa input.

    Args:
        alphafold_cfg (DictConfig): Config object for the model and dataset setup.
        alphafold_model_trainer (Tuple[AlphaFold, plt.Trainer]): Model and Trainer for inference.
    """
    infer_cfg.sequences = DRYRUN_SEQUENCES
    infer_cfg.seq_names = DRYRUN_SEQ_NAMES
    infer_cfg.model.data.generate_templates_if_missing = False

    alphafold_model, trainer = get_alphafold_model_trainer(infer_cfg)
    dataset = get_predict_dataset(infer_cfg)
    assert len(dataset) > 0

    dl = DataLoader(
        dataset,
        batch_size=infer_cfg.model.micro_batch_size,
        num_workers=infer_cfg.model.data.num_workers,
        collate_fn=collate,
    )
    trainer.predict(alphafold_model, dl, return_predictions=False)


# TODO: test dryrun inference with template inputs
#  considerations:
#  - the (sample) pdb_mmcif database has to contain every template in sample hhr
#  - a (sample) pdb70 database is needed to test option - generate_templates_if_missing
#  - realign_when_mismatch requires third party softwares on top of template inputs


def test_sample_data_exists():
    """Test whether sample data for OpenFold unittest exists"""
    if not os.path.exists(SAMPLE_DATA_PATH):
        raise FileNotFoundError(
            "Before testing, users must download openfold sample data through download_artifacts.py."
        )


@pytest.mark.xfail(reason="Issue #195")
@pytest.mark.skipif(not os.path.exists(SAMPLE_DATA_PATH), reason="Test sample data not found")
@pytest.mark.needs_gpu
def test_openfold_inference_lddt_validation_metric_check(infer_cfg: DictConfig):
    """Test that checks whether the structure predicted by OpenFold is similar to the ground truth structure.
    For this, the predicted and ground truth coordinates are both represented in atom37 format and fed into
    the `compute_validation_metrics` function that computes the metric `{"lddt_ca"}`.
    [Atom37 format information](https://huggingface.co/spaces/simonduerr/ProteinMPNN/blame/e65166bd70446c6fddcc1581dbc6dac06e7f8dca/alphafold/alphafold/model/all_atom.py)

    In theory we could compute the metrics `{"lddt_ca", "alignment_rmsd", "gdt_ts", "gdt_ha"} via the `compute_validation_metrics` function.
    We do not do this since lddt_ca is the only SE(3)-invariant metrics; all the others rely on alignments.
    The default `superimpose` function used for this in `compute_validation_metrics` is not very good and gives
    therefore incorrect results. Tests of incorporating Kabsch alignment showed that the metric computation itself
    works when a better alignment algorithm is employed, but was not worth the compute requirement here.

    Args:
        alphafold_cfg (DictConfig): Config Object to restore AlphaFold model from checkpoint and initialise dataset.
    """
    infer_cfg.model.data.generate_templates_if_missing = False

    # load ground truth data from mmcif files
    mmcif_strings = [load_mmcif_file(mmcif_path) for mmcif_path in CIF_PATHS]
    mmcif_dicts = [parse_mmcif_string(mmcif_string) for mmcif_string in mmcif_strings]
    mmcif_features = [
        create_mmcif_features(mmcif_dict, chain_id) for mmcif_dict, chain_id in zip(mmcif_dicts, CIF_CHAIN_IDS)
    ]
    ground_truth_atom_37_coords_list = [
        torch.from_numpy(mmcif_feature["all_atom_positions"]) for mmcif_feature in mmcif_features
    ]
    # reshape by adding first dimension to match with the batch dimension layout of the predicted coords
    ground_truth_atom_37_coords_list = [
        torch.unsqueeze(gt_coords, 0) for gt_coords in ground_truth_atom_37_coords_list
    ]
    # setup for inference
    alphafold_model, trainer = get_alphafold_model_trainer(infer_cfg)
    # get prediction dataset from config file, containing only sequences and MSAs, no structures
    dataset = get_predict_dataset(infer_cfg)
    data_loader = DataLoader(
        dataset,
        batch_size=infer_cfg.model.micro_batch_size,
        num_workers=infer_cfg.model.data.num_workers,
        collate_fn=collate,
    )
    # inference
    predicted_structures = trainer.predict(alphafold_model, data_loader, return_predictions=True)
    # get the predicted atom14 coordinates and atom mask telling which positions are valid
    predicted_atom_37_coords_list = [sample["final_atom_positions"] for sample in predicted_structures]
    all_atom_mask_list = [sample["final_atom_mask"] for sample in predicted_structures]
    # compare ground truth to predicted coordinates and calculate lddt_ca metric
    metrics = {"lddt_ca"}
    for predicted_atom_37_coords, ground_truth_atom_37_coord, all_atom_mask in zip(
        predicted_atom_37_coords_list, ground_truth_atom_37_coords_list, all_atom_mask_list
    ):
        validation_metrics = compute_validation_metrics(
            predicted_atom_37_coords, ground_truth_atom_37_coord, all_atom_mask, metrics
        )
        assert validation_metrics["lddt_ca"] > 90
