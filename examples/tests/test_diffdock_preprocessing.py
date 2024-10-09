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


import hashlib
import os
import pickle
from functools import partial
from pathlib import Path

import e3nn
import numpy as np
import pytest
import torch
from Bio import SeqIO
from pytorch_lightning import seed_everything

from bionemo.data.diffdock.confidence_dataset import diffdock_build_confidence_dataset
from bionemo.data.diffdock.docking_dataset import DataSplit, diffdock_build_dataset
from bionemo.data.diffdock.embedding_preprocess import prep_embedding
from bionemo.model.molecule.diffdock.infer import DiffDockModelInference
from bionemo.model.molecule.diffdock.utils.diffusion import t_to_sigma as t_to_sigma_compl
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import teardown_apex_megatron_cuda


# some pytorch functions use cuBlas or cuDNN with TF32 enabled for acceleration,
# which can potentially result in the loss of precision of the returned values.
# This in turn affect comparing pytorch results with our tp results. Turn off
# TF32 here to make sure we get precise result to compare with
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
# This works in parallel with torch.use_deterministic_algorithms
# see: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
e3nn.set_optimization_defaults(optimize_einsums=False)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
torch.backends.cudnn.enabled = False


def sha256_file(file_path: str):
    """Compute the sha256 checksum of the input file

    Args:
        file_path (str): path to the input file

    Returns: the file's sha256 (str)

    """
    f_hash = hashlib.new("sha256")

    with open(file_path, "rb") as file:
        while chunk := file.read(1024 * 8):
            f_hash.update(chunk)

    return f_hash.hexdigest()


@pytest.fixture(scope="session")
def bionemo_home() -> Path:
    try:
        x = os.environ["BIONEMO_HOME"]
    except KeyError:
        raise ValueError("Need to set BIONEMO_HOME in order to run unit tests! See docs for instructions.")
    else:
        yield Path(x).absolute()


@pytest.fixture(scope="session")
def config_path_for_tests(bionemo_home) -> str:
    yield str(bionemo_home / "examples" / "tests" / "conf")


@pytest.mark.needs_gpu
def test_diffdock_embedding_preprocessing(tmp_path, config_path_for_tests):
    cfg = load_model_config(config_name="diffdock_embedding_test", config_path=config_path_for_tests)
    cfg.protein_data.output_fasta_file = str(tmp_path / "pdb_sequences.fasta")
    cfg.protein_data.esm_embeddings_path = str(tmp_path / "esm2_embeddings")

    seed_everything(cfg.seed)
    prep_embedding(cfg.protein_data)
    fasta = {rec.id: str(rec.seq) for rec in SeqIO.parse(cfg.protein_data.output_fasta_file, "fasta")}
    ref_fasta = {rec.id: str(rec.seq) for rec in SeqIO.parse(cfg.ref_fasta_file, "fasta")}
    for complex_name in ref_fasta.keys():
        assert fasta[complex_name] == ref_fasta[complex_name], f"fasta is incorrect for {complex_name}"

    assert len(os.listdir(cfg.protein_data.esm_embeddings_path)) == 3

    for file in os.listdir(cfg.ref_esm_embeddings_path):
        ref_embedding = torch.load(os.path.join(cfg.ref_esm_embeddings_path, file))
        embedding = torch.load(os.path.join(cfg.protein_data.esm_embeddings_path, file))

        assert torch.allclose(embedding, ref_embedding, atol=1.0e-2)


@pytest.mark.slow
def test_diffdock_prepare_score_dataset(config_path_for_tests, tmp_path):
    cfg = load_model_config(config_name="diffdock_score_preprocessing_test", config_path=config_path_for_tests)
    cfg.data.cache_path = str(tmp_path / "data_cache")

    t_to_sigma = partial(t_to_sigma_compl, cfg=cfg.model)

    dataset = diffdock_build_dataset(
        cfg.data, cfg.model.train_ds, t_to_sigma, _num_conformers=True, mode=DataSplit("train")
    )
    seed_everything(cfg.seed)
    dataset.build_complex_graphs()

    assert len(os.listdir(os.path.join(cfg.data.cache_path, cfg.ref_train_graph_folder_name))) > 0

    attr_dict = {
        "ligand": ["orig_pos", "x", "pos", "edge_mask", "mask_rotate"],
        "receptor": ["x", "pos", "mu_r_norm", "side_chain_vecs"],
        ("ligand", "lig_bond", "ligand"): ["edge_index", "edge_attr"],
        ("receptor", "rec_contact", "receptor"): ["edge_index"],
    }

    for file in os.listdir(cfg.ref_train_graph_file):
        path_file = os.path.join(cfg.ref_train_graph_file, file)
        hash_file = sha256_file(path_file)
        ref_sample = pickle.load(open(path_file, "rb"))
        sample = pickle.load(open(os.path.join(cfg.data.cache_path, cfg.ref_train_graph_folder_name, file), "rb"))

        assert torch.allclose(torch.tensor(sample.rmsd_matching), torch.tensor(ref_sample.rmsd_matching), atol=0.01)
        assert torch.allclose(
            torch.tensor(sample.original_center), torch.tensor(ref_sample.original_center), atol=0.01
        )

        # DEBUG: print the reference value
        torch.set_printoptions(precision=15, sci_mode=True, threshold=2000)
        for key in attr_dict:
            for attr in attr_dict[key]:
                result = torch.tensor(getattr(sample[key], attr))
                expected = torch.tensor(getattr(ref_sample[key], attr))
                torch.testing.assert_close(
                    result,
                    expected,
                    atol=0.01,
                    rtol=0.01,
                    msg=lambda msg: f"Wrong {key}.{attr} \
                        compared with {path_file} whose sha256 is {hash_file} \
                        and the reference data:\n {expected} \n\n{msg}",
                )
        # DEBUG: revert print option
        torch.set_printoptions(precision=4, sci_mode=None, threshold=1000)


@pytest.mark.needs_fork
@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
def test_diffdock_prepare_confidence_dataset(tmp_path, config_path_for_tests):
    cfg = load_model_config(config_name="diffdock_confidence_preprocessing_test", config_path=config_path_for_tests)
    cfg.data.cache_path = str(tmp_path / "data_cache")

    dataset = diffdock_build_confidence_dataset(cfg.data, cfg.model.train_ds, mode=DataSplit("train"))
    seed_everything(cfg.seed)
    dataset.build_complex_graphs()

    assert len(os.listdir(os.path.join(cfg.data.cache_path, cfg.ref_train_graph_folder_name))) == 2

    attr_dict = {
        "ligand": ["orig_pos", "x", "pos", "edge_mask", "mask_rotate"],
        "receptor": ["x", "pos", "mu_r_norm", "side_chain_vecs"],
        "atom": ["x", "pos"],
        ("ligand", "lig_bond", "ligand"): ["edge_index", "edge_attr"],
        ("receptor", "rec_contact", "receptor"): ["edge_index"],
        ("atom", "atom_contact", "atom"): ["edge_index"],
        ("atom", "atom_rec_contact", "receptor"): ["edge_index"],
    }

    for file in os.listdir(cfg.ref_train_graph_file):
        path_file = os.path.join(cfg.ref_train_graph_file, file)
        hash_file = sha256_file(path_file)
        ref_sample = pickle.load(open(path_file, "rb"))
        sample = pickle.load(open(os.path.join(cfg.data.cache_path, cfg.ref_train_graph_folder_name, file), "rb"))

        assert torch.allclose(torch.tensor(sample.rmsd_matching), torch.tensor(ref_sample.rmsd_matching), atol=0.01)
        assert torch.allclose(
            torch.tensor(sample.original_center), torch.tensor(ref_sample.original_center), atol=0.01
        )

        # DEBUG: print the reference value
        torch.set_printoptions(precision=15, sci_mode=True, threshold=2000)
        for key in attr_dict:
            for attr in attr_dict[key]:
                result = torch.tensor(getattr(sample[key], attr))
                expected = torch.tensor(getattr(ref_sample[key], attr))
                torch.testing.assert_close(
                    result,
                    expected,
                    atol=0.01,
                    rtol=0.01,
                    msg=lambda msg: f"Wrong {key}.{attr} \
                        compared with {path_file} whose sha256 is {hash_file} \
                        and the reference data:\n {expected} \n\n{msg}",
                )
        # DEBUG: revert print option
        torch.set_printoptions(precision=4, sci_mode=None, threshold=1000)

    score_model = DiffDockModelInference(cfg.score_infer)
    score_model.eval()

    seed_everything(cfg.seed)
    dataset.generate_ligand_poses(score_model, cfg.score_infer.data)
    teardown_apex_megatron_cuda()

    assert len(os.listdir(os.path.join(cfg.data.cache_path, cfg.ref_ligand_pose_folder_name))) == 2

    for file in os.listdir(cfg.ref_ligand_pose_file):
        ref_sample = pickle.load(open(os.path.join(cfg.ref_ligand_pose_file, file), "rb"))
        sample = pickle.load(open(os.path.join(cfg.data.cache_path, cfg.ref_ligand_pose_folder_name, file), "rb"))

        # complex name
        assert sample[0] == ref_sample[0]

        # ligand positions
        assert np.allclose(sample[1], ref_sample[1], atol=0.1)

        # RMSD
        assert np.allclose(sample[2], ref_sample[2], atol=0.1)
