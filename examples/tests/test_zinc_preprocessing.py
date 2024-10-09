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
import os

import pandas as pd
import pytest
import requests_mock
from omegaconf import OmegaConf
from rdkit import Chem

from bionemo.data import Zinc15Preprocess
from bionemo.utils.tests import get_directory_hash


TEST_DATA_DIR = os.path.join(os.environ["BIONEMO_HOME"], "examples/tests/test_data/molecule/zinc15/")
ROOT_DIR = "zinc15"
CONFIG = {
    "max_smiles_length": 512,
    "train_samples_per_file": 1000,
    "val_samples_per_file": 100,
    "test_samples_per_file": 100,
    "links_file": os.path.join(TEST_DATA_DIR, "ZINC-downloader-test.txt"),
    "pool_size": 4,
    "seed": 0,
}
HEADER = "zinc_id,smiles"
TRAIN_VAL_TEST_HASHES = {
    "train": "fe599545f23995a56dde9e1533b12e35",
    "val": "f9df93c66c777b9e815614577b58fb1b",
    "test": "73af6d6084a7665de7b706fe26593b5c",
}
MAX_LENGTH_URLS = [
    "http://files.docking.org/2D/KA/KAED.txt",  # small mols
    "http://files.docking.org/2D/AI/AIED.txt",
]  # large mols


@pytest.fixture()
def download_directory(tmp_path):
    """Create the temporary directory for testing and the download directory"""
    download_directory = tmp_path / "raw"
    download_directory.mkdir(parents=True, exist_ok=True)
    return download_directory


@pytest.fixture()
def output_directory(tmp_path):
    """Create the directory for processed data"""
    output_dir = tmp_path / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# TODO mocker could probably be made into a fixture
@requests_mock.Mocker(kw="mocker")
@pytest.mark.parametrize("config", [(CONFIG)])
def test_process_files(tmp_path, download_directory, config, **kwargs):
    cfg = OmegaConf.create(config)

    mocker = kwargs["mocker"]
    with open(cfg.links_file, "r") as fl:
        for url in fl:
            url = url.strip()
            data_filename = os.path.basename(url)
            with open(os.path.join(TEST_DATA_DIR, "raw", data_filename), "r") as fh:
                mocker.get(url, text=fh.read())

    preproc = Zinc15Preprocess(root_directory=str(tmp_path))
    preproc.process_files(
        links_file=cfg.links_file,
        download_dir=download_directory,
        pool_size=cfg.pool_size,
        max_smiles_length=cfg.max_smiles_length,
    )

    with open(cfg.links_file, "r") as fh:
        expected_tranche_files = fh.readlines()

    tranche_files = glob.glob(os.path.join(download_directory, "*.txt"))
    assert len(tranche_files) == len(expected_tranche_files)

    expected_tranche_files = sorted([os.path.basename(x.strip()) for x in expected_tranche_files])
    tranche_files = sorted([os.path.basename(x) for x in tranche_files])
    assert tranche_files == expected_tranche_files


@requests_mock.Mocker(kw="mocker")
@pytest.mark.parametrize("config, header, hash_dict", [(CONFIG, HEADER, TRAIN_VAL_TEST_HASHES)])
def test_prepare_dataset(tmp_path, download_directory, output_directory, config, header, hash_dict, **kwargs):
    cfg = OmegaConf.create(config)

    mocker = kwargs["mocker"]
    with open(cfg.links_file, "r") as fl:
        for url in fl:
            url = url.strip()
            data_filename = os.path.basename(url)
            with open(os.path.join(TEST_DATA_DIR, "raw", data_filename), "r") as fh:
                mocker.get(url, text=fh.read())

    preproc = Zinc15Preprocess(root_directory=str(tmp_path))
    preproc.prepare_dataset(
        max_smiles_length=cfg.max_smiles_length,
        train_samples_per_file=cfg.train_samples_per_file,
        val_samples_per_file=cfg.val_samples_per_file,
        test_samples_per_file=cfg.test_samples_per_file,
        links_file=cfg.links_file,
        seed=cfg.seed,
    )

    expected_lines = 0
    tranche_files = glob.glob(os.path.join(download_directory, "*.txt"))
    for file in tranche_files:
        with open(file, "r") as fh:
            expected_lines += len(fh.readlines()) - 1

    total_lines = 0
    for split in ["train", "val", "test"]:
        split_directory = os.path.join(output_directory, split)
        assert get_directory_hash(split_directory) == hash_dict[split]

        csv_file_list = sorted(glob.glob(os.path.join(split_directory, "*.csv")))
        expected_samples = cfg.get(f"{split}_samples_per_file")

        for file in csv_file_list:
            with open(file, "r") as fh:
                lines = fh.readlines()
                num_samples = len(lines) - 1
                if file != csv_file_list[-1]:
                    assert num_samples == expected_samples
                assert lines[0].strip() == header
                total_lines += num_samples

    assert expected_lines == total_lines


@requests_mock.Mocker(kw="mocker")
@pytest.mark.parametrize("url", MAX_LENGTH_URLS)
@pytest.mark.parametrize("max_smiles_length", (20, 100, 200))
def test_filtering(tmp_path, download_directory, url, max_smiles_length, **kwargs):
    data_filename = os.path.basename(url)
    mocker = kwargs["mocker"]
    with open(os.path.join(TEST_DATA_DIR, "raw", data_filename), "r") as fh:
        mocker.get(url, text=fh.read())
    preproc = Zinc15Preprocess(root_directory=str(tmp_path))
    preproc._process_file(url, download_directory, max_smiles_length)
    filtered_data = pd.read_csv(download_directory / data_filename)
    if len(filtered_data) > 0:
        assert filtered_data["SMILES"].map(len).max() <= max_smiles_length

    full_data = pd.read_table(os.path.join(TEST_DATA_DIR, "raw", data_filename))
    num_kept = (full_data["smiles"].map(Chem.CanonSmiles).map(len) <= max_smiles_length).sum()
    assert len(filtered_data) == num_kept
