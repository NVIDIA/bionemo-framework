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

import pytest
from omegaconf import OmegaConf

from bionemo.data import PhysChemPreprocess
from bionemo.utils.tests import get_directory_hash


# Physchem secondary structure benchmark dataset is small and will be fully downloaded in this test
CONFIG = {
    "url": None,
    "links_file": os.path.join(
        os.environ["BIONEMO_HOME"], "examples/molecule/megamolbart/dataset/PhysChem-downloader.txt"
    ),
    "test_frac": 0.15,
    "val_frac": 0.15,
}
DATA_HEADERS = {
    "Lipophilicity": "CMPD_CHEMBLID,exp,smiles",
    "delaney-processed": "Compound ID,ESOL predicted log solubility in mols per litre,Minimum Degree,Molecular Weight,Number of H-Bond Donors,Number of Rings,Number of Rotatable Bonds,Polar Surface Area,measured log solubility in mols per litre,smiles",
    "SAMPL": "iupac,smiles,expt,calc",
}
NUM_ENTRIES = 5973
DATA_HASHES = {
    "all_datasets": "5b39b31dad10a254fb3c6ff8845254f5",
    "Lipophilicity": "d41d8cd98f00b204e9800998ecf8427e",
    "delaney-processed": "d41d8cd98f00b204e9800998ecf8427e",
    "SAMPL": "d41d8cd98f00b204e9800998ecf8427e",
}


@pytest.mark.parametrize("config, header, num_entries, hash_dict", [(CONFIG, DATA_HEADERS, NUM_ENTRIES, DATA_HASHES)])
def test_prepare_dataset(tmp_path, config, header, num_entries, hash_dict):
    cfg = OmegaConf.create(config)
    processed_directory = os.path.join(tmp_path, "processed")
    PhysChemPreprocess().prepare_dataset(links_file=cfg.links_file, output_dir=processed_directory)

    # Check that all three CSV files were downloaded properly
    assert get_directory_hash(processed_directory) == hash_dict["all_datasets"]

    PhysChemPreprocess()._process_split(
        links_file=cfg.links_file, output_dir=processed_directory, test_frac=cfg.test_frac, val_frac=cfg.val_frac
    )

    total_lines = 0
    for file in header.keys():
        with open(os.path.join(processed_directory, file + ".csv"), "r") as fh:
            lines = fh.readlines()
            total_lines += len(lines)

            # Check against expected header of file
            assert lines[0].strip() == header[file]

        # Check that datasets were split correctly
        assert get_directory_hash(os.path.join(processed_directory, file + "_splits")) == hash_dict[file]

    # Check total expected lines from all 3 files
    assert total_lines == NUM_ENTRIES
