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
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from bionemo.data import FLIPPreprocess
from bionemo.utils.tests import get_directory_hash


# FLIP secondary structure benchmark dataset is small and will be fully downloaded in this test
CONFIG = {"url": None, "num_csv_files": 1}
HEADER = "id,sequence,3state,resolved"
NUM_ENTRIES = 11156
TRAIN_VAL_TEST_HASHES = {
    "train": "af653e0e56e63442b4f3bafe68de7d70",
    "val": "fa5d9cbdd729180b8401d18b31ecc0a9",
    "test": "a92c394b0017cd573d25fbd86cc6abf7",
}


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


@pytest.mark.parametrize(
    "config, header, num_entries, hash_dict", [(CONFIG, HEADER, NUM_ENTRIES, TRAIN_VAL_TEST_HASHES)]
)
def test_prepare_dataset(tmp_path, config, header, num_entries, hash_dict):
    cfg = OmegaConf.create(config)
    preproc = FLIPPreprocess(root_directory=str(tmp_path))
    processed_directory = os.path.join(tmp_path, "processed")
    preproc.prepare_dataset(num_csv_files=cfg.num_csv_files, output_dir=processed_directory)

    total_lines = 0
    for split in ["train", "val", "test"]:
        split_directory = os.path.join(processed_directory, split)
        assert get_directory_hash(split_directory) == hash_dict[split]

        csv_file_list = glob.glob(os.path.join(split_directory, "*.csv"))
        assert len(csv_file_list) == cfg.num_csv_files

        for file in csv_file_list:
            with open(file, "r") as fh:
                lines = fh.readlines()
                total_lines += len(lines) - 1
                assert lines[0].strip() == header

    assert total_lines == num_entries
