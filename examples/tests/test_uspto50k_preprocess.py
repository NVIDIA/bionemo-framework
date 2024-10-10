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

import pandas as pd
import pytest

from bionemo.data.preprocess.molecule.uspto50k_preprocess import USPTO50KPreprocess


@pytest.mark.xfail(reason="NGC CLI might not be present in environment.")
def test_uspto50k_preprocess(tmp_path):
    ngc_registry_target = "uspto_50k_dataset"
    ngc_registry_version = "v23.06"
    filename = "uspto_50.pickle"
    data_preprocessor = USPTO50KPreprocess(data_dir=str(tmp_path))
    data_preprocessor.prepare_dataset(
        ngc_registry_target=ngc_registry_target, ngc_registry_version=ngc_registry_version, force=True
    )

    raw_data_filepath = os.path.join(data_preprocessor.download_dir, filename)
    assert raw_data_filepath == data_preprocessor.datapath_raw
    assert os.path.join(tmp_path, "raw", filename) == data_preprocessor.datapath_raw
    assert os.listdir(data_preprocessor.download_dir)[0] == filename
    assert os.path.join(tmp_path, "processed") == data_preprocessor.processed_dir
    # reading raw data
    df_raw = pd.read_pickle(raw_data_filepath)
    # Make sure that if the file already exists, that it can be preprocessed without ngc registry information
    data_preprocessor2 = USPTO50KPreprocess(data_dir=str(tmp_path))
    data_preprocessor2.prepare_dataset(force=True)
    assert raw_data_filepath == data_preprocessor2.datapath_raw
    assert os.path.join(tmp_path, "raw", filename) == data_preprocessor2.datapath_raw
    assert os.listdir(data_preprocessor2.download_dir)[0] == filename
    assert os.path.join(tmp_path, "processed") == data_preprocessor2.processed_dir

    assert all(folder in data_preprocessor.splits for folder in os.listdir(data_preprocessor.processed_dir))

    assert len(df_raw) == 50037
    assert all(col in ["reactants_mol", "products_mol", "reaction_type", "set"] for col in df_raw.columns)

    assert all(s in ["train", "valid", "test"] for s in df_raw.set.unique())
    assert all(s in ["train", "val", "test"] for s in data_preprocessor.splits)

    split_sizes = {"train": 40029, "val": 5004, "test": 5004}
    for split in data_preprocessor.splits:
        filepath = os.path.join(data_preprocessor.get_split_dir(split), data_preprocessor.data_file)
        # reading processed split data
        df = pd.read_csv(filepath)
        assert len(df) == split_sizes[split]
        assert all(col in ["reaction_type", "set", "reactants", "products"] for col in df.columns)

    assert df.products[54] == "O=[N+]([O-])c1ccccc1CCN1CCN(c2ccccc2)CC1"
    assert df.reactants[565] == "Nc1ccc(O)c(C(=O)O)c1.O=C(O)C1CCCCC1"
