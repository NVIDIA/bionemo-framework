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
from pathlib import Path

import numpy as np

from bionemo.scdl.io.single_cell_collection import SingleCellCollection
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset


def test_sccollection_empty(tmp_path):
    coll = SingleCellCollection(tmp_path / "sccz")
    assert len(coll.fname_to_mmap) == 0
    assert coll.number_of_rows() == 0
    assert coll.number_of_variables() == [0]
    assert coll.number_of_values() == 0
    assert coll.number_nonzero_values() == 0
    assert coll.data_path == tmp_path / "sccz"


def test_sccollection_basics(tmp_path, test_directory):
    coll = SingleCellCollection(tmp_path / "sccz")
    coll.load_h5ad(test_directory / "adata_sample0.h5ad")
    assert coll.number_of_rows() == 8
    assert coll.number_of_variables() == [10]
    assert coll.number_of_values() == 80
    assert coll.number_nonzero_values() == 5
    assert np.isclose(coll.sparsity(), 0.9375, rtol=1e-6)
    assert coll.shape() == (8, [10])


def test_sccollection_multi(tmp_path, test_directory):
    coll = SingleCellCollection(tmp_path)

    coll.load_h5ad_multi(test_directory / "", max_workers=4, use_processes=False)
    assert sorted(coll.fname_to_mmap) == [
        Path(tmp_path / "adata_sample0"),
        Path(tmp_path / "adata_sample1"),
        Path(tmp_path / "adata_sample2"),
    ]
    for dir_path in coll.fname_to_mmap:
        for fn in ["col_ptr.npy", "data.npy", "features", "metadata.json", "row_ptr.npy", "version.json"]:
            assert os.path.exists(f"{dir_path}/{fn}")

    assert len(coll.fname_to_mmap) == 3
    assert coll.number_of_rows() == 114
    assert sorted(coll.number_of_variables()) == [2, 10, 20]
    assert coll.number_nonzero_values() == 57
    assert coll.number_of_values() == 2092
    assert isinstance(coll.sparsity(), float)
    assert np.isclose(coll.sparsity(), 0.972753346080306, rtol=1e-6)
    shape = coll.shape()
    assert isinstance(shape[0], int)
    assert isinstance(shape[1], list)
    assert shape[0] == 114


def test_sccollection_serialization(tmp_path, test_directory):
    coll = SingleCellCollection(tmp_path / "sccy")
    coll.load_h5ad_multi(test_directory / "", max_workers=4, use_processes=False)
    coll.flatten(tmp_path / "flattened")
    dat = SingleCellMemMapDataset(tmp_path / "flattened")
    assert dat.number_of_rows() == 114
    assert dat.number_of_values() == 2092
    assert dat.number_nonzero_values() == 57
    assert np.isclose(coll.sparsity(), 0.972753346080306, rtol=1e-6)
    for fn in ["col_ptr.npy", "data.npy", "features", "metadata.json", "row_ptr.npy", "version.json"]:
        assert os.path.exists(tmp_path / "flattened" / fn)
