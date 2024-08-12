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
from typing import Tuple

import numpy as np
import pytest

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset, _swap_mmap_array


first_array_values = [1, 2, 3, 4, 5]
second_array_values = [10, 9, 8, 7, 6, 5, 4, 3]


@pytest.fixture
def generate_dataset(tmpdir) -> SingleCellMemMapDataset:
    """
    Create a SingleCellMemMapDataset, save and reload it

    Args:
        tmpdir: temporary directory fixture
    Returns:
        A SingleCellMemMapDataset
    """
    ds = SingleCellMemMapDataset(f"{tmpdir}/scy", h5ad_path="tests/test_data/adata_sample0.h5ad")
    ds.save()
    del ds
    reloaded = SingleCellMemMapDataset(f"{tmpdir}/scy")
    return reloaded


@pytest.fixture
def create_and_fill_mmap_arrays(tmpdir) -> Tuple[np.memmap, np.memmap]:
    """
    Instantiate and fill two np.memmap arrays.

    Args:
        tmpdir: temporary directory fixture
    Returns:
        Two instantiated np.memmap arrays.
    """
    arr1 = np.memmap(f"{tmpdir}/x.npy", dtype="uint32", shape=(len(first_array_values),), mode="w+")
    arr1[:] = np.array(first_array_values, dtype="uint32")

    arr2 = np.memmap(f"{tmpdir}/y.npy", dtype="uint32", shape=(len(second_array_values),), mode="w+")
    arr2[:] = np.array(second_array_values, dtype="uint32")
    return arr1, arr2


@pytest.fixture
def compare_fn():
    def _compare(dns: SingleCellMemMapDataset, dt: SingleCellMemMapDataset) -> bool:
        """
        Returns whether two SingleCellMemMapDatasets are equal

        Args:
            dns: SingleCellMemMapDataset
            dnt: SingleCellMemMapDataset
        Returns:
            True if these datasets are equal.
        """

        assert dns.number_of_rows() == dt.number_of_rows()
        assert dns.number_of_values() == dt.number_of_values()
        assert dns.number_nonzero_values() == dt.number_nonzero_values()
        assert dns.number_of_variables() == dt.number_of_variables()
        assert dns.number_of_rows() == dt.number_of_rows()
        for row_idx in range(len(dns)):
            assert (dns[row_idx][0] == dt[row_idx][0]).all()
            assert (dns[row_idx][1] == dt[row_idx][1]).all()

    return _compare


def test_empty_dataset_save_and_reload(tmpdir):
    ds = SingleCellMemMapDataset(data_path=f"{tmpdir}/scy", num_rows=2, num_elements=10)
    ds.save()
    del ds
    reloaded = SingleCellMemMapDataset(f"{tmpdir}/scy")
    assert reloaded.number_of_rows() == 0
    assert reloaded.number_of_variables() == [0]
    assert reloaded.number_of_values() == 0
    assert len(reloaded) == 0
    assert len(reloaded[1][0]) == 0


def test_wrong_arguments_for_dataset(tmpdir):
    with pytest.raises(
        ValueError, match=r"An np.memmap path, an h5ad path, or the number of elements and rows is required"
    ):
        SingleCellMemMapDataset(data_path=f"{tmpdir}/scy")


def test_load_h5ad(tmpdir):
    ds = SingleCellMemMapDataset(f"{tmpdir}/scy", h5ad_path="tests/test_data/adata_sample0.h5ad")
    assert ds.number_of_rows() == 8
    assert ds.number_of_variables() == [10]
    assert len(ds) == 8
    assert ds.number_of_values() == 80
    assert ds.number_nonzero_values() == 5
    np.isclose(ds.sparsity(), 0.9375, rtol=1e-6)
    assert len(ds) == 8


def test_h5ad_no_file(tmpdir):
    ds = SingleCellMemMapDataset(data_path=f"{tmpdir}/scy", num_rows=2, num_elements=10)
    with pytest.raises(FileNotFoundError, match=rf"Error: could not find h5ad path {tmpdir}/a"):
        ds.load_h5ad(anndata_path=f"{tmpdir}/a")


def test_swap_mmap_array_result_has_proper_length(tmpdir, create_and_fill_mmap_arrays):
    x_arr, y_arr = create_and_fill_mmap_arrays
    x_path = f"{tmpdir}/x.npy"
    y_path = f"{tmpdir}/y.npy"

    _swap_mmap_array(x_arr, x_path, y_arr, y_path)

    x_now_y = np.memmap(y_path, dtype="uint32", shape=(len(first_array_values),), mode="r+")
    y_now_x = np.memmap(x_path, dtype="uint32", shape=(len(second_array_values),), mode="r+")

    assert len(x_now_y) == len(first_array_values)
    assert np.array_equal(x_now_y, np.array(first_array_values))
    assert len(y_now_x) == len(second_array_values)
    assert np.array_equal(y_now_x, np.array(second_array_values))


def test_swap_mmap_no_file(tmpdir, create_and_fill_mmap_arrays):
    x_arr, y_arr = create_and_fill_mmap_arrays
    with pytest.raises(FileNotFoundError, match=rf"The destination file {tmpdir}/z.npy does not exist"):
        _swap_mmap_array(x_arr, f"{tmpdir}/x.npy", y_arr, f"{tmpdir}/z.npy")


def test_swap_mmap_with_delete_source(tmpdir, create_and_fill_mmap_arrays):
    x_arr, y_arr = create_and_fill_mmap_arrays
    x_path = f"{tmpdir}/x.npy"
    y_path = f"{tmpdir}/y.npy"
    _swap_mmap_array(x_arr, x_path, y_arr, y_path, destroy_src=True)

    assert not os.path.exists(x_path)
    x_now_y = np.memmap(y_path, dtype="uint32", shape=(len(first_array_values),), mode="r+")
    assert len(x_now_y) == len(first_array_values)
    assert np.array_equal(x_now_y, np.array(first_array_values))


def test_SingleCellMemMapDataset_constructor(generate_dataset):
    assert generate_dataset.number_of_rows() == 8
    assert generate_dataset.number_of_variables() == [10]
    assert generate_dataset.number_of_values() == 80
    assert generate_dataset.number_nonzero_values() == 5
    assert np.isclose(generate_dataset.sparsity(), 0.9375, rtol=1e-6)
    assert len(generate_dataset) == 8

    assert generate_dataset.shape() == (8, [10])


def test_SingleCellMemMapDataset_get_row(generate_dataset):
    assert len(generate_dataset[0][0]) == 1
    vals, cols = generate_dataset[0]
    assert vals[0] == 6.0
    assert cols[0] == 2
    assert len(generate_dataset[1][1]) == 0
    assert len(generate_dataset[1][0]) == 0
    vals, cols = generate_dataset[2]
    assert vals[0] == 19.0
    assert cols[0] == 2
    vals, cols = generate_dataset[7]
    assert vals[0] == 1.0
    assert cols[0] == 8


def test_SingleCellMemMapDataset_get_row_colum(generate_dataset):
    assert generate_dataset.get_row_column(0, 0, impute_missing_zeros=True) == 0.0
    assert generate_dataset.get_row_column(0, 0, impute_missing_zeros=False) is None
    assert generate_dataset.get_row_column(0, 2) == 6.0
    assert generate_dataset.get_row_column(6, 3) == 16.0
    assert generate_dataset.get_row_column(3, 2) == 12.0


def test_SingleCellMemMapDataset_get_row_padded(generate_dataset):
    padded_row, feats = generate_dataset.get_row_padded(0, return_features=True, feature_vars="feature_name")
    assert len(padded_row) == 10
    assert padded_row[2] == 6.0
    assert len(feats) == 10
    assert generate_dataset.get_row_padded(0)[0][0] == 0.0
    assert generate_dataset.data[0] == 6.0
    assert generate_dataset.data[1] == 19.0
    assert len(generate_dataset.get_row_padded(2)[0]) == 10


def test_concat_SingleCellMemMapDatasets_same(tmpdir):
    ds = SingleCellMemMapDataset(f"{tmpdir}/scy", h5ad_path="tests/test_data/adata_sample0.h5ad")
    dt = SingleCellMemMapDataset(f"{tmpdir}/sct", h5ad_path="tests/test_data/adata_sample0.h5ad")
    dt.concat(ds)

    assert dt.number_of_rows() == 2 * ds.number_of_rows()
    assert dt.number_of_values() == 2 * ds.number_of_values()
    assert dt.number_nonzero_values() == 2 * ds.number_nonzero_values()


def test_concat_SingleCellMemMapDatasets_diff(tmpdir):
    ds = SingleCellMemMapDataset(f"{tmpdir}/scy", h5ad_path="tests/test_data/adata_sample0.h5ad")
    dt = SingleCellMemMapDataset(f"{tmpdir}/sct", h5ad_path="tests/test_data/adata_sample1.h5ad")

    exp_number_of_rows = ds.number_of_rows() + dt.number_of_rows()
    exp_n_val = ds.number_of_values() + dt.number_of_values()
    exp_nnz = ds.number_nonzero_values() + dt.number_nonzero_values()
    dt.concat(ds)
    assert dt.number_of_rows() == exp_number_of_rows
    assert dt.number_of_values() == exp_n_val
    assert dt.number_nonzero_values() == exp_nnz


def test_concat_SingleCellMemMapDatasets_multi(tmpdir, compare_fn):
    ds = SingleCellMemMapDataset(f"{tmpdir}/scy", h5ad_path="tests/test_data/adata_sample0.h5ad")
    dt = SingleCellMemMapDataset(f"{tmpdir}/sct", h5ad_path="tests/test_data/adata_sample1.h5ad")
    dx = SingleCellMemMapDataset(f"{tmpdir}/sccx", h5ad_path="tests/test_data/adata_sample2.h5ad")
    exp_n_obs = ds.number_of_rows() + dt.number_of_rows() + dx.number_of_rows()
    dt.concat(ds)
    dt.concat(dx)
    assert dt.number_of_rows() == exp_n_obs
    dns = SingleCellMemMapDataset(f"{tmpdir}/scdns", h5ad_path="tests/test_data/adata_sample1.h5ad")

    dns.concat([ds, dx])
    compare_fn(dns, dt)
