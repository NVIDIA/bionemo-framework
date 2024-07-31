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
import pathlib
from unittest.mock import MagicMock

import numpy as np
import scanpy

from bionemo.contrib.data.singlecell.dataset import SingleCellDataset


# TODO(@jstjohn) use fixtures for pulling down data and checkpoints
test_script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
bionemo2_root = pathlib.Path("/workspace/bionemo2")
data_path = bionemo2_root / "test_data/cellxgene_2023-12-15_small/processed_data"


def test_dataset_init_lookup():
    data_path = bionemo2_root / "test_data/cellxgene_2023-12-15_small/processed_data/test"
    input_data_path = bionemo2_root / "test_data/cellxgene_2023-12-15_small/input_data/test"
    # This is a convention based on the way the data is generated, not a general rule for how to look
    #  up raw data paths. In general there is no guarantee that we have the path to the original h5ad data.
    tokenizer = MagicMock()

    dataset = SingleCellDataset(data_path, tokenizer, keep_metadata=True)
    first_metadata = dataset.metadata_lookup(0)  # I think we can open these up directly to test

    scanpy.read_h5ad(
        os.path.join(
            input_data_path,
            first_metadata["file_path"].replace("data/cellxgene_2023-12-15_small/input_data/test/", ""),
        )
    )

    last_metadata = dataset.metadata_lookup(len(dataset) - 1)
    scanpy.read_h5ad(
        os.path.join(
            input_data_path, last_metadata["file_path"].replace("data/cellxgene_2023-12-15_small/input_data/test/", "")
        )
    )

    random_metadata = dataset.metadata_lookup(150)
    scanpy.read_h5ad(
        os.path.join(
            input_data_path,
            random_metadata["file_path"].replace("data/cellxgene_2023-12-15_small/input_data/test/", ""),
        )
    )

    assert len(dataset) == 7689


def test_dataset_ccum():
    data_path = bionemo2_root / "test_data/cellxgene_2023-12-15_small/processed_data/test"
    tokenizer = MagicMock()

    dataset = SingleCellDataset(data_path, tokenizer, keep_metadata=True)
    # should sum to the total length
    assert len(dataset) == sum(
        [m["shape"][0] for m in dataset.metadata.values()]
    )  # A dataset is the shape of the individual metadata shapes
    assert len(dataset.dataset_ccum) == 2  # Two datasets

    # NOTE(SKH) these are the source files that created the memmap stored in the metadata, not actual files that must exist.
    datasets = [
        "data/cellxgene_2023-12-15_small/input_data/test/assay__10x_3_v2/sex__male/development_stage__45-year-old_human_stage/self_reported_ethnicity__unknown/tissue_general__small_intestine/dataset_id__ee195b7d-184d-4dfa-9b1c-51a7e601ac11/sidx_19480503_2689_0.h5ad",
        "data/cellxgene_2023-12-15_small/input_data/test/assay__10x_3_v3/sex__male/development_stage__42-year-old_human_stage/self_reported_ethnicity__European/tissue_general__brain/dataset_id__00476f9f-ebc1-4b72-b541-32f912ce36ea/sidx_29791758_10099_1.h5ad",
    ]
    set(dataset.dataset_map.values()) == set(datasets)

    first_ds_key = datasets[0]
    second_ds_key = datasets[1]

    # Exhaustive search over did lookup, 100 elements for each, should map to the appropriate dataset
    assert all(
        dataset.metadata_lookup(i) == dataset.metadata[first_ds_key]
        for i in range(dataset.metadata[first_ds_key]["shape"][0])
    )
    assert all(
        dataset.metadata_lookup(i) == dataset.metadata[second_ds_key]
        for i in range(dataset.metadata[first_ds_key]["shape"][0], len(dataset))
    )


def test_dataset_process_item():
    tokenizer = MagicMock()

    tokenizer.pad_token = "pad"
    tokenizer.cls_token = "cls"
    tokenizer.mask_token = "mask"
    tokenizer.ukw_token = "ukn"
    tokenizer.gene_tok_to_ens = lambda x: x

    # Need this to mock the underlying dictionary behavior with arbitrary keys
    class gene_to_ens:
        @staticmethod
        def get(x, other):
            return x

    tokenizer.gene_to_ens = gene_to_ens
    tokenizer.vocab = {"GENE0": 1, "GENE1": 2, "GENE2": 3, "ukn": 7, "mask": 7, "cls": 5, "pad": 4}

    def tok_to_id(tok):
        if tok == tokenizer.pad_token:
            return 4
        if tok == tokenizer.cls_token:
            return 5
        if tok == tokenizer.mask_token:
            return 6
        if tok == tokenizer.ukw_token:
            return 7
        if tok == "GENE0":
            return 1
        if tok == "GENE1":
            return 2
        if tok == "GENE2":
            return 3

    tokenizer.token_to_id = tok_to_id
    # Create a sample input item
    input_item = {
        "expression": np.array([1, 2, 3]),
        "indices": np.array([0, 1, 2]),
        "metadata": [f"GENE{i}" for i in range(3)],
    }

    # Process the input item
    from bionemo.contrib.data.singlecell.dataset import process_item

    processed_item = process_item(
        input_item["expression"],
        input_item["indices"],
        input_item["metadata"],
        tokenizer,
        gene_median={"GENE0": 1, "GENE1": 1, "GENE2": 1},
        max_len=5,
        mask_prob=0,
    )
    assert all(processed_item["text"] == [5, 3, 2, 1, tok_to_id("pad")])  # CLS, 1, 2, 3, PAD
    # The following is used as 'attention_mask' in NeMo, so it's probably the opposite of what you think it should be.
    assert all(processed_item["attention_mask"] == [1, 1, 1, 1, 0])  # NO, NO, NO, NO, YES

    ###### Check median rank norm, sorts in ascending order. ######

    # 1/6/1=1/6 , 2/3/6 =2/18=1/9, 3/6/6 =3/36=1/12 => 3, 2, 1
    processed_item = process_item(
        input_item["expression"],
        input_item["indices"],
        input_item["metadata"],
        tokenizer,
        gene_median={"GENE0": 1, "GENE1": 3, "GENE2": 6},
        max_len=4,
        mask_prob=0,
        target_sum=1,
    )
    assert all(processed_item["text"] == [5, 1, 2, 3])

    # Checks median norm, should change the order due to medians.
    # 1/6/.5=1/3, 2/6/1=2/6=1/3, 3/6/2=3/12=1/4
    processed_item = process_item(
        input_item["expression"],
        input_item["indices"],
        input_item["metadata"],
        tokenizer,
        gene_median={"GENE0": 0.5, "GENE1": 1, "GENE2": 2},
        max_len=4,
        mask_prob=0,
        target_sum=1,
    )
    assert all(processed_item["text"] == [5, 1, 2, 3])

    # checks padding is added for a short sequence
    processed_item = process_item(
        input_item["expression"],
        input_item["indices"],
        input_item["metadata"],
        tokenizer,
        gene_median={"GENE0": 1, "GENE1": 1, "GENE2": 1},
        max_len=5,
        mask_prob=0,
        target_sum=1,
    )
    assert all(processed_item["text"] == [5, 3, 2, 1, 4])

    #    Masking - test that no special tokens are masked, all when 100, none when 0
    processed_item = process_item(
        input_item["expression"],
        input_item["indices"],
        input_item["metadata"],
        tokenizer,
        gene_median={"GENE0": 1, "GENE1": 1, "GENE2": 1},
        random_token_prob=0,
        max_len=5,
        mask_prob=1.0,
        mask_token_prob=1.0,
        target_sum=1,
    )
    # NOTE: we need to set masked tokens to MASK so that they are decoded.
    assert all(processed_item["text"] == [5, 6, 6, 6, 4])  # CLS, MASK, MASK, MASK, PAD
    # NOTE: MASKed tokens are the only ones used by loss
    assert all(processed_item["loss_mask"] == [0, 1, 1, 1, 0])  # NO, MASK, MASK, MASK, NO
    # the ARBITRARY labels should be ignored due to loss mask.
    assert all(processed_item["labels"] == [-1, 3, 2, 1, -1])  # ARBITRARY, 3, 2, 1, ARBITRARY
    assert all(processed_item["is_random"] == 0)  # For now we don't support random masking.

    # checks sequence is truncated for a long sequence
    processed_item = process_item(
        input_item["expression"],
        input_item["indices"],
        input_item["metadata"],
        tokenizer,
        gene_median={"GENE0": 1, "GENE1": 1, "GENE2": 1},
        max_len=3,
        mask_prob=0,
        target_sum=1,
    )
    # Randomly permutes the other values, no fixed order
    assert all(processed_item["text"][0] == [5])
    # Truncate to exactly three items
    assert len(processed_item["text"]) == 3
    assert all(processed_item["loss_mask"] == [False, False, False])  # No mask applied


def test_lookup_cell_by_idx():
    import numpy as np
    from scipy.sparse import csr_matrix

    # Function to create a sparse matrix with specified sparsity and a unique identifier
    def create_sparse_matrix(rows, cols, sparsity, identifier):
        dense_matrix = np.zeros((rows, cols))
        num_non_zeros = int((1 - sparsity) * rows * cols)
        positions = np.random.choice(rows * cols, num_non_zeros, replace=False)
        # Add identifier value to each non-sparse position
        np.put(dense_matrix, positions, identifier)
        return dense_matrix

    # Define dimensions and sparsity
    cols = 2000
    rows1, rows2, rows3 = 100, 150, 200
    sparsity = 0.9

    # Create three matrices with unique identifiers
    matrix1 = create_sparse_matrix(rows1, cols, sparsity, 1)
    matrix2 = create_sparse_matrix(rows2, cols, sparsity, 2)
    matrix3 = create_sparse_matrix(rows3, cols, sparsity, 3)

    # Concatenate the matrices and get CSR
    big_matrix = np.vstack([matrix1, matrix2, matrix3])
    csr = csr_matrix(big_matrix)

    mocked_dataset: SingleCellDataset = MagicMock()
    mocked_dataset.lookup_cell_by_idx = SingleCellDataset.lookup_cell_by_idx
    # Disable the guard
    mocked_dataset.assert_increasing_columns = False

    # Assign the sparse index structure
    mocked_dataset.gene_data = csr.data
    mocked_dataset.gene_data_ptr = csr.indptr
    mocked_dataset.gene_data_indices = csr.indices

    gene_data1, _, _ = mocked_dataset.lookup_cell_by_idx(mocked_dataset, 0)
    assert all(gene_data1 == 1)

    # Test boundaries for the second dataset
    gene_data2, _, _ = mocked_dataset.lookup_cell_by_idx(mocked_dataset, 100)
    assert all(gene_data2 == 2)
    gene_data3, _, _ = mocked_dataset.lookup_cell_by_idx(mocked_dataset, 101)
    assert all(gene_data3 == 2)
    # Test boundaries for the third dataset
    gene_data4, _, _ = mocked_dataset.lookup_cell_by_idx(mocked_dataset, 250)
    assert all(gene_data4 == 3)
    gene_data5, _, _ = mocked_dataset.lookup_cell_by_idx(mocked_dataset, 251)
    assert all(gene_data5 == 3)
