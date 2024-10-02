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
from unittest.mock import MagicMock

import numpy as np
import scanpy

from bionemo.data.singlecell.dataset import SingleCellDataset


def test_dataset_init_lookup():
    data_path = "examples/tests/test_data/cellxgene_2023-12-15_small/processed_data/test"
    # This is a convention based on the way the data is generated, not a general rule for how to look
    #  up raw data paths. In general there is no guarantee that we have the path to the original h5ad data.
    input_data_path = data_path.replace("processed_data", "input_data")
    tokenizer = MagicMock()

    dataset = SingleCellDataset(data_path, tokenizer)
    first = dataset.lookup_cell_by_idx(0)  # I think we can open these up directly to test
    print(first[2].keys())

    scanpy.read_h5ad(
        os.path.join(
            input_data_path, first[2]["file_path"].replace("data/cellxgene_2023-12-15_small/input_data/test/", "")
        )
    )

    last = dataset.lookup_cell_by_idx(len(dataset) - 1)
    scanpy.read_h5ad(
        os.path.join(
            input_data_path, last[2]["file_path"].replace("data/cellxgene_2023-12-15_small/input_data/test/", "")
        )
    )

    random = dataset.lookup_cell_by_idx(150)
    scanpy.read_h5ad(
        os.path.join(
            input_data_path, random[2]["file_path"].replace("data/cellxgene_2023-12-15_small/input_data/test/", "")
        )
    )

    assert len(dataset) == 7689


def test_dataset_ccum():
    data_path = "examples/tests/test_data/cellxgene_2023-12-15_small/processed_data/test"
    tokenizer = MagicMock()

    dataset = SingleCellDataset(data_path, tokenizer)
    # should sum to the total length
    assert len(dataset) == sum(
        [m["shape"][0] for m in dataset.metadata.values()]
    )  # A dataset is the shape of the individual metadata shapes
    assert len(dataset.dataset_ccum) == 2  # Two datasets

    # we expect all two of our test files to end up in dataset_map
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
        "metadata": {"feature_ids": [f"GENE{i}" for i in range(3)]},
    }

    # Process the input item
    from bionemo.data.singlecell.dataset import process_item

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
    assert all(processed_item["padding_mask"] == [1, 1, 1, 1, 0])  # NO, NO, NO, NO, YES

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


def test_neighbor_lookup():
    data_path = "/workspace/bionemo/examples/tests/test_data/cellxgene_2023-12-15_small/processed_data/pseudotime/test"
    # ensure to run add_pt_neighbors and memmaps creation on data_path
    data_path.replace("processed_data", "input_data")
    tokenizer = MagicMock()
    medians = MagicMock()
    # Configure the MagicMock to return 1 for any key access
    medians.__getitem__.return_value = 1
    dataset = SingleCellDataset(data_path, tokenizer, next_cell_prediction=True, median_dict=medians)

    len_datasets = [2689, 2689, 5000, 5000]
    offsets = [0, 2689, 5378, 10378]
    no_neighbors = 0
    ind_offsets = [[offsets[i]] * len_datasets[i] for i in range(4)]  # calculate the offset for each dataset sample
    merged_list = sum(ind_offsets, [])

    num_neighbors = [0, 5, 10]  # iterate over number of neighbors
    for i, offset in zip(range(dataset.num_samples), merged_list):
        if i in range(offsets[1]) or i in range(offsets[2], offsets[3]):
            assert not dataset.sample_has_neighbor(i)  # samples from 1 and 3 datasets have no neighbors
            no_neighbors += 1
        else:
            metadata = dataset.metadata_lookup(i)
            i_offset = i - offset
            k_i = num_neighbors[i_offset % 3]
            if k_i == 0:
                assert not dataset.sample_has_neighbor(i)  # every third cell has no neighbors
                no_neighbors += 1
                continue
            neighbor_idx, _, _, _ = dataset.lookup_neighbor_by_idx(i)
            n_samples = metadata["shape"][0]
            permitted_range = [
                offset + (x - offset) % n_samples for x in range(i + 1, i + k_i + 1)
            ]  # defined by the way dummy neighbors were created
            assert neighbor_idx in permitted_range
            sample_data, col_idxs, metadata_neighb = dataset.lookup_cell_by_idx(neighbor_idx)
            assert metadata == metadata_neighb

    assert no_neighbors == (2689 + 5000 + 2689 // 3 + 1 + 5000 // 3 + 1)


def test_dataset_process_ncp_item():
    tokenizer = MagicMock()

    tokenizer.pad_token = "pad"
    tokenizer.cls_token = "cls"
    tokenizer.mask_token = "mask"
    tokenizer.ukw_token = "ukn"
    tokenizer.sep_token = "sep"
    tokenizer.gene_tok_to_ens = lambda x: x

    # Need this to mock the underlying dictionary behavior with arbitrary keys
    class gene_to_ens:
        @staticmethod
        def get(x, other):
            return x

    tokenizer.gene_to_ens = gene_to_ens
    tokenizer.vocab = {"GENE0": 1, "GENE1": 2, "GENE2": 3, "ukn": 7, "mask": 6, "cls": 5, "pad": 4, "sep": 8}

    def tok_to_id(tok):
        if tok == tokenizer.pad_token:
            return 4
        if tok == tokenizer.cls_token:
            return 5
        if tok == tokenizer.mask_token:
            return 6
        if tok == tokenizer.ukw_token:
            return 7
        if tok == tokenizer.sep_token:
            return 8
        if tok == "GENE0":
            return 1
        if tok == "GENE1":
            return 2
        if tok == "GENE2":
            return 3

    tokenizer.token_to_id = tok_to_id
    # Create a sample input item
    input_item_current = {
        "expression": np.array([1, 2, 3]),
        "indices": np.array([0, 1, 2]),
        "metadata": {"feature_ids": [f"GENE{i}" for i in range(3)]},
    }

    input_item_next = {
        "expression": np.array([3, 2, 1]),
        "indices": np.array([0, 1, 2]),
        "metadata": {"feature_ids": [f"GENE{i}" for i in range(3)]},
    }

    # Process the input item
    from bionemo.data.singlecell.dataset import process_item_ncp

    processed_item = process_item_ncp(
        input_item_current["expression"],
        input_item_current["indices"],
        input_item_next["expression"],
        input_item_next["indices"],
        input_item_next["metadata"],
        tokenizer,
        gene_median={"GENE0": 1, "GENE1": 1, "GENE2": 1},
        max_len=10,
        mask_prob=0,
        token_selection_policy="identity",
    )

    assert all(
        processed_item["text"]
        == [tok_to_id("cls"), 3, 2, 1, tok_to_id("pad"), tok_to_id("sep"), 1, 2, 3, tok_to_id("pad")]
    )  # CLS, 3,2,1, PAD, SEP, 1,2,3, PAD
    # The following is used as 'attention_mask' in NeMo, so it's probably the opposite of what you think it should be.
    assert all(
        processed_item["padding_mask"] == [1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    )  # NO, NO, NO, NO, YES,  NO, NO, NO, NO, YES

    ###### Check median rank norm, sorts in ascending order. ######
    # 1/6 = 1/6, 2/3 = 2/3, 3/2=1.5 => [3, 2, 1] for current
    # 3/6 = 1/2, 2/3 = 2/3, 1/2=1/2 => [2, 1, 3] for next

    processed_item = process_item_ncp(
        input_item_current["expression"],
        input_item_current["indices"],
        input_item_next["expression"],
        input_item_next["indices"],
        input_item_current["metadata"],
        tokenizer,
        gene_median={"GENE0": 6, "GENE1": 3, "GENE2": 2},
        max_len=10,
        mask_prob=0,
        target_sum=1,
    )

    assert all(
        processed_item["text"]
        == [tok_to_id("cls"), 3, 2, 1, tok_to_id("pad"), tok_to_id("sep"), 2, 1, 3, tok_to_id("pad")]
    )

    # checks sequence is truncated for a long sequence
    processed_item = process_item_ncp(
        input_item_current["expression"],
        input_item_current["indices"],
        input_item_next["expression"],
        input_item_next["indices"],
        input_item_current["metadata"],
        tokenizer,
        gene_median={"GENE0": 1, "GENE1": 1, "GENE2": 1},
        max_len=6,
        mask_prob=0,
        target_sum=1,
    )
    processed_item["text"]

    assert all(processed_item["text"] == [5, 3, 2, 8, 1, 2])

    # checks sequence is truncated for a long sequence
    processed_item = process_item_ncp(
        input_item_current["expression"],
        input_item_current["indices"],
        input_item_next["expression"],
        input_item_next["indices"],
        input_item_current["metadata"],
        tokenizer,
        gene_median={"GENE0": 1, "GENE1": 1, "GENE2": 1},
        max_len=5,
        mask_prob=0,
        target_sum=1,
    )

    # Truncate to exactly 5 items
    assert len(processed_item["text"]) == 5
    assert all(processed_item["text"] == [5, 3, 2, 8, 1])  # the next cell has -1 element compared to the current cell

    #  Masking - test that no special tokens are masked, all when 100, none when 0
    processed_item = process_item_ncp(
        input_item_current["expression"],
        input_item_current["indices"],
        input_item_next["expression"],
        input_item_next["indices"],
        input_item_current["metadata"],
        tokenizer,
        gene_median={"GENE0": 1, "GENE1": 1, "GENE2": 1},
        random_token_prob=0,
        max_len=10,
        mask_prob=1,
        mask_token_prob=1.0,
        target_sum=1,
    )
    assert all(processed_item["text"] == [5, 3, 2, 1, 4, 8, 6, 6, 6, 4])
    # CLS, GENE2, GENE1, GENE0, PAD, SEP, MASK, MASK, MASK, PAD

    assert all(
        processed_item["loss_mask"] == [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
    )  #  MASKed tokens are the only ones used by loss
    # the ARBITRARY labels should be ignored due to loss mask.
    assert all(processed_item["labels"] == [-1, -1, -1, -1, -1, -1, 1, 2, 3, -1])
