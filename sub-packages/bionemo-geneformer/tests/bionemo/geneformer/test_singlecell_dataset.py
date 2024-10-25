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
import torch

from bionemo.core.utils import random_utils
from bionemo.geneformer.data.singlecell.dataset import SingleCellDataset
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from nemo.utils import logging

def test_load_sc_datasets(tmp_path, test_directory): 
    tokenizer = MagicMock()
    sc_memmap_dataset_path0 = tmp_path / "test_data_0"
    ds_0 = SingleCellMemMapDataset(sc_memmap_dataset_path0, h5ad_path=test_directory / "adata_sample0.h5ad")  # create the memmap dataset format from h5ad for testing purposes
    dataset0 = SingleCellDataset(sc_memmap_dataset_path0, tokenizer)
    assert len(dataset0) == len(ds_0) == 8
    sc_memmap_dataset_path1 = tmp_path / "test_data_1"
    ds_1 = SingleCellMemMapDataset(sc_memmap_dataset_path1, h5ad_path=test_directory / "adata_sample1.h5ad")  # create the memmap dataset format from h5ad for testing purposes
    dataset1 = SingleCellDataset(sc_memmap_dataset_path1, tokenizer)
    assert len(dataset1) == len(ds_1) == 6
    sc_memmap_dataset_path2 = tmp_path / "test_data_2"
    ds_2 = SingleCellMemMapDataset(sc_memmap_dataset_path2, h5ad_path=test_directory / "adata_sample2.h5ad")  # create the memmap dataset format from h5ad for testing purposes
    dataset2 = SingleCellDataset(sc_memmap_dataset_path2, tokenizer)
    assert len(dataset2) ==  len(ds_2) == 100
    

def test_get_item(tmp_path, test_directory): 
    sc_memmap_dataset_path0 = tmp_path / "test_data_0"
    ds_0 = SingleCellMemMapDataset(sc_memmap_dataset_path0, h5ad_path=test_directory / "modified_adata_sample0.h5ad")  # create the memmap dataset format from h5ad for testing purposes
    preprocessor = GeneformerPreprocess(
        download_directory=sc_memmap_dataset_path0,
        medians_file_path=sc_memmap_dataset_path0 / "medians.json",
        tokenizer_vocab_path=sc_memmap_dataset_path0 / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")
    dataset0 = SingleCellDataset(sc_memmap_dataset_path0, tokenizer, median_dict=median_dict)  # type: ignore
    item = dataset0.__getitem__(0)
    assert(all(item["text"]) == torch.tensor([0]))
    assert(all(item["types"]) == torch.tensor([0]))
    assert(item["attention_mask"][0] == torch.tensor([1]))
    assert(item["labels"][0] == torch.tensor([-1]))
    assert(item["loss_mask"][0] == torch.tensor([False]))
    assert(all(item["is_random"]) == torch.tensor([0]))

def test_lookup_row_by_index(tmp_path, sc_test_data_directory): 
    tokenizer = MagicMock()
    dataset = SingleCellDataset(tmp_path / sc_test_data_directory, tokenizer)
    values, feature_ids = dataset.scdl.get_row(0, return_features=True, feature_vars=["feature_id"])
    gene_data, col_idxs = values[0], values[1]
    assert  len(gene_data) == 1594
    assert len(col_idxs) == 1594
    assert len(feature_ids) == 60664

    values, feature_ids = dataset.scdl.get_row(len(dataset) - 1, return_features=True, feature_vars=["feature_id"])
    gene_data, col_idxs = values[0], values[1]
    assert  len(gene_data) == 4930
    assert len(col_idxs) == 4930
    assert len(feature_ids) == 60664

def test_dataset_process_item():
    tokenizer = MagicMock()

    tokenizer.pad_token = "pad"
    tokenizer.cls_token = "cls"
    tokenizer.mask_token = "mask"
    tokenizer.ukw_token = "ukn"
    tokenizer.gene_tok_to_ens = lambda x: x
    tokenizer.mask_token_id = 6

    # Need this to mock the underlying dictionary behavior with arbitrary keys
    class gene_to_ens:
        @staticmethod
        def get(x, other):
            return x

    tokenizer.gene_to_ens = gene_to_ens
    tokenizer.vocab = {"GENE0": 1, "GENE1": 2, "GENE2": 3, "ukn": 7, "mask": 6, "cls": 5, "pad": 4}

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
    from bionemo.geneformer.data.singlecell.dataset import process_item
    seed = 42 
    rng = np.random.default_rng(seed)
    seed= random_utils.get_seed_from_rng(rng)
    idx = 0
    rng = np.random.default_rng([seed, idx])


    processed_item = process_item(
        input_item["expression"],
        input_item["indices"],
        input_item["metadata"],
        tokenizer,
        gene_median={"GENE0": 1, "GENE1": 1, "GENE2": 1},
        max_len=5,
        mask_prob=0,
        rng= rng
    )
    assert all(processed_item["text"] == torch.tensor([5, 3, 2, 1]))# CLS, 1, 2, 3, but in reverse order 
    # The following is used as 'attention_mask' in NeMo, so it's probably the opposite of what you think it should be.
    assert all(processed_item["attention_mask"] == torch.tensor([1, 1, 1, 1])) # this is all 1s 

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
        rng= rng
    )
    assert all(processed_item["text"] == torch.tensor([5, 1, 2, 3]))

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
        rng= rng
    )
    assert all(processed_item["text"] == torch.tensor([5, 1, 2, 3]))

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
        rng= rng
    )
    # NOTE: we need to set masked tokens to MASK so that they are decoded.
    assert all(processed_item["text"] == torch.tensor([5, 6, 6, 6]))  # CLS, MASK, MASK, MASK
    # NOTE: MASKed tokens are the only ones used by loss
    assert all(processed_item["loss_mask"] == torch.tensor([False, True, True, True]))  # NO, MASK, MASK, MASK, NO
    # the ARBITRARY labels should be ignored due to loss mask.
    assert all(processed_item["labels"] == torch.tensor([-1, 3, 2, 1]))  # ARBITRARY, 3, 2, 1, ARBITRARY
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
        rng= rng
    )
    # Randomly permutes the other values, no fixed order
    assert processed_item["text"][0] == torch.tensor([5])
    # Truncate to exactly three items
    assert len(processed_item["text"]) == 3
    assert all(processed_item["loss_mask"] == torch.tensor([False, False, False])) # No mask applied