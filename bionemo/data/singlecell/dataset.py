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


import json
import random
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

import numpy as np
from torch.utils.data import Dataset

from bionemo.data.singlecell.utils import sample_or_truncate_plus_pad
from bionemo.tokenizer.gene_tokenizer import GeneTokenizer


class SingleCellDataset(Dataset):
    """
    A dataset class for single-cell pre-training. These can be generated using the sc_memmap.py script. Future
    updates will contain more comprehensive workflows for generating a Sparse Memmap from scRNA-seq.

    Args:
        data_path (str): Path where the single cell files are stored. It should contain the following files:
            - `metadata.json`: Path containing feature subset associated with each dataset.
            - `features.csv`: Feature subset associated with each sample.
            - Gene expression matrix stored in CSR format as `numpy.memmap`:
                - `gene_expression_data.npy`: Gene expression values.
                - `gene_expression_ind.npy`: Gene indices associated with gene values.
                - `gene_expression_ptr.npy`: Column indices for each sample.
            - Temporal neighbors stored in CSR format as `numpy.memmap` if next cell prediction tasks is enabled (any temporal-aware
            tasks such as predicting future cell masked tokens, or temporal relation between a pair of chosen cells):
                - `pseudotime_neighbors_ind.npy`: Indices of neighbors for each sample.
                - `pseudotime_neighbors_ptr.npy`: Column indices for each sample.
        tokenizer: The tokenizer to use for tokenizing the input data.
        median_dict (dict, optional): A dictionary containing median values for each gene. Defaults to None.
        max_len (int, optional): The maximum length of the input sequence. Defaults to 1024.
        mask_prob (float): Probability of masking a token. Defaults to 0.15.
        mask_token_prob (float): Probability of the token [MASK] out of the tokens selected for masking. Defaults to 0.8.
        random_token_prob (float): Probability of a random token out of the tokens selected for masking. Defaults to 0.1.
        prepend_cls_token (bool): Whether to prepend a [CLS] token. Defaults to True.
        assert_increasing_columns (bool): Check if column indices are increasing for looking up genes. This is a way of
        spotting if the sc_memmap.py script produced properly structured sparse files. Defaults to True.
        next_cell_prediction (bool): Include the next cell prediction tasks. Defaults to False.
        no_neighbor_policy (str, optional): In case next_cell_prediction is on, how to handle samples with no neighbors. Defaults to None.

    Attributes:
        data_path (str): Path where the single cell files are stored.
        max_len (int): The maximum length of the input sequence.
        random_token_prob (float): Probability of a random token out of the tokens selected for masking. Defaults to 0.1.
        mask_token_prob (float): Probability of the token [MASK] out of the tokens selected for masking. Defaults to 0.8.
        mask_prob (float): Probability of masking a token. Defaults to 0.15.
        prepend_cls_token (bool): Whether to prepend a [CLS] token. Defaults to True.
        assert_increasing_columns (bool): Check if column indices are increasing for looking up genes.
        metadata (dict): Metadata loaded from `metadata.json`.
        gene_medians (dict): A dictionary containing median values for each gene. If None, a median of '1' is assumed for all genes.
        num_samples (int): The total number of samples in the dataset.
        gene_data (numpy.memmap): Gene expression values stored in CSR format.
        gene_data_indices (numpy.memmap): Gene indices associated with gene values.
        gene_data_ptr (numpy.memmap): Column indices for each sample.
        tokenizer: The tokenizer used for tokenizing the input data.
        dataset_ccum (numpy.ndarray): Cumulative sum of row counts to map row indices to dataset id.
        dataset_map (dict): Mapping of dataset id to dataset name.
        next_cell_prediction (bool): Include the next cell prediction tasks.
        next_cell_indices (numpy.memmap): Indices of next cell to predict stored in CSR format.
        next_cell_ptr (numpy.memmap): Neighbor column indices for each sample in CSR format.
        no_neighbor_policy (str): Handling samples with no neighbors in case next cell masking is enabled.


    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.

    See Also:
        bionemo/data/singlecell/sc_memmap.py - creates the artifacts required for instantiating a singlecell dataset from hdf5 files.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        median_dict: Optional[dict] = None,
        max_len: int = 1024,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
        prepend_cls_token: bool = True,
        assert_increasing_columns: bool = True,
        next_cell_prediction: bool = False,
        no_neighbor_policy: Optional[str] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.max_len = max_len
        self.random_token_prob = random_token_prob
        self.mask_token_prob = mask_token_prob
        self.mask_prob = mask_prob
        self.prepend_cls_token = prepend_cls_token
        # check if column indices are increasing for looking up genes. This is a way of spotting if the sc_memmap.py
        #  script produced properly strctured sparse files.
        self.assert_increasing_columns = assert_increasing_columns
        path = Path(data_path)

        # - metadata
        self.metadata = json.load(open(path / "metadata.json", "r"))

        # - median dict
        self.gene_medians = median_dict

        # - use next cell prediction tasks
        self.next_cell_prediction = next_cell_prediction
        self.no_neighbor_policy = no_neighbor_policy

        # - train/val idxs sampled contiguously
        total_el = sum([v["num_el"] for _, v in self.metadata.items()])
        self.num_samples = sum([v["shape"][0] for _, v in self.metadata.items()])
        # - load data
        self.gene_data = np.memmap(path / "gene_expression_data.npy", dtype="float32", mode="r", shape=(total_el,))

        self.gene_data_indices = np.memmap(
            path / "gene_expression_ind.npy", dtype="int32", mode="r", shape=(total_el,)
        )

        self.gene_data_ptr = np.memmap(
            path / "gene_expression_ptr.npy", dtype="int64", mode="r", shape=(self.num_samples + 1,)
        )
        # - load indices and pointers to the neighbor cell if NCP is on
        if self.next_cell_prediction:
            total_pt_el = sum([v["num_neighbors_el"] for _, v in self.metadata.items()])
            self.next_cell_indices = np.memmap(
                path / "pseudotime_neighbors_ind.npy", dtype="int32", mode="r", shape=(total_pt_el,)
            )

            self.next_cell_ptr = np.memmap(
                path / "pseudotime_neighbors_ptr.npy", dtype="int64", mode="r", shape=(self.num_samples + 1,)
            )

        self.tokenizer = tokenizer

        # map row indices to dataset id
        self.dataset_ccum = np.zeros(
            len(self.metadata),
        )
        # Maps dataset ids to dataset names (used in the metadata dict)
        self.dataset_map = {}
        count = 0
        for i, k in enumerate(self.metadata.keys()):
            self.dataset_ccum[i] = count
            self.dataset_map[i] = k
            count += self.metadata[k]["shape"][0]
        self.dataset_ccum[0] = -1

    def __len__(self) -> int:
        return self.num_samples

    def metadata_lookup(self, idx) -> dict:
        """Go from a cell idx to the file-level metadata associated with that cell."""
        did = sum(~(self.dataset_ccum > idx)) - 1
        metadata = self.metadata[self.dataset_map[did]]
        return metadata

    def lookup_cell_by_idx(self, idx) -> tuple[np.array, np.array, dict]:
        ptr = slice(int(self.gene_data_ptr[idx]), int(self.gene_data_ptr[idx + 1]))
        # col idxs poin to offsets in the original sparse metadata, this is for looking up metadata eg gene names
        col_idxs = np.asarray(self.gene_data_indices[ptr]).astype(int)  # keyed by ptr
        if self.assert_increasing_columns and len(col_idxs) > 1:
            is_increasing = np.diff(col_idxs) > 0
            if not np.all(is_increasing):
                raise ValueError(f"Column indices are not increasing for {np.sum(~is_increasing)} pairs of genes")
        gene_data = np.asarray(self.gene_data[ptr]).astype(int)  # keyed by ptr
        metadata = self.metadata_lookup(idx)
        return gene_data, col_idxs, metadata

    def sample_has_neighbor(self, idx):
        ptr = slice(int(self.next_cell_ptr[idx]), int(self.next_cell_ptr[idx + 1]))
        if ptr.start > ptr.stop:
            raise ValueError(
                "Neighbor pointer can not be decreasing: self.next_cell_ptr[idx] > self.next_cell_ptr[idx + 1]."
            )
        elif ptr.start == ptr.stop:
            return False  # no neighbors
        else:
            return True

    def ncp_no_neighbor_policy(self, idx: int, type: str = "identity"):
        if type == "identity":
            gene_data, col_idxs, metadata = self.lookup_cell_by_idx(idx)
            return process_item(
                gene_data,
                col_idxs,
                metadata,
                self.tokenizer,
                gene_median=self.gene_medians,
                max_len=self.max_len,
                mask_token_prob=self.mask_token_prob,
                mask_prob=self.mask_prob,
                random_token_prob=self.random_token_prob,
                prepend_cls_token=self.prepend_cls_token,
            )
        else:
            raise NotImplementedError

    def lookup_neighbor_by_idx(self, idx: int) -> tuple[np.array, np.array]:
        """Go from a cell idx to the information about the sampled neighbor"""
        ptr = slice(int(self.next_cell_ptr[idx]), int(self.next_cell_ptr[idx + 1]))  # pointers to all neighbors info
        # check if there are any neighbors
        sampled_ptr = random.randint(ptr.start, ptr.stop - 1)  # Sample one random neighbor
        neighbor_idx = self.next_cell_indices[sampled_ptr]  # keyed by ptr
        return neighbor_idx, *self.lookup_cell_by_idx(neighbor_idx)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Performs a lookup and the required transformation for the model"""
        gene_data, col_idxs, metadata = self.lookup_cell_by_idx(idx)
        if self.next_cell_prediction:
            if self.sample_has_neighbor(idx):
                neighbor_idx, next_gene_data, next_col_idxs, _ = self.lookup_neighbor_by_idx(
                    idx
                )  # sample a neighbor and return associated gene counts and col_idx
                # assuming metadata for current and next cells is shared as they come from the same AnnData
                return process_item_ncp(
                    gene_data,
                    col_idxs,
                    next_gene_data,
                    next_col_idxs,
                    metadata,
                    self.tokenizer,
                    gene_median=self.gene_medians,
                    max_len=self.max_len,
                    mask_token_prob=self.mask_token_prob,
                    mask_prob=self.mask_prob,
                    random_token_prob=self.random_token_prob,
                    prepend_cls_token=self.prepend_cls_token,
                )
            else:
                return self.ncp_no_neighbor_policy(idx, self.no_neighbor_policy)

        else:
            return process_item(
                gene_data,
                col_idxs,
                metadata,
                self.tokenizer,
                gene_median=self.gene_medians,
                max_len=self.max_len,
                mask_token_prob=self.mask_token_prob,
                mask_prob=self.mask_prob,
                random_token_prob=self.random_token_prob,
                prepend_cls_token=self.prepend_cls_token,
            )


class Item(TypedDict):
    text: np.array
    types: np.array
    padding_mask: np.array
    labels: np.array
    loss_mask: np.array
    is_random: np.array


def process_item(
    gene_data: np.array,
    gene_idxs: np.array,
    metadata: dict[str, float],
    tokenizer: GeneTokenizer,
    gene_median: dict,
    max_len: int = 1024,
    mask_prob: float = 0.15,
    mask_token_prob: float = 0.8,
    random_token_prob: float = 0.1,
    target_sum: int = 10000,
    normalize: bool = True,
    prepend_cls_token: bool = True,
) -> Item:
    """Process a single item in the dataset.

    Optionally performs median normalization and rank ordering. The tokenizers CLS token is added to the beginning
    of every sample. Converts gene names to ensemble ids before tokenizing. Expects gene_medians to contain ensembl ids as keys.

    Args:
        gene_data (list): List of gene data, these are expression counts.
        gene_idxs (list): List of gene indices, these are keys in 'metadata['feature_ids']' and correspdong the CSR entry. These are computed by sc_memmap.
        metadata (dict): Metadata dictionary.
        tokenizer (Tokenizer): Tokenizer object.
        gene_median (optional(dict)): Dictionary of gene medians. Defaults to None. Expects ensembl IDs to be keys.
        max_len (int): Maximum length of the item. Defaults to 1024. Applies padding to any sequence shorter than max_len and truncates any sequence longer than max_len.
        mask_prob (float): Probability of masking a token. Defaults to 0.15.
        mask_token_prob (float): Probability of the token [MASK] out of the tokens selected for masking. Defaults to 0.8.
        random_token_prob (float): Probability of a random token out of the tokens selected for masking. Defaults to 0.1.
        target_sum (int): Target sum for normalization. Defaults to 10000.
        normalize (bool): Flag to normalize the gene data. Defaults to True.
            When set, this re-orders the gene tokens by their median expression value.
        prepend_cls_token (bool): Whether to prepend a [CLS] token. Defaults to True.
    Returns:
        dict: Processed item dictionary.

    NOTE: this method is very important and very useful. To generalize thiswwe should add an abstraction for
        Datasets that have some kind of functor transformation.
    """

    if max_len < 1:
        raise ValueError(f"max_len must be greater than 1, {max_len=}")

    if random_token_prob + mask_token_prob > 1.0:
        raise ValueError(
            "Sum of random_token_prob and mask_token_prob must be less than or equal to 1.0, identity_token_prob is any remainder less than 1.0."
        )

    identity_token_prob = 1.0 - (random_token_prob + mask_token_prob)
    assert identity_token_prob >= 0.0

    if gene_median is None:
        raise ValueError("gene_median must be provided for this tokenizer")

    max_len = max_len - 1  # - minus 1 for [CLS] token

    gene_names = [metadata["feature_ids"][idx] for idx in gene_idxs]
    genes, tokens, medians = [], [], []
    for tok, gene in zip(gene_names, gene_data):
        if tok in tokenizer.vocab:
            tokens.append(tokenizer.token_to_id(tok))
            genes.append(gene)
            if normalize:
                med = gene_median.get(tok, 1)  # If not in the dictionary we default to no normalization (1)
                medians.append(med)

    genes = np.asarray(genes)
    token_ids = np.asarray(tokens)
    medians = np.asarray(medians)

    if normalize:
        # re-order according to expression median normalized rank. descending order.

        genes = genes / genes.sum() * target_sum
        genes = genes / medians.astype(float)
        idxs = np.argsort(-genes)  # sort in descending order so that the 0th position is the highest value.
        genes = genes[idxs]
        token_ids = token_ids[idxs]

    # - select max_len subset, set sample to false so it doesnt permute the already rank ordered expression values.
    token_ids = sample_or_truncate_plus_pad(
        token_ids, max_len, tokenizer.token_to_id(tokenizer.pad_token), sample=False
    )

    mask = None
    mask_tokens_positions = None
    random_tokens_positions = None

    # - masked tokens
    if mask_prob > 0.0:
        probs = np.full(token_ids.shape[0], mask_prob)
        probs[token_ids == tokenizer.token_to_id(tokenizer.pad_token)] = 0.0
        mask = np.random.binomial(1, probs).astype(bool)
        mask_tokens_positions = mask & np.random.binomial(1, mask_token_prob, mask.shape).astype(bool)
        random_tokens_positions = (
            mask & np.random.binomial(1, random_token_prob, mask.shape).astype(bool) & (~mask_tokens_positions)
        )
        # - ensure [CLS] token is masked from the loss. Note that we're dealing with 1d arrays so flattening isn't a problem here.
        if prepend_cls_token:
            mask = np.insert(mask, 0, False)
            mask_tokens_positions = np.insert(mask_tokens_positions, 0, False)
            random_tokens_positions = np.insert(random_tokens_positions, 0, False)

    # - add [CLS] token, note that token_ids is a 1d array so flattening isn't a problem here.
    if prepend_cls_token:
        token_ids = np.insert(token_ids, 0, tokenizer.token_to_id(tokenizer.cls_token))
    attention_mask = token_ids != tokenizer.token_to_id(tokenizer.pad_token)

    labels = np.ones(len(token_ids)) * -1

    if mask is None:
        # If prob is set to zero, we get None for our mask, which could have unintended side effects.
        # We abuse the scenario where mask == None
        labels[mask] = token_ids[mask]
        mask = np.zeros(shape=token_ids.shape, dtype=bool)
    else:
        mask[~attention_mask] = False  # make sure that we aren't doing MLM on [PAD] tokens
        labels[mask] = token_ids[mask]
    if mask_tokens_positions is None:
        mask_tokens_positions = np.zeros_like(mask)
    if random_tokens_positions is None:
        random_tokens_positions = np.zeros_like(mask)
    # identity_tokens = mask & (~mask_tokens_positions) & (~random_tokens_positions), not needed because
    token_ids[mask_tokens_positions] = tokenizer.token_to_id(tokenizer.mask_token)
    # There are 5 special tokens in the tokenizer, so we start from 5. TODO make this a parameter of the tokenizer.
    if random_tokens_positions.sum() > 0:
        token_ids[random_tokens_positions] = np.random.randint(5, len(tokenizer.vocab), random_tokens_positions.sum())

    # NeMo megatron assumes this return structure.
    item = {
        "text": token_ids.astype(np.int64),
        "types": np.zeros_like(token_ids).astype(np.int64),
        "padding_mask": attention_mask.astype(
            np.int64
        ),  # NeMo BERT wants the attention mask to be named "padding_mask" in this version.
        "labels": labels.astype(np.int64),
        "loss_mask": mask,
        "is_random": np.zeros_like(token_ids).astype(np.int64),
        "has_neighbor": False,
    }

    return item


def process_item_ncp(
    gene_data: np.array,
    gene_idxs: np.array,
    next_gene_data: np.array,
    next_gene_idxs: np.array,
    metadata: dict,
    tokenizer: GeneTokenizer,
    gene_median: dict,
    max_len: int = 1024,
    mask_prob: float = 0.15,
    mask_token_prob: float = 0.8,
    random_token_prob: float = 0.1,
    target_sum: int = 10000,
    normalize: bool = True,
    prepend_cls_token: bool = True,
    token_selection_policy: str = "identity",
) -> Item:
    """Process a pair of cells in the datasets: the current cell and the next cell to predict.

    Optionally performs median normalization and rank ordering. The tokenizers CLS token is added to the beginning
    of every sample. The SEP token in added in between two cell tokens. Positional embeddings are added to differentiate between the current and the next cell.
    The gene tokens are only masked in the next cell. Converts gene names to ensemble ids before tokenizing. Expects gene_medians to contain ensembl ids as keys.

    Args:
        gene_data (list): List of gene data for the current cell, these are expression counts.
        gene_idxs (list): List of gene indices for the current cell, these are keys in 'metadata['feature_ids']' and correspdong the CSR entry. These are computed by sc_memmap.
        next_gene_data (list): List of gene data for the next cell.
        next_idxs (list):  List of gene indices for the next cell, these are keys in 'metadata['feature_ids']' and correspdong the CSR entry. These are computed by sc_memmap.
        metadata (dict): Metadata dictionary.
        tokenizer (Tokenizer): Tokenizer object.
        gene_median (optional(dict)): Dictionary of gene medians. Defaults to None. Expects ensembl IDs to be keys.
        max_len (int): Maximum length of the item. Defaults to 1024. Applies padding to any sequence shorter than max_len and truncates any sequence longer than max_len.
        mask_prob (float): Probability of masking a token. Defaults to 0.15.
        mask_token_prob (float): Probability of the token [MASK] out of the tokens selected for masking. Defaults to 0.8.
        random_token_prob (float): Probability of a random token out of the tokens selected for masking. Defaults to 0.1.
        target_sum (int): Target sum for normalization. Defaults to 10000.
        normalize (bool): Flag to normalize the gene data. Defaults to True.
            When set, this re-orders the gene tokens by their median expression value.
        prepend_cls_token (bool): Whether to prepend a [CLS] token. Defaults to True.
        token_selection_policy (str): A method for selecting the future cell tokens to predict for a given current cell. Defaults to identity.
    Returns:
        dict: Processed item dictionary.
    """

    if max_len < 1:
        raise ValueError(f"max_len must be greater than 1, {max_len=}")

    if random_token_prob + mask_token_prob > 1.0:
        raise ValueError(
            "Sum of random_token_prob and mask_token_prob must be less than or equal to 1.0, identity_token_prob is any remainder less than 1.0."
        )

    identity_token_prob = 1.0 - (random_token_prob + mask_token_prob)
    assert identity_token_prob >= 0.0

    if gene_median is None:
        raise ValueError("gene_median must be provided for this tokenizer")

    max_len = max_len - 1  # - minus 1 for [CLS] token

    current_token_ids = process_cell(gene_data, gene_idxs, tokenizer, metadata, gene_median, normalize, target_sum)

    next_token_ids = process_cell(
        next_gene_data, next_gene_idxs, tokenizer, metadata, gene_median, normalize, target_sum
    )
    # apply separately to both cells data

    next_token_ids = select_future_tokens(current_token_ids, next_token_ids, token_selection_policy)
    # select which tokens to subset

    # - select max_len subset, set sample to false so it doesnt permute the already rank ordered expression values.
    current_token_ids = sample_or_truncate_plus_pad(
        current_token_ids, max_len // 2, tokenizer.token_to_id(tokenizer.pad_token), sample=False
    )

    next_token_ids = sample_or_truncate_plus_pad(
        next_token_ids, adjust_half_length(max_len), tokenizer.token_to_id(tokenizer.pad_token), sample=False
    )

    token_ids = np.concatenate((current_token_ids, [tokenizer.token_to_id(tokenizer.sep_token)], next_token_ids))
    # Insert [SEP] token and stitch the tokens from two cells together

    mask = None
    mask_tokens_positions = None
    random_tokens_positions = None

    # - masked tokens
    if mask_prob > 0.0:
        probs = np.full(token_ids.shape[0], mask_prob)
        probs[token_ids == tokenizer.token_to_id(tokenizer.pad_token)] = 0.0
        probs[: (max_len // 2 + 1)] = 0.0  # the tokens are masked only on the next cell and SEP is also not masked
        mask = np.random.binomial(1, probs).astype(bool)
        mask_tokens_positions = mask & np.random.binomial(1, mask_token_prob, mask.shape).astype(bool)
        random_tokens_positions = (
            mask & np.random.binomial(1, random_token_prob, mask.shape).astype(bool) & (~mask_tokens_positions)
        )
        # - ensure [CLS] token is masked from the loss. Note that we're dealing with 1d arrays so flattening isn't a problem here.
        if prepend_cls_token:
            mask = np.insert(mask, 0, False)
            mask_tokens_positions = np.insert(mask_tokens_positions, 0, False)
            random_tokens_positions = np.insert(random_tokens_positions, 0, False)

    # - add [CLS] token, note that token_ids is a 1d array so flattening isn't a problem here.
    if prepend_cls_token:
        token_ids = np.insert(token_ids, 0, tokenizer.token_to_id(tokenizer.cls_token))
    attention_mask = token_ids != tokenizer.token_to_id(tokenizer.pad_token)

    labels = np.ones(len(token_ids)) * -1

    if mask is None:
        # If prob is set to zero, we get None for our mask, which could have unintended side effects.
        # We abuse the scenario where mask == None
        labels[mask] = token_ids[mask]
        mask = np.zeros(shape=token_ids.shape, dtype=bool)
    else:
        mask[~attention_mask] = False  # make sure that we aren't doing MLM on [PAD] tokens
        labels[mask] = token_ids[mask]
    if mask_tokens_positions is None:
        mask_tokens_positions = np.zeros_like(mask)
    if random_tokens_positions is None:
        random_tokens_positions = np.zeros_like(mask)
    # identity_tokens = mask & (~mask_tokens_positions) & (~random_tokens_positions), not needed because
    token_ids[mask_tokens_positions] = tokenizer.token_to_id(tokenizer.mask_token)
    # There are 5 special tokens in the tokenizer, so we start from 5. TODO make this a parameter of the tokenizer.
    if random_tokens_positions.sum() > 0:
        token_ids[random_tokens_positions] = np.random.randint(5, len(tokenizer.vocab), random_tokens_positions.sum())

    # NeMo megatron assumes this return structure.
    item = {
        "text": token_ids.astype(np.int64),
        "types": np.zeros_like(token_ids).astype(np.int64),
        "padding_mask": attention_mask.astype(
            np.int64
        ),  # NeMo BERT wants the attention mask to be named "padding_mask" in this version.
        "labels": labels.astype(np.int64),
        "loss_mask": mask,
        "is_random": np.zeros_like(token_ids).astype(np.int64),
        "has_neighbor": True,
    }

    return item


def adjust_half_length(max_length: int) -> int:
    """Helper function to adjust the half length of a cell to make sure the combined sequence (current+SEP+future cell tokens) size is equal to max_length.

    Args:
        max_length (int): Total length of the sequence + [SEP].

    Returns:
        int: Adjusted half length. For odd (max_length - 1) the half size is decremented by 1.
    """
    if (max_length - 1) % 2 == 0:
        return max_length // 2
    else:
        return max_length // 2 - 1


def select_future_tokens(
    current_token_ids: np.array, next_token_ids: np.array, token_selection_policy: Literal["identity"] = "identity"
) -> np.array:
    """Subset future tokens to be included to prediction based on the current cell tokens.

    Args:
        current_token_ids (np.array): Token sequence for the reference cell from which the next cell is predicted.
        next_token_ids (np.array): Token sequence for next cell for downstream tasks.
        token_selection_policy (Literal['identity'], optional): method to subset next cell tokens. Defaults to 'identity'.

    Returns:
        np.array: processed next cell tokens
    """

    if token_selection_policy == "identity":
        return next_token_ids
    else:
        raise NotImplementedError


def process_cell(
    gene_data: np.array,
    gene_idxs: np.array,
    tokenizer: GeneTokenizer,
    metadata: dict,
    gene_median: dict,
    normalize: bool = True,
    target_sum: int = 10000,
) -> np.array:
    """Process a data in a given cell by tokenizing the genes, (optionally) normalizing, sorting according to the values in descending order.

    Args:
        gene_data (np.array): List of gene data, these are expression counts.
        gene_idxs (np.array):  List of gene indices, these are keys in 'metadata['feature_ids']' and correspdong the CSR entry. These are computed by sc_memmap.
        tokenizer (GeneTokenizer): Tokenizer object.
        metadata (dict): Metadata dictionary.
        gene_median (dict): Dictionary of gene medians. Defaults to None. Expects ensembl IDs to be keys.
        normalize (bool): Flag to normalize the gene data. Defaults to True.
        target_sum (int): Target sum for normalization. Defaults to 10000.
    Returns:
        np.array: re-ordered token sequence in descending count order.
    """
    gene_names = [metadata["feature_ids"][idx] for idx in gene_idxs]
    genes, tokens, medians = [], [], []
    for tok, gene in zip(gene_names, gene_data):
        if tok in tokenizer.vocab:
            tokens.append(tokenizer.token_to_id(tok))
            genes.append(gene)
            if normalize:
                med = gene_median.get(tok, 1)  # If not in the dictionary we default to no normalization (1)
                medians.append(med)

    genes = np.asarray(genes)  # gene counts
    token_ids = np.asarray(tokens)  # gene tokens
    medians = np.asarray(medians)  # gene median-normalized counts

    if normalize:
        # re-order according to expression median normalized rank. descending order.
        genes = genes / genes.sum() * target_sum
        genes = genes / medians.astype(float)
        idxs = np.argsort(-genes)  # sort in descending order so that the 0th position is the highest value.
        genes = genes[idxs]
        token_ids = token_ids[idxs]
    return token_ids
