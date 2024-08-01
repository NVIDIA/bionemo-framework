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
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, TypedDict

import numpy as np
from nemo.utils import logging
from torch.utils.data import Dataset

from bionemo.geneformer.data.singlecell.utils import sample_or_truncate_plus_pad
from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer


__all__: Sequence[str] = (
    "SingleCellDataset",
    "Item",
    "process_item",
)


class Item(TypedDict):
    text: np.ndarray
    types: np.ndarray
    padding_mask: np.ndarray
    labels: np.ndarray
    loss_mask: np.ndarray
    is_random: np.ndarray


class SingleCellDataset(Dataset):
    """A dataset class for single-cell pre-training. These can be generated using the sc_memmap.py script. Future
    updates will contain more comprehensive workflows for generating a Sparse Memmap from scRNA-seq.

    Args:
        data_path (str): Path where the single cell files are stored. It should contain the following files:
            - `metadata.json`: Path containing feature subset associated with each dataset.
            - `features.csv`: Feature subset associated with each sample.
            - Gene expression matrix stored in CSR format as `numpy.memmap`:
                - `gene_expression_data.npy`: Gene expression values.
                - `gene_expression_ind.npy`: Gene indices associated with gene values.
                - `gene_expression_ptr.npy`: Column indices for each sample.
        tokenizer: The tokenizer to use for tokenizing the input data.
        median_dict (dict, optional): A dictionary containing median values for each gene. Defaults to None.
        max_len (int, optional): The maximum length of the input sequence. Defaults to 1024.

    Attributes:
        data_path (str): Path where the single cell files are stored.
        max_len (int): The maximum length of the input sequence.
        metadata (dict): Metadata loaded from `metadata.json`.
        gene_medians (dict): A dictionary containing median values for each gene. If None, a median of '1' is assumed for all genes.
        num_train (int): The number of samples in the training split.
        num_val (int): The number of samples in the validation split.
        num_test (int): The number of samples in the test split.
        index_offset (int): The offset to apply to the indices.
        length (int): The total number of samples in the dataset.
        gene_data (numpy.memmap): Gene expression values stored in CSR format.
        gene_data_indices (numpy.memmap): Gene indices associated with gene values.
        gene_data_ptr (numpy.memmap): Column indices for each sample.
        tokenizer: The tokenizer used for tokenizing the input data.
        dataset_ccum (numpy.ndarray): Cumulative sum of row counts to map row indices to dataset id.
        dataset_map (dict): Mapping of dataset id to dataset name.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.

    See Also:
        bionemo/data/singlecell/sc_memmap.py - creates the artifacts required for instantiating a singlecell dataset from hdf5 files.
    """

    def __init__(  # noqa: D107
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
        metadata = json.load(open(path / "metadata.json", "r"))

        # - median dict
        self.gene_medians = median_dict

        # - train/val idxs sampled contiguously
        total_el = sum([v["num_el"] for _, v in metadata.items()])
        self.num_samples = sum([v["shape"][0] for _, v in metadata.items()])
        # - load data
        self.gene_data = np.memmap(path / "gene_expression_data.npy", dtype="float32", mode="r", shape=(total_el,))

        self.gene_data_indices = np.memmap(
            path / "gene_expression_ind.npy", dtype="int32", mode="r", shape=(total_el,)
        )

        self.gene_data_ptr = np.memmap(
            path / "gene_expression_ptr.npy", dtype="int64", mode="r", shape=(self.num_samples + 1,)
        )
        self.tokenizer = tokenizer
        rnd_key = next(iter(metadata))
        feature_ids = np.array(metadata[rnd_key]["feature_ids"])

        # Determine if we need to store the full metadata (per file feature_ids) or just a single feature_id
        #  vector for all files. If we can do the later this is much more memory efficient.
        #  without this change, if num_workers>0, we seem to hit a memory leak after a relatively small number
        #  of steps. Online discussion points to native python objects like dictionaries of a lot of data
        #  being a primary culprit behind large RAM usage in dataloaders that use multiprocessing.
        features_all_same = True
        for m in metadata.values():
            if np.any(np.char.not_equal(np.array(m["feature_ids"]), feature_ids)):
                features_all_same = False
                break

        if not features_all_same:
            # We need to store per-file metadata of feature_ids. Make sure you run with a lot of RAM or few dataset workers.
            #  we need to store per-file metadata in this case because some of the files have different subsets of the
            #  feature_ids.
            logging.warning(
                "Feature ids are not the same across datasets. This can cause heavy RAM usage "
                "for large datasets, try setting num_workers to 0."
            )
            self.metadata = metadata
            self.feature_ids = None

            # map row indices to dataset id
            self.dataset_ccum = np.zeros(
                len(self.metadata),
            )
            # Maps dataset ids to dataset names (used in the metadata dict)
            self.dataset_map = {}
            count = 0
            for i, k in enumerate(self.metadata):
                self.dataset_ccum[i] = count
                self.dataset_map[i] = k
                count += self.metadata[k]["shape"][0]
            self.dataset_ccum[0] = -1
        else:
            # We can store a single feature_id vector for all datasets, and do not need to store the full metadata array.
            logging.warning(
                "Feature ids are the same across datasets. This is good, using the same feature_ids for all datasets."
            )
            self.feature_ids = feature_ids
            self.metadata = None

    def __len__(self):  # noqa: D105
        return self.num_samples

    def metadata_lookup(self, idx) -> Dict[str, np.ndarray]:
        """Go from a cell idx to the file-level metadata associated with that cell."""
        did = sum(~(self.dataset_ccum > idx)) - 1
        metadata = self.metadata[self.dataset_map[did]]
        return metadata

    def lookup_cell_by_idx(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ptr = slice(int(self.gene_data_ptr[idx]), int(self.gene_data_ptr[idx + 1]))
        # col idxs poin to offsets in the original sparse metadata, this is for looking up metadata eg gene names
        col_idxs = np.asarray(self.gene_data_indices[ptr]).astype(int)  # keyed by ptr
        if self.assert_increasing_columns and len(col_idxs) > 1:
            is_increasing = np.diff(col_idxs) > 0
            if not np.all(is_increasing):
                raise ValueError(f"Column indices are not increasing for {np.sum(~is_increasing)} pairs of genes")
        gene_data = np.asarray(self.gene_data[ptr]).astype(int)  # keyed by ptr
        # Get feature_ids for this particular cell. Eitehr lookup by index if we need to, or if we already verified that
        #  metadata is not needed because feature_ids are the same for every file, then we can just use the single feature_ids
        #  vector instead.
        feature_ids: np.ndarray = (
            self.feature_ids if self.metadata is None else self.metadata_lookup(idx)["feature_ids"]
        )
        return gene_data, col_idxs, feature_ids

    def __getitem__(self, idx: int) -> Item:
        """Performs a lookup and the required transformation for the model"""  # noqa: D415
        gene_data, col_idxs, feature_ids = self.lookup_cell_by_idx(idx)
        return process_item(
            gene_data,
            col_idxs,
            feature_ids,
            self.tokenizer,
            gene_median=self.gene_medians,
            max_len=self.max_len,
            mask_token_prob=self.mask_token_prob,
            mask_prob=self.mask_prob,
            random_token_prob=self.random_token_prob,
            prepend_cls_token=self.prepend_cls_token,
        )


def process_item(
    gene_data: np.ndarray,
    gene_idxs: np.ndarray,
    feature_ids: np.ndarray,
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
        feature_ids (list): Feature ids for the full dataset.
        tokenizer (Tokenizer): Tokenizer object.
        gene_median (optional(dict)): Dictionary of gene medians. Defaults to None. Expects ensembl IDs to be keys.
        max_len (int): Maximum length of the item. Defaults to 1024. Applies padding to any sequence shorter than max_len and truncates any sequence longer than max_len.
        mask_prob (float): Probability of masking a token. Defaults to 0.15.
        target_sum (int): Target sum for normalization. Defaults to 10000.
        normalize (bool): Flag to normalize the gene data. Defaults to True.
            When set, this re-orders the gene tokens by their median expression value.
        probabilistic_dirichlet_sampling (bool): Flag to enable probabilistic dirichlet sampling. Defaults to False.
        dirichlet_alpha (float): Alpha value for dirichlet sampling if set by `probabilistic_dirichlet_sampling`. Defaults to 0.5.
        same_length (bool): when true, sample the same length of genes as you originally had before the dirichlet sampler.
        recompute_globals (bool): when true, global arrays are always recomputed. this is only useful for testing.

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

    gene_names = [feature_ids[idx] for idx in gene_idxs]
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
        "attention_mask": attention_mask.astype(np.int64),
        "labels": labels.astype(np.int64),
        "loss_mask": mask,
        "is_random": np.zeros_like(token_ids).astype(np.int64),
    }

    return item
