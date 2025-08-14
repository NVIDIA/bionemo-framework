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


import random
import time
from pathlib import Path
from typing import Any, Optional, Sequence, cast

import numpy as np
import torch
from nemo.utils import logging
from torch.utils.data import Dataset
from tqdm import tqdm

from bionemo.core.data.load import load
from bionemo.core.data.multi_epoch_dataset import EpochIndex
from bionemo.core.utils import random_utils
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.geneformer.data.singlecell.utils import sample_or_truncate
from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.llm.data import masking, types
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset

__all__: Sequence[str] = (
    "TemporalGeneformerDataset",
    "process_item",
    "process_item_ncp",
)


class TemporalGeneformerDataset(Dataset):
    """Temporal next-cell prediction dataset built on SCDL memmap.

    Args:
        data_path (str | Path): Path to SingleCell Memmap directory.
        tokenizer: Tokenizer to convert gene identifiers into tokens.
        median_dict (dict, optional): Gene medians for normalization. Required if normalize is used.
        max_len (int, optional): Maximum total sequence length. Defaults to 1024.
        mask_prob (float, optional): MLM probability for next-cell tokens. Defaults to 0.15.
        mask_token_prob (float, optional): Probability of using [MASK] for masked positions. Defaults to 0.8.
        random_token_prob (float, optional): Probability of using random token for masked positions. Defaults to 0.1.
        prepend_cls_token (bool, optional): Whether to add [CLS] at the start. Defaults to True.
        eos_token (int | None, optional): Optional EOS token id to append. Defaults to None.
        include_unrecognized_vocab_in_dataset (bool, optional): If True, require all genes in vocab. Defaults to False.
        seed (int, optional): RNG seed for deterministic sampling. Defaults to numpy entropy.
        neighbor_key (str, optional): Neighbor key present in SCDL. Defaults to "next_cell_ids".
        fallback_to_identity (bool, optional): If no neighbors, use identity neighbor. Defaults to True.
        no_neighbor_policy (str, optional): Policy to use when no neighbors are available. Defaults to "identity".
        only_cells_with_neighbors (bool, optional): If True, only include cells that have neighbors in the dataset. Defaults to False.
    """  # noqa: D205

    def __init__(  # noqa: D107
        self,
        data_path: str | Path,
        tokenizer: Any,
        median_dict: Optional[dict] = None,
        max_len: int = 1024,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
        prepend_cls_token: bool = True,
        eos_token: int | None = None,
        include_unrecognized_vocab_in_dataset: bool = False,
        seed: int = np.random.SeedSequence().entropy,  # type: ignore
        neighbor_key: str = "next_cell_ids",
        fallback_to_identity: bool = True,
        no_neighbor_policy: str = "identity",
        only_cells_with_neighbors: bool = True,
    ):
        super().__init__()

        self.data_path = data_path
        self.max_len = max_len
        self.random_token_prob = random_token_prob
        self.mask_token_prob = mask_token_prob
        self.mask_prob = mask_prob
        self.prepend_cls_token = prepend_cls_token
        self._seed = seed
        self.eos_token = eos_token

        # Load SCDL with neighbor support enabled
        self.scdl = SingleCellMemMapDataset(
            str(data_path),
            load_neighbors=True,
            neighbor_key=neighbor_key,
            fallback_to_identity=fallback_to_identity,
        )
        # Build valid indices depending on neighbor availability requirement
        if only_cells_with_neighbors:
            if not getattr(self.scdl, "_has_neighbors", False):
                raise ValueError(
                    "Dataset has no neighbor data but only_cells_with_neighbors=True was requested."
                )
            self.valid_indices: list[int] = []
            for i in range(self.scdl.number_of_rows()):
                try:
                    neighbors = self.scdl.get_neighbor_indices_for_cell(i)
                except Exception:
                    neighbors = []
                if len(neighbors) > 0:
                    self.valid_indices.append(i)
            if len(self.valid_indices) == 0:
                raise ValueError("No cells with neighbors found in the dataset.")
        else:
            # default to all rows
            self.valid_indices = list(range(self.scdl.number_of_rows()))
        self.length = len(self.valid_indices)
        # - median dict
        self.gene_medians = median_dict
        self.tokenizer = tokenizer
        self.include_unrecognized_vocab_in_dataset = include_unrecognized_vocab_in_dataset
        self.no_neighbor_policy = no_neighbor_policy
        self.only_cells_with_neighbors = only_cells_with_neighbors

        # Log and print the neighbor filtering flag for verification
        logging.info(f"[TemporalGeneformerDataset] only_cells_with_neighbors={self.only_cells_with_neighbors}")

    def __len__(self):  # noqa: D105
        return self.length

    def __getitem__(self, index: EpochIndex) -> types.BertSample:
        """Lookup a cell and its neighbor and build a temporal NCP example."""
        # Map index into actual SCDL row if we filtered
        cell_idx = self.valid_indices[index.idx]
        rng = np.random.default_rng([self._seed, index.epoch, cell_idx])
        if self.gene_medians is None:
            raise ValueError("gene_median must be provided for this tokenizer")

        # Current cell
        (cur_values, cur_cols), feature_ids = self.scdl.get_row(
            cell_idx, return_features=True, feature_vars=["feature_id"]
        )

        assert (
            len(feature_ids) == 1
        )  # we expect feature_ids to be a list containing one np.array with the row's feature ids
        cur_gene_data, cur_col_idxs = np.array(cur_values), np.array(cur_cols)
        if len(cur_gene_data) == 0:
            raise ValueError(
                "SingleCellMemmap data provided is invalid; the gene expression data parsed for the specified index is empty."
            )

        # Neighbor cell (sample) with per-cell neighbor presence check
        if not self.sample_has_neighbor(cell_idx):
            return self.ncp_no_neighbor_policy(cell_idx, self.no_neighbor_policy, epoch=index.epoch)
        neighbor_idx = self.scdl.sample_neighbor_index(cell_idx, rng=rng)
        (n_values, n_cols), next_feature_ids = self.scdl.get_row(
            neighbor_idx, return_features=True, feature_vars=["feature_id"]
        )
        assert len(next_feature_ids) == 1
        next_gene_data, next_col_idxs = np.array(n_values), np.array(n_cols)

        return process_item_ncp(
            cur_gene_data,
            cur_col_idxs,
            next_gene_data,
            next_col_idxs,
            feature_ids[0],
            next_feature_ids[0] if getattr(self.scdl, "_has_neighbors", False) else feature_ids[0],
            self.tokenizer,
            gene_median=cast(dict, self.gene_medians),
            rng=rng,
            max_len=self.max_len,
            mask_prob=self.mask_prob,
            mask_token_prob=self.mask_token_prob,
            random_token_prob=self.random_token_prob,
            prepend_cls_token=self.prepend_cls_token,
            eos_token=self.eos_token,
            include_unrecognized_vocab_in_dataset=self.include_unrecognized_vocab_in_dataset,
        )

    def ncp_no_neighbor_policy(self, idx: int, type: str = "identity", epoch: int = 0) -> types.BertSample:
        """Handle samples without neighbors according to the chosen policy.

        Currently supports 'identity' which returns a single-cell item for the index.
        """
        if type == "identity":
            rng = np.random.default_rng([self._seed, epoch, idx])
            (values, cols), feature_ids = self.scdl.get_row(idx, return_features=True, feature_vars=["feature_id"])
            assert len(feature_ids) == 1
            gene_data, col_idxs = np.array(values), np.array(cols)
            if len(gene_data) == 0:
                raise ValueError(
                    "SingleCellMemap data provided is invalid; the gene expression data parsed for the specified index is empty."
                )
            return process_item(
                gene_data,
                col_idxs,
                feature_ids[0],
                self.tokenizer,
                gene_median=cast(dict, self.gene_medians),
                rng=rng,
                max_len=self.max_len,
                mask_token_prob=self.mask_token_prob,
                mask_prob=self.mask_prob,
                random_token_prob=self.random_token_prob,
                prepend_cls_token=self.prepend_cls_token,
                eos_token=self.eos_token,
                include_unrecognized_vocab_in_dataset=self.include_unrecognized_vocab_in_dataset,
            )
        else:
            raise NotImplementedError(f"Unknown no_neighbor_policy: {type}")

    def sample_has_neighbor(self, idx: int) -> bool:
        """Return True if the given cell index has at least one neighbor, False otherwise."""
        if not getattr(self.scdl, "_has_neighbors", False):
            return False
        try:
            nbrs = self.scdl.get_neighbor_indices_for_cell(idx)
            return len(nbrs) > 0
        except Exception:
            return False


def _gather_medians(
    gene_names: np.ndarray,
    gene_data: np.ndarray,
    normalize: bool,
    vocab: dict[str, int],
    gene_median: dict[str, float],
    include_unrecognized_vocab_in_dataset: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter out genes that are not in the provided tokenizer vocab, and tokenize the gene names."""
    genes, tokens, medians = [], [], []
    for tok, gene in zip(gene_names, gene_data):
        if tok in vocab:
            tokens.append(vocab[tok])
            genes.append(gene)
            if normalize:
                med = gene_median[tok]  # If not in the dictionary we default to no normalization (1)
                medians.append(med)
        elif include_unrecognized_vocab_in_dataset:
            raise ValueError(f"Provided gene identifier, {str(tok)}, is not in the tokenizer vocab.")
    return np.asarray(genes), np.asarray(tokens), np.asarray(medians)


def process_item(  # noqa: D417
    gene_data: np.ndarray,
    gene_idxs: np.ndarray,
    feature_ids: np.ndarray,
    tokenizer: GeneTokenizer,
    gene_median: dict,
    rng: np.random.Generator,
    max_len: int = 1024,
    mask_prob: float = 0.15,
    mask_token_prob: float = 0.8,
    random_token_prob: float = 0.1,
    target_sum: int = 10000,
    normalize: bool = True,
    prepend_cls_token: bool = True,
    eos_token: None | int = None,
    include_unrecognized_vocab_in_dataset: bool = False,
) -> types.BertSample:
    """Process a single item in the dataset.

    Optionally performs median normalization and rank ordering. The tokenizers CLS token is added to the beginning
    of every sample. Converts gene names to ensemble ids before tokenizing. Expects gene_medians to contain ensembl ids as keys.

    Args:
        gene_data (list): List of gene data, these are expression counts.
        gene_idxs (list): List of gene indices, these are keys in 'metadata['feature_ids']' and corresponding the CSR entry.
        feature_ids (list): Feature ids for the full dataset.
        tokenizer (Tokenizer): Tokenizer object.
        gene_median (optional(dict)): Dictionary of gene medians. Defaults to None. Expects ensembl IDs to be keys.
        rng: Random number generator to ensure deterministic results.
        max_len (int): Maximum length of the item. Defaults to 1024. Applies padding to any sequence shorter than max_len and truncates any sequence longer than max_len.
        mask_prob (float): Probability of masking a token. Defaults to 0.15.
        target_sum (int): Target sum for normalization. Defaults to 10000.
        normalize (bool): Flag to normalize the gene data. Defaults to True.
            When set, this re-orders the gene tokens by their median expression value.
        probabilistic_dirichlet_sampling (bool): Flag to enable probabilistic dirichlet sampling. Defaults to False.
        dirichlet_alpha (float): Alpha value for dirichlet sampling if set by `probabilistic_dirichlet_sampling`. Defaults to 0.5.
        same_length (bool): when true, sample the same length of genes as you originally had before the dirichlet sampler.
        recompute_globals (bool): when true, global arrays are always recomputed. this is only useful for testing.
        include_unrecognized_vocab_in_dataset (bool, optional): If set to True, a hard-check is performed to verify all gene identifers are in the user supplied tokenizer vocab. Defaults to False which means any gene identifier not in the user supplied tokenizer vocab will be excluded.

    Returns:
        dict: Processed item dictionary.

    NOTE: this method is very important and very useful. To generalize thiswwe should add an abstraction for
        Datasets that have some kind of functor transformation.
    """
    if max_len < 1:
        raise ValueError(f"max_len must be greater than 1, {max_len=}")

    if gene_median is None:
        raise ValueError("gene_median must be provided for this tokenizer")

    if prepend_cls_token:
        max_len = max_len - 1  # - minus 1 for [CLS] token
    if eos_token is not None:
        max_len = max_len - 1  # - minus 1 for [EOS] token

    gene_names = feature_ids[gene_idxs]

    gene_expression_cell, token_ids, gene_expression_medians = _gather_medians(
        gene_names,
        gene_data,
        normalize,
        tokenizer.vocab,
        gene_median,
        include_unrecognized_vocab_in_dataset=include_unrecognized_vocab_in_dataset,
    )

    if normalize:
        # re-order according to expression median normalized rank. descending order.

        gene_expression_cell = gene_expression_cell / gene_expression_cell.sum() * target_sum
        gene_expression_cell = gene_expression_cell / gene_expression_medians.astype(float)
        idxs = np.argsort(
            -gene_expression_cell
        )  # sort in descending order so that the 0th position is the highest value.
        gene_expression_cell = gene_expression_cell[idxs]
        token_ids = token_ids[idxs]

    # - select max_len subset, set sample to false so it doesnt permute the already rank ordered expression values.
    token_ids = sample_or_truncate(token_ids, max_len, sample=False)
    with torch.no_grad(), torch.device("cpu"):
        masked_tokens, labels, loss_mask = masking.apply_bert_pretraining_mask(
            tokenized_sequence=torch.from_numpy(token_ids),
            random_seed=int(random_utils.get_seed_from_rng(rng)),
            mask_config=masking.BertMaskConfig(
                tokenizer=tokenizer,
                random_tokens=range(len(tokenizer.special_tokens), len(tokenizer.vocab)),
                mask_prob=mask_prob,
                mask_token_prob=mask_token_prob,
                random_token_prob=random_token_prob,
            ),
        )
        cls_token = tokenizer.token_to_id(tokenizer.cls_token) if prepend_cls_token else None
        if cls_token is not None or eos_token is not None:
            masked_tokens, labels, loss_mask = masking.add_cls_and_eos_tokens(
                sequence=masked_tokens,
                labels=labels,
                loss_mask=loss_mask,
                cls_token=cls_token,
                eos_token=eos_token,
            )

        # NeMo megatron assumes this return structure.
        return {
            "text": masked_tokens,
            "types": torch.zeros_like(masked_tokens, dtype=torch.int64),
            "attention_mask": torch.ones_like(masked_tokens, dtype=torch.int64),
            "labels": labels,
            "loss_mask": loss_mask,
            "is_random": torch.zeros_like(masked_tokens, dtype=torch.int64),
        }


def process_item_ncp(  # noqa: D417
    gene_data: np.ndarray,
    gene_idxs: np.ndarray,
    next_gene_data: np.ndarray,
    next_gene_idxs: np.ndarray,
    feature_ids_current: np.ndarray,
    feature_ids_next: np.ndarray,
    tokenizer: GeneTokenizer,
    gene_median: dict,
    rng: np.random.Generator,
    max_len: int = 1024,
    mask_prob: float = 0.15,
    mask_token_prob: float = 0.8,
    random_token_prob: float = 0.1,
    target_sum: int = 10000,
    normalize: bool = True,
    prepend_cls_token: bool = True,
    eos_token: None | int = None,
    include_unrecognized_vocab_in_dataset: bool = False,
) -> types.BertSample:
    """Process a pair of cells (current, next) to build a temporal NCP sample.

    Builds [CLS] + current + [SEP] + next (+ [EOS]) and applies MLM only on the next-cell segment.
    """
    if max_len < 1:
        raise ValueError(f"max_len must be greater than 1, {max_len=}")
    if gene_median is None:
        raise ValueError("gene_median must be provided for this tokenizer")

    # Reserve for special tokens: optional CLS, required SEP, optional EOS
    reserved = (1 if prepend_cls_token else 0) + 1 + (1 if eos_token is not None else 0)
    if max_len <= reserved:
        raise ValueError(f"max_len must be > reserved tokens ({reserved}), got {max_len}")

    # Gather and optionally normalize/rank current cell
    cur_gene_names = feature_ids_current[gene_idxs]
    cur_expr, cur_tokens, cur_meds = _gather_medians(
        cur_gene_names,
        gene_data,
        normalize,
        tokenizer.vocab,
        gene_median,
        include_unrecognized_vocab_in_dataset=include_unrecognized_vocab_in_dataset,
    )
    if normalize and len(cur_expr) > 0:
        cur_expr = cur_expr / cur_expr.sum() * target_sum
        cur_expr = cur_expr / cur_meds.astype(float)
        idxs = np.argsort(-cur_expr)
        cur_tokens = cur_tokens[idxs]
        cur_expr = cur_expr[idxs]

    # Gather and optionally normalize/rank next cell
    next_gene_names = feature_ids_next[next_gene_idxs]
    next_expr, next_tokens, next_meds = _gather_medians(
        next_gene_names,
        next_gene_data,
        normalize,
        tokenizer.vocab,
        gene_median,
        include_unrecognized_vocab_in_dataset=include_unrecognized_vocab_in_dataset,
    )
    if normalize and len(next_expr) > 0:
        next_expr = next_expr / next_expr.sum() * target_sum
        next_expr = next_expr / next_meds.astype(float)
        idxs = np.argsort(-next_expr)
        next_tokens = next_tokens[idxs]
        next_expr = next_expr[idxs]

    # Split available length between current and next
    from bionemo.geneformer.data.singlecell.utils import sample_or_truncate
    available = max_len - reserved
    cur_max = (available + 1) // 2
    next_max = available - cur_max

    cur_tokens = sample_or_truncate(cur_tokens, cur_max, sample=False)
    next_tokens = sample_or_truncate(next_tokens, next_max, sample=False)

    with torch.no_grad(), torch.device("cpu"):
        # Convert to tensors
        cur_t = torch.from_numpy(cur_tokens)
        next_t = torch.from_numpy(next_tokens)

        # Special tokens
        cls_t = (
            torch.tensor([tokenizer.token_to_id(tokenizer.cls_token)], dtype=torch.long)
            if prepend_cls_token
            else torch.empty(0, dtype=torch.long)
        )
        sep_t = torch.tensor([tokenizer.token_to_id(tokenizer.sep_token)], dtype=torch.long)
        eos_t = torch.tensor([eos_token], dtype=torch.long) if eos_token is not None else torch.empty(0, dtype=torch.long)

        # Combine: [CLS]? + current + [SEP] + next + [EOS]?
        combined = torch.cat([cls_t, cur_t, sep_t, next_t, eos_t])

        # Prepare labels/loss_mask
        labels = torch.full_like(combined, -100)
        loss_mask = torch.zeros_like(combined, dtype=torch.bool)

        # Mask ONLY the next segment
        from bionemo.core.utils import random_utils
        next_start = len(cls_t) + len(cur_t) + len(sep_t)
        next_end = len(combined) - (1 if eos_token is not None else 0)
        if next_end > next_start and mask_prob > 0.0:
            masked_next, next_labels, next_loss_mask = masking.apply_bert_pretraining_mask(
                tokenized_sequence=combined[next_start:next_end],
                random_seed=int(random_utils.get_seed_from_rng(rng)),
                mask_config=masking.BertMaskConfig(
                    tokenizer=tokenizer,
                    random_tokens=range(len(tokenizer.special_tokens), len(tokenizer.vocab)),
                    mask_prob=mask_prob,
                    mask_token_prob=mask_token_prob,
                    random_token_prob=random_token_prob,
                ),
            )
            # Splice masked next back into combined
            combined = torch.cat([combined[:next_start], masked_next, combined[next_end:]])
            labels[next_start:next_end] = next_labels
            loss_mask[next_start:next_end] = next_loss_mask

        # NeMo-compatible return
        return {
            "text": combined,
            "types": torch.zeros_like(combined, dtype=torch.int64),
            "attention_mask": (combined != tokenizer.token_to_id(tokenizer.pad_token)).to(dtype=torch.int64),
            "labels": labels,
            "loss_mask": loss_mask,
            "is_random": torch.zeros_like(combined, dtype=torch.int64),
        }


def _profile_sc_dataset():
    data_path = load("single_cell/testdata-20241203") / "cellxgene_2023-12-15_small_processed_scdl" / "train"
    preprocessor = GeneformerPreprocess(
        download_directory=data_path,
        medians_file_path=data_path / "medians.json",
        tokenizer_vocab_path=data_path / "geneformer.vocab",
    )
    tokenizer = None
    median_dict = None
    match preprocessor.preprocess():
        case {"tokenizer": tk, "median_dict": md}:
            logging.info("*************** Preprocessing Finished ************")
            tokenizer = tk
            median_dict = md
        case _:
            logging.error("Preprocessing failed.")
            return
    if tokenizer is None or median_dict is None:
        raise RuntimeError("Preprocessing did not produce tokenizer/median_dict")
    scd = TemporalGeneformerDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        median_dict=median_dict,
        max_len=2048,
        seed=321,
        fallback_to_identity=True,
    )
    n_epochs = 1
    len_dataset: int = len(scd)
    idxs = list(range(len_dataset * n_epochs))
    random.seed(315)
    random.shuffle(idxs)
    start = time.monotonic()  # Like time.time() but uses the CPU clock rather so subsequent calls will progress.
    for i in tqdm(idxs):
        _ = scd[EpochIndex(idx=i % len_dataset, epoch=i // len_dataset)]
    stop = time.monotonic()
    print(f"Processed {len_dataset * n_epochs} rows in {stop - start} seconds")


if __name__ == "__main__":
    # python -m bionemo.geneformer.data.singlecell.dataset will run this profile.
    _profile_sc_dataset()
