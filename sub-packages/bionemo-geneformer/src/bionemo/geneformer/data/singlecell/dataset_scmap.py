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
from typing import Any, Optional, Sequence, cast

import numpy as np
import pandas as pd
import torch
from nemo.utils import logging
from torch.utils.data import Dataset

from bionemo.core.data.multi_epoch_dataset import EpochIndex
from bionemo.core.utils import random_utils
from bionemo.geneformer.data.singlecell.utils import sample_or_truncate
from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.llm.data import masking, types

__all__: Sequence[str] = ("SingleCellDatasetSCMAP", "process_item", "process_item_ncp", "process_cell")


class SingleCellDatasetSCMAP(Dataset):
    """A dataset class for single-cell pre-training using the sc_memmap format.
    
    This dataset reads from numpy memmap files created by sc_memmap.py script,
    which stores gene expression data in CSR (Compressed Sparse Row) format.
    
    Args:
        data_path (str | Path): Path to the sc_memmap directory containing:
            - metadata.json: Metadata about the datasets and feature IDs
            - gene_expression_data.npy: Non-zero gene expression values (CSR data)
            - gene_expression_ind.npy: Column indices for CSR format
            - gene_expression_ptr.npy: Row pointers for CSR format
            - features.csv (optional): Cell metadata
            For next_cell_prediction=True, also requires:
            - pseudotime_neighbors_ind.npy: Neighbor indices in CSR format
            - pseudotime_neighbors_ptr.npy: Neighbor pointers in CSR format
        tokenizer: The tokenizer to use for tokenizing the input data
        median_dict (dict, optional): A dictionary containing median values for each gene
        max_len (int, optional): The maximum length of the input sequence. Defaults to 1024
        mask_prob (float, optional): Probability of masking tokens. Defaults to 0.15
        mask_token_prob (float, optional): Probability of using [MASK] token. Defaults to 0.8
        random_token_prob (float, optional): Probability of random token replacement. Defaults to 0.1
        prepend_cls_token (bool, optional): Whether to prepend CLS token. Defaults to True
        eos_token (int | None, optional): Optional EOS token ID. Defaults to None
        include_unrecognized_vocab_in_dataset (bool, optional): Whether to raise error for unknown genes. Defaults to False
        seed (int, optional): Random seed for reproducibility
        next_cell_prediction (bool, optional): Enable next cell prediction task. Defaults to False
        no_neighbor_policy (str, optional): Policy for handling cells without neighbors. Defaults to "identity"
        assert_increasing_columns (bool, optional): Check if column indices are increasing (for debugging). Defaults to False
    """

    def __init__(
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
        next_cell_prediction: bool = False,
        no_neighbor_policy: str = "identity",
        assert_increasing_columns: bool = False,
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.max_len = max_len
        self.random_token_prob = random_token_prob
        self.mask_token_prob = mask_token_prob
        self.mask_prob = mask_prob
        self.prepend_cls_token = prepend_cls_token
        self._seed = seed
        self.eos_token = eos_token
        self.gene_medians = median_dict
        self.tokenizer = tokenizer
        self.include_unrecognized_vocab_in_dataset = include_unrecognized_vocab_in_dataset
        self.next_cell_prediction = next_cell_prediction
        self.no_neighbor_policy = no_neighbor_policy
        self.assert_increasing_columns = assert_increasing_columns
        
        # Load metadata
        with open(self.data_path / "metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        # Load memmap arrays for CSR format
        self.gene_data = np.memmap(
            self.data_path / "gene_expression_data.npy",
            dtype="float32",
            mode="r"
        )
        self.gene_indices = np.memmap(
            self.data_path / "gene_expression_ind.npy",
            dtype="int32",
            mode="r"
        )
        self.gene_ptr = np.memmap(
            self.data_path / "gene_expression_ptr.npy",
            dtype="int64",
            mode="r"
        )
        
        # Calculate total number of cells
        # Use the minimum of actual array size and metadata to handle padding
        array_cells = len(self.gene_ptr) - 1
        metadata_cells = sum(meta["shape"][0] for meta in self.metadata.values())
        
        if array_cells != metadata_cells:
            logging.warning(
                f"Array size ({array_cells}) differs from metadata ({metadata_cells}). "
                f"Using metadata count to avoid accessing uninitialized data."
            )
            # Use metadata count as it represents actual valid data
            # Arrays may be pre-allocated larger than needed
            self.num_cells = metadata_cells
        else:
            self.num_cells = array_cells
        
        # Load neighbor data if next_cell_prediction is enabled
        if self.next_cell_prediction:
            neighbor_ind_path = self.data_path / "pseudotime_neighbors_ind.npy"
            neighbor_ptr_path = self.data_path / "pseudotime_neighbors_ptr.npy"
            
            if neighbor_ind_path.exists() and neighbor_ptr_path.exists():
                total_pt_el = sum([v.get("num_neighbors_el", 0) for _, v in self.metadata.items()])
                self.next_cell_indices = np.memmap(
                    neighbor_ind_path,
                    dtype="int32",
                    mode="r",
                    shape=(total_pt_el,) if total_pt_el > 0 else (0,)
                )
                self.next_cell_ptr = np.memmap(
                    neighbor_ptr_path,
                    dtype="int64",
                    mode="r",
                    shape=(self.num_cells + 1,)
                )
                logging.info(f"Loaded neighbor data for next cell prediction with {total_pt_el} total neighbor connections")
            else:
                logging.warning(
                    f"Next cell prediction enabled but neighbor files not found at {neighbor_ind_path} and {neighbor_ptr_path}. "
                    f"Will fall back to {no_neighbor_policy} policy for all cells."
                )
                self.next_cell_indices = None
                self.next_cell_ptr = None
        
        # Build mapping from row indices to dataset id (cumulative counts)
        self.dataset_ccum = np.zeros(len(self.metadata))
        # Maps dataset ids to dataset names (used in the metadata dict)
        self.dataset_map = {}
        count = 0
        for i, k in enumerate(self.metadata.keys()):
            self.dataset_ccum[i] = count
            self.dataset_map[i] = k
            count += self.metadata[k]["shape"][0]
        self.dataset_ccum[0] = -1
        
        # Load features if available
        features_path = self.data_path / "features.csv"
        if features_path.exists():
            self.features = pd.read_csv(features_path)
        else:
            self.features = None
            
        logging.info(f"Loaded SingleCellDatasetSCMAP with {self.num_cells} cells from {self.data_path}")
        if self.next_cell_prediction:
            logging.info(f"Next cell prediction enabled with policy: {self.no_neighbor_policy}")
        
    def metadata_lookup(self, idx: int) -> dict:
        """Go from a cell idx to the file-level metadata associated with that cell."""
        did = sum(~(self.dataset_ccum > idx)) - 1
        metadata = self.metadata[self.dataset_map[did]]
        return metadata
    
    def lookup_cell_by_idx(self, idx: int) -> tuple[np.ndarray, np.ndarray, dict]:
        """Lookup cell data by index and return gene data, column indices, and metadata."""
        start_ptr = int(self.gene_ptr[idx])
        end_ptr = int(self.gene_ptr[idx + 1])
        
        # Extract gene expression values and column indices
        gene_data = self.gene_data[start_ptr:end_ptr]
        col_idxs = self.gene_indices[start_ptr:end_ptr].astype(int)
        
        # Optional check for increasing column indices (for debugging)
        if self.assert_increasing_columns and len(col_idxs) > 1:
            is_increasing = np.diff(col_idxs) > 0
            if not np.all(is_increasing):
                raise ValueError(f"Column indices are not increasing for {np.sum(~is_increasing)} pairs of genes")
        
        # Get metadata for this cell's dataset
        metadata = self.metadata_lookup(idx)
        
        return gene_data, col_idxs, metadata
    
    def sample_has_neighbor(self, idx: int) -> bool:
        """Check if a cell has any neighbors."""
        if not self.next_cell_prediction:
            return False
        
        if self.next_cell_ptr is None or self.next_cell_indices is None:
            return False
        
        # We've verified next_cell_ptr is not None
        assert self.next_cell_ptr is not None
        
        ptr_start = int(self.next_cell_ptr[idx])
        ptr_end = int(self.next_cell_ptr[idx + 1])
        
        # Validate pointers are within bounds
        if ptr_start > ptr_end:
            # This can happen if we're reading uninitialized memory beyond valid data
            logging.debug(f"Invalid neighbor pointers for cell {idx}: {ptr_start} > {ptr_end}")
            return False
        
        # Additional safety check: ensure indices are within neighbor array bounds
        if ptr_end > len(self.next_cell_indices):
            logging.debug(f"Neighbor pointer {ptr_end} exceeds array size {len(self.next_cell_indices)}")
            return False
        
        return ptr_end > ptr_start
    
    def sample_neighbor_index(self, idx: int, rng: np.random.Generator) -> int:
        """Sample a random neighbor for the given cell index."""
        if not self.sample_has_neighbor(idx):
            raise ValueError(f"Cell {idx} has no neighbors")
        
        # At this point we know next_cell_ptr and next_cell_indices are not None
        # because sample_has_neighbor returned True
        assert self.next_cell_ptr is not None
        assert self.next_cell_indices is not None
        
        ptr_start = int(self.next_cell_ptr[idx])
        ptr_end = int(self.next_cell_ptr[idx + 1])
        
        # Sample random neighbor
        sampled_ptr = rng.integers(ptr_start, ptr_end)
        neighbor_idx = int(self.next_cell_indices[sampled_ptr])
        
        return neighbor_idx
    
    def lookup_neighbor_by_idx(self, idx: int, rng: np.random.Generator) -> tuple[int, np.ndarray, np.ndarray, dict]:
        """Go from a cell idx to the information about a sampled neighbor."""
        neighbor_idx = self.sample_neighbor_index(idx, rng)
        neighbor_data, neighbor_cols, neighbor_metadata = self.lookup_cell_by_idx(neighbor_idx)
        return neighbor_idx, neighbor_data, neighbor_cols, neighbor_metadata
    
    def __len__(self) -> int:
        """Return the number of cells in the dataset."""
        return self.num_cells
    
    def __getitem__(self, index: EpochIndex) -> types.BertSample:
        """Get a single cell's data and process it for training.
        
        Args:
            index: EpochIndex containing the cell index and epoch
            
        Returns:
            Processed sample ready for training
        """
        cell_idx = index.idx
        if cell_idx >= self.num_cells or cell_idx < 0:
            raise IndexError(f"Cell index {cell_idx} out of range [0, {self.num_cells})")
        
        # Set up RNG for this specific cell and epoch
        rng = np.random.default_rng([self._seed, index.epoch, cell_idx])
        
        # Get cell data using lookup method
        cell_gene_data, cell_gene_indices, metadata = self.lookup_cell_by_idx(cell_idx)
        
        if len(cell_gene_data) == 0:
            raise ValueError(f"No gene expression data for cell {cell_idx}")
        
        # Check if we should do next cell prediction
        if self.next_cell_prediction:
            if self.sample_has_neighbor(cell_idx):
                # Get neighbor cell data
                neighbor_idx, neighbor_gene_data, neighbor_gene_indices, neighbor_metadata = self.lookup_neighbor_by_idx(cell_idx, rng)
                
                Use process_item_ncp for temporal prediction
                Note: Assuming metadata is shared between current and next cells (from same AnnData)
                return process_item_ncp(
                    cell_gene_data,
                    cell_gene_indices,
                    neighbor_gene_data,
                    neighbor_gene_indices,
                    metadata,  # Use current cell's metadata (should be same as neighbor's)
                    self.tokenizer,
                    cast(dict, self.gene_medians),
                    rng,
                    max_len=self.max_len,
                    mask_prob=self.mask_prob,
                    mask_token_prob=self.mask_token_prob,
                    random_token_prob=self.random_token_prob,
                    prepend_cls_token=self.prepend_cls_token,
                    eos_token=self.eos_token,
                    include_unrecognized_vocab_in_dataset=self.include_unrecognized_vocab_in_dataset,
                )
            else:
                # No neighbor - use policy
                if self.no_neighbor_policy == "identity":
                    # Fall back to single cell processing
                    return process_item(
                        cell_gene_data,
                        cell_gene_indices,
                        metadata,
                        self.tokenizer,
                        cast(dict, self.gene_medians),
                        rng,
                        max_len=self.max_len,
                        mask_prob=self.mask_prob,
                        mask_token_prob=self.mask_token_prob,
                        random_token_prob=self.random_token_prob,
                        prepend_cls_token=self.prepend_cls_token,
                        eos_token=self.eos_token,
                        include_unrecognized_vocab_in_dataset=self.include_unrecognized_vocab_in_dataset,
                    )
                else:
                    raise NotImplementedError(f"Unknown no_neighbor_policy: {self.no_neighbor_policy}")
        else:
            # Regular single cell processing
            return process_item(
                cell_gene_data,
                cell_gene_indices,
                metadata,
                self.tokenizer,
                cast(dict, self.gene_medians),
                rng,
                max_len=self.max_len,
                mask_prob=self.mask_prob,
                mask_token_prob=self.mask_token_prob,
                random_token_prob=self.random_token_prob,
                prepend_cls_token=self.prepend_cls_token,
                eos_token=self.eos_token,
                include_unrecognized_vocab_in_dataset=self.include_unrecognized_vocab_in_dataset,
            )


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
                med = gene_median[tok]
                medians.append(med)
        elif include_unrecognized_vocab_in_dataset:
            raise ValueError(f"Provided gene identifier, {str(tok)}, is not in the tokenizer vocab.")
    return np.asarray(genes), np.asarray(tokens), np.asarray(medians)


def process_item(
    gene_data: np.ndarray,
    gene_idxs: np.ndarray,
    metadata: dict,
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
    
    Optionally performs median normalization and rank ordering. The tokenizer's CLS token is added 
    to the beginning of every sample. Converts gene names to ensemble ids before tokenizing. 
    Expects gene_medians to contain ensembl ids as keys.
    
    Args:
        gene_data: List of gene data (expression counts)
        gene_idxs: List of gene indices (keys in 'metadata["feature_ids"]')
        metadata: Metadata dictionary containing feature_ids
        tokenizer: Tokenizer object
        gene_median: Dictionary of gene medians (expects ensembl IDs as keys)
        rng: Random number generator for deterministic results
        max_len: Maximum length of the item
        mask_prob: Probability of masking a token
        mask_token_prob: Probability of using [MASK] token
        random_token_prob: Probability of random token replacement
        target_sum: Target sum for normalization
        normalize: Flag to normalize and rank order the gene data
        prepend_cls_token: Whether to prepend a [CLS] token
        eos_token: Optional EOS token ID
        include_unrecognized_vocab_in_dataset: Whether to raise error for unknown genes
        
    Returns:
        Processed item dictionary for training
    """
    if max_len < 1:
        raise ValueError(f"max_len must be greater than 1, {max_len=}")
    
    if gene_median is None:
        raise ValueError("gene_median must be provided for this tokenizer")
    
    if prepend_cls_token:
        max_len = max_len - 1  # Reserve space for CLS token
    if eos_token is not None:
        max_len = max_len - 1  # Reserve space for EOS token
    
    # Get gene names from metadata
    gene_names = np.array([metadata["feature_ids"][idx] for idx in gene_idxs])
    
    # Filter genes and get tokens
    gene_expression_cell, token_ids, gene_expression_medians = _gather_medians(
        gene_names,
        gene_data,
        normalize,
        tokenizer.vocab,
        gene_median,
        include_unrecognized_vocab_in_dataset=include_unrecognized_vocab_in_dataset,
    )
    
    if normalize and len(gene_expression_cell) > 0:
        # Normalize and rank order by expression
        gene_expression_cell = gene_expression_cell / gene_expression_cell.sum() * target_sum
        gene_expression_cell = gene_expression_cell / gene_expression_medians.astype(float)
        idxs = np.argsort(-gene_expression_cell)  # Descending order
        gene_expression_cell = gene_expression_cell[idxs]
        token_ids = token_ids[idxs]
    
    # Truncate to max_len
    token_ids = sample_or_truncate(token_ids, max_len, sample=False)
    
    # Apply masking
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
        
        # Add special tokens
        cls_token = tokenizer.token_to_id(tokenizer.cls_token) if prepend_cls_token else None
        if cls_token is not None or eos_token is not None:
            masked_tokens, labels, loss_mask = masking.add_cls_and_eos_tokens(
                sequence=masked_tokens,
                labels=labels,
                loss_mask=loss_mask,
                cls_token=cls_token,
                eos_token=eos_token,
            )
        
        # Return in NeMo format
        return {
            "text": masked_tokens,
            "types": torch.zeros_like(masked_tokens, dtype=torch.int64),
            "attention_mask": torch.ones_like(masked_tokens, dtype=torch.int64),
            "labels": labels,
            "loss_mask": loss_mask,
            "is_random": torch.zeros_like(masked_tokens, dtype=torch.int64),
        }


def process_cell(
    gene_data: np.ndarray,
    gene_idxs: np.ndarray,
    tokenizer: GeneTokenizer,
    metadata: dict,
    gene_median: dict,
    normalize: bool = True,
    target_sum: int = 10000,
) -> np.ndarray:
    """Process data in a given cell by tokenizing the genes, normalizing, and sorting.
    
    Args:
        gene_data: Gene expression counts
        gene_idxs: Gene indices (keys in metadata['feature_ids'])
        tokenizer: Tokenizer object
        metadata: Metadata dictionary containing feature_ids
        gene_median: Dictionary of gene medians
        normalize: Flag to normalize the gene data
        target_sum: Target sum for normalization
        
    Returns:
        Re-ordered token sequence in descending count order
    """
    gene_names = [metadata["feature_ids"][idx] for idx in gene_idxs]
    genes, tokens, medians = [], [], []
    for tok, gene in zip(gene_names, gene_data):
        if tok in tokenizer.vocab:
            tokens.append(tokenizer.token_to_id(tok))
            genes.append(gene)
            if normalize:
                med = gene_median.get(tok, 1)  # Default to 1 if not in dictionary
                medians.append(med)
    
    genes = np.asarray(genes)
    token_ids = np.asarray(tokens)
    medians = np.asarray(medians)
    
    if normalize and len(genes) > 0:
        # Re-order according to expression median normalized rank (descending)
        genes = genes / genes.sum() * target_sum
        genes = genes / medians.astype(float)
        idxs = np.argsort(-genes)
        genes = genes[idxs]
        token_ids = token_ids[idxs]
    
    return token_ids


def process_item_ncp(
    gene_data: np.ndarray,
    gene_idxs: np.ndarray, 
    next_gene_data: np.ndarray,
    next_gene_idxs: np.ndarray,
    metadata: dict,
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
    Note: Assumes metadata for current and next cells is shared (from same dataset).
    """
    if max_len < 1:
        raise ValueError(f"max_len must be greater than 1, {max_len=}")
    if gene_median is None:
        raise ValueError("gene_median must be provided for this tokenizer")
    
    # Reserve for special tokens: optional CLS, required SEP, optional EOS
    reserved = (1 if prepend_cls_token else 0) + 1 + (1 if eos_token is not None else 0)
    if max_len <= reserved:
        raise ValueError(f"max_len must be > reserved tokens ({reserved}), got {max_len}")
    
    # Process current cell
    cur_tokens = process_cell(
        gene_data, 
        gene_idxs, 
        tokenizer, 
        metadata, 
        gene_median, 
        normalize, 
        target_sum
    )
    
    # Process next cell
    next_tokens = process_cell(
        next_gene_data,
        next_gene_idxs,
        tokenizer,
        metadata,
        gene_median,
        normalize,
        target_sum
    )
    
    # Split available length between current and next
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


def _profile_scmap_dataset():
    """Profile function to test the SingleCellDatasetSCMAP implementation."""
    import random
    import time
    from tqdm import tqdm
    from bionemo.core.data.load import load
    from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
    
    # This assumes you have sc_memmap formatted data available
    # You'll need to adjust the path to your actual sc_memmap data
    data_path = Path("path/to/your/scmemmap/data")  # Update this path
    
    if not data_path.exists():
        logging.warning(f"Test data path {data_path} does not exist. Please update the path.")
        return
    
    # Load tokenizer and medians (adjust paths as needed)
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
    
    # Test both regular and NCP modes
    for ncp_mode in [False, True]:
        logging.info(f"\n{'='*50}")
        logging.info(f"Testing with next_cell_prediction={ncp_mode}")
        logging.info(f"{'='*50}")
        
        # Create dataset
        dataset = SingleCellDatasetSCMAP(
            data_path=data_path,
            tokenizer=tokenizer,
            median_dict=median_dict,
            max_len=2048,
            seed=321,
            next_cell_prediction=ncp_mode,
            no_neighbor_policy="identity"
        )
        
        logging.info(f"Dataset loaded with {len(dataset)} cells")
        
        # Test a few samples
        n_epochs = 1
        n_samples = min(100, len(dataset))  # Test first 100 samples
        idxs = list(range(n_samples * n_epochs))
        random.seed(315)
        random.shuffle(idxs)
        
        start = time.monotonic()
        for i in tqdm(idxs, desc=f"Processing samples (NCP={ncp_mode})"):
            try:
                sample = dataset[EpochIndex(idx=i % n_samples, epoch=i // n_samples)]
                # Verify sample has expected keys
                assert "text" in sample
                assert "labels" in sample
                assert "loss_mask" in sample
            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                raise
        
        stop = time.monotonic()
        logging.info(f"Processed {n_samples * n_epochs} samples in {stop - start:.2f} seconds")
        logging.info(f"Average time per sample: {(stop - start) / (n_samples * n_epochs):.4f} seconds")


if __name__ == "__main__":
    # Run profile when executed directly
    _profile_scmap_dataset() 