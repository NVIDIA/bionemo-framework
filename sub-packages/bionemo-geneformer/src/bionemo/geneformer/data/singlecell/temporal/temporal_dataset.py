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
# SPDX-License-Identifier: Apache-2.0
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

"""Temporal dataset for next-cell prediction training using SCDL neighbor data."""

import logging
from pathlib import Path
from typing import Any, Dict, Literal, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from bionemo.llm.data import masking, types
from bionemo.core.data.multi_epoch_dataset import EpochIndex
from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.geneformer.data.singlecell.dataset import process_item


logger = logging.getLogger(__name__)

# TypeVar for Dataset typing
 
 
class TemporalGeneformerDataset(Dataset):
    """Dataset for temporal next-cell prediction training.

    This dataset implements the temporal training strategy where:
    1. Each sample consists of a current cell and its neighbor (next cell)
    2. The current cell provides context (no masking)
    3. The neighbor cell is masked for prediction
    4. Sequences are concatenated with [SEP] token: [CLS] + current + [SEP] + next
    5. Attention is restricted to prevent next cell from attending to itself

    Args:
        data_path: Path to SCDL dataset with neighbor information
        tokenizer: Geneformer tokenizer
        median_dict: Dictionary of median gene expression values
        max_len: Maximum sequence length (default: 2048)
        mask_prob: Probability of masking tokens in next cell (default: 0.15)
        mask_token_prob: Probability of using [MASK] token (default: 0.8)
        random_token_prob: Probability of using random token (default: 0.1)
        neighbor_key: Key for neighbor data in SCDL (default: "neighbors")
        seed: Random seed for reproducibility (default: 42)
        only_cells_with_neighbors: Only include cells that have neighbors (default: True)
        no_neighbor_policy: Policy for handling cells without neighbors (default: "skip")
        token_selection_policy: Policy for selecting tokens from next cell (default: "identity")
        normalize_gene_expression: Whether to normalize gene expression (default: True)
        target_sum: Target sum for normalization (default: 10000)
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Any,
        median_dict: Dict[str, float],
        max_len: int = 2048,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
        prepend_cls_token: bool = True,
        eos_token: int | None = None,
        neighbor_key: str = "next_cell_ids",
        seed: int = 42,
        only_cells_with_neighbors: bool = True,
        no_neighbor_policy: Literal["skip", "identity", "random"] = "skip",
        token_selection_policy: Literal["identity", "intersection", "union"] = "identity",
        normalize_gene_expression: bool = True,
        target_sum: int = 10000,
        include_unrecognized_vocab_in_dataset: bool = False,
    ):
        """Initialize the temporal dataset for next-cell prediction.

        Args:
            data_path: Path to the AnnData file containing single-cell data
            tokenizer: Gene tokenizer for converting gene names to tokens
            median_dict: Dictionary containing median expression values for genes
            max_len: Maximum sequence length for tokenized data
            mask_prob: Probability of masking tokens in the next cell
            mask_token_prob: Probability of replacing masked tokens with [MASK]
            random_token_prob: Probability of replacing masked tokens with random tokens
            neighbor_key: Key in AnnData.obsp containing neighbor information
            seed: Random seed for reproducibility
            only_cells_with_neighbors: Whether to only include cells that have neighbors
            no_neighbor_policy: Policy for handling cells without neighbors ('skip', 'identity', 'random')
            token_selection_policy: Policy for selecting tokens when cells have different genes
            normalize_gene_expression: Whether to normalize gene expression values
            target_sum: Target sum for normalization if normalize_gene_expression is True
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.gene_medians = median_dict
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob
        self.prepend_cls_token = prepend_cls_token
        self.neighbor_key = neighbor_key
        self.seed = seed
        self.eos_token = eos_token
        self.only_cells_with_neighbors = only_cells_with_neighbors
        self.no_neighbor_policy = no_neighbor_policy
        self.token_selection_policy = token_selection_policy
        self.normalize_gene_expression = normalize_gene_expression
        self.target_sum = target_sum
        self.include_unrecognized_vocab_in_dataset = include_unrecognized_vocab_in_dataset

        # Validate parameters
        if random_token_prob + mask_token_prob > 1.0:
            raise ValueError("Sum of random_token_prob and mask_token_prob must be <= 1.0")

        # Load SCDL dataset with neighbor support
        self.scdl = SingleCellMemMapDataset(
            data_path=str(self.data_path),
            load_neighbors=True,
            neighbor_key=self.neighbor_key,
            neighbor_sampling_strategy="random",
            fallback_to_identity=not self.only_cells_with_neighbors,
        )

        # Validate neighbor data was loaded
        if not self.scdl._has_neighbors:
            raise ValueError(f"No neighbor data found in {self.data_path} with key '{self.neighbor_key}'")

        # Filter to only cells with neighbors if requested
        if self.only_cells_with_neighbors:
            self.valid_indices = []
            for i in range(self.scdl.number_of_rows()):
                neighbors = self.scdl.get_neighbor_indices_for_cell(i)
                if len(neighbors) > 0:
                    self.valid_indices.append(i)

            if len(self.valid_indices) == 0:
                raise ValueError("No cells with neighbors found in the dataset")

            logger.info(
                f"Found {len(self.valid_indices)} cells with neighbors out of {self.scdl.number_of_rows()} total cells"
            )
        else:
            self.valid_indices = list(range(self.scdl.number_of_rows()))

        # Token IDs for sequence construction
        self.cls_token_id = self.tokenizer.token_to_id(self.tokenizer.cls_token)
        self.sep_token_id = self.tokenizer.token_to_id(self.tokenizer.sep_token)
        self.pad_token_id = self.tokenizer.token_to_id(self.tokenizer.pad_token)
        self.mask_token_id = self.tokenizer.token_to_id(self.tokenizer.mask_token)

        logger.info(f"Initialized TemporalGeneformerDataset with {len(self)} samples")

    def __len__(self) -> int:
        """Return the number of valid samples."""
        return len(self.valid_indices)

    def __getitem__(self, index: Union[int, EpochIndex]) -> types.BertSample:
        """Get a temporal training sample.

        Args:
            index: Index of the sample (int for direct access, EpochIndex for multi-epoch training)

        Returns:
            Dictionary containing:
            - text: Combined sequence [CLS] + current + [SEP] + next
            - attention_mask: Standard padding mask
            - labels: Target labels for masked tokens
            - loss_mask: Mask indicating which tokens to compute loss on
            - types: Token type IDs (all zeros to match baseline)
            - has_neighbor: Whether this sample has a real neighbor
        """
        # Handle both int and EpochIndex
        if isinstance(index, EpochIndex):
            epoch, idx = index.epoch, index.idx
        else:
            epoch, idx = 0, index
            
        # Get the actual cell index
        cell_idx = self.valid_indices[idx]

        # Check if cell has neighbors
        neighbors = self.scdl.get_neighbor_indices_for_cell(cell_idx)
        has_neighbor = len(neighbors) > 0

        if has_neighbor:
            # Create deterministic RNG for neighbor sampling that varies by epoch
            neighbor_rng = np.random.default_rng([self.seed, epoch, cell_idx])
            # Sample a neighbor using SCDL's deterministic sampling
            neighbor_idx = self.scdl.sample_neighbor_index(cell_idx, rng=neighbor_rng)

            # Process current cell (no masking)
            current_cell_data = self._process_cell(cell_idx, apply_masking=False, epoch=epoch)

            # Process neighbor cell (with masking)
            next_cell_data = self._process_cell(neighbor_idx, apply_masking=True, epoch=epoch)

            # Apply token selection policy
            next_cell_data = self._apply_token_selection_policy(
                current_cell_data, next_cell_data, self.token_selection_policy
            )

            # Create temporal sequence
            temporal_sample = self._create_temporal_sequence(current_cell_data, next_cell_data)
            # temporal_sample["has_neighbor"] = torch.tensor(True, dtype=torch.bool)

            return temporal_sample

        else:
            # Handle cells without neighbors
            # temporal_sample = self._handle_no_neighbor_case(cell_idx)
            # temporal_sample["has_neighbor"] = torch.tensor(False, dtype=torch.bool)
            return self.ncp_no_neighbor_policy(cell_idx, self.no_neighbor_policy, epoch=epoch)


    def ncp_no_neighbor_policy(self, idx: int, type: str = "identity", epoch: int = 0):
        if type == "identity":
            rng = np.random.default_rng([self.seed, epoch, idx])

            # Get raw cell data from SCDL (same format as SingleCellDataset)
            values, feature_ids = self.scdl.get_row(idx, return_features=True, feature_vars=["feature_id"])
            assert len(feature_ids) == 1  # Expect one array with feature IDs

            gene_data, col_idxs = np.array(values[0]), np.array(values[1])

            if len(gene_data) == 0:
                raise ValueError(
                    f"TemporalGeneformerDataset data provided is invalid; the gene expression data parsed for cell {idx} is empty."
                )
            return process_item(
                gene_data,
                col_idxs,
                feature_ids[0],
                self.tokenizer,
                gene_median=self.gene_medians,
                rng=rng,
                max_len=self.max_len,
                mask_token_prob=self.mask_token_prob,
                mask_prob=self.mask_prob,
                random_token_prob=self.random_token_prob,
                prepend_cls_token=self.prepend_cls_token,
                eos_token=self.eos_token,
                target_sum=self.target_sum,
                normalize=self.normalize_gene_expression,
                include_unrecognized_vocab_in_dataset=self.include_unrecognized_vocab_in_dataset,
            )
        else:
            raise NotImplementedError(f"Unknown no_neighbor_policy: {type}")

    def _apply_token_selection_policy(
        self, current_cell_data: Dict[str, Any], next_cell_data: Dict[str, Any], policy: str
    ) -> Dict[str, Any]:
        """Apply token selection policy to determine which tokens from next cell to include.

        Args:
            current_cell_data: Processed current cell data with token_ids
            next_cell_data: Processed next cell data with token_ids
            policy: Token selection policy ("identity", "intersection", "union")

        Returns:
            Modified next_cell_data based on the policy
        """
        if policy == "identity":
            # Use next cell as-is (default behavior)
            return next_cell_data

        elif policy == "intersection":
            # Only include tokens that appear in both current and next cell
            current_tokens = set(current_cell_data["token_ids"])
            next_tokens = next_cell_data["token_ids"]

            # Find intersection (excluding special tokens)
            special_tokens = {
                self.tokenizer.token_to_id(self.tokenizer.cls_token),
                self.tokenizer.token_to_id(self.tokenizer.pad_token),
                self.sep_token_id,
            }

            # Filter next cell tokens to only include those in current cell
            filtered_tokens = []
            for token in next_tokens:
                if token in current_tokens or token in special_tokens:
                    filtered_tokens.append(token)

            # Update next cell data with filtered tokens
            next_cell_data = next_cell_data.copy()
            next_cell_data["token_ids"] = np.array(filtered_tokens)
            return next_cell_data

        elif policy == "union":
            # Include all tokens from both cells (prioritize next cell order)
            # This is more complex and might not be commonly used
            # For now, default to identity behavior
            return next_cell_data

        else:
            raise ValueError(f"Unknown token_selection_policy: {policy}")

    def _process_cell(self, cell_idx: int, apply_masking: bool = False, epoch: int = 0) -> Dict[str, Any]:
        """Process a single cell using core Geneformer processing steps.

        This method extracts the proven core processing from process_item() but stops
        before final masking/padding so we can handle temporal sequence construction:
        - Uses _gather_medians() for gene filtering and tokenization
        - Applies normalization and ranking
        - Returns raw token_ids for temporal concatenation
        - Masking and padding handled at temporal sequence level

        Args:
            cell_idx: Index of the cell to process
            apply_masking: Whether this cell should be masked (only for next cell)
            epoch: Current epoch for proper RNG seeding

        Returns:
            Dictionary with core processed data: token_ids, should_mask flag
        """
        # Create RNG for this specific cell and masking state with epoch information
        rng = np.random.default_rng([self.seed, epoch, cell_idx, int(apply_masking)])

        # Get raw cell data from SCDL (same format as SingleCellDataset)
        values, feature_ids = self.scdl.get_row(cell_idx, return_features=True, feature_vars=["feature_id"])
        assert len(feature_ids) == 1  # Expect one array with feature IDs

        gene_data, col_idxs = np.array(values[0]), np.array(values[1])

        if len(gene_data) == 0:
            raise ValueError(
                f"TemporalGeneformerDataset data provided is invalid; the gene expression data parsed for cell {cell_idx} is empty."
            )

        # Use the core processing steps from process_item()
        gene_names = feature_ids[0][col_idxs]

        # Step 1: _gather_medians (proven gene filtering and tokenization)
        from bionemo.geneformer.data.singlecell.dataset import _gather_medians

        gene_expression_cell, token_ids, gene_expression_medians = _gather_medians(
            gene_names,
            gene_data,
            self.normalize_gene_expression,
            self.tokenizer.vocab,
            self.gene_medians,
            include_unrecognized_vocab_in_dataset=False,
        )

        # Step 2: Normalization and ranking (if enabled)
        if self.normalize_gene_expression:
            # Proven normalization logic from process_item()
            gene_expression_cell = gene_expression_cell / gene_expression_cell.sum() * self.target_sum
            gene_expression_cell = gene_expression_cell / gene_expression_medians.astype(float)
            idxs = np.argsort(-gene_expression_cell)  # Descending order
            gene_expression_cell = gene_expression_cell[idxs]
            token_ids = token_ids[idxs]

        # Return raw token_ids and masking flag for temporal sequence construction
        #NOTE: we don't handle masking here so do we need to track a should_mask flag? re-visit this
        return {
            "token_ids": token_ids,
            "should_mask": apply_masking,
            "rng": rng,  # Pass RNG for consistent masking later
        }

    def _calculate_half_sequence_length(self, max_length: int) -> int:
        """Calculate half sequence length accounting for special tokens.

        Based on the old NCP implementation's adjust_half_length function.

        Args:
            max_length: Total maximum sequence length

        Returns:
            Half sequence length adjusted for special tokens
        """
        # Account for [CLS] and [SEP] tokens
        available_length = max_length - 2

        # Split evenly, but handle odd lengths
        if available_length % 2 == 0:
            return available_length // 2
        else:
            return available_length // 2

    def _create_temporal_sequence(
        self, current_cell_data: Dict[str, Any], next_cell_data: Dict[str, Any]
    ) -> types.BertSample:
        """Create temporal sequence by concatenating current and next cell.

        This method now properly handles the core processing pipeline:
        - Truncates raw token_ids to appropriate lengths
        - Concatenates with CLS and SEP tokens
        - Applies BERT masking only to next cell
        - Creates proper attention masks and labels

        Args:
            current_cell_data: Raw processed current cell data (token_ids, should_mask=False)
            next_cell_data: Raw processed next cell data (token_ids, should_mask=True)

        Returns:
            Combined temporal sequence ready for training
        """
        from bionemo.core.utils import random_utils
        from bionemo.geneformer.data.singlecell.utils import sample_or_truncate
        from bionemo.llm.data import masking

        # Calculate lengths for temporal concatenation accounting for optional CLS and EOS
        tokens_reserved = (1 if self.prepend_cls_token else 0) + 1 + (1 if self.eos_token is not None else 0)  # SEP always present
        available_length = self.max_len - tokens_reserved
        current_max = available_length // 2
        next_max = available_length - current_max

        # Truncate token sequences (no sampling to preserve rank order)
        current_tokens = sample_or_truncate(current_cell_data["token_ids"], current_max, sample=False)
        next_tokens = sample_or_truncate(next_cell_data["token_ids"], next_max, sample=False)

        # Convert to torch tensors
        current_tokens = torch.from_numpy(current_tokens)
        next_tokens = torch.from_numpy(next_tokens)

        # Create optional CLS and required SEP tokens
        cls_token = (
            torch.tensor([self.tokenizer.token_to_id(self.tokenizer.cls_token)])
            if self.prepend_cls_token
            else torch.empty(0, dtype=torch.long)
        )
        sep_token = torch.tensor([self.sep_token_id])
        eos_token_tensor = (
            torch.tensor([self.eos_token], dtype=torch.long)
            if self.eos_token is not None
            else torch.empty(0, dtype=torch.long)
        )

        # Concatenate: optional [CLS] + current + [SEP] + next + optional [EOS]
        combined_tokens = torch.cat([cls_token, current_tokens, sep_token, next_tokens, eos_token_tensor])

        # Apply masking only to next cell portion
        current_len = len(cls_token) + len(current_tokens) + len(sep_token)

        # Create labels: current cell and SEP have no loss (-100), next cell gets proper labels
        labels = torch.full_like(combined_tokens, -100)
        loss_mask = torch.zeros_like(combined_tokens, dtype=torch.bool)

        if next_cell_data["should_mask"]:
            # Apply BERT masking only to next cell tokens
            next_start_idx = current_len
            # exclude EOS if present
            next_end_idx = len(combined_tokens) - (1 if self.eos_token is not None else 0)

            # Extract next cell tokens for masking
            next_tokens_for_masking = combined_tokens[next_start_idx:next_end_idx]

            # Apply BERT masking using proven logic
            with torch.no_grad():
                masked_next_tokens, next_labels, next_loss_mask = masking.apply_bert_pretraining_mask(
                    tokenized_sequence=next_tokens_for_masking,
                    random_seed=int(random_utils.get_seed_from_rng(next_cell_data["rng"])),
                    mask_config=masking.BertMaskConfig(
                        tokenizer=self.tokenizer,
                        random_tokens=range(len(self.tokenizer.special_tokens), len(self.tokenizer.vocab)),
                        mask_prob=self.mask_prob,
                        mask_token_prob=self.mask_token_prob,
                        random_token_prob=self.random_token_prob,
                    ),
                )

            # Update the combined sequence with masked next cell tokens
            combined_tokens[next_start_idx:next_end_idx] = masked_next_tokens
            labels[next_start_idx:next_end_idx] = next_labels
            loss_mask[next_start_idx:next_end_idx] = next_loss_mask
        else:
            # For current cell (no masking), labels remain -100, loss_mask remains False
            pass

        # Create attention mask (1 for real tokens, 0 for padding)
        seq_len = len(combined_tokens)
        attention_mask = torch.ones(seq_len, dtype=torch.long)

        # Create token type IDs (all zeros to match baseline behavior)
        types = torch.zeros_like(combined_tokens, dtype=torch.long)

        # Optional temporal attention mask (not used in standard stack)
        temporal_attention_mask = self._create_1d_temporal_attention_mask(
            current_len=current_len,
            next_len=len(next_tokens),
            total_len=seq_len,
        )

        # attention_mask = (combined_tokens != self.pad_token_id).long()
        pad_positions = (combined_tokens == self.pad_token_id)
        attention_mask[pad_positions] = 0

        # Pad to max_len if needed
        if seq_len < self.max_len:
            pad_len = self.max_len - seq_len

            # Pad all tensors
            combined_tokens = torch.cat([combined_tokens, torch.full((pad_len,), self.pad_token_id)])
            temporal_attention_mask = torch.cat([temporal_attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            types = torch.cat([types, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100)])
            loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=torch.bool)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])

        return {
            "text": combined_tokens,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": loss_mask,
            "types": types,
            "is_random": torch.zeros_like(combined_tokens, dtype=torch.long),
        }

    def _create_1d_temporal_attention_mask(self, current_len: int, next_len: int, total_len: int) -> torch.Tensor:
        """Create 1D attention mask where only current tokens can attend.
        
        This creates a simple 1D mask that allows:
        - Current cell tokens (including CLS and SEP): can attend (mask = 1)
        - Next cell tokens: cannot attend (mask = 0)
        
        Args:
            current_len: Length of current cell sequence (including CLS and SEP)
            next_len: Length of next cell sequence
            total_len: Total sequence length
            
        Returns:
            1D attention mask of shape (total_len,) where 1=can attend, 0=cannot attend
        """
        mask = torch.zeros(total_len, dtype=torch.long)
        
        # Only current cell tokens (including CLS and SEP) can attend
        mask[:current_len] = 1
        
        # Next cell tokens cannot attend (remain 0)
        # mask[current_len:] = 0  # Already zero from initialization
        
        return mask
