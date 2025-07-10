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

"""Temporal data module for Geneformer training."""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.geneformer.data.singlecell.temporal.temporal_dataset import TemporalGeneformerDataset


__all__ = ["TemporalGeneformerDataModule"]

logger = logging.getLogger(__name__)


class TemporalGeneformerDataModule(SingleCellDataModule):
    """Data module for temporal Geneformer training.

    This extends the standard SingleCellDataModule to support temporal
    datasets with neighbor information for next-cell prediction.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Tokenizer,
        median_dict: dict[str, float],
        seq_length: int = 2048,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
        neighbor_key: str = "neighbors",
        micro_batch_size: int = 4,
        global_batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        seed: int = 42,
        only_cells_with_neighbors: bool = True,
        no_neighbor_policy: str = "skip",
        token_selection_policy: str = "identity",
        normalize_gene_expression: bool = True,
        target_sum: int = 10000,
        **kwargs,
    ):
        """Initialize the temporal data module.

        Args:
            data_path: Path to the SCDL dataset
            tokenizer_vocab_path: Path to the tokenizer vocabulary
            median_dict_path: Path to the median dictionary
            seq_length: Maximum sequence length
            mask_prob: Probability of masking tokens
            mask_token_prob: Probability of using mask token
            random_token_prob: Probability of using random token
            neighbor_key: Key for neighbor data in SCDL
            micro_batch_size: Micro batch size
            global_batch_size: Global batch size
            num_workers: Number of data loader workers
            pin_memory: Whether to pin memory
            persistent_workers: Whether to use persistent workers
            seed: Random seed
            only_cells_with_neighbors: Whether to only use cells with neighbors
            no_neighbor_policy: Policy for handling cells without neighbors ("skip", "identity", "random")
            token_selection_policy: Policy for selecting tokens from next cell ("identity", "intersection", "union")
            normalize_gene_expression: Whether to normalize gene expression
            target_sum: Target sum for normalization
            **kwargs: Additional arguments
        """
        # Initialize parent class without calling setup yet
        super().__init__(
            seq_length=seq_length,
            tokenizer=None,  # Will be set during preprocessing
            train_dataset_path=str(data_path),
            val_dataset_path=str(data_path),
            test_dataset_path=str(data_path),
            random_token_prob=random_token_prob,
            median_dict=None,  # Will be set during preprocessing
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **kwargs,
        )

        # Store temporal-specific parameters
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.median_dict = median_dict
        self.seq_length = seq_length  # Store seq_length for compatibility
        self.mask_prob = mask_prob
        self.mask_token_prob = mask_token_prob
        self.neighbor_key = neighbor_key
        self.seed = seed
        self.only_cells_with_neighbors = only_cells_with_neighbors
        self.no_neighbor_policy = no_neighbor_policy
        self.token_selection_policy = token_selection_policy
        self.normalize_gene_expression = normalize_gene_expression
        self.target_sum = target_sum

        # Datasets will be created during setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        logger.info(f"Initialized TemporalGeneformerDataModule with data_path: {self.data_path}")
        logger.info(
            f"Enhanced features: no_neighbor_policy={no_neighbor_policy}, token_selection_policy={token_selection_policy}"
        )

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation, and testing.

        Args:
            stage: Stage of training (fit, validate, test, predict)
        """
        logger.info(f"Setting up temporal datasets for stage: {stage}")

        # Create temporal datasets
        if stage == "fit" or stage is None:
            self.train_dataset = TemporalGeneformerDataset(
                data_path=self.data_path,
                tokenizer=self.tokenizer,
                median_dict=self.median_dict,
                max_len=self.seq_length,
                mask_prob=self.mask_prob,
                mask_token_prob=self.mask_token_prob,
                random_token_prob=self.random_token_prob,
                neighbor_key=self.neighbor_key,
                seed=self.seed,
                only_cells_with_neighbors=self.only_cells_with_neighbors,
                no_neighbor_policy=self.no_neighbor_policy,
                token_selection_policy=self.token_selection_policy,
                normalize_gene_expression=self.normalize_gene_expression,
                target_sum=self.target_sum,
            )

            # For validation, we can use the same dataset with a different seed
            # or create a separate validation split
            self.val_dataset = TemporalGeneformerDataset(
                data_path=self.data_path,
                tokenizer=self.tokenizer,
                median_dict=self.median_dict,
                max_len=self.seq_length,
                mask_prob=self.mask_prob,
                mask_token_prob=self.mask_token_prob,
                random_token_prob=self.random_token_prob,
                neighbor_key=self.neighbor_key,
                seed=self.seed + 1,  # Different seed for validation
                only_cells_with_neighbors=self.only_cells_with_neighbors,
                no_neighbor_policy=self.no_neighbor_policy,
                token_selection_policy=self.token_selection_policy,
                normalize_gene_expression=self.normalize_gene_expression,
                target_sum=self.target_sum,
            )

            logger.info(f"Created training dataset with {len(self.train_dataset)} samples")
            logger.info(f"Created validation dataset with {len(self.val_dataset)} samples")

        if stage == "test" or stage is None:
            self.test_dataset = TemporalGeneformerDataset(
                data_path=self.data_path,
                tokenizer=self.tokenizer,
                median_dict=self.median_dict,
                max_len=self.seq_length,
                mask_prob=self.mask_prob,
                mask_token_prob=self.mask_token_prob,
                random_token_prob=self.random_token_prob,
                neighbor_key=self.neighbor_key,
                seed=self.seed + 2,  # Different seed for testing
                only_cells_with_neighbors=self.only_cells_with_neighbors,
                no_neighbor_policy=self.no_neighbor_policy,
                token_selection_policy=self.token_selection_policy,
                normalize_gene_expression=self.normalize_gene_expression,
                target_sum=self.target_sum,
            )

            logger.info(f"Created test dataset with {len(self.test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.data_sampler.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.data_sampler.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.data_sampler.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        """Custom collate function for temporal batches.

        Handles batching of temporal training samples where each sample contains:
        - text: Combined sequence [CLS] + current + [SEP] + next
        - attention_mask: 1D temporal attention mask (current tokens=1, next tokens=0)
        - labels: Target labels for masked tokens (-100 for no loss)
        - loss_mask: Boolean mask indicating which tokens to compute loss on
        - types: Token type IDs (0 for current, 1 for next)
        - is_random: Random token indicators
        - has_neighbor: Boolean flag indicating if sample has real neighbor

        Args:
            batch: List of samples from the temporal dataset

        Returns:
            Batched data dictionary ready for next-cell training
        """
        # Standard collation for most fields
        collated = {}

        # Get all keys from the first sample
        keys = batch[0].keys()

        for key in keys:
            if key in ["text", "attention_mask", "labels", "loss_mask", "types", "is_random"]:
                # Stack tensor fields - these are the core training tensors
                collated[key] = torch.stack([sample[key] for sample in batch])
            elif key == "has_neighbor":
                # Handle has_neighbor boolean flag
                collated[key] = torch.stack([sample[key] for sample in batch])
            else:
                # Keep any other fields as lists (for debugging/metadata)
                collated[key] = [sample[key] for sample in batch]

        return collated

    @property
    def vocab_size(self):
        """Return the vocabulary size of the tokenizer."""
        if self.tokenizer is None:
            self.prepare_data()
        return self.tokenizer.vocab_size

    def get_dataset_info(self) -> Dict:
        """Get information about the datasets.

        Returns:
            Dictionary with dataset information
        """
        info = {
            "data_path": str(self.data_path),
            "tokenizer_vocab_path": str(self.tokenizer_vocab_path),
            "median_dict_path": str(self.median_dict_path),
            "seq_length": self.seq_length,
            "mask_prob": self.mask_prob,
            "neighbor_key": self.neighbor_key,
            "only_cells_with_neighbors": self.only_cells_with_neighbors,
            "no_neighbor_policy": self.no_neighbor_policy,
            "token_selection_policy": self.token_selection_policy,
            "normalize_gene_expression": self.normalize_gene_expression,
            "target_sum": self.target_sum,
        }

        if self.train_dataset is not None:
            info["train_size"] = len(self.train_dataset)
        if self.val_dataset is not None:
            info["val_size"] = len(self.val_dataset)
        if self.test_dataset is not None:
            info["test_size"] = len(self.test_dataset)

        return info
