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
from typing import Dict, Optional, Union, Literal

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
        tokenizer: Tokenizer,
        median_dict: dict[str, float],
        train_dataset_path: Optional[Union[str, Path]] = None,
        val_dataset_path: Optional[Union[str, Path]] = None,
        test_dataset_path: Optional[Union[str, Path]] = None,
        predict_dataset_path: Optional[Union[str, Path]] = None,
        seq_length: int = 2048,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
        neighbor_key: str = "next_cell_ids",
        micro_batch_size: int = 4,
        global_batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        seed: int = 42,
        only_cells_with_neighbors: bool = True,
        no_neighbor_policy: Literal["skip", "identity", "random"] = "skip",
        token_selection_policy: Literal["identity", "intersection", "union"] = "identity",
        normalize_gene_expression: bool = True,
        target_sum: int = 10000,
        **kwargs,
    ):
        """Initialize the temporal data module.

        Args:
            tokenizer: Tokenizer for gene names
            median_dict: Dictionary containing median values for normalization
            train_dataset_path: Path to the training SCDL dataset
            val_dataset_path: Path to the validation SCDL dataset
            test_dataset_path: Path to the test SCDL dataset
            predict_dataset_path: Path to the prediction SCDL dataset
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
        # Validate path arguments (same logic as parent class)
        if predict_dataset_path is None:
            assert train_dataset_path is not None and val_dataset_path is not None and test_dataset_path is not None, (
                "Provide either predict_dataset_path or (train_dataset_path, val_dataset_path, and test_dataset_path)"
            )
        elif train_dataset_path is None:
            assert val_dataset_path is None and test_dataset_path is None, (
                "Provide either predict_dataset_path or (train_dataset_path, val_dataset_path, and test_dataset_path)"
            )
            assert predict_dataset_path is not None, (
                "Provide either predict_dataset_path or (train_dataset_path, val_dataset_path, and test_dataset_path)"
            )

        # Initialize parent class with separate paths
        super().__init__(
            tokenizer=tokenizer,
            median_dict=median_dict,
            train_dataset_path=train_dataset_path,
            val_dataset_path=val_dataset_path,
            test_dataset_path=test_dataset_path,
            predict_dataset_path=predict_dataset_path,
            seq_length=seq_length,
            mask_prob=mask_prob,
            mask_token_prob=mask_token_prob,
            random_token_prob=random_token_prob,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            seed=seed,
            **kwargs,
        )

        # Store separate dataset paths
        self.data_path_train = Path(train_dataset_path) if train_dataset_path is not None else None
        self.data_path_val = Path(val_dataset_path) if val_dataset_path is not None else None
        self.data_path_test = Path(test_dataset_path) if test_dataset_path is not None else None
        self.data_path_predict = Path(predict_dataset_path) if predict_dataset_path is not None else None

        # Store temporal-specific parameters
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
        self.predict_dataset = None

        if self.data_path_predict is not None:
            logger.info(f"Initialized TemporalGeneformerDataModule for prediction with data_path: {self.data_path_predict}")
        else:
            logger.info(f"Initialized TemporalGeneformerDataModule with train: {self.data_path_train}, val: {self.data_path_val}, test: {self.data_path_test}")
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
            if self.data_path_train is not None:
                self.train_dataset = TemporalGeneformerDataset(
                    data_path=self.data_path_train,
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
                logger.info(f"Created training dataset with {len(self.train_dataset)} samples")

            if self.data_path_val is not None:
                # For validation, we can use the same dataset with a different seed
                # or create a separate validation split
                self.val_dataset = TemporalGeneformerDataset(
                    data_path=self.data_path_val,
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
                logger.info(f"Created validation dataset with {len(self.val_dataset)} samples")

        if stage == "test" or stage is None:
            if self.data_path_test is not None:
                self.test_dataset = TemporalGeneformerDataset(
                    data_path=self.data_path_test,
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

        if stage == "predict" or stage is None:
            if self.data_path_predict is not None:
                self.predict_dataset = TemporalGeneformerDataset(
                    data_path=self.data_path_predict,
                    tokenizer=self.tokenizer,
                    median_dict=self.median_dict,
                    max_len=self.seq_length,
                    mask_prob=self.mask_prob,
                    mask_token_prob=self.mask_token_prob,
                    random_token_prob=self.random_token_prob,
                    neighbor_key=self.neighbor_key,
                    seed=self.seed + 3,  # Different seed for prediction #NOTE: check if this is correct
                    only_cells_with_neighbors=self.only_cells_with_neighbors,
                    no_neighbor_policy=self.no_neighbor_policy,
                    token_selection_policy=self.token_selection_policy,
                    normalize_gene_expression=self.normalize_gene_expression,
                    target_sum=self.target_sum,
                )
                logger.info(f"Created prediction dataset with {len(self.predict_dataset)} samples")

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
            "train_dataset_path": str(self.data_path_train) if self.data_path_train is not None else None,
            "val_dataset_path": str(self.data_path_val) if self.data_path_val is not None else None,
            "test_dataset_path": str(self.data_path_test) if self.data_path_test is not None else None,
            "predict_dataset_path": str(self.data_path_predict) if self.data_path_predict is not None else None,
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
        if hasattr(self, 'predict_dataset') and self.predict_dataset is not None:
            info["predict_size"] = len(self.predict_dataset)

        return info
