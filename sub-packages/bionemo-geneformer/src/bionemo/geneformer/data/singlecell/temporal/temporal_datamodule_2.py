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
import functools

import torch
import numpy as np
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from bionemo.core.utils import random_utils
from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.geneformer.data.singlecell.temporal.temporal_dataset_2 import TemporalDatasetNew
from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.llm.data import collate


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
        no_neighbor_policy: str = "identity",
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
        # Initialize parent class first
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

        # Store temporal-specific parameters
        self.neighbor_key = neighbor_key
        self.only_cells_with_neighbors = only_cells_with_neighbors
        self.no_neighbor_policy = no_neighbor_policy
        self.token_selection_policy = token_selection_policy
        self.normalize_gene_expression = normalize_gene_expression
        self.target_sum = target_sum

        # Create temporal datasets following parent pattern
        rng = np.random.default_rng(seed)
        
        if self.data_path_train is not None:
            assert self.data_path_val is not None and self.data_path_test is not None
            self._train_dataset_ori = TemporalDatasetNew(
                data_path=self.data_path_train,
                tokenizer=tokenizer,
                median_dict=median_dict,
                max_len=seq_length,
                mask_prob=mask_prob,
                mask_token_prob=mask_token_prob,
                random_token_prob=random_token_prob,
                neighbor_key=neighbor_key,
                seed=random_utils.get_seed_from_rng(rng),
                fallback_to_identity=not only_cells_with_neighbors,
                no_neighbor_policy=no_neighbor_policy,
                only_cells_with_neighbors=only_cells_with_neighbors,
            )
            self._val_dataset_ori = TemporalDatasetNew(
                data_path=self.data_path_val,
                tokenizer=tokenizer,
                median_dict=median_dict,
                max_len=seq_length,
                mask_prob=mask_prob,
                mask_token_prob=mask_token_prob,
                random_token_prob=random_token_prob,
                neighbor_key=neighbor_key,
                seed=random_utils.get_seed_from_rng(rng),
                fallback_to_identity=not only_cells_with_neighbors,
                no_neighbor_policy=no_neighbor_policy,
                only_cells_with_neighbors=only_cells_with_neighbors,
            )
            self._test_dataset_ori = TemporalDatasetNew(
                data_path=self.data_path_test,
                tokenizer=tokenizer,
                median_dict=median_dict,
                max_len=seq_length,
                mask_prob=mask_prob,
                mask_token_prob=mask_token_prob,
                random_token_prob=random_token_prob,
                neighbor_key=neighbor_key,
                seed=random_utils.get_seed_from_rng(rng),
                fallback_to_identity=not only_cells_with_neighbors,
                no_neighbor_policy=no_neighbor_policy,
                only_cells_with_neighbors=only_cells_with_neighbors,
            )
            self._predict_dataset_ori = None
        else:
            assert self.data_path_predict is not None
            self._predict_dataset_ori = TemporalDatasetNew(
                data_path=self.data_path_predict,
                tokenizer=tokenizer,
                median_dict=median_dict,
                max_len=seq_length,
                mask_prob=mask_prob,
                mask_token_prob=mask_token_prob,
                random_token_prob=random_token_prob,
                neighbor_key=neighbor_key,
                seed=random_utils.get_seed_from_rng(rng),
                fallback_to_identity=not only_cells_with_neighbors,
                no_neighbor_policy=no_neighbor_policy,
                only_cells_with_neighbors=only_cells_with_neighbors,
            )
            self._train_dataset_ori = None
            self._val_dataset_ori = None
            self._test_dataset_ori = None

        if self.data_path_predict is not None:
            logger.info(f"Initialized TemporalGeneformerDataModule for prediction with data_path: {self.data_path_predict}")
        else:
            logger.info(f"Initialized TemporalGeneformerDataModule with train: {self.data_path_train}, val: {self.data_path_val}, test: {self.data_path_test}")
        logger.info(
            f"Enhanced features: no_neighbor_policy={no_neighbor_policy}, token_selection_policy={token_selection_policy}"
        )

        # Explicitly indicate we're using the new temporal dataset implementation
        logger.info("[TemporalDataModule] Using TemporalDatasetNew from temporal_dataset_2.py")

        # Report number of valid indices in each dataset split
        if hasattr(self, "_train_dataset_ori") and self._train_dataset_ori is not None:
            train_len = len(self._train_dataset_ori)
            logger.info(f"[TemporalDataModule] train valid indices: {train_len}")
            print(f"[TemporalDataModule] train valid indices: {train_len}")
        if hasattr(self, "_val_dataset_ori") and self._val_dataset_ori is not None:
            val_len = len(self._val_dataset_ori)
            logger.info(f"[TemporalDataModule] val valid indices: {val_len}")
            print(f"[TemporalDataModule] val valid indices: {val_len}")
        if hasattr(self, "_test_dataset_ori") and self._test_dataset_ori is not None:
            test_len = len(self._test_dataset_ori)
            logger.info(f"[TemporalDataModule] test valid indices: {test_len}")
            print(f"[TemporalDataModule] test valid indices: {test_len}")
        if hasattr(self, "_predict_dataset_ori") and self._predict_dataset_ori is not None:
            predict_len = len(self._predict_dataset_ori)
            logger.info(f"[TemporalDataModule] predict valid indices: {predict_len}")
            print(f"[TemporalDataModule] predict valid indices: {predict_len}")

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation, and testing.

        This now follows the parent class pattern by calling super().setup()
        which wraps the temporal datasets in MultiEpochDatasetResampler.

        Args:
            stage: Stage of training (fit, validate, test, predict)
        """
        logger.info(f"Setting up temporal datasets for stage: {stage}")
        # Let parent class handle MultiEpochDatasetResampler wrapping
        super().setup(stage or "")

    def _create_dataloader(self, dataset, mode, **kwargs):
        """Create dataloader for train, validation, and test stages.
        
        Uses the standard bert_padding_collate_fn which handles temporal data properly.

        Args:
            dataset: The dataset to create the dataloader for.
            mode: Stage of training.
            **kwargs: Additional arguments to pass to the dataloader.
        """
        self.update_init_global_step()
        from nemo.lightning.data import WrappedDataLoader
        
        return WrappedDataLoader(
            mode=mode,
            dataset=dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=functools.partial(
                collate.bert_padding_collate_fn,
                padding_value=self.tokenizer.token_to_id(GeneTokenizer.pad_token),
                min_length=self.max_len,
                max_length=self.max_len,
            ),
            **kwargs,
        )

    @property
    def vocab_size(self):
        """Return the vocabulary size of the tokenizer."""
        return self.tokenizer.get_vocab_size()

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
            "seq_length": self.max_len,
            "mask_prob": self.mask_prob,
            "neighbor_key": self.neighbor_key,
            "only_cells_with_neighbors": self.only_cells_with_neighbors,
            "no_neighbor_policy": self.no_neighbor_policy,
            "token_selection_policy": self.token_selection_policy,
            "normalize_gene_expression": self.normalize_gene_expression,
            "target_sum": self.target_sum,
        }

        if hasattr(self, '_train_ds') and self._train_ds is not None:
            info["train_size"] = len(self._train_ds)
        if hasattr(self, '_validation_ds') and self._validation_ds is not None:
            info["val_size"] = len(self._validation_ds)
        if hasattr(self, '_test_ds') and self._test_ds is not None:
            info["test_size"] = len(self._test_ds)
        if hasattr(self, '_predict_ds') and self._predict_ds is not None:
            info["predict_size"] = len(self._predict_ds)

        return info
