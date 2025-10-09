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


import functools
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import numpy as np
from nemo.lightning.data import WrappedDataLoader
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from tokenizers import Tokenizer

from bionemo.core.data.multi_epoch_dataset import MultiEpochDatasetResampler
from bionemo.core.utils import random_utils
from bionemo.geneformer.data.singlecell.dataset_scmap import SingleCellDatasetSCMAP
from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule
from bionemo.geneformer.data.mapped_dataset import FilteredIdxMappedDataset
from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.llm.data import collate
from bionemo.llm.utils.datamodule_utils import infer_num_samples


Mode = Literal["train", "validation", "test", "predict"]

__all__: Sequence[str] = ("SingleCellDataModuleSCMAP",)


class SingleCellDataModuleSCMAP(SingleCellDataModule):
    """LightningDataModule wrapper for SingleCellDatasetSCMAP with temporal training support.
    
    This data module integrates the SCMAP dataset format which uses memory-mapped files
    for efficient data loading and supports next cell prediction (temporal training).
    
    Args:
        data_path (Union[str, Path]): Path to sc_memmap directory containing the data files
        tokenizer (Tokenizer): Maps gene names to ids and vice-versa
        median_dict (dict[str, float]): Dictionary containing median values for normalization
        train_dataset_path (str | Path | None): Path to training SCMAP dataset
        val_dataset_path (str | Path | None): Path to validation SCMAP dataset
        test_dataset_path (str | Path | None): Path to test SCMAP dataset
        predict_dataset_path (str | Path | None): Path to prediction SCMAP dataset
        mask_prob (float): Probability of masking a token
        mask_token_prob (float): Probability of using [MASK] token
        random_token_prob (float): Probability of random token replacement
        seq_length (int): Maximum sequence length
        micro_batch_size (int): Micro batch size
        global_batch_size (int): Global batch size
        rampup_batch_size (Optional[List[int]]): Batch size ramp-up schedule
        seed (int): Random seed for reproducibility
        num_workers (int): Number of data loader workers
        persistent_workers (bool): Whether to use persistent workers
        pin_memory (bool): Whether to pin memory for data loading
        include_unrecognized_vocab_in_dataset (bool): Whether to raise error for unknown genes
        next_cell_prediction (bool): Enable next cell prediction (temporal training)
        no_neighbor_policy (str): Policy for handling cells without neighbors ("identity", "skip", etc.)
        filter_no_neighbors (bool): Whether to filter out cells without neighbors
        assert_increasing_columns (bool): Check if column indices are increasing (for debugging)
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        median_dict: dict[str, float],
        train_dataset_path: str | Path | None = None,
        val_dataset_path: str | Path | None = None,
        test_dataset_path: str | Path | None = None,
        predict_dataset_path: str | Path | None = None,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
        seq_length: int = 2048,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        seed: int = 42,
        num_workers: int = 10,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        include_unrecognized_vocab_in_dataset: bool = False,
        next_cell_prediction: bool = False,
        no_neighbor_policy: str = "identity",
        filter_no_neighbors: bool = False,
        assert_increasing_columns: bool = False,
        prepend_cls_token: bool = True,
        eos_token: int | None = None,
    ) -> None:
        # Store SCMAP-specific parameters
        self.next_cell_prediction = next_cell_prediction
        self.no_neighbor_policy = no_neighbor_policy
        self.filter_no_neighbors = filter_no_neighbors
        self.assert_increasing_columns = assert_increasing_columns
        self.prepend_cls_token = prepend_cls_token
        self.eos_token = eos_token
        
        # Initialize base MegatronDataModule (skip SingleCellDataModule's __init__)
        from bionemo.llm.data.datamodule import MegatronDataModule
        MegatronDataModule.__init__(self)
        
        # Set up parameters from SingleCellDataModule
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
        
        self.data_path_predict = predict_dataset_path
        self.data_path_train = train_dataset_path
        self.data_path_val = val_dataset_path
        self.data_path_test = test_dataset_path
        self.tokenizer = tokenizer
        self.median_dict = median_dict
        self.max_len = seq_length
        self.mask_prob = mask_prob
        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob
        self.seed = seed
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.include_unrecognized_vocab_in_dataset = include_unrecognized_vocab_in_dataset
        
        # Initialize data sampler
        from nemo.lightning.pytorch.plugins import MegatronDataSampler
        if self.data_path_predict is not None:
            # For predict mode, we need to handle smaller batch sizes
            self.data_sampler = MegatronDataSampler(
                seq_len=self.max_len,
                micro_batch_size=micro_batch_size,
                global_batch_size=global_batch_size,
                rampup_batch_size=rampup_batch_size,
                output_log=False,  # this is needed for predict step to work
            )
        else:
            self.data_sampler = MegatronDataSampler(
                seq_len=self.max_len,
                micro_batch_size=micro_batch_size,
                global_batch_size=global_batch_size,
                rampup_batch_size=rampup_batch_size,
            )
        
        # Create datasets with SCMAP format
        rng = np.random.default_rng(seed)
        
        if self.data_path_train is not None:
            assert self.data_path_val is not None and self.data_path_test is not None
            self._train_dataset_ori = self._create_scmap_dataset(
                self.data_path_train,
                random_utils.get_seed_from_rng(rng)
            )
            self._val_dataset_ori = self._create_scmap_dataset(
                self.data_path_val,
                random_utils.get_seed_from_rng(rng)
            )
            self._test_dataset_ori = self._create_scmap_dataset(
                self.data_path_test,
                random_utils.get_seed_from_rng(rng)
            )
            self._predict_dataset_ori = None
        else:
            assert self.data_path_predict is not None
            self._predict_dataset_ori = self._create_scmap_dataset(
                self.data_path_predict,
                random_utils.get_seed_from_rng(rng)
            )
            self._train_dataset_ori = None
            self._val_dataset_ori = None
            self._test_dataset_ori = None
        
        # Log dataset information
        if self.next_cell_prediction:
            logging.info(f"SingleCellDataModuleSCMAP initialized with next_cell_prediction={self.next_cell_prediction}, "
                        f"no_neighbor_policy={self.no_neighbor_policy}, filter_no_neighbors={self.filter_no_neighbors}")
    
    def _create_scmap_dataset(self, data_path: Path, seed: int) -> SingleCellDatasetSCMAP:
        """Create a SingleCellDatasetSCMAP instance with the configured parameters.
        
        Args:
            data_path: Path to the SCMAP data directory
            seed: Random seed for this dataset
            
        Returns:
            SingleCellDatasetSCMAP instance
        """
        return SingleCellDatasetSCMAP(
            data_path=data_path,
            tokenizer=self.tokenizer,
            median_dict=self.median_dict,
            max_len=self.max_len,
            mask_prob=self.mask_prob,
            mask_token_prob=self.mask_token_prob,
            random_token_prob=self.random_token_prob,
            prepend_cls_token=self.prepend_cls_token,
            eos_token=self.eos_token,
            include_unrecognized_vocab_in_dataset=self.include_unrecognized_vocab_in_dataset,
            seed=seed,
            next_cell_prediction=self.next_cell_prediction,
            no_neighbor_policy=self.no_neighbor_policy,
            assert_increasing_columns=self.assert_increasing_columns,
        )
    
    def setup(self, stage: str = "") -> None:
        """Set up datasets with optional filtering for cells without neighbors.
        
        This method wraps the datasets in MultiEpochDatasetResampler and optionally
        filters out cells without neighbors when filter_no_neighbors is True.
        """
        assert getattr(self, "trainer", None) is not None, "Please only call setup after trainer is attached."
        
        if self._train_dataset_ori is not None:
            assert self._val_dataset_ori is not None and self._test_dataset_ori is not None
            
            # Optionally filter datasets to only include cells with neighbors
            if self.filter_no_neighbors and self.next_cell_prediction:
                logging.info("Filtering datasets to only include cells with neighbors")
                
                # Create filtered versions of the datasets
                train_dataset = self._filter_dataset_for_neighbors(self._train_dataset_ori)
                val_dataset = self._filter_dataset_for_neighbors(self._val_dataset_ori)
                test_dataset = self._filter_dataset_for_neighbors(self._test_dataset_ori)
            else:
                train_dataset = self._train_dataset_ori
                val_dataset = self._val_dataset_ori
                test_dataset = self._test_dataset_ori
            
            # Log filtered dataset sizes
            if self.filter_no_neighbors and self.next_cell_prediction:
                logging.info(f"After filtering - Train: {len(train_dataset)} samples "
                           f"(from {len(self._train_dataset_ori)})")
                logging.info(f"After filtering - Val: {len(val_dataset)} samples "
                           f"(from {len(self._val_dataset_ori)})")
                logging.info(f"After filtering - Test: {len(test_dataset)} samples "
                           f"(from {len(self._test_dataset_ori)})")
            
            # Trainer API
            max_train_steps = self.trainer.max_steps
            if self.trainer.max_epochs > 1:
                logging.warning(
                    "Trainer is set to run for multiple epochs. This is not recommended due to the same shuffle being used in each. "
                    "Instead set max_epochs to 1 and increase the number of max_steps."
                )
            assert max_train_steps > 0, "Please specify trainer.max_steps"
            
            num_train_samples = int(max_train_steps * self.data_sampler.global_batch_size)
            
            # Wrap datasets in MultiEpochDatasetResampler
            self._train_ds = MultiEpochDatasetResampler(
                train_dataset,
                num_samples=num_train_samples,
                shuffle=True,
                seed=self.seed,
            )
            
            if self.trainer.limit_val_batches == 0:  # disable validation
                logging.info("Skip creating validation dataset because trainer.limit_val_batches=0.")
            else:
                num_val_samples = infer_num_samples(
                    limit_batches=self.trainer.limit_val_batches,
                    num_samples_in_dataset=len(val_dataset),
                    global_batch_size=self.data_sampler.global_batch_size,
                    stage="val",
                )
                self._validation_ds = MultiEpochDatasetResampler(
                    val_dataset,
                    num_samples=num_val_samples,
                    shuffle=False,
                    seed=self.seed,
                )
            
            if self.trainer.limit_test_batches == 0:  # disable testing
                logging.info("Skip creating test dataset because trainer.limit_test_batches=0.")
            else:
                num_test_samples = infer_num_samples(
                    limit_batches=self.trainer.limit_test_batches,
                    num_samples_in_dataset=len(test_dataset),
                    global_batch_size=self.data_sampler.global_batch_size,
                    stage="test",
                )
                self._test_ds = MultiEpochDatasetResampler(
                    test_dataset,
                    num_samples=num_test_samples,
                    shuffle=False,
                    seed=self.seed,
                )
        else:
            assert self._predict_dataset_ori is not None
            
            # Optionally filter predict dataset
            if self.filter_no_neighbors and self.next_cell_prediction:
                predict_dataset = self._filter_dataset_for_neighbors(self._predict_dataset_ori)
                logging.info(f"After filtering - Predict: {len(predict_dataset)} samples "
                           f"(from {len(self._predict_dataset_ori)})")
            else:
                predict_dataset = self._predict_dataset_ori
            
            self._predict_ds = MultiEpochDatasetResampler(
                predict_dataset,
                shuffle=False,
                seed=self.seed,
            )
    
    def _filter_dataset_for_neighbors(self, dataset: SingleCellDatasetSCMAP):
        """Filter dataset to only include cells with neighbors.
        
        Args:
            dataset: The SCMAP dataset to filter
            
        Returns:
            Filtered dataset (using FilteredIdxMappedDataset) or original dataset
        """
        if not hasattr(dataset, 'sample_has_neighbor'):
            logging.warning("Dataset does not have sample_has_neighbor method, skipping filtering")
            return dataset
        
        # Create criterion function for filtering
        def has_neighbor_criterion(idx: int) -> bool:
            return dataset.sample_has_neighbor(idx)
        
        # Use FilteredIdxMappedDataset to filter the dataset
        filtered_dataset = FilteredIdxMappedDataset(
            dataset=dataset,
            criterion_fn=has_neighbor_criterion,
            num_samples=None  # Keep all samples that match the criterion
        )
        
        return filtered_dataset
    
    def get_dataset_statistics(self) -> dict:
        """Get statistics about the datasets, including neighbor information.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {}
        
        if self._train_dataset_ori is not None and self.next_cell_prediction:
            train_with_neighbors = sum(
                1 for i in range(len(self._train_dataset_ori))
                if self._train_dataset_ori.sample_has_neighbor(i)
            )
            stats['train_cells_with_neighbors'] = train_with_neighbors
            stats['train_total_cells'] = len(self._train_dataset_ori)
            stats['train_neighbor_ratio'] = train_with_neighbors / len(self._train_dataset_ori)
        
        if self._val_dataset_ori is not None and self.next_cell_prediction:
            val_with_neighbors = sum(
                1 for i in range(len(self._val_dataset_ori))
                if self._val_dataset_ori.sample_has_neighbor(i)
            )
            stats['val_cells_with_neighbors'] = val_with_neighbors
            stats['val_total_cells'] = len(self._val_dataset_ori)
            stats['val_neighbor_ratio'] = val_with_neighbors / len(self._val_dataset_ori)
        
        return stats 