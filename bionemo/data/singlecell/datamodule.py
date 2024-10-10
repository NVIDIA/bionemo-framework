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


import os
from functools import lru_cache
from pathlib import Path
from typing import List, Literal

import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from scanpy import AnnData
from torch.utils.data import DataLoader

from bionemo.core import BioNeMoDataModule
from bionemo.data.mapped_dataset import FilteredIdxMappedDataset, IndexMappedDataset, ResamplingMappedDataset
from bionemo.data.singlecell.adamson import AdamsonDataset, _parse_pert
from bionemo.data.singlecell.dataset import SingleCellDataset
from tokenizers import Tokenizer


class SingleCellDataModule(BioNeMoDataModule):
    """LightningDataModule wrapper of `SingleCellDataset`

     Args:
        cfg (DictConfig): Configuration file.
        trainer (Trainer): Pytorch Lightning Trainer.
        tokenizer (Tokenizer): Maps gene names to ids and vice-versa.
        median_dict (dict[str, float]): Dictionary containing median values.
        mask_prob (float, optional): Probability of masking a token. Defaults to 0.15.
        mask_token_prob (float, optional): Probability of the token [MASK] out of the tokens selected for masking. Defaults to 0.8.
        random_token_prob (float, optional): Probability of a random token out of the tokens selected for masking. Defaults to 0.1.
        max_len (int, optional): he maximum length of the input sequence. Defaults to 1024.

    Attributes:
        cfg (Config): Configuration object
        data_path (Union[str, PosixPath]): Path to preprocessed single-cell data files
        median_dict (dict): Dictionary containing median values
        tokenizer (Tokenizer): Tokenizer object
        setup_called (bool): Flag indicating if the setup method has been called
        dataset (SingleCellDataset): Single-cell dataset object

    """

    # Nothing says we cant pass in the dataset...
    def __init__(
        self,
        cfg: DictConfig,
        trainer: Trainer,
        tokenizer: Tokenizer,
        median_dict: dict[str, float],
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__(cfg, trainer)
        self.cfg = cfg.data
        self.data_path_train = self.cfg.train_dataset_path
        self.data_path_val = self.cfg.val_dataset_path
        self.data_path_test = self.cfg.test_dataset_path
        self.ncp = cfg.next_cell_prediction
        self.filter_no_neighbors = cfg.filter_no_neighbors
        self.no_neighbor_policy = cfg.no_neighbor_policy
        self.tokenizer = tokenizer
        self.median_dict = median_dict
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob
        self.index_mapping_dir = cfg.data.get("index_mapping_dir", str(Path(self.data_path_train).parent))
        self._train_dataset = SingleCellDataset(
            self.data_path_train,
            self.tokenizer,
            self.median_dict,
            self.max_len,
            mask_prob=self.mask_prob,
            mask_token_prob=self.mask_token_prob,
            random_token_prob=self.random_token_prob,
            next_cell_prediction=self.ncp,
            no_neighbor_policy=self.no_neighbor_policy,
        )
        self._val_dataset = SingleCellDataset(
            self.data_path_val,
            self.tokenizer,
            self.median_dict,
            self.max_len,
            mask_prob=self.mask_prob,
            mask_token_prob=self.mask_token_prob,
            random_token_prob=self.random_token_prob,
            next_cell_prediction=self.ncp,
            no_neighbor_policy=self.no_neighbor_policy,
        )
        self._test_dataset = SingleCellDataset(
            self.data_path_test,
            self.tokenizer,
            self.median_dict,
            self.max_len,
            mask_prob=self.mask_prob,
            mask_token_prob=self.mask_token_prob,
            random_token_prob=self.random_token_prob,
            next_cell_prediction=self.ncp,
            no_neighbor_policy=self.no_neighbor_policy,
        )
        self.init_num_samples()

    def sample_train_dataset(self, dataset):
        """Sample the training dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from

        Returns:
            ResamplingMappedDataset: Resampled dataset

        """
        # This is where re-sampling occurs.
        os.makedirs(self.index_mapping_dir, exist_ok=True)
        if self.filter_no_neighbors:

            def criterion(idx):
                return dataset.sample_has_neighbor(idx)

            dataset = FilteredIdxMappedDataset(dataset, criterion_fn=criterion)
        return ResamplingMappedDataset(
            dataset,
            num_samples=self.train_num_samples,
            cfg=self.cfg,
            name=f"train_{self.train_num_samples}",
            index_mapping_dir=self.index_mapping_dir,
        )

    def sample_val_dataset(self, dataset):
        os.makedirs(self.index_mapping_dir, exist_ok=True)
        if self.filter_no_neighbors:

            def criterion(idx):
                return dataset.sample_has_neighbor(idx)

            dataset = FilteredIdxMappedDataset(dataset, criterion_fn=criterion)
        return ResamplingMappedDataset(
            dataset,
            num_samples=self.val_num_samples,
            cfg=self.cfg,
            name=f"val_{self.val_num_samples}",
            index_mapping_dir=self.index_mapping_dir,
        )

    def sample_test_dataset(self, dataset):
        os.makedirs(self.index_mapping_dir, exist_ok=True)
        if self.filter_no_neighbors:

            def criterion(idx):
                return dataset.sample_has_neighbor(idx)

            dataset = FilteredIdxMappedDataset(dataset, criterion_fn=criterion)
        return ResamplingMappedDataset(
            dataset,
            num_samples=self.test_num_samples,
            cfg=self.cfg,
            name=f"test_{self.test_num_samples}",
            index_mapping_dir=self.index_mapping_dir,
        )

    def train_dataset(self):
        """Get the training dataset.

        Returns:
            torch.utils.data.Dataset: Training dataset

        """
        return self._train_dataset

    def val_dataset(self):
        """Get the validation dataset.

        Returns:
            torch.utils.data.Dataset: Validation dataset

        """
        return self._val_dataset

    def test_dataset(self):
        """Get the test dataset.

        Returns:
            torch.utils.data.Dataset: Test dataset

        """
        return self._test_dataset


class AdamsonDataModule(BioNeMoDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        trainer: Trainer,
        tokenizer: Tokenizer,
        median_dict: dict[str, float],
        train_ratio: float = 0.98,
        val_ratio: float = 0.01,
        max_len: int = 1024,
    ):
        super().__init__(cfg, trainer)
        self.data_path = self.cfg.dataset_path
        self.preprocessed_anndata_fn = cfg.data.preprocessed_anndata_fn
        self.target_gep_fn = cfg.data.target_gep_fn
        self.tokenizer = tokenizer
        self.median_dict = median_dict
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.max_len = max_len
        self.seed = cfg.data.seed + 1  # Give it a slightly different number than before..
        self.split_type = cfg.data.split_type
        self.index_mapping_dir = cfg.data.get("index_mapping_dir", os.path.dirname(self.data_path))

    def onetime_init_num_samples(self):
        self.init_num_samples()

    @lru_cache
    def onetime_init_backed_dataset(self):
        # No shuffling this time. Also can this happen in the constructor?
        self.dataset = AdamsonDataset(
            self.preprocessed_anndata_fn, self.target_gep_fn, self.tokenizer, self.median_dict, self.max_len
        )
        self.train_idxs, self.val_idxs, self.test_idxs = create_adamson_splits(
            self.dataset.data, self.split_type, seed=self.seed
        )

    def sample_train_dataset(self, dataset):
        # Need to setup distributed state to compute num_samples, need num_samples to not deadlock.
        self.onetime_init_num_samples()
        os.makedirs(self.index_mapping_dir, exist_ok=True)
        return ResamplingMappedDataset(
            dataset,
            num_samples=self.train_num_samples,
            cfg=self.cfg,
            name=f"train_{self.train_num_samples}",
            index_mapping_dir=self.index_mapping_dir,
        )

    def train_dataset(self):
        self.onetime_init_backed_dataset()
        return IndexMappedDataset(self.dataset, self.train_idxs)

    def val_dataset(self):
        self.onetime_init_backed_dataset()
        return IndexMappedDataset(self.dataset, self.val_idxs, copy=True)

    def test_dataset(self):
        self.onetime_init_backed_dataset()
        return IndexMappedDataset(self.dataset, self.test_idxs, copy=True)

    def test_dataloader(self) -> DataLoader:
        # Expected for certain PTL functionality (e.g. Trainer.test())

        return DataLoader(self.test_dataset(), batch_size=self.cfg.micro_batch_size, pin_memory=True, shuffle=False)


def create_adamson_splits(
    data: AnnData,
    split_type: Literal["single", "single_only", "double", "double-0", "double-1"] = "single",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1337,
) -> tuple[np.array, np.array, np.array]:
    """
    Create train, validation, and test splits for the given AnnData object. Test ratio is inferred from train and validation ratios.

    Parameters:
        data (AnnData): The input AnnData object.
        split_type (Literal['single', 'single_only', 'double', 'double-0', 'double-1']): The type of split to perform. Defaults to 'single'.
            single: test set contains single perturbations not seen in train set, train contains both single and double not seen in test set.
            single_only: test set contains single pertubations not seen in train set, train contains single pertubations only.
            double:  test set contains double perturbations not seen in train set, train contains both double and single.
            double-0: filters pertubations in double pertubations that have both genes not in test genes. i.e. train set will only include double pertubations with atleast one gene not in test_genes
            double-1: filter pertubations in double pertubations that have both genes not in `test_genes`. i.e. train set will include double pertubations such that both perturbs are not in test set.
        train_ratio (float): The ratio of samples to include in the training set. Defaults to 0.8.
        val_ratio (float): The ratio of samples to include in the validation set. Defaults to 0.1.
        seed (int): random seed to use for splitting.

    Returns:
        tuple[np.array, np.array, np.array]: A tuple containing the indices of the training, validation, and test indices, respectively.
    """
    perts = [p for p in data.obs.condition.unique() if p != "ctrl"]
    rng = np.random.RandomState(seed)
    train_pert, val_pert, test_pert = _split_data(
        split_type, perts, rng=rng, train_ratio=train_ratio, val_ratio=val_ratio
    )
    idxs = np.arange(len(data))
    train_idxs = idxs[data.obs.condition.isin(train_pert)]
    validation_idxs = idxs[data.obs.condition.isin(val_pert)]
    test_idxs = idxs[data.obs.condition.isin(test_pert)]
    return train_idxs, validation_idxs, test_idxs


def _split_data(
    split_type: Literal["single", "single_only", "double", "double-0", "double-1"],
    perts: List[str],
    rng: np.random.RandomState,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """Split the data into train, validation, and test sets based on the specified split type.

    Args:
        split_type (str): The type of split to perform. Options are "single", "single_only", "double", "double-0", "double-1".
            single: test set contains single perturbations not seen in train set, train contains both single and double not seen in test set.
            single_only: test set contains single pertubations not seen in train set, train contains single pertubations only.
            double:  test set contains double perturbations not seen in train set, train contains both double and single.
            double-0: filters pertubations in double pertubations that have both genes not in test genes. i.e. train set will only include double pertubations with atleast one gene not in test_genes
            double-1: filter pertubations in double pertubations that have both genes not in `test_genes`. i.e. train set will include double pertubations such that both perturbs are not in test set.
        perts (list[str]): List of pertubation targets geneids.
        rng (np.random.RandomState): Initialized random state object to use when creating splits. Using a rng object preserves reproducability by keeping other processes from interfering with this.

    Returns:
        train_perts (list): List of perturbations in the train set.
        val_perts (list): List of perturbations in the validation set.
        test_perts (list): List of perturbations in the test set.

    Raises:
        ValueError: If the split_type is not recognized.
    """

    test_ratio = 1 - (train_ratio + val_ratio)

    def split(perts, test_ratio):
        double_perts = [p for p in perts if ("ctrl" not in p)]
        all_genes = _get_all_genes_in_perts(perts)
        test_genes = rng.choice(all_genes, size=int(len(all_genes) * test_ratio), replace=False)
        test_singles = _get_perts_with_genes(test_genes, perts, type_="single")
        test_doubles = _get_perts_with_genes(test_genes, perts, type_="double")
        if split_type == "single":
            test_perts = test_singles
            train_perts = [p for p in perts if p not in test_singles and p not in test_doubles]
        elif split_type == "single_only":
            test_perts = test_singles
            train_perts = [p for p in perts if p not in test_singles and p not in double_perts]
        elif split_type == "double":
            test_perts = np.random.choice(double_perts, size=int(len(double_perts) * test_ratio), replace=False)
            train_perts = [p for p in perts if p not in test_perts]
        elif split_type == "double-0":
            single_or_both_unseen_perts = [
                p for p in test_doubles if len([pp for pp in p.split("+") if pp not in test_genes]) > 0
            ]
            test_doubles = [p for p in test_doubles if p not in single_or_both_unseen_perts]
            test_perts = test_singles + test_doubles
            train_perts = [p for p in perts if p not in test_perts]
        elif split_type == "double-1":
            # - filter perturbations in double perturbations that have both genes not in `test_genes`
            # i.e. train set will include double perturbations that have both not in the test set
            both_unseen_perts = [
                p for p in test_doubles if len([pp for pp in p.split("+") if pp not in test_genes]) > 1
            ]
            test_doubles = [p for p in test_doubles if p not in both_unseen_perts]
            test_perts = test_singles + test_doubles
            train_perts = [p for p in perts if p not in test_perts]
        else:
            raise ValueError("split_type not recognized")

        return train_perts, test_perts

    train_perts, test_perts = split(perts, test_ratio)
    train_perts, val_perts = split(train_perts, val_ratio)
    return train_perts, val_perts, test_perts


def _get_all_genes_in_perts(perts):
    """gets all the unique genes in the perturbations list"""
    all_genes = []
    for pert in perts:
        all_genes.extend(_parse_pert(pert))
    all_genes = list(set(all_genes))
    return all_genes


def _get_perts_with_genes(genes: List[str], perts: List[str], type_: str = "both") -> List[str]:
    """Gets the perturbations with the given genes.

    Args:
        genes (List[str]): List of genes.
        perts (List[str]): List of perturbations.
        type_ (str, optional): Type of perturbations to consider.
            Possible values are "single", "double", or "both".
            Defaults to "both".

    Returns:
        List[str]: List of perturbations that contain the given genes.

    Raises:
        ValueError: If the type_ argument is not recognized.
    """
    single_perts = [p for p in perts if ("ctrl" in p) and (p != "ctrl")]
    double_perts = [p for p in perts if ("ctrl" not in p)]

    pert_list = []

    if type_ == "single":
        pert_list = single_perts
    elif type_ == "double":
        pert_list = double_perts
    elif type_ == "both":
        pert_list = perts
    else:
        raise ValueError("type_ not recognized")

    perts_w_gene = []

    for pert in pert_list:
        for gene in genes:
            # This assumes the Adamson data structure
            if gene in _parse_pert(pert):
                perts_w_gene.append(pert)

    return perts_w_gene
