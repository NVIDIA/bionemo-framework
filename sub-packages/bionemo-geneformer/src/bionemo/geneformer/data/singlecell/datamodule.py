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

from bionemo.core.data.resamplers import PRNGResampleDataset
from bionemo.core.utils import random_utils
from bionemo.geneformer.data.singlecell.dataset import SingleCellDataset
from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.llm.data import collate
from bionemo.llm.data.datamodule import MegatronDataModule
from bionemo.llm.utils.datamodule_utils import infer_num_samples


Mode = Literal["train", "validation", "test"]

__all__: Sequence[str] = ("SingleCellDataModule",)


class SingleCellDataModule(MegatronDataModule):
    """LightningDataModule wrapper of `SingleCellDataset`

    Args:
        data_path (Union[str, PosixPath]): Path to preprocessed single-cell data files
        tokenizer (Tokenizer): Maps gene names to ids and vice-versa
        collator: Used to batch samples
        process_item: Function defining how each item should be processed
        num_workers (int): Number of workers to use
        num_mask_per_sample (int): Number of masked versions of a single sample to be returned by each worker
        train_batch_size (int): Batch size for training
        val_batch_size (int): Batch size for validation

    Attributes:
        cfg (Config): Configuration object
        data_path (Union[str, PosixPath]): Path to preprocessed single-cell data files
        median_dict (dict): Dictionary containing median values
        tokenizer (Tokenizer): Tokenizer object
        setup_called (bool): Flag indicating if the setup method has been called
        dataset (SingleCellDataset): Single-cell dataset object

    """  # noqa: D415

    # Nothing says we cant pass in the dataset...
    def __init__(  # noqa: D107
        self,
        tokenizer: Tokenizer,
        train_dataset_path: str,
        val_dataset_path: str,
        test_dataset_path: str,
        median_dict: dict[str, float],
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,  # 80% mask token
        random_token_prob: float = 0.1,  # 10% random token, remaining 1-(mask+random) will be identity.
        seq_length: int = 2048,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        index_mapping_dir: Optional[str] = None,
        rampup_batch_size: Optional[List[int]] = None,
        seed: int = 42,
        num_workers: int = 10,  # TODO can this be automatically set?
        persistent_workers: bool = True,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
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
        self.index_mapping_dir = index_mapping_dir or str(Path(self.data_path_train).parent)

        rng = np.random.default_rng(seed)
        self._train_dataset_ori = SingleCellDataset(
            self.data_path_train,
            self.tokenizer,
            self.median_dict,
            self.max_len,
            mask_prob=self.mask_prob,
            mask_token_prob=self.mask_token_prob,
            random_token_prob=self.random_token_prob,
            seed=random_utils.get_seed_from_rng(rng),
        )
        self._val_dataset_ori = SingleCellDataset(
            self.data_path_val,
            self.tokenizer,
            self.median_dict,
            self.max_len,
            mask_prob=self.mask_prob,
            mask_token_prob=self.mask_token_prob,
            random_token_prob=self.random_token_prob,
            seed=random_utils.get_seed_from_rng(rng),
        )
        self._test_dataset_ori = SingleCellDataset(
            self.data_path_test,
            self.tokenizer,
            self.median_dict,
            self.max_len,
            mask_prob=self.mask_prob,
            mask_token_prob=self.mask_token_prob,
            random_token_prob=self.random_token_prob,
            seed=random_utils.get_seed_from_rng(rng),
        )

        # This is needed here, or you need to specify it in the megatron adapter thing TODO name?
        #  Note that this sampler is sequential, meaning it does not do any shuffling. Let's wrap our data in a shuffler.
        self.data_sampler = MegatronDataSampler(
            seq_len=self.max_len,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:  # noqa: D102
        assert getattr(self, "trainer", None) is not None, "Please only call setup after trainer is attached."

        # Trainer API
        max_train_steps = self.trainer.max_steps
        if self.trainer.max_epochs > 1:
            logging.warning(
                "Trainer is set to run for multiple epochs. This is not recommended due to the same shuffle being used in each. Instead set max_epochs to 1 and increase the number of max_steps."
            )
        assert max_train_steps > 0, "Please specify trainer.max_steps"

        num_train_samples = int(max_train_steps * self.data_sampler.global_batch_size)
        num_val_samples = infer_num_samples(
            limit_batches=self.trainer.limit_val_batches,
            num_samples_in_dataset=len(self._val_dataset_ori),
            global_batch_size=self.data_sampler.global_batch_size,
            stage="val",
        )
        num_test_samples = infer_num_samples(
            limit_batches=self.trainer.limit_test_batches,
            num_samples_in_dataset=len(self._test_dataset_ori),
            global_batch_size=self.data_sampler.global_batch_size,
            stage="test",
        )

        # This happens exactly once during setup.
        self._train_ds = self._sample_and_shuffle_dataset(self._train_dataset_ori, num_train_samples, "train")
        self._validation_ds = self._sample_and_shuffle_dataset(self._val_dataset_ori, num_val_samples, "val")
        self._test_ds = self._sample_and_shuffle_dataset(self._test_dataset_ori, num_test_samples, "test")

    def train_dataloader(self) -> TRAIN_DATALOADERS:  # noqa: D102
        return self._create_dataloader(self._train_ds, mode="train")

    def val_dataloader(self) -> EVAL_DATALOADERS:  # noqa: D102
        return self._create_dataloader(self._validation_ds, mode="validation")

    def test_dataloader(self) -> EVAL_DATALOADERS:  # noqa: D102
        return self._create_dataloader(self._test_ds, mode="test")

    def _create_dataloader(self, dataset, mode: Mode, **kwargs) -> WrappedDataLoader:
        """Create dataloader for train, validation, and test stages.

        Args:
            dataset: The dataset to create the dataloader for.
            mode: Stage of training, which is used to determined if consumed_samples in MegatronPretrainingSampler should be initialized to 0 (validation/test), or be set to the previous value from state_dict in case of checkpoint resumption (train).
            **kwargs: Additional arguments to pass to the dataloader.
        """
        self.update_init_global_step()
        return WrappedDataLoader(
            mode=mode,
            dataset=dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=functools.partial(
                collate.bert_padding_collate_fn,
                padding_value=self.tokenizer.token_to_id(GeneTokenizer.pad_token),
                min_length=None,
                max_length=self.max_len,
            ),
            **kwargs,
        )

    def _sample_and_shuffle_dataset(self, dataset: SingleCellDataset, num_samples: int, stage: str):  # noqa: D417
        """Sample the training dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from

        Returns:
            ResamplingMappedDataset: Resampled dataset

        """
        # This is where re-sampling occurs.
        return PRNGResampleDataset(
            dataset,
            num_samples=num_samples,
            seed=self.seed + len(stage),
        )
