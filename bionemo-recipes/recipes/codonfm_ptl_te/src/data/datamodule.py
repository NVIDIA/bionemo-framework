# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import logging
from typing import Any, Callable, Dict, Optional

import lightning as L  # noqa: N812
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .stateful_dataset import StatefulDataset


logger = logging.getLogger(__name__)


class CodonFMDataModule(L.LightningDataModule):
    """Unified Lightning DataModule that combines functionality from all existing datamodules.

    This datamodule can handle:
    - Pretraining with CodonMemmapDataset
    - Finetuning with any dataset
    - Evaluation/prediction tasks

    Args:
        dataset (Callable): A callable that returns a dataset.
        seed (int): Random seed.
        world_size (int): Number of distributed processes.
        train_iters (int): Number of training iterations.
        collate_fn (Callable): Collate function for DataLoader.
        num_workers (int): Number of workers for DataLoader.
        num_mask_per_sample (int): Number of masked versions per sample.
        train_batch_size (int): Training batch size.
        val_batch_size (int): Validation batch size.
        shuffle (bool): Whether to shuffle training data.
        pin_memory (bool): Whether to pin memory.
        persistent_workers (bool): Whether to use persistent workers.
        process_item (Callable): Function to process each item.
        is_evaluation (bool): Whether this is for evaluation/prediction only.
    """

    def __init__(  # noqa: D107
        self,
        dataset: Callable,
        seed: int = 123,
        world_size: int = 1,
        train_iters: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
        num_workers: int = 8,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        gradient_accumulation_steps: int = 1,
        shuffle: bool = True,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        process_item: Callable = lambda *x: x,
        is_evaluation: bool = False,  # if True, whole dataset will be used for evaluation.
    ):
        super().__init__()

        self.seed = seed
        self.init_consumed_samples = 0
        self.init_global_step = 0
        self.num_workers = num_workers
        self.dataset = dataset
        self.is_evaluation = is_evaluation
        if self.is_evaluation:
            shuffle = False

        self.shuffle = shuffle
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.setup_called = False

        self.world_size = world_size
        self.train_iters = train_iters

        self.micro_batch_size = self.train_batch_size
        self.global_batch_size = self.train_batch_size * self.world_size * self.gradient_accumulation_steps

        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.save_hyperparameters(logger=False)

    def setup(self, stage: str):  # noqa: D102
        if not self.setup_called:
            self.dataset = self.dataset(seed=self.seed)
            self.setup_called = True

    def get_stateful_dataset(self, dataset, total_samples, consumed_samples=0, global_batch_size=None, shuffle=False):
        """Wrap dataset with StatefulDataset for upsampling/resampling."""
        if not self.is_evaluation and total_samples > len(dataset):
            logger.info(f"Resampling dataset with {len(dataset)} samples to {total_samples}")
            dataset = StatefulDataset(
                dataset=dataset,
                total_samples=total_samples,
                global_batch_size=global_batch_size,
                shuffle=shuffle,
                seed=self.seed,
                consumed_samples=consumed_samples,
            )
        return dataset

    def train_dataloader(self) -> DataLoader:  # noqa: D102
        if self.is_evaluation:
            # For evaluation mode, return test dataloader
            return self.test_dataloader()

        train_ds = self.dataset.get_train(self.hparams.process_item)

        if self.train_iters:
            train_samples = self.train_iters * self.global_batch_size
        else:
            train_samples = len(train_ds)

        consumed_samples = self.calc_consumed_samples()
        train_ds = self.get_stateful_dataset(
            train_ds,
            total_samples=train_samples,
            consumed_samples=consumed_samples,
            global_batch_size=self.global_batch_size,
            shuffle=self.shuffle,
        )

        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(train_ds, shuffle=False, drop_last=True)

        # Ensure that the dataloader is not shuffled if StatefulDataset is used.
        dataloader_shuffle = self.shuffle and sampler is None and not isinstance(train_ds, StatefulDataset)

        dl = DataLoader(
            train_ds,
            shuffle=dataloader_shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            batch_size=self.train_batch_size,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            pin_memory=self.pin_memory,
        )
        return dl

    def val_dataloader(self) -> DataLoader:  # noqa: D102
        if self.is_evaluation:
            return self.test_dataloader()
        val_ds = self.dataset.get_validation(self.hparams.process_item)
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(val_ds)

        dl = DataLoader(
            val_ds,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            batch_size=self.val_batch_size,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )
        return dl

    def test_dataloader(self) -> DataLoader:  # noqa: D102
        test_ds = self.dataset.get_test(self.hparams.process_item)
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(test_ds, shuffle=False)

        dl = DataLoader(
            test_ds,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            batch_size=self.val_batch_size,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        return dl

    def predict_dataloader(self) -> DataLoader:
        """Return test dataloader for prediction."""
        return self.test_dataloader()

    def calc_consumed_samples(self) -> int:
        """Calculate consumed samples for resuming training/evaluation."""
        consumed_samples = 0
        if hasattr(self, "trainer") and self.trainer is not None and self.train_iters is not None:
            # Training mode - use trainer global step
            total_samples = self.train_iters * self.global_batch_size
            consumed_samples = min(
                (self.trainer.global_step - self.init_global_step) * self.global_batch_size, total_samples
            )

        consumed_samples = self.init_consumed_samples + consumed_samples
        return consumed_samples

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint.

        This method is implemented to generate and save the datamodule state.

        Returns:
            Dict[Any, Any]: A dictionary containing the datamodule state that you want to save.
        """
        state_dict = {}
        state_dict["consumed_samples"] = self.calc_consumed_samples()
        if hasattr(self, "trainer") and self.trainer is not None:
            state_dict["global_step"] = self.trainer.global_step

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the state of the datamodule from a checkpoint.

        This method is called when loading a checkpoint and is used to reload
        the datamodule state given the `state_dict`.

        Args:
            state_dict (Dict[str, Any]): The state dictionary containing the
            datamodule state returned by `self.state_dict()`.
        """
        if "consumed_samples" in state_dict:
            self.init_consumed_samples = state_dict["consumed_samples"]

        if "global_step" in state_dict:
            self.init_global_step = state_dict["global_step"]
