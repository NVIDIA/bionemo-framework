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

import pytorch_lightning as pl
from nemo.lightning.pytorch.plugins import MegatronDataSampler

from bionemo.contrib.data.esm2.dataset import create_train_dataset, create_valid_dataset
from bionemo.contrib.data.types import BertMaskConfig


def create_dataloader(): ...


class ESMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_cluster_path: str | os.PathLike,
        train_database_path: str | os.PathLike,
        valid_cluster_path: str | os.PathLike,
        valid_database_path: str | os.PathLike,
        seed: int | None = 42,
        max_seq_length: int = 1024,
        mask_config: BertMaskConfig = BertMaskConfig(),
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        num_workers: int = 10,  # TODO(@jomitchell) can this be automatically set?
        persistent_workers: bool = True,
        pin_memory: bool = True,
        rampup_batch_size: list[int] | None = None,
    ) -> None:
        super().__init__()
        self._train_cluster_path = train_cluster_path
        self._train_database_path = train_database_path
        self._valid_cluster_path = valid_cluster_path
        self._valid_database_path = valid_database_path
        self._seed = seed
        self._max_seq_length = max_seq_length
        self._mask_config = mask_config
        self._micro_batch_size = micro_batch_size
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._pin_memory = pin_memory

        self.data_sampler = MegatronDataSampler(
            seq_len=max_seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            dataloader_type="cyclic",  # This should attach a `MegatronPretrainingRandomSampler`.
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        del stage  # Unused.

        create_train_dataset()
        create_valid_dataset()

        assert (
            hasattr(self, "trainer") and self.trainer is not None
        ), "Setup should be completed when trainer and config are attached."
