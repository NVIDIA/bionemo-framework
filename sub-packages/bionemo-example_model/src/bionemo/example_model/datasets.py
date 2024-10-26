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

from typing import Sequence, TypedDict

import pytorch_lightning as pl
import torch
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from bionemo.core.data.resamplers import PRNGResampleDataset


__all__: Sequence[str] = (
    "MNISTCustom",
    "MNISTDataModule",
    "MnistItem",
)


class MnistItem(TypedDict):
    """Training input for the MNIST dataset."""

    data: Tensor
    label: Tensor
    idx: int


class MNISTCustom(MNIST):
    """Custom dictionary that returns data for the Mnist dataset."""

    def __getitem__(self, index: int) -> MnistItem:
        """Wraps the getitem method of the MNIST dataset such that we return a Dict
        instead of a Tuple or tensor.

        Args:
            index: The index we want to grab, an int.

        Returns:
            A dict containing the data ("x"), label ("y"), and index ("idx").
        """  # noqa: D205
        x, y = super().__getitem__(index)

        return {
            "data": x,
            "label": y,
            "idx": index,
        }


class MNISTDataModule(pl.LightningDataModule):
    """Data module for the MNIST data set."""

    def __init__(self, data_dir: str = "./", batch_size: int = 32, output_log=False) -> None:
        """Instantiate the directoy."""
        super().__init__()
        self.micro_batch_size = batch_size
        self.global_batch_size = batch_size
        self.max_len = 1048
        self.data_dir = data_dir

        # Wraps the datasampler with the MegatronDataSampler.
        self.data_sampler = MegatronDataSampler(
            seq_len=self.max_len,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=None,
            output_log=output_log,
        )

    def setup(self, stage: str) -> None:
        """Sets up the datasets.

        Args:
            stage: can be one of train / test / predict.
        """
        self.mnist_test = PRNGResampleDataset(
            MNISTCustom(self.data_dir, download=True, transform=transforms.ToTensor(), train=False), seed=43
        )
        mnist_full = MNISTCustom(self.data_dir, download=True, transform=transforms.ToTensor(), train=True)
        mnist_train, mnist_val = torch.utils.data.random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        self.mnist_train = PRNGResampleDataset(mnist_train, seed=44)
        self.mnist_val = PRNGResampleDataset(mnist_val, seed=45)

    def train_dataloader(self) -> DataLoader:
        """Training Dataloader."""
        return DataLoader(self.mnist_train, batch_size=self.micro_batch_size, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(self.mnist_val, batch_size=self.micro_batch_size, num_workers=0)

    def predict_dataloader(self) -> DataLoader:
        """Test/ prediction dataloader."""
        return DataLoader(self.mnist_test, batch_size=self.micro_batch_size, num_workers=0)
