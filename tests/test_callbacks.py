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


import pytest
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset

from bionemo.callbacks.scheduling_callbacks import ParameterMultiplicativeScheduler


FUNNY_BUSINESS_VAL = 42.0


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self, train_dl, val_dl):
        super().__init__()
        self.cfg = DictConfig({"micro_batch_size": 2})
        self.layer = torch.nn.Linear(32, 2)
        self._train_dl = train_dl
        self._validation_dl = val_dl
        self._test_dl = None
        self.layer.funny_business = FUNNY_BUSINESS_VAL

    def train_dataloader(self):
        if self._train_dl is not None:
            return self._train_dl

    def val_dataloader(self):
        if self._validation_dl is not None:
            return self._validation_dl

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    def forward(self, x):
        return self.layer(x)

    def training_step(self, dataloader_iter, batch_idx):
        # dataloader_iter, done = self._val_iterator_done(dataloader_iter)
        # if done:
        #     return None
        batch = next(dataloader_iter)
        loss = self(batch).sum()
        self.log("train_loss", loss, batch_size=len(batch))
        return {"loss": loss}

    def validation_step(self, dataloader_iter, batch_idx):
        # dataloader_iter, done = self._val_iterator_done(dataloader_iter)
        # if done:
        #     return None
        batch = next(dataloader_iter)
        ibatch, obatch = batch
        loss = self(ibatch).sum()
        self.log("valid_loss", loss, batch_size=2)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


@pytest.mark.parametrize("target_fraction", [0.1, 0.5, 1.0])
def test_scheduling_callback(tmp_path, target_fraction: float):
    train_data = DataLoader(RandomDataset(32, length=64), batch_size=2)
    num_train_steps_per_epoch = 64 // 2
    assert len(train_data) == num_train_steps_per_epoch
    val_data = DataLoader(RandomDataset(32, length=10), batch_size=2)
    assert len(val_data) == 10 // 2
    num_epochs = 2
    expected_final_value = FUNNY_BUSINESS_VAL * target_fraction

    factor = (expected_final_value / (num_train_steps_per_epoch * num_epochs - 1)) / FUNNY_BUSINESS_VAL

    model = BoringModel(train_data, val_data)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=num_epochs,
        num_sanity_val_steps=2,
        callbacks=[
            ParameterMultiplicativeScheduler(
                factor=factor, module_parameter_path="layer.funny_business", enable_progress_bar=True
            )
        ],
        accelerator="cpu",
        enable_progress_bar=True,
    )
    trainer.fit(model)
    assert trainer.model.layer.funny_business == pytest.approx(expected_final_value)
