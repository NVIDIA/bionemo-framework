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

from unittest import mock

import pandas as pd
import pytest
import torch.utils.data

from bionemo.esm2.data.datamodule import ESMDataModule


def _create_dummy_parquet_inputs(tmp_path):
    train_cluster_path = tmp_path / "train_clusters.parquet"
    train_clusters = pd.DataFrame(
        {
            "ur90_id": [["UniRef90_A"], ["UniRef90_B", "UniRef90_C"]],
        }
    )
    train_clusters.to_parquet(train_cluster_path)

    valid_cluster_path = tmp_path / "valid_clusters.parquet"
    valid_clusters = pd.DataFrame(
        {
            "ur50_id": ["UniRef50_A", "UniRef50_B"],
        }
    )
    valid_clusters.to_parquet(valid_cluster_path)


def test_create_esm_datamodule_raises_without_trainer(tmp_path, dummy_protein_dataset):
    _create_dummy_parquet_inputs(tmp_path)

    # Initialize the data module.
    data_module = ESMDataModule(
        train_cluster_path=tmp_path / "train_clusters.parquet",
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=tmp_path / "valid_clusters.parquet",
        valid_database_path=dummy_protein_dataset,
    )
    assert data_module is not None

    with pytest.raises(RuntimeError, match="Setup should be completed when trainer and config are attached."):
        data_module.setup()


def test_create_esm_datamodule_raises_without_trainer_max_steps(tmp_path, dummy_protein_dataset):
    _create_dummy_parquet_inputs(tmp_path)

    # Initialize the data module.
    data_module = ESMDataModule(
        train_cluster_path=tmp_path / "train_clusters.parquet",
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=tmp_path / "valid_clusters.parquet",
        valid_database_path=dummy_protein_dataset,
    )
    assert data_module is not None

    data_module.trainer = mock.Mock()
    data_module.trainer.max_epochs = 1
    data_module.trainer.max_steps = 0

    with pytest.raises(RuntimeError, match="Please specify trainer.max_steps"):
        data_module.setup()


def test_create_esm_datamodule_creates_valid_dataloaders(tmp_path, dummy_protein_dataset):
    _create_dummy_parquet_inputs(tmp_path)

    # Initialize the data module.
    data_module = ESMDataModule(
        train_cluster_path=tmp_path / "train_clusters.parquet",
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=tmp_path / "valid_clusters.parquet",
        valid_database_path=dummy_protein_dataset,
        global_batch_size=8,
        micro_batch_size=4,
        min_seq_length=36,
        max_seq_length=36,
    )
    assert data_module is not None

    data_module.trainer = mock.Mock()
    data_module.trainer.max_epochs = 1
    data_module.trainer.max_steps = 10
    data_module.trainer.val_check_interval = 2
    data_module.trainer.limit_val_batches = 1

    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)

    val_dataloader = data_module.val_dataloader()
    assert isinstance(val_dataloader, torch.utils.data.DataLoader)

    assert len(train_dataloader) == 10 * 8  # max steps * global batch size
    assert len(val_dataloader) == (10 // 2 + 1) * 8  # number of eval steps * global batch size

    for batch in train_dataloader:
        assert isinstance(batch, dict)
        assert isinstance(batch["text"], torch.Tensor)
        assert isinstance(batch["attention_mask"], torch.Tensor)
        assert isinstance(batch["labels"], torch.Tensor)
        assert isinstance(batch["loss_mask"], torch.Tensor)
        assert isinstance(batch["is_random"], torch.Tensor)

    for batch in val_dataloader:
        assert isinstance(batch, dict)
        assert isinstance(batch["text"], torch.Tensor)
        assert isinstance(batch["attention_mask"], torch.Tensor)
        assert isinstance(batch["labels"], torch.Tensor)
        assert isinstance(batch["loss_mask"], torch.Tensor)
        assert isinstance(batch["is_random"], torch.Tensor)
