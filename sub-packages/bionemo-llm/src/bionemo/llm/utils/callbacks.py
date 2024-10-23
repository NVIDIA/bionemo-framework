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
from typing import Any, Optional, Sequence

import pytorch_lightning as pl
import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class PredictionWriter(BasePredictionWriter, pl.Callback):
    """A callback that writes predictions to disk at specified intervals during training."""

    def __init__(self, output_dir, write_interval):
        """Initializes the callback.

        Args:
            output_dir (str): The directory where predictions will be written.
            write_interval (str): The interval at which predictions will be written. (batch, epoch)

        """
        super().__init__(write_interval)
        self.output_dir = str(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Sequence[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Writes predictions to disk at the end of each batch.

        Args:
            trainer (pl.Trainer): The Trainer instance.
            pl_module (pl.LightningModule): The LightningModule instance.
            prediction (Any): The prediction made by the model.
            batch_indices (Sequence[int]): The indices of the batch.
            batch (Any): The batch data.
            batch_idx (int): The index of the batch.
            dataloader_idx (int): The index of the dataloader.
        """
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        result_path = os.path.join(self.output_dir, f"predictions__rank_{trainer.global_rank}__batch_{batch_idx}.pt")

        if len(batch_indices) > 0 and len(batch_indices[0]) > 0:
            batch_indices = batch_indices
        else:
            batch_indices: Optional[Sequence[int]] = trainer.predict_dataloaders.batch_sampler.seen_batch_indices

        torch.save(
            {"prediction": prediction, "batch_indices": batch_indices},
            result_path,
        )

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Any,
        batch_indices: Sequence[int],
    ) -> None:
        """Writes predictions to disk at the end of each epoch.

        Args:
            trainer (pl.Trainer): The Trainer instance.
            pl_module (pl.LightningModule): The LightningModule instance.
            predictions (Any): The predictions made by the model.
            batch_indices (Sequence[int]): The indices of the batch.
        """
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        result_path = os.path.join(self.output_dir, f"predictions__rank_{trainer.global_rank}.pt")

        if len(batch_indices) > 0 and len(batch_indices[0]) > 0:
            batch_indices = batch_indices
        else:
            batch_indices = trainer.predict_dataloaders.batch_sampler.seen_batch_indices

        torch.save(
            {"prediction": predictions, "batch_indices": batch_indices},
            result_path,
        )
