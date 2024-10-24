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


import pathlib
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from overrides import override
from pytorch_lightning import Callback, LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader


def compute_biobert_loss_singlegpu(model, dl: DataLoader, limit_val_batches: int = 1):
    """Computes the loss for BioBert models on a single GPU.

    This will not function in multi-gpu settings nor with models that do not conform to BioBert.

    Args:
        model (torch.nn.Module): The Biobert model.
        dl (torch.utils.data.DataLoader): The data loader.
        limit_val_batches (int): The number of batches to use for validation.

    Returns:
        float: The mean loss.

    See Also:
    - :class: BioBertModel
    """
    n, loss = 0, 0.0
    model.eval()
    for i, batch in enumerate(dl):
        batch = model.data_step(iter(dl))
        result = model(
            input_ids=batch["text"].cuda(),  # 'tokens' also a valid input for MockGPTDataModule
            attention_mask=batch["attention_mask"].cuda(),
        )
        loss_mask = batch["loss_mask"].cuda()
        # Not guaranteed i guess?
        logits = result["token_logits"]
        target = batch["labels"].cuda()
        loss += F.cross_entropy(logits[loss_mask].float(), target[loss_mask], reduction="sum")
        n += loss_mask.sum()

        if i >= limit_val_batches:
            break

    mean_loss: float = (loss / n).detach().cpu().numpy().item()
    model.train()

    return mean_loss


class StopAndGoException(Exception):  # noqa: D101
    pass


class RaiseAfterMetadataCallback(Callback):
    """A callback that raises a StopAndGoException kills it if the metadata from the MetadataSaveCallback was saved successfully beforehand.

    Use this callback for pytest based Stop and go tests.
    """

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):  # noqa: D102
        if trainer.sanity_checking:
            return
        raise StopAndGoException()


class AbstractStopAndGoCallback(ABC, Callback):
    """Abstract base class for stop-and-go callback to compare metadata before pausing and after resuming training."""

    def __init__(self, pickle_directory: str | pathlib.Path, mode: str = "stop"):
        """Initialize StopAndGoCallback.

        Args:
            pickle_directory (str | pathlib.Path): Directory at which the pickle file to save metadata to.
            mode (str, optional): Mode to run in. Must be either "stop" or "go". Defaults to "stop".

        Notes:
            User must override get_metadata and compare_metadata, within which self.has_compared should be set to True after comparison to manually indicate that the method is overrode.
        """
        if mode not in ["stop", "go"]:
            raise ValueError(f"mode must be 'stop' or 'go', got {mode}")

        if not isinstance(pickle_directory, pathlib.Path):
            pickle_directory = pathlib.Path(pickle_directory)

        pickle_directory.mkdir(parents=True, exist_ok=True)
        self.pickle_directory = pickle_directory
        self.mode = mode

    def write_pickle(self, mode: str, data: Any):
        """Write metadata to pickle file."""
        pickle_file_path = self.pickle_directory / f"metadata_{mode}.pkl"
        with open(pickle_file_path, "wb") as f:
            pickle.dump(data, f)

    def load_pickle(self, mode: str) -> Any:
        """Load metadata from pickle file."""
        pickle_file_path = self.pickle_directory / f"metadata_{mode}.pkl"
        with open(pickle_file_path, "rb") as f:
            return pickle.load(f)

    def load_stop_and_go_pickles(self) -> Tuple[Any, Any]:
        """Load both stop and go metadata from pickle files."""
        return self.load_pickle(mode="stop"), self.load_pickle(mode="go")

    @abstractmethod
    def get_metadata(self, trainer: Trainer, pl_module: LightningModule) -> Any:
        """Get metadata from trainer and pl_module."""
        raise NotImplementedError

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):  # noqa: D102
        if self.mode == "go":
            self.write_pickle(mode="go", data=self.get_metadata(trainer, pl_module))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):  # noqa: D102
        if not trainer.sanity_checking and self.mode == "stop":
            metadata_stop = self.get_metadata(trainer, pl_module)
            self.write_pickle(mode="stop", data=metadata_stop)


class LearningRateStateStopAndGoCallback(AbstractStopAndGoCallback):
    """Stop-and-go callback for learning rate before pausing and after resuming training."""

    @override
    def get_metadata(self, trainer: Trainer, pl_module: LightningModule) -> float:
        """Get learning rate as metadata."""
        return trainer.optimizers[0].param_groups[0]["lr"]


class GlobalStepStateStopAndGoCallback(AbstractStopAndGoCallback):
    """Stop-and-go callback for global_step before pausing and after resuming training."""

    @override
    def get_metadata(self, trainer: Trainer, pl_module: LightningModule) -> int:
        """Get global_step as metadata."""
        return trainer.global_step


class OptimizerStateStopAndGoCallback(AbstractStopAndGoCallback):
    """Stop-and-go callback to check optimizer states before pausing and after resuming training."""

    @override
    def get_metadata(self, trainer: Trainer, pl_module: LightningModule) -> List[Dict[str, torch.Tensor]]:
        """Get optimizer states as metadata."""
        return [optimizer.mcore_optimizer.optimizer.state_dict()["state"] for optimizer in trainer.optimizers]


class ComsumedSamplesStopAndGoCallback(AbstractStopAndGoCallback):
    """Stop-and-go callback to check consumed samples before pausing and after resuming training."""

    @override
    def get_metadata(self, trainer: Trainer, pl_module: LightningModule) -> Any:
        """Get consumed samples as metadata."""
        # return trainer.datamodule.state_dict()["consumed_samples"]  # TODO why state_dict can be empty despite working lines below
        data_sampler = trainer.datamodule.data_sampler
        consumed_samples = data_sampler.compute_consumed_samples(
            trainer.global_step - trainer.datamodule.init_global_step
        )
        return consumed_samples


class ManualValLossStopAndGoCallback(AbstractStopAndGoCallback):
    """Stop-and-go callback to check validation loss manually before pausing and after resuming training."""

    def compute_biobert_loss_singlegpu_on_first_validation_batch(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> float:
        """Computes the loss for BioBert models on a single GPU on the first validation batch.

        This will not function in multi-gpu settings nor with models that do not conform to BioBert.

        Args:
            trainer (pl.Trainer): The Lightning Trainer object.
            pl_module (pl.LightningModule): The LightningModule being trained.

        Returns:
            float: The mean loss.

        See Also:
        - :class: BioBertModel
        """
        is_training = pl_module.training
        dl = trainer.datamodule.val_dataloader()

        with torch.no_grad():
            n, loss = -1, 0.0
            pl_module.eval()  # turn off dropout, etc.
            # batch = next(iter(dl))
            batch = pl_module.data_step(iter(dl))
            result = pl_module(
                input_ids=batch["text"].cuda(),  # 'tokens' also a valid input for MockGPTDataModule
                attention_mask=batch["attention_mask"].cuda(),
            )
            loss_mask = batch["loss_mask"].cuda()
            # Not guaranteed i guess?
            logits = result["token_logits"]
            target = batch["labels"].cuda()
            loss += F.cross_entropy(logits[loss_mask].float(), target[loss_mask], reduction="sum")
            n += loss_mask.sum()
            mean_loss: float = (loss / n).detach().cpu().numpy().item()

        if is_training:
            pl_module.train()

        return mean_loss

    @override
    def get_metadata(self, trainer: Trainer, pl_module: LightningModule) -> Any:
        """Get validation loss as metadata."""
        return self.compute_biobert_loss_singlegpu_on_first_validation_batch(trainer, pl_module)
