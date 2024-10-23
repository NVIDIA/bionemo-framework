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
import pathlib
import pickle
from typing import Any, Callable

import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader

from bionemo.llm.model.biobert.testing_utils import get_optimizers_state


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

    def __init__(self, metadata_path: pathlib.Path):  # noqa: D107
        self.metadata_path = metadata_path

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int):
        """PTL callback that raises a StopAndGoException if metadata exists."""
        pickle_file_path = os.path.join(self.metadata_path, "checkpoints/metadata.pkl")
        if os.path.exists(pickle_file_path):
            # Register the signal handler
            raise StopAndGoException("Terminating early, checkpoint exists.")
            # kill job afterwards


class MetadataSaveCallback(Callback):
    """A callback that saves metadata about the current training at the second validation epoch."""

    def __init__(
        self, metadata_path: pathlib.Path, metrics_getter: dict[str, Callable[[pl.Trainer, pl.LightningModule], Any]]
    ):
        """Initialises callback with path and called information.

        Args:
            metadata_path (pathlib.Path): Path where the metadata will be saved.
            metrics_getter (dict[str, Callable[[pl.Trainer, pl.LightningModule], Any]]): A dictionary of metadata keys and their corresponding functions.

        See Also: bionemo.testing.stop_and_go
        """
        self.metadata_path = metadata_path
        self.pickle_file_path = os.path.join(self.metadata_path, "checkpoints/metadata.pkl")
        self.called = False  # indicates if callback was already called
        self.metrics_getter = metrics_getter

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        """Set up the testing callbacks and removes lingering metadata."""
        super().setup(trainer, pl_module, stage)
        if trainer.is_global_zero and os.path.exists(self.pickle_file_path):
            os.remove(self.pickle_file_path)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Stores requisite metadata at the end of the first non-warmup validation epoch.

        Executes on the second validation epoch -only- due to how warmups are handled. May not work as intended in the
        absence of a warmup.

        Args:
            trainer (Trainer): The Lightning Trainer object.
            pl_module (LightningModule): The LightningModule being trained.

        Notes:
            - If `called` is True and `trainer.is_global_zero` is True, the function saves metadata to compare after resuming with a checkpoint.
            - The metadata is obtained using the `metrics_getter` dict and results are saved as a pickle file.

        """
        if self.called and trainer.is_global_zero:
            # save metadata to compare to after resuming with checkpoint
            metadata = {
                "optimizer_states": get_optimizers_state(
                    trainer, pl_module
                ),  # handle separately due to different test logic
            }
            for metadata_key, func in self.metrics_getter.items():
                metadata_value = func(trainer, pl_module)
                metadata[metadata_key] = metadata_value

            # prepare paths for metadata save
            pickle_file_path = self.pickle_file_path
            os.makedirs(os.path.dirname(pickle_file_path), exist_ok=True)
            with open(pickle_file_path, "wb") as metadata_file:
                pickle.dump(metadata, metadata_file)

            # check that pickle file was saved correctly
            assert os.path.isfile(pickle_file_path), f"No file found at {pickle_file_path}"
        else:
            # first time this callback is called is before the ModelCheckpoint callback
            # since that one is always executed last. Therefore, we skip the first validation
            # round and only save metadata at the second validation round
            self.called = True


class TestCheckpointIntegrityCallback(Callback):
    """Callback that tests if current metrics match those saved in the associated metadata file.

    This callback expects to be invoked _only_ after resuming a model that used the MetadataSaveCallback. When training begins, it checks the value of each metric and compares to the metadata stored in the metadata pickle file. Any deviances are assumed to be a failure in restoration.
    """

    def __init__(
        self, metadata_path: pathlib.Path, metrics_getter: dict[str, Callable[[pl.Trainer, pl.LightningModule], Any]]
    ):
        """Initialises callback with path and called information.

        Args:
            metadata_path (pathlib.Path): Path where the metadata will be saved.
            metrics_getter (dict[str, Callable[[pl.Trainer, pl.LightningModule], Any]]): A dictionary of metadata keys and their corresponding functions. Must be a subset of the dictionary passed to MetadataSaveCallback.
        """
        self.metadata_path = metadata_path
        self.metrics_getter = metrics_getter

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        """Loads associated metadata and compares with current metrics."""
        pickle_file_path = os.path.join(self.metadata_path, "checkpoints/metadata.pkl")
        # check that pickle file exists
        assert os.path.isfile(pickle_file_path), f"No file found at {pickle_file_path}"
        with open(pickle_file_path, "rb") as metadata_file:
            metadata_dict = pickle.load(metadata_file)

        current_metadata = {}
        for metadata_key, func in self.metrics_getter.items():
            current_value = func(trainer, pl_module)
            current_metadata[metadata_key] = current_value

        # TODO (SKH): Ideally this would collect _all_ failures instead of failing on the first one.
        for metadata_key in current_metadata:
            expected_value = metadata_dict[metadata_key]
            current_value = current_metadata[metadata_key]
            assert (
                expected_value == current_value
            ), f"Value mismatch for key {metadata_key}: stored_value={expected_value}, current_value={current_value}"

        current_optimizer_states = get_optimizers_state(trainer, pl_module)
        assert str(current_optimizer_states) == str(
            metadata_dict["optimizer_states"]
        ), "Optimizer state mismatch."  # TODO @sichu: implement nested tensor_dict_hash comparison
