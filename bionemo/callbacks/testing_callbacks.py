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
import signal
from typing import Any, List

from nemo.utils import logging
from pytorch_lightning import Callback, LightningModule, Trainer


def get_global_step(trainer: Trainer, pl_module: LightningModule) -> int:
    return trainer.global_step


def get_learning_rate(trainer: Trainer, pl_module: LightningModule) -> float:
    return trainer.optimizers[0].param_groups[0]["lr"]


getter_function_map = {
    "learning_rate": get_learning_rate,
    "global_step": get_global_step,
}


class KillAfterSignalCallback(Callback):
    """A callback that sends a SIGTERM signal to the process and kills it if the metadata
    from the MetadataSaveCallback was saved successfully beforehand.
    """

    def __init__(self, metadata_path: pathlib.Path):
        self.metadata_path = metadata_path

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int):
        def terminate_process(signum, frame):
            logging.info("\nReceived SIGTERM signal, terminating process...\n")
            exit(0)

        pickle_file_path = os.path.join(self.metadata_path, "checkpoints/metadata.pkl")

        if not trainer.is_global_zero:
            # Sleep the process if not global zero to clean up hanging procs
            import time

            time.sleep(5)

        if os.path.exists(pickle_file_path):
            # Register the signal handler
            signal.signal(signal.SIGTERM, terminate_process)
            # kill job afterwards
            os.kill(os.getpid(), signal.SIGTERM)


class MetadataSaveCallback(Callback):
    """A callback that saves metadata about the current training at the second validation epoch."""

    def __init__(self, metadata_path: pathlib.Path, metadata_keys: List[str]):
        """Initialises callback with path and called information.
        Args:
            metadata_path (pathlib.Path): Path where the metadata will be saved.
            metadata_keys (List[str]): keys for metadata to be checked.
        """
        self.metadata_path = metadata_path
        self.metadata_keys = metadata_keys
        self.called = False  # indicates if callback was already called

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.called and trainer.is_global_zero:
            # save metadata to compare to after resuming with checkpoint
            metadata = {}
            for metadata_key in self.metadata_keys:
                metadata_value = getter_function_map[metadata_key](trainer, pl_module)
                metadata[metadata_key] = metadata_value
            # prepare paths for metadata save
            pickle_file_path = os.path.join(self.metadata_path, "checkpoints/metadata.pkl")
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
    """A callback that, after resuming from a checkpoint, checks that attributes of this
    resumed checkpoint are the same as the metadata that was saved at the time of
    checkpoint creation as part of stop-and-go tests.
    """

    def __init__(self, metadata_path: pathlib.Path, metadata_keys: List[str]):
        """Initialises callback with path and called information.
        Args:
            metadata_path (pathlib.Path): Path where the metadata will be saved.
            metadata_keys (List[str]): keys for metadata to be checked.
        """
        self.metadata_path = metadata_path
        self.metadata_keys = metadata_keys

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.is_global_zero:
            pickle_file_path = os.path.join(self.metadata_path, "checkpoints/metadata.pkl")
            # check that pickle file exists
            assert os.path.isfile(pickle_file_path), f"No file found at {pickle_file_path}"
            with open(pickle_file_path, "rb") as metadata_file:
                metadata_dict = pickle.load(metadata_file)

            for metadata_key in self.metadata_keys:
                expected_value = metadata_dict[metadata_key]
                current_value = getter_function_map[metadata_key](trainer, pl_module)
                assert (
                    expected_value == current_value
                ), f"Value mismatch for key {metadata_key}: stored_value={expected_value}, current_value={current_value}"
