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
import warnings
from collections import defaultdict
from typing import Callable, Dict, List, TypeVar, Union

import torch
from lightning import pytorch as pl


Batch = TypeVar("Batch")


class MeasureMemoryCallback(pl.Callback):
    """
    A PyTorch Lightning callback to measure and log memory usage and data size info during training and validation.

    The goal of this callback is to measure the memory usage during the train/validation step and extract all necessary
    size information in each batch related to the memory usage.

    In the train/validation stage, the batch size info is collected in `on_train/validation_batch_start()` with `extract_sizes_on_batch_start`,
    and in `on_train/validation_batch_end()` through callable `extract_sizes_on_batch_end`.
    As an example, the callable can be like `extract_sizes_on_batch_start = lambda batch: {"num_samples": len(batch)}`.
    These two callables will take a batch, and return a dictionary of samples, two callables are used as model may modify the batch data.
    The keywords: "stage", "device", "memory_allocated[MiB]", "max_memory_allocated[MiB]" are reserved to log the step stage, device info, allocated memory in MiB,
    and maximal allocated memory in MiB, and shouldn't be used in these two callables' output dictionaries.

    The memory are measured in `on_before_zero_grad()` during training step, and in `on_validation_batch_end() during validation step, and reset
    in the `on_train/validation_batch_start()` for train/validation accordingly.

    The logged info is written and appended to `output_filepath` in `on_train/validation_batch_end()` in every train/validation step.
    """

    def __init__(
        self,
        output_filepath: Union[str, bytes, os.PathLike],
        extract_sizes_on_batch_start: Callable[[Batch], Dict] = None,
        extract_sizes_on_batch_end: Callable[[Batch], Dict] = None,
        measure_allocated: bool = True,
        measure_max_allocated: bool = True,
        warmup: int = 1,
    ):
        """
        Initializes the callback.

        Args:
            output_filepath (Union[str, bytes, os.PathLike]): The file path to write the memory usage logs in CSV format.
            extract_sizes_on_batch_start (Callable, optional): A callable that takes a batch as input on train/validation
                batch start and returns a dictionary of sizes to log. Defaults to None.
            extract_sizes_on_batch_end (Callable, optional): A callable that takes a batch as input on train/validation
                batch end and returns a dictionary of sizes to log. Defaults to None.
            measure_allocated (bool, optional): Whether to measure and log the allocated memory. Defaults to True.
            measure_max_allocated (bool, optional): Whether to measure and log the maximum allocated memory. Defaults to True.
            warmup (int, optional): The number of warmup steps before starting to measure memory usage. Defaults to 1.

        """

        super().__init__()
        self.output_filepath = output_filepath
        self.measure_allocated = measure_allocated
        self.measure_max_allocated = measure_max_allocated
        self.extract_sizes_on_batch_start = (
            extract_sizes_on_batch_start if callable(extract_sizes_on_batch_start) else lambda _: {}
        )
        self.extract_sizes_on_batch_end = (
            extract_sizes_on_batch_end if callable(extract_sizes_on_batch_end) else lambda _: {}
        )
        self.warmup = warmup

        self.output_header: List[str] = None
        if os.path.exists(self.output_filepath):
            with open(self.output_filepath, "r") as f:
                self.output_header = f.readline().strip().split(",")
            warnings.warn(
                f"Appending memory usage info to the existing file {self.output_filepath} "
                f"with header: {self.output_header}"
            )

        self.logged_info: Dict = defaultdict(lambda: None)
        self.enabled: bool = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """
        Called at the start of each training batch.

        Enables memory measurement after the warmup period and extract sizes for the current batch.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The PyTorch Lightning module.
            batch: The current batch.
            batch_idx: The index of the current batch.
        """
        # trainer.global_step is updated between `on_train_batch_start` and `on_train_batch_end`
        # use self.enabled flag across train/val step
        self.enabled = trainer.global_step > self.warmup
        if self.enabled:
            self._log_on_batch_start(batch, pl_module.device, "train")

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        """
        Called before zeroing the gradients.

        Measures memory usage after the warmup period.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The PyTorch Lightning module.
            optimizer: The optimizer.
        """
        if self.enabled:
            self._measure_memory(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Called at the end of each training batch.

        Extract sizes with the current batch and writes output after the warmup period.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The PyTorch Lightning module.
            outputs: The outputs of the current batch.
            batch: The current batch.
            batch_idx: The index of the current batch.
        """
        if self.enabled:
            self._log_on_batch_end(batch, "train")
            self._write_output()

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        """
        Called at the start of each validation batch.

        Enables memory measurement after the warmup period and and extract sizes for the current batch.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The PyTorch Lightning module.
            batch: The current batch.
            batch_idx: The index of the current batch.
        """
        self.enabled = trainer.global_step > self.warmup
        if self.enabled:
            self._log_on_batch_start(batch, pl_module.device, "val")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Called at the end of each validation batch.

        Extract sizes with the current batch, measures memory, and writes output if enabled.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The PyTorch Lightning module.
            outputs: The outputs of the current batch.
            batch: The current batch.
            batch_idx: The index of the current batch.
        """
        if self.enabled:
            self._measure_memory(pl_module.device)
            self._log_on_batch_end(batch, "val")
            self._write_output()

    def _reset_peak_memory_stats(self, device: torch.device):
        if self.measure_max_allocated:
            torch.cuda.reset_peak_memory_stats(device)

    def _measure_memory(self, device: torch.device):
        if self.measure_allocated:
            self.logged_info["memory_allocated[MiB]"] = torch.cuda.memory_allocated(device) / 2**20
        if self.measure_max_allocated:
            self.logged_info["max_memory_allocated[MiB]"] = torch.cuda.max_memory_allocated(device) / 2**20

    def _write_output(self):
        keys = list(self.logged_info.keys())
        if self.output_header is None:
            self.output_header = keys
        else:
            if set(keys).issubset(set(self.output_header)):
                raise ValueError("")
        if not os.path.exists(self.output_filepath):
            with open(self.output_filepath, "w") as f:
                f.write(",".join(self.output_header) + "\n")
        with open(self.output_filepath, "a") as f:
            f.write(",".join([str(self.logged_info[key]) for key in self.output_header]) + "\n")

    def _check_duplicate_keys(self, log_dict: Dict):
        # check if exists duplicate keys
        duplicated_keys = list(set(self.logged_info.keys()).intersection(set(log_dict.keys())))
        if len(duplicated_keys) > 0:
            return True, duplicated_keys
        else:
            return False, duplicated_keys

    def _log_on_batch_start(self, batch: Batch, device: torch.device, stage: str):
        self._reset_peak_memory_stats(device)

        self.logged_info["stage"] = stage
        self.logged_info["device"] = str(device)
        start_size_dict = self.extract_sizes_on_batch_start(batch)
        exist_duplicated_keys, duplicate_keys = self._check_duplicate_keys(start_size_dict)
        if exist_duplicated_keys:
            raise KeyError(
                f"Duplicate key(s) used in the output from `extract_sizes_on_batch_start` in {stage} step: {duplicate_keys}. "
            )

        self.logged_info.update(start_size_dict)

    def _log_on_batch_end(self, batch: Batch, stage: str):
        end_size_dict = self.extract_sizes_on_batch_end(batch)
        exist_duplicated_keys, duplicated_keys = self._check_duplicate_keys(end_size_dict)
        if exist_duplicated_keys:
            raise KeyError(
                f"Duplicated key(s) used in the output from `extract_sizes_on_batch_end` in {stage} step: {duplicated_keys}. "
            )
        self.logged_info.update(end_size_dict)
