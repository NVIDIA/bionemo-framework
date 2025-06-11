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


"""OneLogger callback for NeMo training.

This module provides a callback that integrates OneLogger telemetry with NeMo training.
OneLogger is NVIDIA's internal telemetry system for tracking training metrics and performance.
"""

from typing import Any, Optional

import pytorch_lightning as pl
from pydantic import BaseModel
from pytorch_lightning.callbacks import Callback


class OneLoggerConfig(BaseModel):
    """Configuration class for OneLogger metadata fields.

    This class contains all the metadata fields that can be used with OneLogger,
    including both basic metadata common to all run types and training-specific metadata.

    Attributes:
        enable_for_current_rank: Boolean indicating whether to enable logging for current rank in distributed training.
            This value should be calculated dynamically for each rank.
        one_logger_async: Boolean indicating whether to use async logging. Set to True if you are already using WandB Logger.
        one_logger_project: String representing the project name for where to track your OneLogger metrics in WandB.
        one_logger_run_name: String representing a unique identifier for the current run. If not given, will use random string.
        log_every_n_train_iterations: Integer specifying how often to log training metrics.
        app_tag_run_version: String representing the version of the model/application.
        summary_data_schema_version: String representing the version of the OneLogger data schema. Use 1.0.0 if not specially told.
        app_run_type: String indicating the type of job (currently only 'training' is supported).
        app_tag: String used for performance tracking - jobs with same expected performance should have same tag.
        app_tag_run_name: String used for tracking overall training progress across multiple jobs.
        world_size: Integer representing the number of processes in distributed training.
        global_batch_size: Integer representing the total batch size across all devices.
        batch_size: Integer representing the batch size per iteration.
        micro_batch_size: Integer representing the batch size per device.
        seq_length: Integer representing the sequence length for the model.
        train_iterations_target: Integer representing the target number of training iterations.
        train_samples_target: Integer representing the target number of training samples.
        is_train_iterations_enabled: Boolean indicating whether to log training iterations.
        is_baseline_run: Boolean indicating whether this is a baseline run for comparison.
        is_test_iterations_enabled: Boolean indicating whether to log test iterations.
        is_validation_iterations_enabled: Boolean indicating whether to log validation iterations.
        is_save_checkpoint_enabled: Boolean indicating whether to save checkpoints.
        is_log_throughput_enabled: Boolean indicating whether to track and log TFLOPS.
        save_checkpoint_strategy: String indicating checkpoint saving strategy ('sync' or 'async').
    """

    # Basic Configuration
    enable_for_current_rank: bool
    one_logger_async: bool
    one_logger_project: str = "bionemo-framework"
    one_logger_run_name: Optional[str] = None

    # Logging Configuration
    log_every_n_train_iterations: Optional[int] = None
    app_tag_run_version: str
    summary_data_schema_version: str = "1.0.0"
    app_run_type: str = "training"

    # Run Identification
    app_tag: str
    app_tag_run_name: str

    # Training Configuration
    world_size: int
    global_batch_size: Optional[int] = None
    batch_size: Optional[int] = None
    micro_batch_size: Optional[int] = None
    seq_length: Optional[int] = None

    # Training Targets
    train_iterations_target: Optional[int] = None
    train_samples_target: Optional[int] = None

    # Feature Flags
    is_train_iterations_enabled: bool = False
    is_baseline_run: bool = False
    is_test_iterations_enabled: bool = False
    is_validation_iterations_enabled: bool = False
    is_save_checkpoint_enabled: bool = False
    is_log_throughput_enabled: bool = False

    # Checkpoint Configuration
    save_checkpoint_strategy: str = "async"


class OneLoggerCallback(Callback):
    """A callback that integrates with OneLogger for tracking metrics.

    This callback provides integration with OneLogger (NVIDIA's internal telemetry system)
    for tracking various training metrics and performance indicators. It automatically
    forwards most method calls to the underlying OneLogger instance while providing
    custom logic for PyTorch Lightning-specific events.

    Read more about OneLogger here: https://confluence.nvidia.com/x/6EIar

    TODO: consider replacing with the callback from NeMo when this PR is merged:
    https://github.com/NVIDIA/NeMo/pull/13437/
    """

    def __init__(self, config: OneLoggerConfig, rank: int):
        """Initialize the OneLogger callback.

        Args:
            config: OneLoggerConfig instance containing all metadata fields
            rank: The rank of the current process
        """
        super().__init__()

        # Create OneLogger instance with the converted metadata
        try:
            from one_logger_utils.core import OneLoggerUtils as OneLogger

            self.one_logger = OneLogger(config=config.model_dump())

        except Exception:
            print(
                "Error: the internal one_logger_utils package is required to enable one_logger "
                "tracking of e2e metrics. Please go to https://confluence.nvidia.com/x/6EIar for details to install it"
            )
        self.world_size = config.world_size
        self.rank = rank

    def __getattr__(self, name: str) -> Any:
        """Automatically forward any undefined method calls to the underlying OneLogger instance.

        This eliminates the need for manually writing pass-through methods for each OneLogger API.
        Only methods that need custom logic (like those interacting with the trainer) need to be
        explicitly defined in this class.

        Args:
            name: The name of the method being called

        Returns:
            The method from the underlying OneLogger instance

        Raises:
            AttributeError: If the method is not found in the OneLogger instance
        """
        # Check if the method exists on the OneLogger instance
        if hasattr(self.one_logger, name):
            return getattr(self.one_logger, name)

        # If not found, raise AttributeError as normal
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training begins.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
        """
        # Extract necessary information from the trainer
        current_step = trainer.global_step

        # Call OneLogger's on_train_start with the extracted information
        self.one_logger.on_train_start(train_iterations_start=current_step)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training ends.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
        """
        self.one_logger.on_train_end()

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int
    ) -> None:
        """Called when a training batch begins.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
            batch: The current batch of data
            batch_idx: The index of the current batch
        """
        self.one_logger.on_train_batch_start()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when a training batch ends.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
            outputs: The outputs from the training step
            batch: The current batch of data
            batch_idx: The index of the current batch
        """
        self.one_logger.on_train_batch_end()

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when validation begins.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
        """
        self.one_logger.on_validation_start()

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when validation ends.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
        """
        self.one_logger.on_validation_end()

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0) -> None:
        """Called when validation batch begins.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
            batch: The current batch of data
            batch_idx: The index of the current batch
            dataloader_idx: The index of the current dataloader (default: 0)
        """
        self.one_logger.on_validation_batch_start()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        """Called when validation batch ends.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
            outputs: The outputs from the validation step
            batch: The current batch of data
            batch_idx: The index of the current batch
            dataloader_idx: The index of the current dataloader (default: 0)
        """
        self.one_logger.on_validation_batch_end()

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when testing begins.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
        """
        self.one_logger.on_test_start()

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when testing ends.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
        """
        self.one_logger.on_test_end()


def configure_one_logger(
    trainer: pl.Trainer,
    experiment_name: str,
    global_batch_size: int,
    micro_batch_size: int,
    seq_length: int,
    create_tflops_callback: bool,
    save_checkpoint_strategy: str = "async",
    one_logger_project: str = "bionemo-framework",
    app_tag_run_version: str = "0.0.0",
    app_run_type: str = "training",
    app_tag: Optional[str] = None,
    app_tag_run_name: Optional[str] = None,
) -> OneLoggerCallback:
    """Configure OneLogger for the current run.

    This function sets up OneLogger with appropriate configuration for tracking training metrics
    and performance indicators. It automatically calculates distributed training parameters
    and sets up feature flags based on the provided configuration.

    Args:
        trainer: The PyTorch Lightning trainer instance
        experiment_name: Name of the experiment, used for run identification
        global_batch_size: Total batch size across all devices in distributed training
        micro_batch_size: Batch size per device in distributed training
        seq_length: Sequence length for the model
        create_tflops_callback: Whether to enable TFLOPS tracking and logging
        save_checkpoint_strategy: Checkpoint saving strategy ('sync' or 'async')
        one_logger_project: Project name in WandB where OneLogger metrics will be tracked
        app_tag_run_version: Version of the model/application
        app_run_type: Type of job (currently only 'training' is supported)
        app_tag: Tag for performance tracking - jobs with same expected performance should have same tag
        app_tag_run_name: Name for tracking overall training progress across multiple jobs

    Returns:
        OneLoggerCallback: Configured callback instance for OneLogger integration
    """
    world_size = trainer.num_nodes * trainer.num_devices
    rank = trainer.global_rank
    num_iterations = trainer.max_steps * (trainer.max_epochs or 1)
    config = OneLoggerConfig(
        # Basic Configuration
        enable_for_current_rank=bool((rank == world_size) - 1),
        one_logger_async=True,
        one_logger_run_name=experiment_name,
        # Logging Configuration
        log_every_n_train_iterations=trainer.log_every_n_steps,
        app_tag_run_version=app_tag_run_version,
        one_logger_project=one_logger_project,
        app_run_type=app_run_type,
        # Run Identification
        app_tag=app_tag or f"{experiment_name}_{world_size}",
        app_tag_run_name=app_tag_run_name or experiment_name,
        # Training Configuration
        world_size=world_size,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        batch_size=global_batch_size,
        seq_length=seq_length,
        # Training Targets
        train_iterations_target=num_iterations,
        train_samples_target=num_iterations * global_batch_size,
        # Feature Flags
        is_train_iterations_enabled=True,
        is_validation_iterations_enabled=True,
        is_save_checkpoint_enabled=trainer.checkpoint_callback is not None,
        # TODO dorotat: enable tflops logging via onelogger. Require to pass model flops to the onelogger callback init
        is_log_throughput_enabled=False,  # create_tflops_callback
        # Checkpoint Configuration
        save_checkpoint_strategy=save_checkpoint_strategy,
    )

    callback = OneLoggerCallback(config=config, rank=rank)
    return callback


def add_one_logger_args(parser):
    """Add OneLogger-specific arguments to the argument parser.

    Args:
        parser: The argument parser to add arguments to
    """
    group = parser.add_argument_group("OneLogger Arguments")
    group.add_argument(
        "--one_logger_project",
        type=str,
        default="bionemo-framework",
        help="Project name in WandB where OneLogger metrics will be tracked",
    )
    group.add_argument(
        "--one_logger_app_tag_run_version", type=str, default="0.0.0", help="Version of the model/application"
    )
    group.add_argument(
        "--one_logger_app_tag",
        type=str,
        default=None,
        help="Tag for performance tracking - jobs with same expected performance should have same tag. If None, OneLoggerConfig will be constructed in OneLoggerConfig.",
    )
    group.add_argument(
        "--one_logger_app_tag_run_name",
        type=str,
        default=None,
        help="Name for tracking overall training progress across multiple jobs. If None, OneLoggerConfig will be constructed in OneLoggerConfig.",
    )
    return parser
