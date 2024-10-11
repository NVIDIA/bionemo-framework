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


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, TypedDict

import nemo.lightning as nl
import pytorch_lightning as pl
from nemo.collections import llm
from nemo.lightning import nemo_logger, resume
from nemo.lightning.pytorch import callbacks as nl_callbacks

from bionemo.core import BIONEMO_CACHE_DIR
from bionemo.testing import testing_callbacks
from bionemo.testing.megatron_parallel_state_utils import distributed_model_parallel_state


__all__: Sequence[str] = (
    "get_learning_rate",
    "get_global_step",
    "StopAndGoHarness",
    "MetricsFn",
    "MetricsDict",
)

MetricsFn = Callable[[pl.Trainer, pl.LightningModule], Any]
"""A metrics producing function."""


class MetricsDict(TypedDict):
    """Default metrics dict."""

    global_step: MetricsFn
    learning_rate: MetricsFn


def get_learning_rate(trainer: pl.Trainer, model: pl.LightningModule) -> Any:
    """Returns the learning rate of the model.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning trainer.
        model (pl.LightningModule): The PyTorch Lightning model.

    Returns:
        Any: The learning rate of the model.
    """
    return trainer.optimizers[0].param_groups[0]["lr"]


def get_global_step(trainer: pl.Trainer, model: pl.LightningModule) -> Any:
    """Returns the global step of the model.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning trainer.
        model (pl.LightningModule): The PyTorch Lightning model.

    Returns:
        Any: The global step of the model.
    """
    return trainer.global_step


class StopAndGoHarness(ABC):
    """Abstract base class for a stop-and-go harness.

    Stop and go tests act as follows:
        - setup a clean model for a brief training run, select metrics to track.
        - interrupt training via the StopAndGoException in the callback InterruptAfterMetadataCallback.
        - setup a model to be resumed from the checkpoint, with the same metrics.
        - Restore training and check that metadta matches the stored metrics in the callback CheckpointIntegrityCallback.
      Useful metrics to check are things like learning rate, global step, validation loss, training loss, and anything
        else that is important to the training process. If there is an unavailable metrics, a method for fetching the
        metric should be provided in the bionemo.testing.callbacks module.

    Considerations when implementing this class:
        - devices, pipeline_model_parallel, and tensor_model_parallel may impact the setup of DataModule. Certain
            datasets expect a known global batch size, which depends on the number of devices and conditional
            tensor model parallel/ pipeline model parallel settings.
        - 'mode' is useful in some cases, but not in all cases. Implement conditions based on these when useful. As an
            example, it may be useful to implement a test that stops and resumes with different parallelism settings.
            - changing callbacks to test metadata integrity (core feature of stop-and-go tests).
            - changing trainer behavior to use multiple GPUs
            - changing the model construction to use different hyperparameters.
            - ... etc
            Each of the above tests cases may be useful for automated testing of various expected behavior.
        - stop(), go(), and run_test() are provided methods which execute the actual tests, leveraging the conditions
            in the various setup methods, respecting 'mode' where necessary.

    Attributes:
        root_di: The root directory.
        val_check_interval: The validation check interval. Stored as an attribute to ensure consistency.
        exp_name: The experiment name.
        extra_metrics_dict: A dictionary of metrics and their corresponding functions.

    See Also: bionemo.testing.callbacks.
    """

    def __init__(
        self,
        root_dir: Path | str = BIONEMO_CACHE_DIR,
        val_check_interval: int = 2,
        exp_name: str = "stop_and_go_harness",
        extra_metrics_dict: dict[str, MetricsFn] | None = None,
    ):
        """Initializes the StopAndGoHarness object.

        Args:
            root_dir: The root directory. Defaults to Path("./").
            val_check_interval: The validation check interval. Defaults to 2.
            exp_name: The experiment name. Defaults to "stop_and_go_harness".
            extra_metrics_dict: A dictionary that maps keys to 'functions capable of computing metrics in a callback.'
                Callbacks typically have an interface where both the Trainer and LightningModule are available, meaning any metric that
                can be computed using these are viable functions to pass in to this dictionary. By default 'global_step' and 'learning_rate' are available.
        """
        self.root_dir = Path(root_dir)  # Set to bionemo2_home ideally.
        self.exp_name = exp_name
        self.metadata_dir = self.root_dir / self.exp_name
        self.metrics_getter: dict[str, MetricsFn] = dict(**self.get_default_metrics_dict())
        if extra_metrics_dict is not None:
            self.metrics_getter.update(extra_metrics_dict)
        self.val_check_interval = val_check_interval
        self.nemo_logger: nemo_logger.NeMoLogger = nemo_logger.NeMoLogger(
            log_dir=str(self.root_dir),
            name=self.exp_name,
            use_datetime_version=False,
            version=None,
            tensorboard=None,
            wandb=None,
            ckpt=None,
        )

    @abstractmethod
    def setup_model(
        self, mode: Literal["stop", "go"]
    ) -> tuple[pl.LightningModule, pl.LightningDataModule, nl.MegatronOptimizerModule]:
        """Constructs the model, data, and optimizer for the test harness.

        Optionally supports separate code paths for 'stop'/'go', although implementors are
        encouraged to use the same code path for both.

        Args:
            mode: The mode indicating whether to stop or go.

        Returns:
            tuple: A tuple containing the model, data, and optimizer.
        """
        raise NotImplementedError()

    @abstractmethod
    def setup_trainer_and_strategy(
        self, mode: Literal["stop", "go"], metrics_getter: dict[str, Callable[[pl.Trainer, pl.LightningModule], Any]]
    ) -> pl.Trainer:
        """Constructs the trainer object for the stop and go test.

        This method invokes the get_callbacks method to get the appropriate callbacks for the mode and passes it to the trainer.

        Args:
            mode: The mode indicating whether to stop or go.
            metrics_getter: A dictionary of functions that computes the metrics.
        """
        raise NotImplementedError()

    def get_default_metrics_dict(self) -> MetricsDict:
        """Returns a dictionary of default metrics that can be used in the StopAndGoHarness.

        Returns:
            dict: A dictionary of default metrics that can be used in the StopAndGoHarness.
        """
        return {"global_step": get_global_step, "learning_rate": get_learning_rate}

    def get_callbacks(self, mode: Literal["stop", "go"]) -> list[pl.Callback]:
        """Returns a list of callbacks based on the specified mode. Base implemention provides reasonable defaults.

        To extend this method, call the super and append to the callbacks, depending on which mode you are in:

        ```python
        callbacks = super().get_callbacks(mode, metrics)
        callbacks.append(MyCustomCallback())
        return callbacks
        ```

        Args:
            mode: The mode indicating whether to stop or go.

        Returns:
            list: A list of callbacks based on the specified mode.

        Raises:
            ValueError: If the mode is neither 'stop' nor 'go'.
        """
        if mode == "stop":
            callbacks = [
                testing_callbacks.MetadataSaveCallback(
                    metadata_path=self.metadata_dir,
                    metrics_getter=self.metrics_getter,
                ),
                testing_callbacks.RaiseAfterMetadataCallback(metadata_path=self.metadata_dir),
                nl_callbacks.ModelCheckpoint(
                    save_last=True,
                    monitor="reduced_train_loss",
                    save_top_k=2,
                    every_n_train_steps=self.val_check_interval,
                    always_save_context=True,
                    try_restore_best_ckpt=False,
                ),
            ]
        elif mode == "go":
            # we must setup the integrity callback.
            callbacks = [
                testing_callbacks.TestCheckpointIntegrityCallback(
                    metadata_path=self.metadata_dir, metrics_getter=self.metrics_getter
                ),
                nl_callbacks.ModelCheckpoint(
                    save_last=True,
                    monitor="reduced_train_loss",
                    save_top_k=2,
                    every_n_train_steps=self.val_check_interval,
                    always_save_context=True,
                    try_restore_best_ckpt=False,
                ),
            ]
        else:
            raise ValueError("mode must be 'stop' or 'go'")

        return callbacks

    # stop() and go() are provided methods and run the requisite methods with the appropriate mode.
    def stop(self) -> None:
        """Runs pre-training and 'stops' after the first checkpoint is saved.

        This method sets up the model, data, and optimizer for the "stop" mode.
        It then sets up the trainer and strategy for the "stop" mode with the given metrics.
        The training process is executed using the `llm.train` function, passing the model, data, trainer, logger, optimizer, and resume options.
        If a `testing_callbacks.StopAndGoException` is raised during training, it is caught and no action is taken.

        Raises:
            testing_callbacks.StopAndGoException: If a stop and go exception occurs during training.
        """
        model, data, opt = self.setup_model(mode="stop")
        trainer = self.setup_trainer_and_strategy("stop", self.metrics_getter)
        with distributed_model_parallel_state():
            try:
                llm.train(
                    model=model,
                    data=data,
                    trainer=trainer,
                    log=self.nemo_logger,
                    optim=opt,
                    resume=resume.AutoResume(
                        resume_if_exists=False,  # Looks for the -last checkpoint to continue training.
                        resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
                    ),
                )
            except testing_callbacks.StopAndGoException:
                raise

    def go(self) -> None:
        """Resumes the model from the checkpoint saved at the end of `stop()` and verifies the metadata integrity."""
        model, data, opt = self.setup_model(mode="go")
        trainer = self.setup_trainer_and_strategy("go", self.metrics_getter)
        with distributed_model_parallel_state():
            llm.train(
                model=model,
                data=data,
                trainer=trainer,
                log=self.nemo_logger,
                optim=opt,
                resume=resume.AutoResume(
                    resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
                    resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
                ),
            )

    # Finally, execution is a simple stop => go.
    def run_test(self):
        """Executes the stop => go process."""
        self.stop()
        self.go()
