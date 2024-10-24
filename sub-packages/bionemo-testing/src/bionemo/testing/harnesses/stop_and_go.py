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


import unittest
from abc import ABC
from typing import Any, Callable, Literal, Sequence, TypedDict

import nemo.lightning as nl
import pytorch_lightning as pl
from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks

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
    consumed_samples: MetricsFn


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


def get_consumed_samples(trainer: pl.Trainer, model: pl.LightningModule) -> Any:
    """Returns the consumed samples of the model.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning trainer.
        model (pl.LightningModule): The PyTorch Lightning model.

    Returns:
        Any: The consumed samples of the model.
    """
    # TODO why state_dict can be empty despite working lines below
    # return trainer.datamodule.state_dict()["consumed_samples"]
    data_sampler = trainer.datamodule.data_sampler
    consumed_samples = data_sampler.compute_consumed_samples(trainer.global_step - trainer.datamodule.init_global_step)
    return consumed_samples


class StopAndGoHarness(ABC, unittest.TestCase):
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

    @classmethod
    # @abstractmethod
    def setup_model(
        cls, mode: Literal["stop", "go"]
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

    @classmethod
    def get_default_metrics_dict(cls) -> MetricsDict:
        """Returns a dictionary of default metrics that can be used in the StopAndGoHarness.

        Returns:
            dict: A dictionary of default metrics that can be used in the StopAndGoHarness.
        """
        return {
            "global_step": get_global_step,
            "learning_rate": get_learning_rate,
            "consumed_samples": get_consumed_samples,
        }

    @classmethod
    def get_default_callbacks(cls, mode: Literal["stop", "go"]) -> list[pl.Callback]:
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
                nl_callbacks.ModelCheckpoint(
                    save_last=True,
                    monitor="reduced_train_loss",
                    save_top_k=2,
                    every_n_train_steps=cls.val_check_interval,
                    always_save_context=True,
                ),
                testing_callbacks.LearningRateStateStopAndGoCallback(
                    pickle_directory=cls.metadata_dir / "learning_rate.pkl",
                    mode="stop",
                ),
                testing_callbacks.GlobalStepStateStopAndGoCallback(
                    pickle_directory=cls.metadata_dir / "global_step.pkl",
                    mode="stop",
                ),
                testing_callbacks.OptimizerStateStopAndGoCallback(
                    pickle_directory=cls.metadata_dir / "optimizer_state.pkl",
                    mode="stop",
                ),
                testing_callbacks.ComsumedSamplesStopAndGoCallback(
                    pickle_directory=cls.metadata_dir / "consumed_samples.pkl",
                    mode="stop",
                ),
                testing_callbacks.ManualValLossStopAndGoCallback(
                    pickle_directory=cls.metadata_dir / "manual_val_loss.pkl",
                    mode="stop",
                ),
                testing_callbacks.RaiseAfterMetadataCallback(),
            ]
        elif mode == "go":
            # we must setup the integrity callback.
            callbacks = [
                nl_callbacks.ModelCheckpoint(
                    save_last=True,
                    monitor="reduced_train_loss",
                    save_top_k=2,
                    every_n_train_steps=cls.val_check_interval,
                    always_save_context=True,
                ),
                testing_callbacks.LearningRateStateStopAndGoCallback(
                    pickle_directory=cls.metadata_dir / "learning_rate.pkl",
                    mode="go",
                ),
                testing_callbacks.GlobalStepStateStopAndGoCallback(
                    pickle_directory=cls.metadata_dir / "global_step.pkl",
                    mode="go",
                ),
                testing_callbacks.OptimizerStateStopAndGoCallback(
                    pickle_directory=cls.metadata_dir / "optimizer_state.pkl",
                    mode="go",
                ),
                testing_callbacks.ComsumedSamplesStopAndGoCallback(
                    pickle_directory=cls.metadata_dir / "consumed_samples.pkl",
                    mode="go",
                ),
                testing_callbacks.ManualValLossStopAndGoCallback(
                    pickle_directory=cls.metadata_dir / "manual_val_loss.pkl",
                    mode="go",
                ),
            ]
        else:
            raise ValueError("mode must be 'stop' or 'go'")

        return callbacks

    # stop() and go() are provided methods and run the requisite methods with the appropriate mode.
    @classmethod
    def stop(cls) -> None:
        """Runs pre-training and 'stops' after the first checkpoint is saved.

        This method sets up the model, data, and optimizer for the "stop" mode.
        It then sets up the trainer and strategy for the "stop" mode with the given metrics.
        The training process is executed using the `llm.train` function, passing the model, data, trainer, logger, optimizer, and resume options.
        If a `testing_callbacks.StopAndGoException` is raised during training, it is caught and no action is taken.

        Raises:
            testing_callbacks.StopAndGoException: If a stop and go exception occurs during training.
        """
        model, data, opt = cls.setup_model(mode="stop")
        trainer = cls.setup_trainer_and_strategy("stop")
        with distributed_model_parallel_state():
            try:
                llm.train(
                    model=model,
                    data=data,
                    trainer=trainer,
                    log=cls.nemo_logger,
                    optim=opt,
                    resume=resume.AutoResume(
                        resume_if_exists=False,  # Looks for the -last checkpoint to continue training.
                        resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
                    ),
                )
            except testing_callbacks.StopAndGoException:
                return

    @classmethod
    def go(cls) -> None:
        """Resumes the model from the checkpoint saved at the end of `stop()` and verifies the metadata integrity."""
        model, data, opt = cls.setup_model(mode="go")
        trainer = cls.setup_trainer_and_strategy("go")
        with distributed_model_parallel_state():
            llm.train(
                model=model,
                data=data,
                trainer=trainer,
                log=cls.nemo_logger,
                optim=opt,
                resume=resume.AutoResume(
                    resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
                    resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
                ),
            )

    # Finally, execution is a simple stop => go.
    @classmethod
    def stop_and_go(cls):
        """Executes the stop => go process."""
        cls.stop()
        cls.go()

    # should we hide shared tests?
    def test_learning_rate_stop_and_go(self):
        """Tests the learning rate stop and go functionality."""
        callback: testing_callbacks.LearningRateStateStopAndGoCallback = self.__class__.go_callbacks[1]
        callback.compare_metadata()

    def test_global_step_stop_and_go(self):
        """Tests the global step in stop-and-go scenario."""
        callback: testing_callbacks.GlobalStepStateStopAndGoCallback = self.__class__.go_callbacks[2]
        callback.compare_metadata()

    def test_optimizer_state_stop_and_go(self):
        """Tests the optimizer state in stop-and-go scenario."""
        callback: testing_callbacks.OptimizerStateStopAndGoCallback = self.__class__.go_callbacks[3]
        callback.compare_metadata()

    def test_consumed_samples_stop_and_go(self):
        """Tests the consumed samples in stop-and-go scenario."""
        callback: testing_callbacks.ComsumedSamplesStopAndGoCallback = self.__class__.go_callbacks[4]
        callback.compare_metadata()

    def test_manual_val_loss_stop_and_go(self):
        """Tests validation loss of the first batch in non-sanity-check validation epoch in stop-and-go scenario."""
        callback: testing_callbacks.ManualValLossStopAndGoCallback = self.__class__.go_callbacks[5]
        callback.compare_metadata()
