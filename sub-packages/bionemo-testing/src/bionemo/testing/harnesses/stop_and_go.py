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
import tempfile
import unittest
from abc import ABC, abstractmethod
from typing import Dict, Literal, Sequence

import nemo.lightning as nl
import pytest
import pytorch_lightning as pl
from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.nemo_logger import NeMoLogger
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.strategies import MegatronStrategy
from nemo.utils import logging

from bionemo.llm.utils.datamodule_utils import tensor_dict_hash
from bionemo.testing import testing_callbacks
from bionemo.testing.harnesses.mode import Mode
from bionemo.testing.megatron_parallel_state_utils import distributed_model_parallel_state
from bionemo.testing.torch import recursive_assert_approx_equal


__all__: Sequence[str] = ("StopAndGoHarness",)


class StopAndGoHarness(ABC, unittest.TestCase):
    """Abstract base class for a stop-and-go harness.

    Users should override cls.setup_model and update cls.setUpClass to customize the downstream test cases. Metadata are
    collected through callbacks and users can add new unit tests by comparing the metadata in stop/go stages.

    By default, learning rate, global step, optimizer state, consumed samples, model weights through validation loss are
    tested, and are accessible through cls.{stop,go}_callbacks.

    Stop and go tests act as follows:
        - setup a clean model for a brief training run, set StopAndGoCallback(s) to track.
        - interrupt training via the StopAndGoException in the callback RaiseAfterMetadataCallback.
        - train the model resumed from the checkpoint with the same StopAndGoCallback(s).
        - compare each pair of stop and go metadata in a test function for each StopAndGoCallback.

    Considerations when implementing this class:
        - devices, pipeline_model_parallel, and tensor_model_parallel may impact the setup of DataModule. Certain
            datasets expect a known global batch size, which depends on the number of devices and conditional tensor
            model parallel/ pipeline model parallel settings. By default, we are testing only on single device without
            parallelism.
        - 'mode' is useful in some cases, but not in all cases. Implement conditions based on these when useful. As an
            example, it may be useful to implement a test that stops and resumes.
            - changing callbacks to test metadata integrity (core feature of stop-and-go tests).
            - changing the model construction to use different hyperparameters.
            - ... etc
            Each of the above tests cases may be useful for automated testing of various expected behavior.
        - stop() and resume(), or collectively stop_and_go() are provided methods which execute the actual tests,
          leveraging the conditions in the various setup methods, respecting 'mode' where necessary.

    Attributes:
        root_dir: The root directory.
        val_check_interval: The validation check interval. Stored as an attribute to ensure consistency.
        exp_name: The experiment name.
        extra_metrics_dict: A dictionary of metrics and their corresponding functions.

    See Also: bionemo.testing.callbacks.
    """

    # class variables that need to be overridden
    num_steps: int
    val_check_interval: int
    limit_val_batches: int
    lr: float = 1e-4
    precision: Literal["16-mixed", "bf16-mixed", "32"]

    # class variables that will be setup in setUpClass
    tempdir: tempfile.TemporaryDirectory
    metadata_dir: pathlib.Path
    exp_name: str
    stop_callbacks: Dict[str, pl.Callback]
    resume_callbacks: Dict[str, pl.Callback]
    nemo_logger: NeMoLogger

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up the class by creating a temporary directory, metadata_dir, exp_name and callbacks."""
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.metadata_dir = pathlib.Path(cls.tempdir.name) / "metadata"
        cls.exp_name = cls.__name__

        cls.interrupted_io_callback = testing_callbacks.InputAndOutputIdentityCallback()
        cls.continuous_io_callback = testing_callbacks.InputAndOutputIdentityCallback()

        cls.stop_callbacks: Dict[str, pl.Callback] = cls.get_default_callbacks(mode=Mode.STOP)
        cls.resume_callbacks: Dict[str, pl.Callback] = cls.get_default_callbacks(mode=Mode.RESUME)
        cls.continuous_callbacks: Dict[str, pl.Callback] = cls.get_default_callbacks(mode=Mode.CONTINUOUS)

        cls.nemo_logger = NeMoLogger(
            log_dir=cls.tempdir.name,
            name=cls.exp_name,
            use_datetime_version=False,
            version=None,
            tensorboard=None,
            wandb=None,
            ckpt=None,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Tears down the class by cleaning up the temporary directory."""
        cls.tempdir.cleanup()

    @classmethod
    @abstractmethod
    def setup_model(cls, mode: Mode) -> tuple[pl.LightningModule, pl.LightningDataModule, nl.MegatronOptimizerModule]:
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
    def setup_trainer(
        cls,
        mode: Mode,
    ) -> nl.Trainer:
        """Setup trainer by passing stop/go callbacks according to mode.

        Args:
            mode (Mode): The mode indicating whether to stop or go.

        Returns:
            (nl.Trainer): NeMo Lightning trainer object.
        """
        strategy = MegatronStrategy(
            ddp="megatron",
            find_unused_parameters=True,
            ckpt_include_optimizer=True,
        )

        callbacks = {
            Mode.STOP: cls.stop_callbacks,
            Mode.RESUME: cls.resume_callbacks,
            Mode.CONTINUOUS: cls.continuous_callbacks,
        }[mode]

        callbacks = list(callbacks.values())
        trainer = nl.Trainer(
            devices=1,
            max_steps=cls.num_steps,
            accelerator="gpu",
            strategy=strategy,
            limit_val_batches=cls.limit_val_batches,
            val_check_interval=cls.val_check_interval,
            log_every_n_steps=cls.val_check_interval,
            num_nodes=1,
            callbacks=callbacks,
            plugins=nl.MegatronMixedPrecision(precision=cls.precision),
        )
        return trainer

    @classmethod
    def get_default_callbacks(cls, mode: Mode) -> Dict[str, pl.Callback]:
        """Returns a list of callbacks based on the specified mode. Base implementation provides reasonable defaults.

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
        callbacks: Dict[str, pl.Callback]

        match mode:
            case Mode.STOP | mode.RESUME:
                callbacks = {
                    "ModelCheckpoint": nl_callbacks.ModelCheckpoint(
                        save_last=True,
                        monitor="reduced_train_loss",
                        save_top_k=2,
                        every_n_train_steps=cls.val_check_interval,
                        always_save_context=True,
                    ),
                    "LearningRateStateStopAndGoCallback": testing_callbacks.LearningRateStateStopAndGoCallback(
                        pickle_directory=cls.metadata_dir / "learning_rate",
                        mode=mode,
                    ),
                    "GlobalStepStateStopAndGoCallback": testing_callbacks.GlobalStepStateStopAndGoCallback(
                        pickle_directory=cls.metadata_dir / "global_step",
                        mode=mode,
                    ),
                    "OptimizerStateStopAndGoCallback": testing_callbacks.OptimizerStateStopAndGoCallback(
                        pickle_directory=cls.metadata_dir / "optimizer_state",
                        mode=mode,
                    ),
                    "ComsumedSamplesStopAndGoCallback": testing_callbacks.ConsumedSamplesStopAndGoCallback(
                        pickle_directory=cls.metadata_dir / "consumed_samples",
                        mode=mode,
                    ),
                    "TrainValInitComsumedSamplesStopAndGoCallback": testing_callbacks.TrainValInitConsumedSamplesStopAndGoCallback(
                        pickle_directory=cls.metadata_dir / "train_val_init_consumed_samples",
                        mode=mode,
                    ),
                    "ManualValLossStopAndGoCallback": testing_callbacks.ManualValLossStopAndGoCallback(
                        pickle_directory=cls.metadata_dir / "manual_val_loss",
                        mode=mode,
                    ),
                    "InputAndOutputIdentityCallback": cls.interrupted_io_callback,
                }

            case Mode.CONTINUOUS:
                callbacks = {
                    "InputAndOutputIdentityCallback": cls.continuous_io_callback,
                }

            case _:
                raise ValueError(f"Invalid mode {mode}")

        if mode == Mode.STOP:
            callbacks["RaiseAfterMetadataCallback"] = testing_callbacks.RaiseAfterMetadataCallback()

        return callbacks

    # stop() and resume() are provided methods and run the requisite methods with the appropriate mode.
    @classmethod
    def stop(cls) -> None:
        """Runs pre-training and 'stops' after the first checkpoint is saved.

        This method sets up the model, data, and optimizer for the Mode.STOP mode.
        It then sets up the trainer and strategy for the Mode.STOP mode with the given metrics.
        The training process is executed using the `llm.train` function, passing the model, data, trainer, logger, optimizer, and resume options.
        If a `testing_callbacks.StopAndGoException` is raised during training, it is caught and no action is taken.

        Raises:
            testing_callbacks.StopAndGoException: If a stop and go exception occurs during training.
        """
        logging.info("Running stop()...")

        model, data, opt = cls.setup_model(mode=Mode.STOP)
        trainer = cls.setup_trainer(Mode.STOP)
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
    def resume(cls) -> None:
        """Resumes the model from the checkpoint saved at the end of `stop()` and verifies the metadata integrity."""
        logging.info("Running resume()...")

        model, data, opt = cls.setup_model(mode=Mode.RESUME)
        trainer = cls.setup_trainer(Mode.RESUME)
        with distributed_model_parallel_state():
            llm.train(
                model=model,
                data=data,
                trainer=trainer,
                log=cls.nemo_logger,
                optim=opt,
                resume=resume.AutoResume(
                    resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
                    resume_ignore_no_checkpoint=False,  # When false this will throw an error with no existing checkpoint.
                ),
            )

    @classmethod
    def continuous(cls) -> None:
        """Trains the model in one continuous path without stopping."""
        logging.info("Running continuous()...")

        model, data, opt = cls.setup_model(mode=Mode.CONTINUOUS)
        trainer = cls.setup_trainer(Mode.CONTINUOUS)
        with distributed_model_parallel_state():
            llm.train(model=model, data=data, trainer=trainer, log=cls.nemo_logger, optim=opt)

    # Finally, execution is a simple stop => go.
    @classmethod
    def run_stop_and_go(cls):
        """Executes the stop => go process."""
        cls.stop()
        cls.resume()
        cls.continuous()

    # should we hide shared tests?
    def test_learning_rate_stop_and_go(self):
        """Tests the learning rate stop and go functionality."""
        callback: testing_callbacks.LearningRateStateStopAndGoCallback = self.resume_callbacks[
            "LearningRateStateStopAndGoCallback"
        ]
        lr_stop, lr_go = callback.load_stop_and_go_pickles()
        assert lr_stop == lr_go

    def test_global_step_stop_and_go(self):
        """Tests the global step in stop-and-go scenario."""
        callback: testing_callbacks.GlobalStepStateStopAndGoCallback = self.resume_callbacks[
            "GlobalStepStateStopAndGoCallback"
        ]
        global_step_stop, global_step_go = callback.load_stop_and_go_pickles()
        assert global_step_stop == global_step_go

    def test_optimizer_state_stop_and_go(self):
        """Tests the optimizer state in stop-and-go scenario."""
        callback: testing_callbacks.OptimizerStateStopAndGoCallback = self.resume_callbacks[
            "OptimizerStateStopAndGoCallback"
        ]
        state_dicts_stop, state_dicts_go = callback.load_stop_and_go_pickles()
        for state_dict_go, state_dict_stop in zip(state_dicts_stop, state_dicts_go):
            assert tensor_dict_hash(state_dict_go) == tensor_dict_hash(state_dict_stop)

    def test_consumed_samples_stop_and_go(self):
        """Tests the consumed samples in stop-and-go scenario."""
        callback: testing_callbacks.ConsumedSamplesStopAndGoCallback = self.resume_callbacks[
            "ComsumedSamplesStopAndGoCallback"
        ]
        consumed_samples_stop, consumed_samples_go = callback.load_stop_and_go_pickles()
        assert consumed_samples_stop == consumed_samples_go
        # Make sure we do not trivially pass.
        assert consumed_samples_stop > 0

    def test_manual_val_loss_stop_and_go(self):
        """Tests validation loss of the first batch in non-sanity-check validation epoch in stop-and-go scenario."""
        callback: testing_callbacks.ManualValLossStopAndGoCallback = self.resume_callbacks[
            "ManualValLossStopAndGoCallback"
        ]
        val_loss_stop, val_loss_go = callback.load_stop_and_go_pickles()
        assert val_loss_stop == val_loss_go

    def test_train_val_init_consumed_samples(self):
        """Tests the initial consumed samples in stop-and-go scenario."""
        callback: testing_callbacks.TrainValInitConsumedSamplesStopAndGoCallback = self.resume_callbacks[
            "TrainValInitComsumedSamplesStopAndGoCallback"
        ]
        (
            (train_consumed_stop, val_consumed_stop),
            (train_consumed_go, val_consumed_go),
        ) = callback.load_stop_and_go_pickles()
        assert val_consumed_stop == 0
        assert val_consumed_go == 0
        assert train_consumed_stop == 0
        assert train_consumed_go > 0

    def test_train_inputs_are_identical_for_interrupted_test(self):
        """Ensures that the input tensors for training are identical for the interrupted and continuous tests."""
        assert len(self.interrupted_io_callback.train_inputs), "No train inputs found."
        recursive_assert_approx_equal(
            self.interrupted_io_callback.train_inputs, self.continuous_io_callback.train_inputs
        )

    def test_train_outputs_are_identical_for_interrupted_test(self):
        """Ensures that the output tensors for training are identical for the interrupted and continuous tests."""
        assert len(self.interrupted_io_callback.train_outputs), "No train outputs found."
        recursive_assert_approx_equal(
            self.interrupted_io_callback.train_outputs, self.continuous_io_callback.train_outputs
        )

    def test_train_losses_are_identical_for_interrupted_test(self):
        """Ensures that the training losses are identical for all microbatches."""
        assert len(self.interrupted_io_callback.train_losses), "No train outputs found."
        recursive_assert_approx_equal(
            self.interrupted_io_callback.train_losses, self.continuous_io_callback.train_losses
        )

    # TODO: For some reason, validation in NeMo runs an extra batch in the case when the training is stopped and
    # resumed. Hopefully we can fix this upstream and remove the indexing based on the length of the continuous
    # validation batches.
    @pytest.mark.xfail(reason="Validation runs an extra batch in the case when training is stopped and resumed.")
    def test_identical_number_of_validation_batches(self):
        """Ensures that the input tensors for training are identical for the interrupted and continuous tests."""
        assert len(self.interrupted_io_callback.valid_inputs) == len(self.continuous_io_callback.valid_inputs)

    def test_valid_inputs_are_identical_for_interrupted_test(self):
        """Ensures that the input tensors for training are identical for the interrupted and continuous tests."""
        assert len(self.interrupted_io_callback.valid_inputs), "No valid inputs found."
        num_continuous_batches = len(self.continuous_io_callback.valid_inputs)
        recursive_assert_approx_equal(
            self.interrupted_io_callback.valid_inputs[-num_continuous_batches:],
            self.continuous_io_callback.valid_inputs,
        )

    def test_valid_outputs_are_identical_for_interrupted_test(self):
        """Ensures that the input tensors for training are identical for the interrupted and continuous tests."""
        assert len(self.interrupted_io_callback.valid_outputs), "No valid outputs found."
        num_continuous_batches = len(self.continuous_io_callback.valid_inputs)
        recursive_assert_approx_equal(
            self.interrupted_io_callback.valid_outputs[-num_continuous_batches:],
            self.continuous_io_callback.valid_outputs,
        )

    def test_valid_losses_are_identical_for_interrupted_test(self):
        """Ensures that the validation losses are identical for all microbatches."""
        assert len(self.interrupted_io_callback.valid_losses), "No train outputs found."
        num_continuous_batches = len(self.continuous_io_callback.valid_inputs)
        recursive_assert_approx_equal(
            self.interrupted_io_callback.valid_losses[-num_continuous_batches:],
            self.continuous_io_callback.valid_losses,
        )
