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
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from nemo.lightning import io
from nemo.lightning.data import MegatronPretrainingSampler
from nemo.lightning.megatron_parallel import CallbackMethods, DataT, MegatronLossReduction, MegatronStep
from overrides import override
from pytorch_lightning import Callback, LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader

from bionemo.testing.harnesses.mode import Mode
from bionemo.testing.torch import recursive_detach


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


class BaseInterruptedVsContinuousCallback(Callback, CallbackMethods, io.IOMixin):
    """Base class for serializable stop-and-go callback to compare continuous to interrupted training."""

    def __init__(self):
        """Initializes the callback."""
        self.data = []

    def __deepcopy__(self, memo):
        """Don't actually attempt to copy this data when this callback is being serialized."""
        ...


class LearningRateCallback(BaseInterruptedVsContinuousCallback):
    """Stop-and-go callback for learning rate before pausing and after resuming training."""

    def on_megatron_step_start(self, step: MegatronStep) -> MegatronStep:
        """Get learning rate as metadata."""
        if step.trainer.training:
            self.data.append(np.array(step.trainer.optimizers[0].param_groups[0]["lr"]))
        return step


class GlobalStepStateCallback(BaseInterruptedVsContinuousCallback):
    """Stop-and-go callback for global_step before pausing and after resuming training."""

    def on_megatron_step_start(self, step: MegatronStep) -> MegatronStep:
        """Get learning rate as metadata."""
        if step.trainer.training:
            self.data.append(np.array(step.trainer.global_step))
        return step


class ConsumedSamplesCallback(BaseInterruptedVsContinuousCallback):
    """Stop-and-go callback to check consumed samples before pausing and after resuming training."""

    def on_megatron_step_start(self, step: MegatronStep) -> MegatronStep:
        """Get consumed samples as metadata."""
        # return trainer.datamodule.state_dict()["consumed_samples"]  # TODO why state_dict can be empty despite working lines below
        if step.trainer.training:
            data_sampler = step.trainer.datamodule.data_sampler
            consumed_samples = data_sampler.compute_consumed_samples(
                step.trainer.global_step - step.trainer.datamodule.init_global_step
            )
            self.data.append(np.array(consumed_samples))
        return step


class InputAndOutputIdentityCallback(BaseInterruptedVsContinuousCallback):
    """Callback to store input and output data for comparison."""

    def __init__(self):
        """Initializes the callback."""
        super().__init__()
        self.train_inputs = []
        self.train_outputs = []
        self.train_losses = []

        self.valid_inputs = []
        self.valid_outputs = []
        self.valid_losses = []

    def on_megatron_microbatch_end(
        self,
        step: MegatronStep,
        batch: DataT,
        forward_callback: "MegatronLossReduction",
        output: Any,
    ) -> None:
        """Store the input and output data for later comparison."""
        if step.trainer.sanity_checking:
            return

        elif step.trainer.training:
            self.train_inputs.append(recursive_detach(batch))
            self.train_outputs.append(recursive_detach(output))

        elif step.trainer.validating:
            self.valid_inputs.append(recursive_detach(batch))
            self.valid_outputs.append(recursive_detach(output))

        else:
            raise RuntimeError(f"Unexpected Mode: {step.trainer.state.stage}")

    def on_megatron_step_end(
        self,
        step: MegatronStep,
        microbatch_outputs: List[Any],
        reduced: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
    ) -> None:
        """Collect the reduced data for comparison."""
        if step.trainer.sanity_checking:
            return

        elif step.trainer.training:
            self.train_losses.append(recursive_detach(reduced))

        elif step.trainer.validating:
            self.valid_losses.append(recursive_detach(reduced))

        else:
            raise RuntimeError(f"Unexpected Mode: {step.trainer.state.stage}")


class AbstractStopAndGoCallback(ABC, BaseInterruptedVsContinuousCallback):
    """Abstract base class for stop-and-go callback to compare metadata before pausing and after resuming training.

    This base class provides utility methods to help streamline stop and go comparison.

    Provided methods:
        - __init__: initializes the callback with the given mode.
        - get_metadata: abstract method that should be overridden to get metadata from the trainer and pl_module.

    Default behaviors:
        - in stop mode, metadata is gotten and compared on_validation_epoch_end.
        - in go mode, metadata is gotten and saved on_train_epoch_start.

    Override these behaviors if necessary.
    """

    def __init__(self, mode: Mode = Mode.STOP):
        """Initialize StopAndGoCallback.

        Args:
            mode (str, optional): Mode to run in. Must be either Mode.STOP or Mode.RESUME. Defaults to Mode.STOP.

        Notes:
            User must override get_metadata to get metadata from the trainer and pl_module.
        """
        if mode not in [Mode.STOP, Mode.RESUME]:
            raise ValueError(f"mode must be 'stop' or 'go', got {mode}")
        self.mode = mode
        super().__init__()

    @abstractmethod
    def get_metadata(self, trainer: Trainer, pl_module: LightningModule) -> Any:
        """Get metadata from trainer and pl_module."""
        raise NotImplementedError

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule):  # noqa: D102
        if self.mode == Mode.RESUME:
            self.data = self.get_metadata(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):  # noqa: D102
        if not trainer.sanity_checking and self.mode == Mode.STOP:
            self.data = self.get_metadata(trainer, pl_module)


class OptimizerStateStopAndGoCallback(AbstractStopAndGoCallback):
    """Stop-and-go callback to check optimizer states before pausing and after resuming training."""

    @override
    def get_metadata(self, trainer: Trainer, pl_module: LightningModule) -> List[Dict[str, torch.Tensor]]:
        """Get optimizer states as metadata."""
        return [optimizer.mcore_optimizer.optimizer.state_dict()["state"] for optimizer in trainer.optimizers]


class TrainValInitConsumedSamplesStopAndGoCallback(AbstractStopAndGoCallback):
    """Stop-and-go callback to check consumed samples before pausing and after resuming training."""

    @override
    def get_metadata(self, trainer: Trainer, pl_module: LightningModule) -> Any:
        """Get consumed samples as metadata."""
        # return trainer.datamodule.state_dict()["consumed_samples"]  # TODO why state_dict can be empty despite working lines below
        train_data_sampler: MegatronPretrainingSampler = trainer.train_dataloader.batch_sampler
        val_data_sampler: MegatronPretrainingSampler = trainer.val_dataloaders.batch_sampler
        return train_data_sampler.consumed_samples, val_data_sampler.consumed_samples
