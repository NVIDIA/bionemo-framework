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
from typing import Generic, Protocol, Sequence, Type, TypeVar

from torch import Tensor


__all__: Sequence[str] = (
    "BionemoModelConfig",
    "BionemoTrainableModelConfig",
    "LossType",
    "Model",
    "ModelOutput",
    "ModelType",
)

ModelOutput = TypeVar("ModelOutput", Tensor, list[Tensor], tuple[Tensor], dict[str, Tensor], covariant=True)
"""A Model's forward pass may produce a tensor, multiple tensors, or named tensors.
"""

LossType = TypeVar("LossType")
"""Stand-in for a loss function; no constraints.
"""


class Model(Protocol[ModelOutput]):
    """Lightweight interface for a model: must have a forward method."""

    def forward(self, *args, **kwargs) -> ModelOutput:
        """Prediction / forward-step for a model."""
        ...


ModelType = TypeVar("ModelType", bound=Model)
"""Generic type for things that have a forward pass.
"""


class BionemoModelConfig(Generic[ModelType], ABC):
    """An abstract class for model configuration."""

    @abstractmethod
    def configure_model(self, *args, **kwargs) -> Model:
        """Configures the model."""
        raise NotImplementedError()


class BionemoTrainableModelConfig(Generic[ModelType, LossType], BionemoModelConfig[ModelType]):
    """An abstract class for trainable model configuration."""

    @abstractmethod
    def get_loss_reduction_class(self) -> Type[LossType]:
        """Returns the loss reduction class."""
        raise NotImplementedError()
