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
from enum import Enum
from typing import Optional, Type, TypeVar, Union

import torch
from jaxtyping import Bool
from torch import Tensor

from bionemo.moco.distributions.prior.distribution import PriorDistribution
from bionemo.moco.distributions.time.distribution import TimeDistribution


# Define a generic type for Enum
AnyEnum = TypeVar("AnyEnum", bound=Enum)


def string_to_enum(value: Union[str, AnyEnum], enum_type: Type[AnyEnum]) -> AnyEnum:
    """Converts a string to an enum value of the specified type. If the input is already an enum instance, it is returned as-is.

    Args:
        value (Union[str, E]): The string to convert or an existing enum instance.
        enum_type (Type[E]): The enum type to convert to.

    Returns:
        E: The corresponding enum value.

    Raises:
        ValueError: If the string does not correspond to any enum member.
    """
    if isinstance(value, enum_type):
        # If the value is already an enum, return it
        return value

    try:
        # Match the value to the Enum, case-insensitively
        return enum_type(value)
    except ValueError:
        # Raise a helpful error if the value is invalid
        valid_values = [e.value for e in enum_type]
        raise ValueError(f"Invalid value '{value}'. Expected one of {valid_values}.")


def pad_like(source: Tensor, target: Tensor) -> Tensor:
    """Pads the dimensions of the source tensor to match the dimensions of the target tensor.

    Args:
        source (Tensor): The tensor to be padded.
        target (Tensor): The tensor that the source tensor should match in dimensions.

    Returns:
        Tensor: The padded source tensor.

    Raises:
        ValueError: If the source tensor has more dimensions than the target tensor.

    Example:
        >>> source = torch.tensor([1, 2, 3])  # shape: (3,)
        >>> target = torch.tensor([[1, 2], [4, 5], [7, 8]])  # shape: (3, 2)
        >>> padded_source = pad_like(source, target)  # shape: (3, 1)
    """
    if source.ndim == target.ndim:
        return source
    elif source.ndim > target.ndim:
        raise ValueError(f"Cannot pad {source.shape} to {target.shape}")
    return source.view(list(source.shape) + [1] * (target.ndim - source.ndim))


class PredictionType(Enum):
    """An enumeration representing the type of prediction a Denoising Diffusion Probabilistic Model (DDPM) can be used for.

    DDPMs are versatile models that can be utilized for various prediction tasks, including:

    - **Data**: Predicting the original data distribution from a noisy input.
    - **Noise**: Predicting the noise that was added to the original data to obtain the input.
    - **Velocity**: Predicting the velocity or rate of change of the data, particularly useful for modeling temporal dynamics.

    These prediction types can be used to train neural networks for specific tasks, such as denoising, image synthesis, or time-series forecasting.
    """

    DATA = "data"
    NOISE = "noise"
    VELOCITY = "velocity"


# Adding useful aliases for Flow Matching
PredictionType._value2member_map_["vector_field"] = PredictionType.VELOCITY
PredictionType._value2member_map_["flow"] = PredictionType.VELOCITY


class Interpolant(ABC):
    """An abstract base class representing an Interpolant.

    This class serves as a foundation for creating interpolants that can be used
    in various applications, providing a basic structure and interface for
    interpolation-related operations.
    """

    def __init__(
        self,
        time_distribution: TimeDistribution,
        prior_distribution: PriorDistribution,
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Initializes the Interpolant class.

        Args:
            time_distribution (TimeDistribution): The distribution of time steps.
            prior_distribution (PriorDistribution): The prior distribution of the variable.
            device (Union[str, torch.device], optional): The device on which to operate. Defaults to "cpu".
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
        """
        self.time_distribution = time_distribution
        self.prior_distribution = prior_distribution
        self.device = device
        self.rng_generator = rng_generator

    @abstractmethod
    def interpolate(self, *args, **kwargs) -> Tensor:
        """Get x(t) with given time t from noise and data.

        Interpolate between x0 and x1 at the given time t.
        """
        pass

    @abstractmethod
    def step(self, *args, **kwargs) -> Tensor:
        """Do one step integration."""
        pass

    def general_step(self, method_name: str, kwargs: dict):
        """Calls a step method of the class by its name, passing the provided keyword arguments.

        Args:
            method_name (str): The name of the step method to call.
            kwargs (dict): Keyword arguments to pass to the step method.

        Returns:
            The result of the step method call.

        Raises:
            ValueError: If the provided method name does not start with 'step'.
            Exception: If the step method call fails. The error message includes a list of available step methods.

        Note:
            This method allows for dynamic invocation of step methods, providing flexibility in the class's usage.
        """
        if not method_name.startswith("step"):
            raise ValueError(f"Method name '{method_name}' does not start with 'step'")

        try:
            # Get the step method by its name
            func = getattr(self, method_name)
            # Call the step method with the provided keyword arguments
            return func(**kwargs)
        except Exception as e:
            # Get a list of available step methods
            available_methods = "\n".join([f"  - {attr}" for attr in dir(self) if attr.startswith("step")])
            # Create a detailed error message
            error_message = f"Error calling method '{method_name}': {e}\nAvailable step methods:\n{available_methods}"
            # Re-raise the exception with the detailed error message
            raise type(e)(error_message)

    def sample_prior(self, *args, **kwargs) -> Tensor:
        """Sample from prior distribution.

        This method generates a sample from the prior distribution specified by the
        `prior_distribution` attribute.

        Returns:
            Tensor: The generated sample from the prior distribution.
        """
        # Ensure the device is specified, default to self.device if not provided
        if "device" not in kwargs:
            kwargs["device"] = self.device
        kwargs["rng_generator"] = self.rng_generator
        # Sample from the prior distribution
        return self.prior_distribution.sample(*args, **kwargs)

    def sample_time(self, *args, **kwargs) -> Tensor:
        """Sample from time distribution."""
        # Ensure the device is specified, default to self.device if not provided
        if "device" not in kwargs:
            kwargs["device"] = self.device
        kwargs["rng_generator"] = self.rng_generator
        # Sample from the time distribution
        return self.time_distribution.sample(*args, **kwargs)

    def to_device(self, device: str):
        """Moves all internal tensors to the specified device and updates the `self.device` attribute.

        Args:
            device (str): The device to move the tensors to (e.g. "cpu", "cuda:0").

        Note:
            This method is used to transfer the internal state of the DDPM interpolant to a different device.
            It updates the `self.device` attribute to reflect the new device and moves all internal tensors to the specified device.
        """
        self.device = device
        for attr_name in dir(self):
            if attr_name.startswith("_") and isinstance(getattr(self, attr_name), torch.Tensor):
                setattr(self, attr_name, getattr(self, attr_name).to(device))
        return self

    def clean_mask_center(self, data: Tensor, mask: Optional[Tensor] = None, center: Bool = False) -> Tensor:
        """Returns a clean tensor that has been masked and/or centered based on the function arguments.

        Args:
            data: The input data with shape (..., nodes, features).
            mask: An optional mask to apply to the data with shape (..., nodes). If provided, it is used to calculate the CoM. Defaults to None.
            center: A boolean indicating whether to center the data around the calculated CoM. Defaults to False.

        Returns:
            The data with shape (..., nodes, features) either centered around the CoM if `center` is True or unchanged if `center` is False.
        """
        if mask is not None:
            data = data * mask.unsqueeze(-1)
        if not center:
            return data
        if mask is None:
            num_nodes = torch.tensor(data.shape[1], device=data.device)
        else:
            num_nodes = torch.clamp(mask.sum(dim=-1), min=1)  # clamp used to prevent divide by 0
        com = data.sum(dim=-2) / num_nodes.unsqueeze(-1)
        return data - com.unsqueeze(-2)
