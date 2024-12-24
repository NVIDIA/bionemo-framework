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
from typing import Optional, Tuple, Union

import torch
from torch import Tensor


class PriorDistribution(ABC):
    """An abstract base class representing a prior distribution."""

    @abstractmethod
    def sample(self, shape: Tuple, mask: Optional[Tensor] = None, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Generates a specified number of samples from the time distribution.

        Args:
        shape (Tuple): The shape of the samples to generate.
        mask (Optional[Tensor], optional): A tensor indicating which samples should be masked. Defaults to None.
        device (str, optional): The device on which to generate the samples. Defaults to "cpu".

        Returns:
            Float: A tensor of samples.
        """
        pass


class DiscretePriorDistribution(PriorDistribution):
    """An abstract base class representing a discrete prior distribution."""

    def __init__(self, num_classes: int, prior_dist: Tensor):
        """Initializes a DiscretePriorDistribution instance.

        Args:
        num_classes (int): The number of classes in the discrete distribution.
        prior_dist (Tensor): The prior distribution over the classes.

        Returns:
        None
        """
        self.num_classes = num_classes
        self.prior_dist = prior_dist

    def get_num_classes(self) -> int:
        """Getter for num_classes."""
        return self.num_classes

    def get_prior_dist(self) -> Tensor:
        """Getter for prior_dist."""
        return self.prior_dist
