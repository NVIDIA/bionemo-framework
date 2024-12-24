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


from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from bionemo.moco.distributions.prior.distribution import DiscretePriorDistribution


class DiscreteUniformPrior(DiscretePriorDistribution):
    """A subclass representing a discrete uniform prior distribution."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initializes a discrete uniform prior distribution.

        Args:
            num_classes (int): The number of classes in the discrete uniform distribution. Defaults to 10.
        """
        prior_dist = torch.ones((num_classes)) * 1 / num_classes
        super().__init__(num_classes, prior_dist)
        if torch.sum(self.prior_dist).item() - 1.0 > 1e-5:
            raise ValueError("Prior distribution probabilities do not sum up to 1.0")

    def sample(
        self,
        shape: Tuple,
        mask: Optional[Tensor] = None,
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Generates a specified number of samples.

        Args:
            shape (Tuple): The shape of the samples to generate.
            device (str): cpu or gpu.
            mask (Optional[Tensor]): An optional mask to apply to the samples. Defaults to None.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

        Returns:
            Float: A tensor of samples.
        """
        samples = torch.randint(0, self.num_classes, shape, device=device, generator=rng_generator)
        if mask is not None:
            samples = samples * mask[(...,) + (None,) * (len(samples.shape) - len(mask.shape))]
        return samples
