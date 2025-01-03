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


import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from bionemo.moco.distributions.prior.distribution import DiscretePriorDistribution


class DiscreteCustomPrior(DiscretePriorDistribution):
    """A subclass representing a discrete custom prior distribution.

    This class allows for the creation of a prior distribution with a custom
    probability mass function defined by the `prior_dist` tensor. For example if my data has 4 classes and I want [.3, .2, .4, .1] as the probabilities of the 4 classes.
    """

    def __init__(self, prior_dist: Tensor, num_classes: int = 10) -> None:
        """Initializes a DiscreteCustomPrior distribution.

        Args:
            prior_dist: A tensor representing the probability mass function of the prior distribution.
            num_classes: The number of classes in the prior distribution. Defaults to 10.

        Note:
            The `prior_dist` tensor should have a sum close to 1.0, as it represents a probability mass function.
        """
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
        """Samples from the discrete custom prior distribution.

        Args:
            shape: A tuple specifying the shape of the samples to generate.
            mask: An optional tensor mask to apply to the samples, broadcastable to the sample shape. Defaults to None.
            device: The device on which to generate the samples, specified as a string or a :class:`torch.device`. Defaults to "cpu".
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

        Returns:
            A tensor of samples drawn from the prior distribution.
        """
        samples = (
            torch.multinomial(self.prior_dist, math.prod(shape), replacement=True, generator=rng_generator)
            .to(device)
            .reshape(shape)
        )
        if mask is not None:
            samples = samples * mask[(...,) + (None,) * (len(samples.shape) - len(mask.shape))]
        return samples
