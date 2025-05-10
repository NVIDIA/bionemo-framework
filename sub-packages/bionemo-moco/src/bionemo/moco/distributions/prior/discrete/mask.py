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


class DiscreteMaskedPrior(DiscretePriorDistribution):
    """A subclass representing a Discrete Masked prior distribution."""

    def __init__(self, num_classes: int = 10, mask_dim: Optional[int] = None, inclusive: bool = True) -> None:
        """Discrete Masked prior distribution.

        Theres 3 ways I can think of defining the problem that are hard to mesh together.

        1. [..., M, ....] inclusive anywhere --> exisiting LLM tokenizer where the mask has a specific location not at the end
        2. [......, M] inclusive on end --> mask_dim = None with inclusive set to True default stick on the end
        3. [.....] + [M] exclusive --> the number of classes representes the number of data classes and one wishes to add a separate MASK dimension.
            - Note the pad_sample function is provided to help add this extra external dimension.

        Args:
            num_classes (int): The number of classes in the distribution. Defaults to 10.
            mask_dim (int): The index for the mask token. Defaults to num_classes - 1 if inclusive or num_classes if exclusive.
            inclusive (bool): Whether the mask is included in the specified number of classes.
                                If True, the mask is considered as one of the classes.
                                If False, the mask is considered as an additional class. Defaults to True.
        """
        if inclusive:
            if mask_dim is None:
                mask_dim = num_classes - 1
            else:
                if mask_dim >= num_classes:
                    raise ValueError(
                        "As Inclusive accounts for the mask as one of the specified num_classes, the provided mask_dim cannot be >= to num_classes"
                    )
            prior_dist = torch.zeros((num_classes))
            prior_dist[-1] = 1.0
            super().__init__(num_classes, prior_dist)
            self.mask_dim = mask_dim
        else:
            prior_dist = torch.zeros((num_classes + 1))
            prior_dist[-1] = 1.0
            super().__init__(num_classes + 1, prior_dist)
            self.mask_dim = num_classes
        if torch.sum(self.prior_dist).item() - 1.0 >= 1e-5:
            raise ValueError("Invalid probability distribution. Must sum to 1.0")

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
        samples = torch.ones(shape, dtype=torch.int64, device=device) * self.mask_dim
        if mask is not None:
            samples = samples * mask[(...,) + (None,) * (len(samples.shape) - len(mask.shape))]
        return samples

    def is_masked(self, sample: Tensor) -> Tensor:
        """Creates a mask for whether a state is masked.

        Args:
            sample (Tensor): The sample to check.

        Returns:
            Tensor: A float tensor indicating whether the sample is masked.
        """
        return (sample == self.mask_dim).float()

    def pad_sample(self, sample: Tensor) -> Tensor:
        """Pads the input sample with zeros along the last dimension.

        Args:
            sample (Tensor): The input sample to be padded.

        Returns:
            Tensor: The padded sample.
        """
        # Create a zeros tensor with the same shape as the original tensor, except the last dimension is 1
        zeros = torch.zeros((*sample.shape[:-1], 1), dtype=torch.float, device=sample.device)
        # Concatenate along the last dimension to make the shape (..., N+1)
        padded_sample = torch.cat((sample, zeros), dim=-1)
        return padded_sample
