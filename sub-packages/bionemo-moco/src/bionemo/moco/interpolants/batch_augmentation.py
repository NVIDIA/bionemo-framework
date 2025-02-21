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


from bionemo.moco.interpolants.continuous_time.continuous.data_augmentation.augmentation_types import AugmentationType
from bionemo.moco.interpolants.continuous_time.continuous.data_augmentation.equivariant_ot_sampler import (
    EquivariantOTSampler,
)
from bionemo.moco.interpolants.continuous_time.continuous.data_augmentation.kabsch_augmentation import (
    KabschAugmentation,
)
from bionemo.moco.interpolants.continuous_time.continuous.data_augmentation.ot_sampler import OTSampler


class BatchDataAugmentation:
    """Facilitates the creation of batch augmentation objects based on specified optimal transport types.

    Args:
        device (str): The device to use for computations (e.g., 'cpu', 'cuda').
        num_threads (int): The number of threads to utilize.
    """

    def __init__(self, device, num_threads):
        """Initializes a BatchAugmentation instance.

        Args:
            device (str): Device for computation.
            num_threads (int): Number of threads to use.
        """
        self.device = device
        self.num_threads = num_threads

    def create(self, method_type: AugmentationType):
        """Creates a batch augmentation object of the specified type.

        Args:
            method_type (AugmentationType): The type of optimal transport method.

        Returns:
            The augmentation object if the type is supported, otherwise **None**.
        """
        if method_type == AugmentationType.EXACT_OT:
            augmentation = OTSampler(method="exact", device=self.device, num_threads=self.num_threads)
        elif method_type == AugmentationType.KABSCH:
            augmentation = KabschAugmentation()
        elif method_type == AugmentationType.EQUIVARIANT_OT:
            augmentation = EquivariantOTSampler(method="exact", device=self.device, num_threads=self.num_threads)
        else:
            return None
        return augmentation
