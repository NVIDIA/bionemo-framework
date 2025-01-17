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


import torch
from torch import Tensor


def safe_index(tensor: Tensor, index: Tensor, device: torch.device):
    """Safely indexes a tensor using a given index and returns the result on a specified device.

    Note can implement forcing with  return tensor[index.to(tensor.device)].to(device) but has costly migration.

    Args:
        tensor (Tensor): The tensor to be indexed.
        index (Tensor): The index to use for indexing the tensor.
        device (torch.device): The device on which the result should be returned.

    Returns:
        Tensor: The indexed tensor on the specified device.

    Raises:
        ValueError: If tensor, index, and device are not all on the same device.
    """
    if not (tensor.device == index.device == device):
        raise ValueError(
            f"Tensor, index, and device must all be on the same device. "
            f"Got tensor.device={tensor.device}, index.device={index.device}, and device={device}."
        )

    return tensor[index]
