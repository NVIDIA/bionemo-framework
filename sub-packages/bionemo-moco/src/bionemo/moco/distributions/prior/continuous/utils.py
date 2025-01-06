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


from typing import Optional

from torch import Tensor


def remove_center_of_mass(data: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Calculates the center of mass (CoM) of the given data.

    Args:
        data: The input data with shape (..., nodes, features).
        mask: An optional binary mask to apply to the data with shape (..., nodes) to mask out interaction from CoM calculation. Defaults to None.

    Returns:
    The CoM of the data with shape (..., 1, features).
    """
    if mask is None:
        com = data.mean(dim=-2, keepdim=True)
    else:
        masked_data = data * mask.unsqueeze(-1)
        num_nodes = mask.sum(dim=-1, keepdim=True).unsqueeze(-1)
        com = masked_data.sum(dim=-2, keepdim=True) / num_nodes
    return data - com
