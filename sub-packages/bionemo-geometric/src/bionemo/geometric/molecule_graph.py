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
from torch_geometric.data import Data


class MoleculeGraph(Data):
    """A torch_geometric.data.Data class for storing and processing molecular graphs."""

    def __init__(self, x_n: torch.tensor, edges: torch.tensor, x_e, x_g: torch.tensor):
        """Initializes MoleculeGraph data point object.

        Initializes MoleculeGraph data point with node attributes `x_n`, edges, edge attributes `e_n` and graph attributes `x_g`
        """
        self.x_g = x_g
        super().__init__(x_n, x_n, x_e)
