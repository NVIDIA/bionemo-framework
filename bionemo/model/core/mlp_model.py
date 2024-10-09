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


from typing import List, Optional

import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(
        self,
        layer_sizes: Optional[List[int]] = None,
        dropout: float = 0.25,
        activation_function: Optional[nn.Module] = None,
    ):
        """
        Simple MLP Model for validation on benchmark datasets

        Params
            layer_sizes: List of layer sizes. By default: [256, 128, 1]
            dropout: float
            activation_function: PyTorch activation function. Uses ReLU if not provided
        """
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [256, 128, 1]
        self.linear_layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.layer_norm = nn.LayerNorm(layer_sizes[0])
        self.act = nn.ReLU() if activation_function is None else activation_function
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for layer in self.linear_layers[:-1]:
            x = self.dropout(self.act(layer(x)))

        x = self.linear_layers[-1](x)
        return x
