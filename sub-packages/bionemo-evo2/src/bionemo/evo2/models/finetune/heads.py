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

"""Task-specific heads for Evo2 fine-tuning."""

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class MegatronMLPHead(MegatronModule):
    """MLP head for sequence-level tasks.

    This head consists of linear layers with ReLU activation and dropout,
    suitable for both regression and classification tasks.
    """

    def __init__(self, config: TransformerConfig):
        """Initialize the MLP head.

        Args:
            config: TransformerConfig containing hidden_size, mlp_hidden_size,
                   mlp_target_size, and ft_dropout parameters
        """
        super().__init__(config)
        layer_sizes = [config.hidden_size, config.mlp_hidden_size, config.mlp_target_size]
        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])]
        )
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=config.ft_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP head.

        Args:
            hidden_states: Input tensor of shape [batch_size, hidden_size]

        Returns:
            Output tensor of shape [batch_size, mlp_target_size]
        """
        for layer in self.linear_layers[:-1]:
            hidden_states = self.dropout(self.act(layer(hidden_states)))
        output = self.linear_layers[-1](hidden_states)
        return output


class MegatronConvHead(MegatronModule):
    """1D Convolutional head for token-level tasks.

    This head uses 1D convolutions to process sequence features,
    suitable for per-token classification tasks.
    """

    def __init__(self, config: TransformerConfig):
        """Initialize the convolutional head.

        Args:
            config: TransformerConfig containing hidden_size, cnn_hidden_size,
                   cnn_num_classes, and cnn_dropout parameters
        """
        super().__init__(config)
        self.conv1 = torch.nn.Conv1d(config.hidden_size, config.cnn_hidden_size, kernel_size=5, padding=2)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=config.cnn_dropout)
        self.conv2 = torch.nn.Conv1d(config.cnn_hidden_size, config.cnn_num_classes, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolutional head.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor of shape [batch_size, seq_len, num_classes]
        """
        # hidden_states: [b, s, h] -> [b, h, s]
        x = hidden_states.transpose(1, 2)
        x = self.dropout(self.act(self.conv1(x)))
        x = self.conv2(x)
        # [b, num_classes, s] -> [b, s, num_classes]
        return x.transpose(1, 2)
