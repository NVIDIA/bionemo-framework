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


import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch_scatter import scatter_mean, scatter_softmax


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


@pytest.mark.needs_gpu
def test_backward_with_scatter():
    """
    Test function for performing a backward pass with scatter operations to catch errors with
    torch-scatter versioning. Passes with v2.1.2 and Fails with v2.0.9.

    This function does the following steps:
    1. Creates a random input tensor.
    2. Moves the input tensor to GPU if available.
    3. Instantiates a Multi-Layer Perceptron (MLP) model and moves it to GPU.
    4. Performs a forward pass through the MLP.
    5. Performs scatter softmax along dimension 0 on the output.
    6. Creates a random target tensor.
    7. Calculates the mean squared error loss with log softmax.
    8. Calculates the mean loss with scatter mean.
    9. Performs a backward pass.
    10. Checks that gradients are not None for all model parameters.
    11. Optionally, performs an optimization step using stochastic gradient descent (SGD).

    Returns:
        None
    """
    # Create random input tensor
    input_tensor = torch.randn(64, 100)  # Assuming input size of 100

    # Move input tensor to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)

    # Instantiate MLP and move it to GPU
    model = MLP()
    model.to(device)

    # Forward pass through the MLP
    output = model(input_tensor)
    # Perform scatter softmax along dimension 0
    softmax_output = scatter_softmax(output, torch.tensor([0], device="cuda"), dim=0)

    # Create target tensor (random for demonstration)
    target = torch.randn((output.size(0), output.size(1)), device="cuda")  # Assuming output size of 10

    # Calculate negative log likelihood loss with log_softmax
    loss = nn.functional.mse_loss(torch.log(softmax_output), target, reduction="none")

    # Calculate mean loss with scatter_mean
    mean_loss = scatter_mean(loss, torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 64).view(64, 10).cuda(), dim=0).mean()

    # Backward pass
    mean_loss.backward()

    # Check that gradients are not None
    assert all(param.grad is not None for param in model.parameters())

    # Optionally, perform optimization step
    optimizer = SGD(model.parameters(), lr=0.01)
    optimizer.step()
