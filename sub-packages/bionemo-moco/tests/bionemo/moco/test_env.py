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
import torch.nn as nn


def test_torch_import():
    assert torch is not None


def test_gpu_availability():
    assert torch.cuda.is_available()


def test_tensor_creation_on_gpu():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor = torch.randn(2, 2, device=device)
    assert tensor.is_cuda


def test_loss_calculation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(2, 2, device=device)
    target_tensor = torch.randn(2, 2, device=device)
    criterion = nn.MSELoss()
    loss = criterion(input_tensor, target_tensor)
    assert loss is not None


def test_backpropagation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(2, 2, device=device, requires_grad=True)
    target_tensor = torch.randn(2, 2, device=device)
    criterion = nn.MSELoss()
    loss = criterion(input_tensor, target_tensor)
    loss.backward()
    assert input_tensor.grad is not None
