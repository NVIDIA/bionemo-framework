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


# Copyright (c) 2023, NVIDIA CORPORATION.
"""This file tests some of the utility functions that are used during unit tests."""

import torch

from bionemo.utils.tests import (
    list_to_tensor,
)


def test_list_to_tensor_simple_list():
    data = [1, 2, 3, 4, 5]
    tensor = list_to_tensor(data)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.tolist() == data


def test_list_to_tensor_nested_list():
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    tensor = list_to_tensor(data)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.tolist() == data


def test_list_to_tensor_mixed_list():
    data = [1, 2, [3, 4, 5], [6, [7, 8], 9]]
    tensor = list_to_tensor(data)
    assert isinstance(tensor, list)
    assert isinstance(tensor[3], list)


def test_list_to_tensor_non_list_input():
    data = 42
    tensor = list_to_tensor(data)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.item() == data
