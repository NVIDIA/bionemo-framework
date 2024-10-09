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
from torch.utils.data import TensorDataset

from bionemo.data.mapped_dataset import SliceDataset


test_dataset = TensorDataset(torch.arange(24 * 3).reshape(24, 3))


start_end_length = [
    (0, 9, 9),
    (1, 7, 6),
    (3, 3, 0),
    (9, 24, 15),
    (3, -1, 20),
    (-5, -2, 3),
]


@pytest.mark.parametrize("start,end,length", start_end_length)
def test_slice_dataset(start, end, length):
    dataset = SliceDataset(test_dataset, start, end)
    assert len(dataset) == length
