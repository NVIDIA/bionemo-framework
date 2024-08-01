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

from bionemo.core.utils.dtypes import get_autocast_dtype


@pytest.mark.parametrize(
    "precision, expected_dtype",
    [
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
        ("fp32", torch.float32),
        ("bf16-mixed", torch.bfloat16),
        ("fp32-mixed", torch.float32),
    ],
)
def test_get_autocast_dtype(precision: str, expected_dtype: torch.dtype):
    assert get_autocast_dtype(precision) == expected_dtype


def test_unsupported_autocast_dtype():
    with pytest.raises(ValueError):
        get_autocast_dtype("unsupported")
