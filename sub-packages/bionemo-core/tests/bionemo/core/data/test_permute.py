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

from bionemo.core.data.permute import permute


def test_permute_with_invalid_range():
    with pytest.raises(ValueError, match="The length of the permuted range must be greater than 1."):
        permute(0, 0, 0)


def test_permute_with_index_greater_than_range():
    with pytest.raises(ValueError, match="The index to permute must be in the range"):
        permute(10, 5, 1)


def test_permute_with_invalid_seed():
    with pytest.raises(ValueError, match="The permutation seed must be greater than or equal to 0."):
        permute(5, 10, -1)


def test_permute_valid_permutation():
    result = permute(5, 10, 1)
    assert 0 <= result < 10


def test_permute_is_exhaustive():
    l = 100
    p = 42
    indices = {permute(i, l, p) for i in range(l)}
    assert len(indices) == l


def test_permute_seed_changes_order():
    l = 100
    indices_42 = tuple(permute(i, l, 42) for i in range(l))
    indices_24 = tuple(permute(i, l, 24) for i in range(l))
    assert indices_24 != indices_42
