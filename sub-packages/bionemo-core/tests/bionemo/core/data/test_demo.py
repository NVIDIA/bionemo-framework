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


from bionemo.core.data.demo import some_function_with_test_coverage


def test_negative_input():
    """Test that negative input returns -1"""
    assert some_function_with_test_coverage(-5) == -1
    assert some_function_with_test_coverage(-1) == -1


def test_large_input():
    """Test that input > 100 returns 100"""
    assert some_function_with_test_coverage(101) == 100
    assert some_function_with_test_coverage(200) == 100


def test_even_input():
    """Test that even input between 0 and 100 returns double the input"""
    assert some_function_with_test_coverage(2) == 4
    assert some_function_with_test_coverage(50) == 100
