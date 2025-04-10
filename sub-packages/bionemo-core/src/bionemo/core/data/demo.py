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


def some_function_with_test_coverage(input_value: int) -> int:
    """A demo function with branching logic to demonstrate test coverage.

    Args:
        input_value: An integer input to demonstrate branching logic

    Returns:
        An integer result after applying branching logic
    """
    if input_value < 0:
        return -1
    elif input_value == 0:
        return 0
    else:
        if input_value > 100:
            return 100
        elif input_value % 2 == 0:
            return input_value * 2
        else:
            return input_value + 1
