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

from typing import Sequence


__all__: Sequence[str] = ("ask_yes_or_no",)


def ask_yes_or_no(message: str) -> bool:
    """Prompt user via STDIN for a boolean response: 'yes'/'y' is True and 'no'/'n' is False.

    Note that the input gathered from STDIN is stripped of all surrounding whitespace and converted to lowercase.
    While the user is prompted on STDOUT to supply 'y' or 'n', note that 'yes' and 'no' are accepted, respectively.
    An affirmative response ('yes' or 'y') will result in True being returned. A negative response ('no' or 'n')
    results in a False being returned.

    This function loops forever until it reads an unambiguous affirmative ('y') or negative ('n') response via STDIN.

    Args:
        message: Added to the STDOUT prompt for the user.

    Returns:
        True if user responds in the affirmative via STDIN. False if user responds in the negative.

    Raises:
        ValueError iff message is the empty string or only consists of whitespace.
    """
    if len(message) == 0 or len(message.strip()) == 0:
        raise ValueError("Must supply non-empty message for STDOUT user prompt.")

    while True:
        print(f"{message} [y/n]\n>> ", end="")
        response = input().strip().lower()
        match response:
            case "y" | "yes":
                return True
            case "n" | "no":
                return False
            case _:
                print(f'😱 ERROR: must supply "y" or "n", not "{response}". Try again!\n')
