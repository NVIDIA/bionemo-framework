# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
