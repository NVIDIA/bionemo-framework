# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Genomic sequence utilities for data preprocessing and masking.

Functions adapted from NeMo's Evo2 implementation for genomic sequence training.
"""

from typing import ClassVar

import torch


def make_upper_case(tokens, lowercase_start=97, lowercase_end=122, case_diff=32):
    """Replace lowercase ASCII characters with uppercase.

    Adapted from: nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils.make_upper_case

    Args:
        tokens: Input tensor containing token IDs (ASCII values)
        lowercase_start: ASCII value for 'a' (default: 97)
        lowercase_end: ASCII value for 'z' (default: 122)
        case_diff: Difference between lowercase and uppercase (default: 32)

    Returns:
        tuple: (uppercase_tensor, lowercase_mask)
    """
    lowercase_mask = (tokens >= lowercase_start) & (tokens <= lowercase_end)
    uppercase_tensor = torch.where(lowercase_mask, tokens - case_diff, tokens)
    return uppercase_tensor, lowercase_mask


class Evo2MaskingConstants:
    """Constants used in Evo2 genomic sequence masking."""

    # Standard DNA tokens: A, C, G, T (both uppercase and lowercase)
    DNA_TOKENS: ClassVar[list[int]] = [65, 67, 71, 84, 97, 99, 103, 116]

    # Control characters used in data formatting
    CONTROL_TAGS: ClassVar[list[int]] = [64, 35]  # '@', '#'
