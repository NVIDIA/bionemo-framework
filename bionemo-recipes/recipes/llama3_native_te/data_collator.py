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

"""Data collator for genomic sequence training with custom masking (BSHD format only).

This module provides a simple collator for standard batching (BSHD format) with:
- Uppercase labels (while keeping inputs mixed case)
- Mask degenerate bases (non-ACGT characters)
- Mask control characters (@, #)
"""

import logging
from typing import ClassVar

import torch
from genomic_utils import make_upper_case
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling


logger = logging.getLogger(__name__)


class GenomicDataCollatorForCLM(DataCollatorForLanguageModeling):
    """Data collator for genomic CLM training in BSHD format (standard batching with padding).

    This collator applies genomic-specific masking:
    1. Uppercase labels (model learns to translate mixed-case input to uppercase output)
    2. Mask degenerate/ambiguous bases (N, R, Y, etc.)
    3. Mask control characters (@, #)

    Args:
        tokenizer: The tokenizer for the genomic sequences.
        uppercase_labels: Whether to uppercase labels. Default: True.
        mask_degenerate_bases: Whether to mask non-ACGT bases. Default: True.

    Example:
        >>> collator = GenomicDataCollatorForCLM(
        ...     tokenizer=tokenizer,
        ...     uppercase_labels=True,
        ...     mask_degenerate_bases=True,
        ... )
    """

    # Standard DNA tokens: A, C, G, T (both uppercase and lowercase)
    DNA_TOKENS: ClassVar[list[int]] = [65, 67, 71, 84, 97, 99, 103, 116]

    # Control characters used in data formatting
    CONTROL_TAGS: ClassVar[list[int]] = [64, 35]  # '@', '#'

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        uppercase_labels: bool = True,
        mask_degenerate_bases: bool = True,
        **kwargs,
    ):
        """Initialize the genomic data collator for BSHD format."""
        super().__init__(tokenizer=tokenizer, mlm=False, **kwargs)
        self.uppercase_labels = uppercase_labels
        self.mask_degenerate_bases = mask_degenerate_bases

    def __call__(self, features):
        """Apply standard batching, then genomic masking."""
        # Parent handles batching and CLM label creation
        batch = super().__call__(features)

        labels = batch["labels"]

        # Step 1: Uppercase labels (inputs stay mixed case)
        if self.uppercase_labels:
            labels, _ = make_upper_case(labels)

        # Step 2: Mask degenerate bases and control characters
        if self.mask_degenerate_bases:
            dna_tokens_tensor = torch.tensor(self.DNA_TOKENS, device=labels.device)
            control_tensor = torch.tensor(self.CONTROL_TAGS, device=labels.device)

            # Identify non-DNA tokens
            not_dna = ~torch.isin(labels, dna_tokens_tensor)
            is_control = torch.isin(labels, control_tensor)

            # Mask both, but preserve existing -100 values
            labels[(not_dna | is_control) & (labels != -100)] = -100

        batch["labels"] = labels
        return batch
