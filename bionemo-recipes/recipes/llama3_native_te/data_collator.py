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

"""Data collator for genomic sequence training with custom masking.

This module provides a collator that wraps any base collator and adds genomic masking:
- Uppercase labels (while keeping inputs mixed case)
- Mask degenerate bases (non-ACGT characters)
- Mask control characters (@, #)

The composition design allows easy extension to THD and other formats.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
from genomic_masking_functions import Evo2MaskingConstants, make_upper_case, mask_phylogenetic_tags


logger = logging.getLogger(__name__)


@dataclass
class GenomicDataCollator:
    """Wrapper collator that adds genomic-specific masking to any base collator.

    This collator uses composition to wrap any base collator (BSHD, THD, etc.) and
    applies genomic masking to the labels after batching.

    Args:
        base_collator: The underlying collator (e.g., DataCollatorForLanguageModeling)
        uppercase_labels: Whether to uppercase labels. Default: False.
        mask_degenerate_bases: Whether to mask non-ACGT bases. Default: True.
        mask_phylo_tags: Whether to mask phylogenetic tags. Default: False (Milestone 2).
        dna_tokens: Tuple of valid DNA token IDs (A, C, G, T upper+lowercase)
        control_tags: Tuple of control character token IDs (@, #)

    Example:
        >>> from transformers.data.data_collator import DataCollatorForLanguageModeling
        >>> base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        >>> collator = GenomicDataCollator(
        ...     base_collator=base,
        ...     uppercase_labels=False,
        ...     mask_degenerate_bases=True,
        ...     mask_phylo_tags=False,
        ... )
    """

    base_collator: Any
    uppercase_labels: bool = False
    mask_degenerate_bases: bool = True
    mask_phylo_tags: bool = False
    dna_tokens: tuple[int, ...] = (65, 67, 71, 84, 97, 99, 103, 116)  # A, C, G, T (upper+lower)
    control_tags: tuple[int, ...] = (64, 35)  # '@', '#'

    def __call__(self, features: list) -> dict[str, Any]:
        """Apply base collator, then add genomic masking.

        Order of operations (IMPORTANT):
        1. Mask degenerate bases (simple character check)
        2. Mask phylogenetic tags (needs lowercase to detect!)
        3. Uppercase labels (after detection, since phylo relies on case)
        """
        # Base collator handles batching and CLM label creation
        batch = self.base_collator(features)

        labels = batch["labels"]

        # Step 1: Mask degenerate bases and control characters
        if self.mask_degenerate_bases:
            dna_tokens_tensor = torch.tensor(self.dna_tokens, device=labels.device)
            control_tensor = torch.tensor(self.control_tags, device=labels.device)

            # Identify non-DNA tokens
            not_dna = ~torch.isin(labels, dna_tokens_tensor)
            is_control = torch.isin(labels, control_tensor)

            # Mask both, but preserve existing -100 values
            labels[(not_dna | is_control) & (labels != -100)] = -100

        # Step 2: Mask phylogenetic tags (BEFORE uppercase!)
        # Phylo detection relies on lowercase letters after '|' to identify tags
        if self.mask_phylo_tags:
            phylo_mask = mask_phylogenetic_tags(
                tokenized_sequence=labels,
                terminal_tag_char=Evo2MaskingConstants.TAG_BOUNDS,
                other_tag_chars=Evo2MaskingConstants.TAG_CHARS,
                eod_token_id=Evo2MaskingConstants.DEFAULT_EOD,
                max_tag_len=Evo2MaskingConstants.MAX_TAG_LEN,
            )
            # Where mask is 0, set label to -100 (but preserve existing -100)
            labels[(phylo_mask == 0) & (labels != -100)] = -100

        # Step 3: Uppercase labels (AFTER phylo detection!)
        if self.uppercase_labels:
            labels, _ = make_upper_case(labels)

        batch["labels"] = labels
        return batch
