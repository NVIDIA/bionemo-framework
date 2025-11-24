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

"""Data collators for genomic sequence training with custom masking.

This module provides collators for both BSHD (standard batching) and THD (sequence packing)
formats, with shared genomic-specific masking logic:
- Uppercase labels (while keeping inputs mixed case)
- Mask degenerate bases (non-ACGT characters)
- Mask control characters (@, #)
- Mask phylogenetic tags (|d__domain;p__phylum|, etc.)

The masking functions are imported from NeMo's Evo2 implementation for consistency
with production genomic models.
"""

import logging
from dataclasses import dataclass
from typing import ClassVar

import datasets
import torch
from genomic_utils import Evo2MaskingConstants, make_upper_case, mask_phylogenetic_tags
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling, DataCollatorWithFlattening


logger = logging.getLogger(__name__)


class GenomicMaskingMixin:
    """Mixin providing shared genomic masking logic for both BSHD and THD collators.

    This mixin implements the masking strategy used in Evo2 training:
    1. Uppercase labels (model learns to translate mixed-case input to uppercase output)
    2. Mask degenerate/ambiguous bases (N, R, Y, K, M, S, W, etc.) - model doesn't predict these
    3. Mask control characters (@, #) - special tokens for data formatting
    4. Mask phylogenetic tags (|d__domain;...| patterns) - metadata, not DNA

    All masking is applied to labels (targets), not inputs. Masked positions are set to -100,
    which PyTorch's CrossEntropyLoss ignores by default.

    Constants are from Evo2Dataset for consistency with production models.
    """

    # Standard DNA tokens: A, C, G, T (both uppercase and lowercase)
    DNA_TOKENS: ClassVar[list[int]] = [65, 67, 71, 84, 97, 99, 103, 116]

    # Control characters used in data formatting
    CONTROL_TAGS: ClassVar[list[int]] = [64, 35]  # '@' (splice/window markers), '#' (contig splits)

    def apply_genomic_masking(
        self,
        labels: torch.Tensor,
        uppercase: bool = True,
        mask_degenerate: bool = True,
        mask_phylo: bool = False,
    ) -> torch.Tensor:
        """Apply all genomic-specific masking transformations to labels.

        This function works on any tensor shape: [B, S] (BSHD) or [1, T] (THD).

        Args:
            labels: Label tensor to mask. Shape: [batch_size, seq_length] or [1, total_tokens]
            uppercase: Whether to uppercase labels (default: True)
            mask_degenerate: Whether to mask non-ACGT bases (default: True)
            mask_phylo: Whether to mask phylogenetic tags (default: False, enable for Milestone 2)

        Returns:
            Modified labels tensor with masking applied.
        """
        # Step 1: Uppercase labels (inputs stay mixed case)
        # This teaches the model to translate: lowercase input → uppercase output
        if uppercase:
            labels, _ = make_upper_case(labels)

        # Step 2: Mask degenerate bases and control characters
        # Only keep standard DNA tokens (A, C, G, T in upper/lowercase)
        if mask_degenerate:
            dna_tokens_tensor = torch.tensor(self.DNA_TOKENS, device=labels.device)
            control_tensor = torch.tensor(self.CONTROL_TAGS, device=labels.device)

            # Identify non-DNA tokens (degenerate bases like N, R, Y, K, M, S, W, etc.)
            not_dna = ~torch.isin(labels, dna_tokens_tensor)
            # Identify control characters (@, #)
            is_control = torch.isin(labels, control_tensor)

            # Mask both, but preserve existing -100 values (padding, etc.)
            labels[(not_dna | is_control) & (labels != -100)] = -100

        # Step 3: Mask phylogenetic tags (e.g., |d__Bacteria;p__Proteobacteria|)
        # NOTE: This must be done AFTER uppercasing, as the function relies on original case
        # to detect tag boundaries (lowercase 'd' after pipe indicates tag start)
        if mask_phylo:
            phylo_mask = mask_phylogenetic_tags(
                tokenized_sequence=labels,
                terminal_tag_char=Evo2MaskingConstants.TAG_BOUNDS,  # '|' pipe character
                other_tag_chars=Evo2MaskingConstants.TAG_CHARS,  # '_', ';', space
                eod_token_id=Evo2MaskingConstants.DEFAULT_EOD,  # End-of-document token
                max_tag_len=Evo2MaskingConstants.MAX_TAG_LEN,  # Maximum tag length
            )
            # Where mask is 0, set label to -100 (ignore in loss)
            labels[phylo_mask == 0] = -100

        return labels


class GenomicDataCollatorForCLM(DataCollatorForLanguageModeling, GenomicMaskingMixin):
    """Data collator for genomic CLM training in BSHD format (standard batching with padding).

    This collator:
    1. Uses standard DataCollatorForLanguageModeling for batching/padding
    2. Applies genomic-specific masking via GenomicMaskingMixin
    3. Produces BSHD format: [batch_size, seq_length]

    Use this collator for:
    - Milestone 1 (metagenomics): uppercase_labels=True, mask_degenerate_bases=True
    - Milestone 2 (full dataset): Also enable mask_phylo_tags=True
    - Testing and debugging (simpler format than THD)

    Args:
        tokenizer: The tokenizer for the genomic sequences.
        uppercase_labels: Whether to uppercase labels. Default: True.
        mask_degenerate_bases: Whether to mask non-ACGT bases. Default: True.
        mask_phylo_tags: Whether to mask phylogenetic tags. Default: False (enable for Milestone 2).
        **kwargs: Additional arguments passed to DataCollatorForLanguageModeling.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("path/to/nucleotide/tokenizer")
        >>>
        >>> collator = GenomicDataCollatorForCLM(
        ...     tokenizer=tokenizer,
        ...     uppercase_labels=True,
        ...     mask_degenerate_bases=True,
        ...     mask_phylo_tags=False,  # Milestone 1: metagenomics only
        ... )
        >>>
        >>> # Features from dataset
        >>> features = [
        ...     {"input_ids": [97, 67, 103, 84, 78]},  # "aCgtN" (mixed case, has degenerate)
        ...     {"input_ids": [71, 82, 65, 65]},       # "GRAA" (has degenerate R)
        ... ]
        >>>
        >>> batch = collator(features)
        >>> # batch["input_ids"]: Mixed case preserved
        >>> # batch["labels"]: Uppercase, with N and R masked as -100
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        uppercase_labels: bool = True,
        mask_degenerate_bases: bool = True,
        mask_phylo_tags: bool = False,
        **kwargs,
    ):
        """Initialize the genomic data collator for BSHD format."""
        super().__init__(tokenizer=tokenizer, mlm=False, **kwargs)  # mlm=False for CLM
        self.uppercase_labels = uppercase_labels
        self.mask_degenerate_bases = mask_degenerate_bases
        self.mask_phylo_tags = mask_phylo_tags

    def __call__(self, features):
        """Apply standard batching, then genomic masking.

        Args:
            features: List of dicts with 'input_ids' and optionally 'labels'

        Returns:
            Batch dict with genomic masking applied to labels.
        """
        # Parent handles batching and CLM label creation
        batch = super().__call__(features)

        # Apply genomic masking to labels
        batch["labels"] = self.apply_genomic_masking(
            labels=batch["labels"],
            uppercase=self.uppercase_labels,
            mask_degenerate=self.mask_degenerate_bases,
            mask_phylo=self.mask_phylo_tags,
        )

        return batch


class GenomicCLMCollatorWithFlattening(DataCollatorWithFlattening, GenomicMaskingMixin):
    """Data collator for genomic CLM training in THD format (sequence packing).

    This collator:
    1. Uses DataCollatorWithFlattening for sequence packing
    2. Applies genomic-specific masking via GenomicMaskingMixin
    3. Produces THD format: [1, total_tokens] with cu_seq_lens

    Use this collator for:
    - Efficient training with variable-length sequences
    - Reduces padding waste
    - Enables Flash Attention optimizations

    The masking logic is identical to GenomicDataCollatorForCLM, but works on flattened format.

    Args:
        tokenizer: The tokenizer for the genomic sequences.
        uppercase_labels: Whether to uppercase labels. Default: True.
        mask_degenerate_bases: Whether to mask non-ACGT bases. Default: True.
        mask_phylo_tags: Whether to mask phylogenetic tags. Default: False (enable for Milestone 2).
        pad_to_multiple_of: Pad total tokens to multiple of this value for efficiency. Default: None.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("path/to/nucleotide/tokenizer")
        >>>
        >>> collator = GenomicCLMCollatorWithFlattening(
        ...     tokenizer=tokenizer,
        ...     uppercase_labels=True,
        ...     mask_degenerate_bases=True,
        ...     pad_to_multiple_of=16,  # For FP8 training
        ... )
        >>>
        >>> # Features from dataset
        >>> features = [
        ...     {"input_ids": [97, 67, 103, 84]},    # "aCgt" (4 tokens)
        ...     {"input_ids": [71, 82, 65, 65, 67]}, # "GRAAC" (5 tokens)
        ... ]
        >>>
        >>> batch = collator(features)
        >>> # batch["input_ids"].shape: [1, 9] (flattened)
        >>> # batch["cu_seq_lens_q"]: [0, 4, 9] (sequence boundaries)
        >>> # batch["labels"]: Uppercase, R masked as -100
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        uppercase_labels: bool = True,
        mask_degenerate_bases: bool = True,
        mask_phylo_tags: bool = False,
        pad_to_multiple_of: int | None = None,
    ):
        """Initialize the genomic data collator for THD format."""
        # DataCollatorWithFlattening doesn't take tokenizer in __init__,
        # but we need it for decoding. Store it manually.
        self.tokenizer = tokenizer

        super().__init__(
            return_position_ids=True,
            separator_id=-100,  # Standard ignore index for labels
            return_flash_attn_kwargs=True,  # Returns cu_seq_lens_q/k, max_length_q/k
            return_seq_idx=True,  # Track which sequence each token belongs to
        )
        self.uppercase_labels = uppercase_labels
        self.mask_degenerate_bases = mask_degenerate_bases
        self.mask_phylo_tags = mask_phylo_tags
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features, return_tensors=None):
        """Apply sequence packing (flattening), then genomic masking.

        Args:
            features: List of dicts with 'input_ids' and optionally 'labels'
            return_tensors: Tensor format (default: "pt")

        Returns:
            Batch dict with genomic masking applied to labels in THD format.
        """
        # Parent handles flattening and CLM label creation
        batch = super().__call__(features, return_tensors)

        # Apply genomic masking to flattened labels
        batch["labels"] = self.apply_genomic_masking(
            labels=batch["labels"],
            uppercase=self.uppercase_labels,
            mask_degenerate=self.mask_degenerate_bases,
            mask_phylo=self.mask_phylo_tags,
        )

        return batch


@dataclass
class TokenPackingDataset(torch.utils.data.IterableDataset):
    """Dataset that uses sequence packing to construct batches with variable length up to a maximum number of tokens.

    Adapted from ESM2's TokenPackingDataset implementation.
    """

    dataset: datasets.IterableDataset
    """Dataset to pack."""
    max_tokens_per_batch: int
    """Maximum number of tokens per batch."""
    drop_last: bool = True
    """Whether to drop the last batch if it's less than max_length."""

    def __iter__(self):
        """Yield batches of samples, each with a variable number of tokens up to the maximum length.

        Returns:
            A generator of batches of samples, each with a variable number of tokens up to the maximum length.
        """
        samples = []
        current_length = 0
        for sample in iter(self.dataset):
            current_length += len(sample["input_ids"])
            if current_length > self.max_tokens_per_batch:
                yield samples
                samples = [sample]
                current_length = len(sample["input_ids"])
            else:
                samples.append(sample)

        if not self.drop_last and samples:
            yield samples

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset."""
        self.dataset.set_epoch(epoch)
