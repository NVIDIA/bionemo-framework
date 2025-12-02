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

"""Genomic sequence masking functions for data preprocessing.

Core functions for genomic data preprocessing during training:
- make_upper_case: Convert lowercase tokens to uppercase
- Evo2MaskingConstants: Standard DNA tokens and control characters

Adapted from NeMo's Evo2 implementation.
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


def mask_phylogenetic_tags(  # noqa: C901
    tokenized_sequence: torch.Tensor,
    terminal_tag_char: int = 124,  # '|' (pipe)
    other_tag_chars: set[int] | None = None,  # '_', ';', space
    eod_token_id: int = 0,
    max_tag_len: int = 2048,
) -> torch.Tensor:
    """Create a binary mask for sequences containing phylogenetic tags.

    Adapted from: nemo.collections.llm.gpt.data.megatron.hyena.evo2_dataset.Evo2Dataset.mask_phylogenetic_tags

    Phylogenetic tags have format: |d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria|

    Detection rules:
    - Tags are enclosed in pipes (|)
    - Contain taxonomy separators (_, ;, space)
    - Start with lowercase letter after pipe (d__, p__, c__, etc.)

    Args:
        tokenized_sequence: Token IDs. Shape: (seq_len,) or (batch_size, seq_len)
        terminal_tag_char: ASCII for '|' (default: 124)
        other_tag_chars: ASCII values for tag separators (default: {95, 59, 32} = '_', ';', space)
        eod_token_id: End-of-document token (default: 0)
        max_tag_len: Maximum tag length (default: 2048)

    Returns:
        Binary mask: 1 = keep (DNA), 0 = mask (tag). Same shape as input.
    """
    if other_tag_chars is None:
        other_tag_chars = {95, 59, 32}  # '_', ';', space

    device = tokenized_sequence.device

    # Handle empty sequences
    if tokenized_sequence.numel() == 0:
        return torch.ones(0, device=device, dtype=torch.int)

    # Handle single token
    if tokenized_sequence.numel() == 1:
        mask = torch.ones(1, device=device, dtype=torch.int)
        token = tokenized_sequence.item()
        if token == terminal_tag_char or token in other_tag_chars:
            mask[0] = 0
        return mask

    # Ensure 2D (batch, seq_len)
    batched = tokenized_sequence.ndim == 2
    if not batched:
        tokenized_sequence = tokenized_sequence.unsqueeze(0)
    batch_size, seq_len = tokenized_sequence.shape

    # Valid DNA + degenerate bases + control chars
    valid_dna_and_degenerate = {
        45,
        65,
        66,
        67,
        68,
        71,
        72,
        75,
        77,
        78,
        82,
        83,
        84,
        85,
        86,
        87,
        89,  # Uppercase
        97,
        98,
        99,
        100,
        103,
        104,
        107,
        109,
        110,
        114,
        115,
        116,
        117,
        118,
        119,
        121,  # Lowercase
    }
    control_tags_set = {64, 35}  # '@', '#'
    valid_dna_or_control_tensor = torch.tensor(
        list(valid_dna_and_degenerate | control_tags_set), device=device, dtype=tokenized_sequence.dtype
    )

    # Initialize mask to all ones (keep everything)
    out_mask = torch.ones_like(tokenized_sequence, dtype=torch.int)

    def region_all_valid_or_control(region: torch.Tensor) -> bool:
        """Check if all tokens in region are valid DNA or control chars."""
        if region.numel() == 0:
            return True
        return bool(torch.all(torch.isin(region, valid_dna_or_control_tensor)).cpu().item())

    def process_segment(seg_seq: torch.Tensor) -> torch.Tensor:  # noqa: C901
        """Process one EOD-free segment."""
        seg_len = seg_seq.size(0)
        seg_mask = torch.ones(seg_len, device=device, dtype=torch.int)

        # Find pipe positions
        pipe_pos = (seg_seq == terminal_tag_char).nonzero(as_tuple=True)[0].cpu().tolist()

        if len(pipe_pos) == 0:
            # No pipes: mask if contains tag chars or invalid DNA and short enough
            if seg_len < max_tag_len and not region_all_valid_or_control(seg_seq):
                seg_mask.zero_()
            return seg_mask

        # Mask all pipe positions
        seg_mask[pipe_pos] = 0

        # Determine if tag starts before first pipe (state machine)
        first_pipe = pipe_pos[0]
        if first_pipe < seg_len - 1:
            # Check token after first pipe
            if seg_len > first_pipe + 2:
                first_tok = seg_seq[first_pipe + 1].item()
                next_tok = seg_seq[first_pipe + 2].item()
                # If pattern is [char]_ or starts with ;, tag is AFTER pipe
                is_tag = not (next_tok == 95 or first_tok == 59)
            elif seg_len > first_pipe + 1:
                next_tok = seg_seq[first_pipe + 1].item()
                # Check for d, D, r, R (domain/realm) or ; (missing field)
                is_tag = next_tok not in {68, 100, 82, 114, 59}
            elif first_pipe >= max_tag_len or region_all_valid_or_control(seg_seq[:first_pipe]):
                is_tag = False
            else:
                is_tag = True
        else:
            # Sequence ends with pipe
            if first_pipe >= max_tag_len or region_all_valid_or_control(seg_seq[:first_pipe]):
                return seg_mask
            else:
                seg_mask[:first_pipe] = 0
                return seg_mask

        # Process regions between pipes (state machine)
        start = 0
        for end in pipe_pos:
            seg_region_len = end - start
            if is_tag and seg_region_len < max_tag_len:
                seg_mask[start:end] = 0
            elif is_tag and seg_region_len >= max_tag_len:
                # Too long to be a tag, must be DNA
                is_tag = False
            # Flip state for next region
            is_tag = not is_tag
            start = end + 1

        # Process region after last pipe
        seg_region_len = seg_len - start
        if is_tag and seg_region_len < max_tag_len:
            seg_mask[start:] = 0

        return seg_mask

    # Process each batch row, splitting on EOD tokens
    for b in range(batch_size):
        row = tokenized_sequence[b]
        eod_positions = (row == eod_token_id).nonzero(as_tuple=True)[0].cpu().tolist()

        start_idx = 0
        for pos in eod_positions:
            if pos > start_idx:
                seg = row[start_idx:pos]
                seg_mask = process_segment(seg)
                out_mask[b, start_idx:pos] = seg_mask
            # Leave EOD unmasked
            start_idx = pos + 1

        # Process remaining after last EOD
        if start_idx < seq_len:
            seg = row[start_idx:]
            seg_mask = process_segment(seg)
            out_mask[b, start_idx:] = seg_mask

    # Safety: mask any non-valid DNA tokens that slipped through
    out_mask[~torch.isin(tokenized_sequence, valid_dna_or_control_tensor)] = 0

    # Force EOD tokens to be unmasked
    out_mask[tokenized_sequence == eod_token_id] = 1

    if not batched:
        out_mask = out_mask.squeeze(0)

    return out_mask


class Evo2MaskingConstants:
    """Constants used in Evo2 genomic sequence masking."""

    # Standard DNA tokens: A, C, G, T (both uppercase and lowercase)
    DNA_TOKENS: ClassVar[list[int]] = [65, 67, 71, 84, 97, 99, 103, 116]

    # Control characters used in data formatting
    CONTROL_TAGS: ClassVar[list[int]] = [64, 35]  # '@', '#'

    # Phylogenetic tag constants
    TAG_BOUNDS: ClassVar[int] = 124  # '|' pipe character
    TAG_CHARS: ClassVar[set[int]] = {95, 59, 32}  # '_', ';', space
    MAX_TAG_LEN: ClassVar[int] = 2048
    DEFAULT_EOD: ClassVar[int] = 0
