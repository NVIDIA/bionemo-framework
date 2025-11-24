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

This module contains utility functions for processing genomic sequences during training:
- Uppercasing labels while preserving mixed-case inputs
- Masking phylogenetic tags in OpenGenome2 data

These functions are adapted from NeMo's Evo2 implementation:
- make_upper_case: From nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils
- mask_phylogenetic_tags: From nemo.collections.llm.gpt.data.megatron.hyena.evo2_dataset

For the latest versions, see:
https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/llm/gpt/model/megatron/hyena/hyena_utils.py
https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/llm/gpt/data/megatron/hyena/evo2_dataset.py
"""

from typing import ClassVar

import torch


def make_upper_case(tokens, lowercase_start=97, lowercase_end=122, case_diff=32):
    """Replace lowercase ASCII characters with uppercase.

    This function performs vectorized uppercasing on token tensors using ASCII math.
    Lowercase 'a'=97 and uppercase 'A'=65, difference is 32.

    Adapted from: nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils.make_upper_case

    Args:
        tokens: Input tensor containing token IDs (ASCII values)
        lowercase_start: ASCII value for 'a' (default: 97)
        lowercase_end: ASCII value for 'z' (default: 122)
        case_diff: Difference between lowercase and uppercase (default: 32)

    Returns:
        tuple: (uppercase_tensor, lowercase_mask)
            - uppercase_tensor: Tokens with lowercase converted to uppercase
            - lowercase_mask: Boolean mask indicating which tokens were lowercase

    Example:
        >>> tokens = torch.tensor([[97, 67, 103, 84]])  # "aCgT"
        >>> upper, mask = make_upper_case(tokens)
        >>> upper
        tensor([[65, 67, 71, 84]])  # "ACGT"
        >>> mask
        tensor([[True, False, True, False]])
    """
    lowercase_mask = (tokens >= lowercase_start) & (tokens <= lowercase_end)
    uppercase_tensor = torch.where(lowercase_mask, tokens - case_diff, tokens)
    return uppercase_tensor, lowercase_mask


class Evo2MaskingConstants:
    """Constants used in Evo2 genomic sequence masking.

    These constants define which characters are valid DNA, degenerate bases,
    or control/tag characters. Used for loss masking during training.

    Adapted from: nemo.collections.llm.gpt.data.megatron.hyena.evo2_dataset.Evo2Dataset
    """

    # Control characters used in data formatting
    CONTROL_TAGS: ClassVar[list[int]] = [64, 35]  # '@' for splice/window markers, '#' for contig splits

    # Phylogenetic tag delimiters and special characters
    TAG_BOUNDS = 124  # '|' pipe character (start/end delimiter)
    TAG_CHARS: ClassVar[set[int]] = {95, 59, 32}  # '_', ';', space (only appear in tags)

    # Default end-of-document token
    DEFAULT_EOD = 0

    # Maximum phylogenetic tag length (to avoid over-masking long DNA sequences)
    MAX_TAG_LEN = 2048

    # Standard DNA tokens: A, C, G, T (both uppercase and lowercase)
    DNA_TOKENS: ClassVar[list[int]] = [65, 67, 71, 84, 97, 99, 103, 116]  # A, C, G, T, a, c, g, t

    # Valid DNA including degenerate/ambiguous bases and RNA
    # Includes: A-Z, a-z, -, and degenerate bases (N, R, Y, K, M, S, W, B, D, H, V)
    VALID_DNA_AND_DEGENERATE: ClassVar[set[int]] = {
        45,  # -
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


def mask_phylogenetic_tags(  # noqa: C901
    tokenized_sequence: torch.Tensor,
    terminal_tag_char: int = 124,
    other_tag_chars: set[int] | None = None,
    eod_token_id: int = 0,
    max_tag_len: int = 2048,
) -> torch.Tensor:
    """Create a binary mask for sequences containing phylogenetic tags and DNA.

    This function identifies and masks phylogenetic taxonomy tags in genomic sequences.
    Tags have the format: |d__domain;p__phylum;c__class;o__order;f__family;g__genus;s__species|

    The masking rules (applied per contiguous sub-sequence between EOD tokens):
    - Any token equal to terminal_tag_char ('|') is masked
    - Regions containing tag-specific characters ('_', ';', space) are masked
    - Regions starting with taxonomy prefixes (d, p, c, o, f, g, s) after '|' are masked
    - Pure DNA regions between tags are NOT masked
    - EOD tokens are always unmasked (they break tag regions)

    Adapted from: nemo.collections.llm.gpt.data.megatron.hyena.evo2_dataset.Evo2Dataset.mask_phylogenetic_tags

    Args:
        tokenized_sequence: Tensor of ASCII token IDs. Shape: (seq_len,) or (batch_size, seq_len)
        terminal_tag_char: ASCII value for pipe '|' (default: 124)
        other_tag_chars: ASCII values that only appear in tags: '_', ';', space (default: {95, 59, 32})
        eod_token_id: Token ID for end-of-document (default: 0)
        max_tag_len: Maximum tag length to prevent over-masking (default: 2048)

    Returns:
        Binary mask tensor (same shape as input):
            - 1 = keep (valid DNA to include in loss)
            - 0 = mask (tag or control character, ignore in loss)

    Example:
        >>> sequence = "ATG|d__Bacteria|TCGA"
        >>> tokens = torch.tensor([ord(c) for c in sequence])
        >>> mask = mask_phylogenetic_tags(tokens)
        >>> # mask: [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]
        >>> #       ATG (keep) |d__Bacteria| (mask) TCGA (keep)

    Notes:
        - Tag format: |d__{};p__{};c__{};o__{};f__{};g__{};s__{}|
        - Tags can be incomplete: |d__Bacteria;p__;;| (some fields missing)
        - Partial tags at sequence boundaries are handled gracefully
        - EOD tokens break sequences so tags never span across them
    """
    if other_tag_chars is None:
        other_tag_chars = {95, 59, 32}  # '_', ';', space

    device = tokenized_sequence.device
    dtype = tokenized_sequence.dtype

    # Handle empty or single-token sequences
    if tokenized_sequence.numel() == 0:
        return torch.ones(0, device=device, dtype=torch.int)
    if tokenized_sequence.numel() == 1:
        mask = torch.ones(1, device=device, dtype=torch.int)
        token = tokenized_sequence.item()
        if token == terminal_tag_char or token in other_tag_chars:
            mask[0] = 0
        return mask

    # Ensure input is 2D (batch, seq_len)
    batched = tokenized_sequence.ndim == 2
    if not batched:
        tokenized_sequence = tokenized_sequence.unsqueeze(0)
    batch_size, seq_len = tokenized_sequence.shape

    valid_dna_or_control_tensor = torch.tensor(
        list(Evo2MaskingConstants.VALID_DNA_AND_DEGENERATE | set(Evo2MaskingConstants.CONTROL_TAGS)),
        device=device,
        dtype=dtype,
    )

    # Initialize output mask to all ones (keep everything by default)
    out_mask = torch.ones_like(tokenized_sequence, dtype=torch.int)

    # Helper: Check if all tokens in a region are valid DNA or control
    def region_all_valid_or_control(region: torch.Tensor) -> bool:
        if region.numel() == 0:
            return True
        return bool(torch.all(torch.isin(region, valid_dna_or_control_tensor)).cpu().item())

    # Process one EOD-free segment using phylo tag detection logic
    def process_segment(seg_seq: torch.Tensor) -> torch.Tensor:  # noqa: C901
        """Process a segment between EOD tokens, identifying and masking phylo tags."""
        seg_len = seg_seq.size(0)
        seg_mask = torch.ones(seg_len, device=device, dtype=torch.int)

        # Find all pipe positions
        pipe_pos = (seg_seq == terminal_tag_char).nonzero(as_tuple=True)[0].cpu().tolist()

        if len(pipe_pos) == 0:
            # No pipes: if this looks like a tag (has tag chars or non-DNA) and is short enough, mask it
            if seg_len < max_tag_len and not region_all_valid_or_control(seg_seq):
                seg_mask.zero_()
            return seg_mask

        # Always mask pipe positions
        seg_mask[pipe_pos] = 0

        # Determine if sequence starts in a tag or DNA region
        # Check first token after first pipe to determine state
        first_pipe = pipe_pos[0]
        if first_pipe >= 0 and first_pipe < seg_len - 1:
            # Look at tokens after the first pipe to determine if tag starts before or after pipe
            if seg_len > first_pipe + 2:
                first_tok = seg_seq[first_pipe + 1].item()
                next_tok = seg_seq[first_pipe + 2].item()
                # If format is [char]_ or starts with ;, tag is AFTER pipe (so before pipe is DNA)
                if next_tok == 95 or first_tok == 59:  # ord('_') = 95, ord(';') = 59
                    is_tag = False  # Before first pipe is DNA
                else:
                    is_tag = True  # Before first pipe is a tag
            elif seg_len > first_pipe + 1:
                next_tok = seg_seq[first_pipe + 1].item()
                # Weak signal: d, D, r, R, or ; suggest tag after pipe
                if next_tok in {68, 100, 82, 114, 59}:  # D, d, R, r, ;
                    is_tag = False
                else:
                    is_tag = True
            elif first_pipe >= max_tag_len or region_all_valid_or_control(seg_seq[:first_pipe]):
                is_tag = False
            else:
                is_tag = True
        else:
            # Sequence ends with pipe
            assert first_pipe == seg_len - 1
            if first_pipe >= max_tag_len or region_all_valid_or_control(seg_seq[:first_pipe]):
                return seg_mask
            else:
                seg_mask[:first_pipe] = 0
                return seg_mask

        # Process regions between pipes using state machine
        start = 0
        for end in pipe_pos:
            seg_len_region = end - start
            # If we're in tag state and region is short enough, mask it
            if is_tag and seg_len_region < max_tag_len:
                seg_mask[start:end] = 0
            elif is_tag and seg_len_region >= max_tag_len:
                # Region too long to be a tag, must be DNA
                is_tag = False
            # Flip state for next region (tag → DNA → tag → DNA...)
            is_tag = not is_tag
            start = end + 1

        # Process final region after last pipe
        seg_len_region = len(seg_mask) - start
        if is_tag and seg_len_region < max_tag_len:
            seg_mask[start:] = 0

        return seg_mask

    # Process each row, splitting on EOD tokens
    for b in range(batch_size):
        row = tokenized_sequence[b]
        # Find all EOD positions
        eod_positions = (row == eod_token_id).nonzero(as_tuple=True)[0].cpu().tolist()

        start_idx = 0
        for pos in eod_positions:
            if pos > start_idx:
                # Process segment between EODs
                seg = row[start_idx:pos]
                seg_mask = process_segment(seg)
                out_mask[b, start_idx:pos] = seg_mask
            # EOD itself stays unmasked
            start_idx = pos + 1

        # Process remaining tokens after last EOD
        if start_idx < seq_len:
            seg = row[start_idx:]
            seg_mask = process_segment(seg)
            out_mask[b, start_idx:] = seg_mask

    # Safety: mask any non-valid DNA/control tokens even if masking logic missed them
    out_mask[~torch.isin(tokenized_sequence, valid_dna_or_control_tensor)] = 0

    # Force EOD tokens to stay unmasked (user decides separately if they want EOD masked)
    out_mask[tokenized_sequence == eod_token_id] = 1

    # Return to original shape if input was 1D
    if not batched:
        out_mask = out_mask.squeeze(0)

    return out_mask
