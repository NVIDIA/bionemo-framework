# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Synthetic ragged batches for the flat index contract.

The flat contract exists because of packed streams: one batch row holds
several records, each contributing a different number of soft tokens. Neither
consumer's real collator produces that at this layer — a dense collator emits a
rectangular ``[B, T]`` grid — so the ragged fixture is synthetic. It carries
uneven per-row token counts, one row with no soft tokens at all, and
non-contiguous target positions, in both the flat and dense (``-1`` padded)
spellings, plus the expected scatter result built by hand.
"""

from __future__ import annotations

from typing import TypedDict

import torch

SEED = 20260820


class RaggedBatch(TypedDict):
    """Typed form of the tensors and scalar used by the scatter fixture."""

    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    soft_tokens: torch.Tensor
    flat_indices: torch.Tensor
    dense_indices: torch.Tensor
    expected: torch.Tensor
    placeholder_token_id: int


def ragged_batch() -> RaggedBatch:
    """A fixed ragged batch: B=3, S=12, C=8, placeholder token id 99.

    Row 0: three soft tokens at non-contiguous positions 2, 5, 9.
    Row 1: no soft tokens.
    Row 2: two soft tokens at contiguous positions 3, 4 (flat indices 27, 28).
    """
    batch, sequence, width = 3, 12, 8
    generator = torch.Generator().manual_seed(SEED)
    input_ids = torch.randint(1000, 2000, (batch, sequence), generator=generator)
    inputs_embeds = torch.randn(batch, sequence, width, generator=generator)

    row_positions = [torch.tensor([2, 5, 9]), torch.tensor([], dtype=torch.long), torch.tensor([3, 4])]
    counts = [len(p) for p in row_positions]
    total = sum(counts)
    soft_tokens = torch.randn(total, width, generator=generator)

    flat_indices = torch.cat([row * sequence + positions for row, positions in enumerate(row_positions)])
    for row, positions in enumerate(row_positions):
        input_ids[row, positions] = 99

    dense_indices = torch.full((batch, max(counts)), -1, dtype=torch.long)
    for row, positions in enumerate(row_positions):
        dense_indices[row, : len(positions)] = positions

    expected = inputs_embeds.clone()
    cursor = 0
    for row, positions in enumerate(row_positions):
        for position in positions:
            expected[row, position] = soft_tokens[cursor]
            cursor += 1

    return {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
        "soft_tokens": soft_tokens,
        "flat_indices": flat_indices,
        "dense_indices": dense_indices,
        "expected": expected,
        "placeholder_token_id": 99,
    }
