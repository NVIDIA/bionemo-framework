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

"""Unit-axis chunk planning and fp32 overlap stitching."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Lazy at runtime (the functions import torch inside); annotations only.
    import torch


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def plan_chunks(length: int, window: int, overlap: int) -> tuple[Chunk, ...]:
    length, window, overlap = int(length), int(window), int(overlap)
    if length < 0 or window <= 0 or overlap < 0 or overlap >= window:
        raise ValueError("invalid length/window/overlap")
    if length == 0:
        return ()
    starts = list(range(0, max(1, length - overlap), window - overlap))
    if starts[-1] + window < length:
        starts.append(length - window)
    starts = sorted(set(max(0, min(start, max(0, length - window))) for start in starts))
    return tuple(Chunk(index=i, start=start, end=min(length, start + window)) for i, start in enumerate(starts))


def coverage_weights(length: int, chunks: Sequence[Chunk]) -> tuple[int, ...]:
    weights = [0] * int(length)
    for chunk in chunks:
        if not (0 <= chunk.start < chunk.end <= length):
            raise ValueError("chunk is outside sequence")
        for position in range(chunk.start, chunk.end):
            weights[position] += 1
    if length and any(weight == 0 for weight in weights):
        raise ValueError("chunk plan leaves uncovered positions")
    return tuple(weights)


def stitch_hidden_states(
    length: int, chunks: Sequence[Chunk], states: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, tuple[float, ...]]:
    if len(chunks) != len(states):
        raise ValueError("chunks and states must have equal length")
    import torch

    if not states:
        return torch.empty((0, 0), dtype=torch.float32), ()
    hidden_size = int(states[0].shape[-1])
    total = torch.zeros((length, hidden_size), dtype=torch.float32, device=states[0].device)
    weights = torch.zeros((length,), dtype=torch.float32, device=states[0].device)
    for chunk, state in zip(chunks, states, strict=True):
        if tuple(state.shape) != (chunk.length, hidden_size):
            raise ValueError(f"chunk {chunk.index} state shape mismatch")
        total[chunk.start : chunk.end] += state.to(torch.float32)
        weights[chunk.start : chunk.end] += 1
    if length and torch.any(weights == 0):
        raise ValueError("stitch leaves uncovered positions")
    return total / weights.clamp_min(1).unsqueeze(-1), tuple(float(x) for x in weights.cpu())


def strip_special_tokens(hidden: torch.Tensor, special_mask: Iterable[bool], expected_units: int) -> torch.Tensor:
    import torch

    mask = torch.as_tensor(list(special_mask), dtype=torch.bool, device=hidden.device)
    if hidden.ndim != 2 or mask.numel() != hidden.shape[0]:
        raise ValueError("special-token mask must align with hidden states")
    result = hidden[~mask]
    if result.shape[0] != expected_units:
        raise ValueError(f"token alignment produced {result.shape[0]} rows, expected {expected_units}")
    return result
