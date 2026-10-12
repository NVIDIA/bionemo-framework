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

"""Scatter primitives for the flat index contract.

The canonical contract is flat: soft tokens ``[N, H_lm]`` and int64 indices
into the flattened ``(B*S)`` sequence, with negative entries dropped. The
dense ``[B, T]`` form  lowers to flat at
the projector boundary via ``dense_to_flat``.
"""

from __future__ import annotations

import torch


def scatter_flat(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    soft_tokens: torch.Tensor,
    flat_indices: torch.Tensor,
    *,
    placeholder_token_id: int | None = None,
) -> torch.Tensor:
    """Scatter ``[N, C]`` soft tokens into ``[B, S, C]`` embeds by flattened index.

    Negative indices are dropped. When a placeholder token id is supplied,
    every scattered position must point at that token. Indices must be in
    range and unique: a silently wrong scatter produces a plausible loss
    curve, so every violation raises.
    """
    if input_ids.ndim != 2 or inputs_embeds.ndim != 3 or soft_tokens.ndim != 2:
        raise ValueError("expected input_ids [B,S], inputs_embeds [B,S,C], soft_tokens [N,C]")
    batch, sequence = input_ids.shape
    if tuple(inputs_embeds.shape[:2]) != (batch, sequence):
        raise ValueError("input_ids and inputs_embeds batch/sequence dimensions differ")
    if soft_tokens.shape[-1] != inputs_embeds.shape[-1]:
        raise ValueError("soft tokens must match the embedding width")
    if flat_indices.ndim != 1 or flat_indices.shape[0] != soft_tokens.shape[0]:
        raise ValueError("flat_indices must be [N] and match the soft-token count")
    flat_indices = flat_indices.to(device=input_ids.device, dtype=torch.long)
    active = flat_indices.ge(0)
    active_indices = flat_indices[active]
    if active_indices.numel():
        if int(active_indices.max()) >= batch * sequence:
            raise ValueError("token index is outside the flattened LM sequence")
        if active_indices.unique().numel() != active_indices.numel():
            raise ValueError("token indices contain duplicates")
        if placeholder_token_id is not None:
            observed = input_ids.reshape(-1)[active_indices]
            if not torch.all(observed.eq(int(placeholder_token_id))):
                raise ValueError("token index points at a non-placeholder token")
    flat = inputs_embeds.reshape(batch * sequence, -1)
    values = soft_tokens[active].to(device=flat.device, dtype=flat.dtype)
    scattered = flat.index_copy(0, active_indices.to(flat.device), values)
    return scattered.reshape_as(inputs_embeds)


def dense_to_flat(
    token_indices: torch.Tensor,
    soft_tokens: torch.Tensor,
    *,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower dense ``[B, T]`` indices (``-1`` padded) and ``[B, T, C]`` tokens to flat form."""
    if token_indices.shape != soft_tokens.shape[:2]:
        raise ValueError(
            f"token_indices {tuple(token_indices.shape)} must match "
            f"the dense token shape {tuple(soft_tokens.shape[:2])}"
        )
    token_indices = token_indices.to(dtype=torch.long)
    active = token_indices.ge(0)
    # A dense position must address its own row: positions are row-local, so
    # anything >= sequence_length would silently alias into another row once
    # flattened. Reject before lowering.
    if active.any() and int(token_indices[active].max()) >= sequence_length:
        raise ValueError("token index is outside the LM sequence")
    batch_indices = torch.arange(token_indices.shape[0], device=token_indices.device)
    batch_indices = batch_indices.unsqueeze(1).expand_as(token_indices)
    flat_indices = (batch_indices * sequence_length + token_indices)[active]
    return flat_indices, soft_tokens[active]


def explicit_to_flat(
    token_indices: torch.Tensor,
    soft_tokens: torch.Tensor,
    *,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower an explicit index override to the canonical flat pair.

    Flat ``[N]`` indices with flat ``[N, C]`` tokens are the canonical
    contract itself and pass through (the count check is what keeps a
    truncated payload loud); dense ``[B, T]`` indices with dense ``[B, T, C]``
    tokens lower via ``dense_to_flat``. The two geometries do not mix, and
    anything else fails closed.
    """
    if token_indices.ndim == 1 and soft_tokens.ndim == 2:
        if token_indices.shape[0] != soft_tokens.shape[0]:
            raise ValueError(
                f"flat token_indices ({token_indices.shape[0]}) must match "
                f"the soft-token count ({soft_tokens.shape[0]})"
            )
        return token_indices.to(dtype=torch.long), soft_tokens
    if token_indices.ndim == 2 and soft_tokens.ndim == 3:
        return dense_to_flat(token_indices, soft_tokens, sequence_length=sequence_length)
    raise ValueError(
        f"explicit token indices {tuple(token_indices.shape)} do not match the soft-token "
        f"geometry {tuple(soft_tokens.shape)}: pair flat [N] indices with flat [N, C] "
        "tokens, or dense [B, T] indices with dense [B, T, C] tokens"
    )


def derive_indices(
    input_ids: torch.Tensor,
    placeholder_token_ids: dict[str, int],
    tokens_per_projector: dict[str, int],
) -> dict[str, torch.Tensor]:
    """Derive dense per-projector target positions from per-projector placeholder masks.

    Each projector owns a distinct placeholder token id, so routing is by token
    identity rather than position: segments of different projectors interleaved
    in one prompt cannot be confused. Per-row slot counts must be uniform
    across the batch and equal the projector's emitted soft-token count — those
    two assertions are what make a silently truncated prompt fail loudly.
    """
    if set(placeholder_token_ids) != set(tokens_per_projector):
        raise ValueError(
            "placeholder_token_ids and soft-token counts must cover the same projectors: "
            f"{sorted(placeholder_token_ids)} != {sorted(tokens_per_projector)}"
        )
    indices = {}
    for name in sorted(tokens_per_projector):
        mask = input_ids.eq(int(placeholder_token_ids[name]))
        row_positions = [row.nonzero(as_tuple=False).flatten() for row in mask]
        counts = {int(len(row)) for row in row_positions}
        if len(counts) != 1:
            raise ValueError(f"placeholder slot counts vary within a batch for projector {name!r}: {sorted(counts)}")
        found = counts.pop()
        expected = int(tokens_per_projector[name])
        if found != expected:
            raise ValueError(
                f"projector {name!r} emitted {expected} soft tokens per row but {found} placeholder slots were found"
            )
        indices[name] = torch.stack(row_positions)
    return indices


def derive_flat_indices(
    input_ids: torch.Tensor,
    placeholder_token_ids: dict[str, int],
    token_counts: dict[str, int],
) -> dict[str, torch.Tensor]:
    """Derive canonical flat target positions from per-projector placeholder masks.

        The flat companion to ``derive_indices`` for flat ``[N, H]`` payloads
    : a projector's targets are its placeholder occurrences in
        the flattened ``(B*S)`` sequence, in row-major order — the same order a
        packed collator concatenates per-record features. The total slot count
        must equal the projector's emitted soft-token count exactly, so a
        truncated or over-long prompt fails loudly.
    """
    if set(placeholder_token_ids) != set(token_counts):
        raise ValueError(
            "placeholder_token_ids and soft-token counts must cover the same projectors: "
            f"{sorted(placeholder_token_ids)} != {sorted(token_counts)}"
        )
    indices = {}
    for name in sorted(token_counts):
        mask = input_ids.eq(int(placeholder_token_ids[name])).reshape(-1)
        found = mask.nonzero(as_tuple=False).flatten()
        expected = int(token_counts[name])
        if found.numel() != expected:
            raise ValueError(
                f"projector {name!r} emitted {expected} soft tokens but {found.numel()} placeholder slots were found"
            )
        indices[name] = found
    return indices
