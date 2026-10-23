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

"""Dense-to-flat conversion and validated scatter behavior."""

from __future__ import annotations

import pytest
import torch

from nemotron_stitch.projector import (
    dense_to_flat,
    derive_flat_indices,
    derive_indices,
    explicit_to_flat,
    scatter_flat,
)
from nemotron_stitch.testing.fixtures import ragged_batch


def test_scatter_replaces_only_placeholders_and_preserves_gradient():
    input_ids = torch.tensor([[1, 18, 18, 2], [18, 3, 18, 4]])
    text = torch.randn(2, 4, 6, requires_grad=True)
    modality = torch.randn(2, 2, 6, requires_grad=True)
    indices = torch.tensor([[1, 2], [0, 2]])
    flat_indices, flat_tokens = dense_to_flat(indices, modality, sequence_length=4)
    result = scatter_flat(input_ids, text, flat_tokens, flat_indices, placeholder_token_id=18)
    torch.testing.assert_close(result[0, [1, 2]], modality[0])
    torch.testing.assert_close(result[1, [0, 2]], modality[1])
    torch.testing.assert_close(result[0, [0, 3]], text[0, [0, 3]])
    result.sum().backward()
    assert modality.grad is not None and torch.count_nonzero(modality.grad) == modality.numel()
    assert text.grad is not None
    assert torch.count_nonzero(text.grad[input_ids == 18]) == 0


def test_scatter_supports_variable_length_rows():
    ids = torch.tensor([[32, 32, 1], [2, 32, 3]])
    text = torch.zeros(2, 3, 4)
    modality = torch.tensor(
        [
            [[1.0] * 4, [2.0] * 4],
            [[3.0] * 4, [99.0] * 4],
        ]
    )
    indices = torch.tensor([[0, 1], [1, -1]])
    flat_indices, flat_tokens = dense_to_flat(indices, modality, sequence_length=3)
    result = scatter_flat(ids, text, flat_tokens, flat_indices, placeholder_token_id=32)
    torch.testing.assert_close(result[0, 0], modality[0, 0])
    torch.testing.assert_close(result[0, 1], modality[0, 1])
    torch.testing.assert_close(result[1, 1], modality[1, 0])
    torch.testing.assert_close(result[1, 2], torch.zeros(4))


def test_ragged_flat_batch_matches_handbuilt_expected():
    fixture = ragged_batch()
    result = scatter_flat(
        fixture["input_ids"],
        fixture["inputs_embeds"],
        fixture["soft_tokens"],
        fixture["flat_indices"],
        placeholder_token_id=fixture["placeholder_token_id"],
    )
    torch.testing.assert_close(result, fixture["expected"])


def test_dense_lowering_matches_the_flat_spelling():
    fixture = ragged_batch()
    dense_indices = fixture["dense_indices"]
    batch, max_tokens = dense_indices.shape
    width = fixture["soft_tokens"].shape[-1]
    dense_tokens = torch.zeros(batch, max_tokens, width)
    cursor = 0
    for row in range(batch):
        count = int(dense_indices[row].ge(0).sum())
        dense_tokens[row, :count] = fixture["soft_tokens"][cursor : cursor + count]
        cursor += count
    flat_indices, flat_tokens = dense_to_flat(dense_indices, dense_tokens, sequence_length=12)
    torch.testing.assert_close(flat_indices, fixture["flat_indices"])
    torch.testing.assert_close(flat_tokens, fixture["soft_tokens"])


def test_scatter_rejects_duplicate_indices():
    fixture = ragged_batch()
    indices = fixture["flat_indices"].clone()
    indices[-1] = indices[0]
    with pytest.raises(ValueError, match="duplicates"):
        scatter_flat(
            fixture["input_ids"],
            fixture["inputs_embeds"],
            fixture["soft_tokens"],
            indices,
            placeholder_token_id=99,
        )


def test_scatter_rejects_out_of_range_indices():
    fixture = ragged_batch()
    indices = fixture["flat_indices"].clone()
    indices[0] = 3 * 12  # one past the flattened sequence
    with pytest.raises(ValueError, match="outside the flattened"):
        scatter_flat(
            fixture["input_ids"],
            fixture["inputs_embeds"],
            fixture["soft_tokens"],
            indices,
            placeholder_token_id=99,
        )


def test_scatter_rejects_non_placeholder_positions():
    fixture = ragged_batch()
    indices = fixture["flat_indices"].clone()
    indices[0] = 0  # row 0 position 0 is a text token
    with pytest.raises(ValueError, match="non-placeholder"):
        scatter_flat(
            fixture["input_ids"],
            fixture["inputs_embeds"],
            fixture["soft_tokens"],
            indices,
            placeholder_token_id=99,
        )


def test_dense_to_flat_rejects_row_aliasing_positions():
    # A dense position >= sequence_length would silently alias into the next
    # row once flattened; reject before lowering.
    indices = torch.tensor([[0, 7]])
    tokens = torch.randn(1, 2, 4)
    with pytest.raises(ValueError, match="outside the LM sequence"):
        dense_to_flat(indices, tokens, sequence_length=5)


def test_derive_indices_routes_by_placeholder_id_not_position():
    # b's segment precedes a's in the prompt; routing follows token identity.
    input_ids = torch.tensor([[99, 99, 1, 98, 98], [2, 99, 99, 98, 98]])
    indices = derive_indices(input_ids, {"a": 98, "b": 99}, {"a": 2, "b": 2})
    torch.testing.assert_close(indices["a"], torch.tensor([[3, 4], [3, 4]]))
    torch.testing.assert_close(indices["b"], torch.tensor([[0, 1], [1, 2]]))


def test_derive_indices_rejects_varying_slot_counts():
    input_ids = torch.tensor([[98, 1, 5], [98, 98, 1]])
    with pytest.raises(ValueError, match="vary within a batch"):
        derive_indices(input_ids, {"a": 98}, {"a": 1})


def test_derive_indices_rejects_slot_count_mismatch():
    input_ids = torch.tensor([[98, 98, 98, 1]])
    with pytest.raises(ValueError, match="emitted 1 soft tokens per row but 3"):
        derive_indices(input_ids, {"a": 98}, {"a": 1})
    input_ids = torch.tensor([[98, 1, 1, 1]])
    with pytest.raises(ValueError, match="emitted 2 soft tokens per row but 1"):
        derive_indices(input_ids, {"a": 98}, {"a": 2})


def test_derive_indices_rejects_mismatched_projector_sets():
    input_ids = torch.tensor([[98, 1]])
    with pytest.raises(ValueError, match="same projectors"):
        derive_indices(input_ids, {"a": 98}, {"b": 1})


def test_explicit_to_flat_passes_the_canonical_pair_through():
    tokens = torch.randn(3, 4)
    flat_indices, flat_tokens = explicit_to_flat(torch.tensor([0, 5, 9], dtype=torch.int32), tokens, sequence_length=6)
    assert flat_indices.dtype == torch.long
    torch.testing.assert_close(flat_indices, torch.tensor([0, 5, 9]))
    assert flat_tokens is tokens


def test_explicit_to_flat_lowers_the_dense_pair():
    tokens = torch.randn(2, 2, 4)
    # Row 1's dense slot 0 targets sequence position 1; slot 1 is padding.
    flat_indices, flat_tokens = explicit_to_flat(torch.tensor([[0, 1], [1, -1]]), tokens, sequence_length=3)
    torch.testing.assert_close(flat_indices, torch.tensor([0, 1, 4]))
    torch.testing.assert_close(flat_tokens, torch.cat([tokens[0], tokens[1, :1]]))


def test_explicit_to_flat_rejects_a_truncated_flat_payload():
    with pytest.raises(ValueError, match="must match the soft-token count"):
        explicit_to_flat(torch.tensor([0, 1]), torch.randn(3, 4), sequence_length=6)


def test_explicit_to_flat_rejects_mixed_geometry():
    # Flat indices with dense tokens (and vice versa) are a collator bug, not
    # a spelling the contract defines.
    with pytest.raises(ValueError, match="do not match the soft-token geometry"):
        explicit_to_flat(torch.tensor([0, 1]), torch.randn(1, 2, 4), sequence_length=6)
    with pytest.raises(ValueError, match="do not match the soft-token geometry"):
        explicit_to_flat(torch.tensor([[0, 1]]), torch.randn(2, 4), sequence_length=6)


def test_derive_flat_indices_spans_ragged_rows_in_row_major_order():
    # Per-row slot counts vary (2 then 1): the flat contract counts in total.
    input_ids = torch.tensor([[99, 1, 99], [2, 99, 3]])
    indices = derive_flat_indices(input_ids, {"a": 99}, {"a": 3})
    torch.testing.assert_close(indices["a"], torch.tensor([0, 2, 4]))


def test_derive_flat_indices_routes_by_placeholder_id_not_position():
    input_ids = torch.tensor([[99, 98, 1], [98, 99, 98]])
    indices = derive_flat_indices(input_ids, {"a": 98, "b": 99}, {"a": 3, "b": 2})
    torch.testing.assert_close(indices["a"], torch.tensor([1, 3, 5]))
    torch.testing.assert_close(indices["b"], torch.tensor([0, 4]))


def test_derive_flat_indices_rejects_a_total_count_mismatch():
    input_ids = torch.tensor([[99, 1, 99], [2, 99, 3]])
    with pytest.raises(ValueError, match="emitted 4 soft tokens but 3"):
        derive_flat_indices(input_ids, {"a": 99}, {"a": 4})


def test_derive_flat_indices_rejects_mismatched_projector_sets():
    with pytest.raises(ValueError, match="same projectors"):
        derive_flat_indices(torch.tensor([[98, 1]]), {"a": 98}, {"b": 1})
