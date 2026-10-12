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

"""Unit-axis chunk planning, coverage, and fp32 overlap stitching."""

from __future__ import annotations

import pytest
import torch

from nemotron_stitch.features.chunking import (
    coverage_weights,
    plan_chunks,
    stitch_hidden_states,
    strip_special_tokens,
)


@pytest.mark.parametrize("length,window,overlap", [(17, 1022, 128), (1022, 1022, 128), (5344, 1022, 128)])
def test_chunk_plan_covers_every_position(length, window, overlap):
    chunks = plan_chunks(length, window, overlap)
    weights = coverage_weights(length, chunks)
    assert len(weights) == length
    assert min(weights) == 1
    assert max(weights) <= 2


def test_chunk_plan_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="invalid length/window/overlap"):
        plan_chunks(10, 4, 4)
    with pytest.raises(ValueError, match="invalid length/window/overlap"):
        plan_chunks(10, 0, 0)
    assert plan_chunks(0, 8, 2) == ()


def test_coverage_rejects_uncovered_positions():
    chunks = plan_chunks(8, 6, 2)
    with pytest.raises(ValueError, match="uncovered"):
        coverage_weights(10, chunks)


def test_short_stitch_is_identical_to_direct_path():
    direct = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    chunks = plan_chunks(7, 20, 4)
    stitched, weights = stitch_hidden_states(7, chunks, [direct])
    torch.testing.assert_close(stitched, direct)
    assert weights == (1.0,) * 7


def test_overlap_stitch_averages_in_fp32():
    chunks = plan_chunks(8, 6, 2)
    states = [torch.ones((chunk.length, 3), dtype=torch.bfloat16) * (index + 1) for index, chunk in enumerate(chunks)]
    stitched, weights = stitch_hidden_states(8, chunks, states)
    assert stitched.dtype == torch.float32
    assert weights[4:6] == (2.0, 2.0)
    torch.testing.assert_close(stitched[4:6], torch.full((2, 3), 1.5))


def test_stitch_rejects_shape_mismatch():
    chunks = plan_chunks(7, 20, 4)
    with pytest.raises(ValueError, match="shape mismatch"):
        stitch_hidden_states(7, chunks, [torch.zeros(6, 5)])


def test_strip_special_tokens_aligns_rows():
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    kept = strip_special_tokens(hidden, [True, False, False, True], 2)
    torch.testing.assert_close(kept, hidden[1:3])
    with pytest.raises(ValueError, match="expected 3"):
        strip_special_tokens(hidden, [True, False, False, True], 3)
    with pytest.raises(ValueError, match="must align"):
        strip_special_tokens(hidden, [True, False], 2)
