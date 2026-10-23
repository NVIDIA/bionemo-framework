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

"""Coverage-weighted mean pooling to a soft-token count.

Width and cap are caller-supplied policy; these tests pin only the shape of
the computation. The consumer's per-unit widths and default cap stay in the
consumer.
"""

from __future__ import annotations

import pytest
import torch

from nemotron_stitch.features.pooling import pool_positions


def test_fixed_width_bins_average_in_fp32():
    hidden = torch.arange(21 * 3 * 2, dtype=torch.float32).reshape(21 * 3, 2)
    pooled = pool_positions(hidden, pooling_width=21)
    assert pooled.shape == (3, 2)
    torch.testing.assert_close(pooled[0], hidden[:21].mean(dim=0))
    torch.testing.assert_close(pooled[-1], hidden[-21:].mean(dim=0))


def test_partial_trailing_bin_is_its_own_soft_token():
    hidden = torch.arange(25 * 2, dtype=torch.float32).reshape(25, 2)
    pooled = pool_positions(hidden, pooling_width=21)
    assert pooled.shape == (2, 2)
    torch.testing.assert_close(pooled[1], hidden[21:].mean(dim=0))


def test_uncapped_pooling_preserves_every_fixed_width_bin():
    hidden = torch.arange(64 * 130 * 2, dtype=torch.float32).reshape(64 * 130, 2)
    pooled = pool_positions(hidden, pooling_width=64, max_soft_tokens=None)
    assert pooled.shape == (130, 2)
    torch.testing.assert_close(pooled[0], hidden[:64].mean(dim=0))


def test_cap_reduces_bins_by_adaptive_boundaries():
    hidden = torch.arange(64 * 130 * 2, dtype=torch.float32).reshape(64 * 130, 2)
    uncapped = pool_positions(hidden, pooling_width=64)
    pooled = pool_positions(hidden, pooling_width=64, max_soft_tokens=128)
    assert uncapped.shape == (130, 2)
    assert pooled.shape == (128, 2)
    assert torch.isfinite(pooled).all()


def test_cap_above_bin_count_is_a_no_op():
    hidden = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    pooled = pool_positions(hidden, pooling_width=2, max_soft_tokens=128)
    assert pooled.shape == (3, 2)


def test_empty_input_returns_empty():
    hidden = torch.zeros((0, 4), dtype=torch.float32)
    assert pool_positions(hidden, pooling_width=8).shape == (0, 4)


def test_invalid_geometry_fails_closed():
    hidden = torch.zeros((4, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match=r"\[units, hidden\]"):
        pool_positions(torch.zeros(4), pooling_width=2)
    with pytest.raises(ValueError, match="pooling_width must be positive"):
        pool_positions(hidden, pooling_width=0)
    with pytest.raises(ValueError, match="max_soft_tokens must be positive or null"):
        pool_positions(hidden, pooling_width=2, max_soft_tokens=0)
