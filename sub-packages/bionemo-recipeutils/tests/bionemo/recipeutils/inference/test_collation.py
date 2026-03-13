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

"""Tests for bionemo.recipeutils.inference.collation.batch_collator."""

import pytest
import torch

from bionemo.recipeutils.inference.collation import batch_collator


def test_collate_simple_tensors():
    """Concatenate same-shape tensors along batch dim."""
    b1 = torch.ones(2, 5)
    b2 = torch.zeros(3, 5)
    result = batch_collator([b1, b2], batch_dim=0, seq_dim=1, batch_dim_key_defaults={}, seq_dim_key_defaults={})
    assert result.shape == (5, 5)
    assert (result[:2] == 1).all()
    assert (result[2:] == 0).all()


def test_collate_pads_sequence_dim():
    """Tensors with different sequence lengths are padded to the max."""
    b1 = torch.ones(2, 10)
    b2 = torch.ones(2, 15)
    result = batch_collator([b1, b2], batch_dim=0, seq_dim=1, batch_dim_key_defaults={}, seq_dim_key_defaults={})
    assert result.shape == (4, 15)
    # First batch padded with zeros from col 10..14
    assert (result[0, :10] == 1).all()
    assert (result[0, 10:] == 0).all()
    # Second batch untouched
    assert (result[2, :15] == 1).all()


def test_collate_1d_tensors():
    """1-D tensors are concatenated without padding logic."""
    b1 = torch.tensor([1, 2, 3])
    b2 = torch.tensor([4, 5])
    result = batch_collator([b1, b2], batch_dim=0, seq_dim=1, batch_dim_key_defaults={}, seq_dim_key_defaults={})
    assert result.shape == (5,)
    assert result.tolist() == [1, 2, 3, 4, 5]


def test_collate_dict_batches():
    """Dict batches are collated key-by-key."""
    b1 = {"logits": torch.randn(2, 10, 4), "mask": torch.ones(2, 10)}
    b2 = {"logits": torch.randn(3, 10, 4), "mask": torch.zeros(3, 10)}
    result = batch_collator([b1, b2], batch_dim=0, seq_dim=1, batch_dim_key_defaults={}, seq_dim_key_defaults={})
    assert isinstance(result, dict)
    assert result["logits"].shape == (5, 10, 4)
    assert result["mask"].shape == (5, 10)


def test_collate_dict_pads_uneven_seqs():
    """Dict batches with differing sequence lengths are padded."""
    b1 = {"tokens": torch.ones(1, 8), "mask": torch.ones(1, 8)}
    b2 = {"tokens": torch.ones(1, 12), "mask": torch.ones(1, 12)}
    result = batch_collator([b1, b2], batch_dim=0, seq_dim=1, batch_dim_key_defaults={}, seq_dim_key_defaults={})
    assert result["tokens"].shape == (2, 12)
    assert result["mask"].shape == (2, 12)


def test_collate_tuple_batches():
    """Tuple batches are collated element-by-element."""
    b1 = (torch.ones(2, 5), torch.zeros(2, 5))
    b2 = (torch.ones(3, 5), torch.zeros(3, 5))
    result = batch_collator([b1, b2], batch_dim=0, seq_dim=1, batch_dim_key_defaults={}, seq_dim_key_defaults={})
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].shape == (5, 5)
    assert result[1].shape == (5, 5)


def test_collate_list_batches():
    """List batches are collated element-by-element."""
    b1 = [torch.ones(2, 5)]
    b2 = [torch.ones(3, 5)]
    result = batch_collator([b1, b2], batch_dim=0, seq_dim=1, batch_dim_key_defaults={}, seq_dim_key_defaults={})
    assert isinstance(result, list)
    assert result[0].shape == (5, 5)


def test_collate_none_returns_none():
    """If the first batch is None, return None."""
    result = batch_collator([None, None])
    assert result is None


def test_collate_empty_raises():
    """Empty batch list raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        batch_collator([])


def test_collate_per_key_dim_overrides():
    """batch_dim_key_defaults and seq_dim_key_defaults override dims for specific keys."""
    # token_logits has shape [seq, batch, vocab] — batch dim is 1, seq dim is 0
    b1 = {"token_logits": torch.randn(10, 2, 256), "mask": torch.ones(2, 10)}
    b2 = {"token_logits": torch.randn(10, 3, 256), "mask": torch.ones(3, 10)}
    result = batch_collator(
        [b1, b2],
        batch_dim=0,
        seq_dim=1,
        batch_dim_key_defaults={"token_logits": 1},
        seq_dim_key_defaults={"token_logits": 0},
    )
    assert result["token_logits"].shape == (10, 5, 256)
    assert result["mask"].shape == (5, 10)
