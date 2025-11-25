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

import pytest
import torch
from data_collator import GenomicDataCollatorForCLM
from genomic_utils import make_upper_case


@pytest.fixture
def tokenizer_path(recipe_path):
    """Get the path to the nucleotide tokenizer."""
    return str(recipe_path / "example_checkpoint")


@pytest.fixture
def tokenizer(tokenizer_path):
    """Load the nucleotide tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path)


# Tests for make_upper_case()
def test_make_upper_case_mixed():
    """Test make_upper_case handles mixed case correctly."""
    tokens = torch.tensor([[97, 67, 103, 84]])  # "aCgT"
    upper, mask = make_upper_case(tokens)

    expected_upper = torch.tensor([[65, 67, 71, 84]])  # "ACGT"
    expected_mask = torch.tensor([[True, False, True, False]])

    assert torch.equal(upper, expected_upper)
    assert torch.equal(mask, expected_mask)


# Tests for GenomicDataCollatorForCLM
def test_collator_basic(tokenizer):
    """Test basic collator functionality."""
    collator = GenomicDataCollatorForCLM(
        tokenizer=tokenizer,
        uppercase_labels=False,
        mask_degenerate_bases=False,
    )

    features = [{"input_ids": [65, 67, 71, 84]}]
    batch = collator(features)

    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] == 1


def test_collator_uppercases(tokenizer):
    """Test that collator uppercases labels."""
    collator = GenomicDataCollatorForCLM(
        tokenizer=tokenizer,
        uppercase_labels=True,
        mask_degenerate_bases=False,
    )

    features = [{"input_ids": [97, 67, 103, 116]}]  # "aCgt"
    batch = collator(features)

    assert "labels" in batch
    assert batch["labels"].ndim == 2


def test_collator_masks_degenerate(tokenizer):
    """Test that collator masks degenerate bases."""
    collator = GenomicDataCollatorForCLM(
        tokenizer=tokenizer,
        uppercase_labels=False,
        mask_degenerate_bases=True,
    )

    features = [{"input_ids": [65, 67, 71, 84, 78]}]  # "ACGTN"
    batch = collator(features)

    assert "labels" in batch
    assert batch["labels"].ndim == 2


def test_collator_combined(tokenizer):
    """Test collator with both uppercase and degenerate masking."""
    collator = GenomicDataCollatorForCLM(
        tokenizer=tokenizer,
        uppercase_labels=True,
        mask_degenerate_bases=True,
    )

    features = [{"input_ids": [97, 67, 103, 84, 78]}]  # "aCgTN"
    batch = collator(features)

    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].ndim == 2
