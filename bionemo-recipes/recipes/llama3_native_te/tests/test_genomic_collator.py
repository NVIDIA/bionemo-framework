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
from data_collator import GenomicDataCollator
from genomic_masking_functions import make_upper_case
from transformers.data.data_collator import DataCollatorForLanguageModeling


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
    base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = GenomicDataCollator(
        base_collator=base,
        uppercase_labels=False,
        mask_degenerate_bases=False,
    )

    features = [{"input_ids": [65, 67, 71, 84]}]
    batch = collator(features)

    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] == 1


def test_collator_uppercases(tokenizer):
    """Test that collator uppercases labels while keeping inputs mixed case."""
    base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = GenomicDataCollator(
        base_collator=base,
        uppercase_labels=True,
        mask_degenerate_bases=False,
    )

    features = [{"input_ids": [97, 67, 103, 116]}]  # "aCgt"
    batch = collator(features)

    # Verify inputs unchanged (still mixed case)
    input_ids = batch["input_ids"]
    assert input_ids[0, 0].item() == 97, "Input 'a' (97) should stay lowercase"
    assert input_ids[0, 2].item() == 103, "Input 'g' (103) should stay lowercase"

    # Verify labels uppercased
    # Parent doesn't shift: labels = [97, 67, 103, 116] (same as input_ids)
    # Our uppercase: [97, 67, 103, 116] → [65, 67, 71, 84]
    #                 a   C   g    t   →   A   C   G   T
    labels = batch["labels"]
    expected_labels = torch.tensor([[65, 67, 71, 84]])  # All uppercase
    assert torch.equal(labels, expected_labels), f"Expected {expected_labels}, got {labels}"


def test_collator_masks_degenerate(tokenizer):
    """Test that collator masks degenerate bases (N, R, Y, etc.)."""
    base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = GenomicDataCollator(
        base_collator=base,
        uppercase_labels=False,
        mask_degenerate_bases=True,
    )

    features = [{"input_ids": [65, 67, 71, 84, 78]}]  # "ACGTN" (N is degenerate)
    batch = collator(features)

    # Parent creates labels = input_ids (no shift): [65, 67, 71, 84, 78]
    # Our degenerate masking: 78 (N) → -100
    # Expected: [65, 67, 71, 84, -100]
    #           A   C   G   T   MASKED
    labels = batch["labels"]
    expected_labels = torch.tensor([[65, 67, 71, 84, -100]])
    assert torch.equal(labels, expected_labels), f"Expected {expected_labels}, got {labels}"


def test_collator_combined(tokenizer):
    """Test collator with both uppercase and degenerate masking."""
    base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = GenomicDataCollator(
        base_collator=base,
        uppercase_labels=True,
        mask_degenerate_bases=True,
    )

    features = [{"input_ids": [97, 67, 103, 84, 78]}]  # "aCgTN" (mixed case + degenerate)
    batch = collator(features)

    # Verify inputs unchanged (still mixed case)
    input_ids = batch["input_ids"]
    assert input_ids[0, 0].item() == 97, "Input 'a' should stay lowercase"
    assert input_ids[0, 2].item() == 103, "Input 'g' should stay lowercase"

    # Verify labels after BOTH uppercase AND degenerate masking
    # Parent: labels = input_ids = [97, 67, 103, 84, 78]
    # Step 1 uppercase: [97, 67, 103, 84, 78] → [65, 67, 71, 84, 78]
    #                   a   C   g    T   N   →   A   C   G   T   N
    # Step 2 mask degenerate: [65, 67, 71, 84, 78] → [65, 67, 71, 84, -100]
    #                         A   C   G   T   N   →   A   C   G   T   MASKED
    # Expected: [65, 67, 71, 84, -100]
    labels = batch["labels"]
    expected_labels = torch.tensor([[65, 67, 71, 84, -100]])
    assert torch.equal(labels, expected_labels), f"Expected {expected_labels}, got {labels}"


def test_collator_handles_lowercase_degenerate(tokenizer):
    """Test that lowercase degenerate bases are handled correctly (uppercase then mask).

    Tests the order of operations: lowercase 'n' should be uppercased to 'N',
    then masked to -100.
    """
    base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = GenomicDataCollator(
        base_collator=base,
        uppercase_labels=True,
        mask_degenerate_bases=True,
    )

    # Input has lowercase degenerate: "ACn" (n=110)
    features = [{"input_ids": [65, 67, 110]}]
    batch = collator(features)

    # Verify exact output:
    # Parent: labels = input_ids = [65, 67, 110]
    # Step 1 uppercase: [65, 67, 110] → [65, 67, 78]
    #                   A   C   n    →   A   C   N (110→78)
    # Step 2 mask degenerate: [65, 67, 78] → [65, 67, -100]
    #                         A   C   N   →   A   C   MASKED
    # Expected: [65, 67, -100]
    labels = batch["labels"]
    expected_labels = torch.tensor([[65, 67, -100]])
    assert torch.equal(labels, expected_labels), f"Expected {expected_labels}, got {labels}"


def test_collator_masks_phylo_tags(tokenizer):
    """Test that collator masks phylogenetic tags."""
    base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = GenomicDataCollator(
        base_collator=base,
        uppercase_labels=False,
        mask_degenerate_bases=False,
        mask_phylo_tags=True,
    )

    # Input: "AC|d__Bacteria|TG" (simplified phylo tag)
    # ASCII: A=65, C=67, |=124, d=100, _=95, B=66, a=97, c=99, t=116, e=101, r=114, i=105, T=84, G=71
    features = [{"input_ids": [65, 67, 124, 100, 95, 95, 66, 97, 99, 124, 84, 71]}]
    batch = collator(features)

    labels = batch["labels"]

    # DNA before tag (AC) should be present
    assert 65 in labels, "A before tag should be present"
    assert 67 in labels, "C before tag should be present"

    # Tag characters should NOT be present (masked to -100)
    assert 100 not in labels, "d from d__Bacteria should be masked"
    assert 66 not in labels, "B from Bacteria should be masked"

    # DNA after tag (TG) should be present
    assert 84 in labels, "T after tag should be present"
    assert 71 in labels, "G after tag should be present"


def test_collator_phylo_middle_of_sequence(tokenizer):
    """Test phylo tag in middle of sequence (inspired by Evo2 tests)."""
    base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = GenomicDataCollator(
        base_collator=base,
        uppercase_labels=False,
        mask_degenerate_bases=False,
        mask_phylo_tags=True,
    )

    # "ATG|d__Bacteria|TCGA"
    sequence_str = "ATG|d__tag|TCGA"
    features = [{"input_ids": [ord(c) for c in sequence_str]}]
    batch = collator(features)

    labels = batch["labels"]
    # DNA before tag (ATG) should be present
    assert 65 in labels  # A
    assert 84 in labels  # T
    assert 71 in labels  # G
    # Tag chars should NOT be present
    assert 100 not in labels  # d
    assert 95 not in labels  # _
    # DNA after tag (TCGA) should be present
    assert 67 in labels  # C


def test_collator_all_three_features(tokenizer):
    """Test all three masking features together (production scenario).

    Tests: uppercase + degenerate + phylo all enabled
    Input: Mix of all scenarios
    """
    base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = GenomicDataCollator(
        base_collator=base,
        uppercase_labels=True,
        mask_degenerate_bases=True,
        mask_phylo_tags=True,
    )

    # Simpler test: "aCgTN" (mixed case + degenerate, no phylo)
    # This tests uppercase + degenerate together
    # a=97, C=67, g=103, T=84, N=78
    features = [{"input_ids": [97, 67, 103, 84, 78]}]
    batch = collator(features)

    # Verify inputs unchanged (mixed case preserved)
    input_ids = batch["input_ids"]
    assert input_ids[0, 0].item() == 97, "Input 'a' should stay lowercase"
    assert input_ids[0, 2].item() == 103, "Input 'g' should stay lowercase"

    labels = batch["labels"]

    # After all processing:
    # Degenerate: [97,67,103,84,78] → [97,67,103,84,-100] (N masked)
    # Phylo: no tags, so no change
    # Uppercase: [97,67,103,84,-100] → [65,67,71,84,-100]
    expected = torch.tensor([[65, 67, 71, 84, -100]])
    assert torch.equal(labels, expected), f"Expected {expected}, got {labels}"


def test_collator_partial_tag_at_end(tokenizer):
    """Test partial phylo tag at end of sequence (edge case from Evo2)."""
    base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = GenomicDataCollator(
        base_collator=base,
        uppercase_labels=False,
        mask_degenerate_bases=False,
        mask_phylo_tags=True,
    )

    # "ATG|r_" (partial tag at end)
    sequence_str = "ATG|r_"
    features = [{"input_ids": [ord(c) for c in sequence_str]}]
    batch = collator(features)

    labels = batch["labels"]
    # DNA (ATG) should be present
    assert 65 in labels  # A
    assert 84 in labels  # T
    assert 71 in labels  # G
    # Partial tag should be masked
    assert 114 not in labels  # r
    assert 95 not in labels  # _
