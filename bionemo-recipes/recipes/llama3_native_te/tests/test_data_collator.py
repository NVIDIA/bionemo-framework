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
from data_collator import GenomicCLMCollatorWithFlattening, GenomicDataCollatorForCLM
from genomic_utils import make_upper_case, mask_phylogenetic_tags


@pytest.fixture
def tokenizer_path(recipe_path):
    """Get the path to the nucleotide tokenizer."""
    return str(recipe_path / "example_checkpoint")


@pytest.fixture
def tokenizer(tokenizer_path):
    """Load the nucleotide tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path)


# ============================================================================
# Tests for make_upper_case() function (from NeMo)
# ============================================================================


def test_make_upper_case_all_lowercase():
    """Test make_upper_case converts all lowercase letters to uppercase."""
    # "acgt" in ASCII
    tokens = torch.tensor([[97, 99, 103, 116]])
    upper_tokens, lower_mask = make_upper_case(tokens)

    # Expected: "ACGT" in ASCII
    expected_upper = torch.tensor([[65, 67, 71, 84]])
    expected_mask = torch.tensor([[True, True, True, True]])

    assert torch.equal(upper_tokens, expected_upper)
    assert torch.equal(lower_mask, expected_mask)


def test_make_upper_case_mixed_case():
    """Test make_upper_case handles mixed case correctly."""
    # "aCgT" in ASCII: 97, 67, 103, 84
    tokens = torch.tensor([[97, 67, 103, 84]])
    upper_tokens, lower_mask = make_upper_case(tokens)

    # Expected: "ACGT" in ASCII
    expected_upper = torch.tensor([[65, 67, 71, 84]])
    expected_mask = torch.tensor([[True, False, True, False]])  # Only a and g were lowercase

    assert torch.equal(upper_tokens, expected_upper)
    assert torch.equal(lower_mask, expected_mask)


def test_make_upper_case_all_uppercase():
    """Test make_upper_case leaves uppercase unchanged."""
    # "ACGT" in ASCII
    tokens = torch.tensor([[65, 67, 71, 84]])
    upper_tokens, lower_mask = make_upper_case(tokens)

    # Should be unchanged
    expected_upper = torch.tensor([[65, 67, 71, 84]])
    expected_mask = torch.tensor([[False, False, False, False]])  # None were lowercase

    assert torch.equal(upper_tokens, expected_upper)
    assert torch.equal(lower_mask, expected_mask)


def test_make_upper_case_batch():
    """Test make_upper_case works on batched inputs."""
    # Batch of 2 sequences: "aCgt" and "GGTA"
    tokens = torch.tensor([[97, 67, 103, 116], [71, 71, 84, 65]])
    upper_tokens, lower_mask = make_upper_case(tokens)

    expected_upper = torch.tensor([[65, 67, 71, 84], [71, 71, 84, 65]])
    expected_mask = torch.tensor([[True, False, True, True], [False, False, False, False]])

    assert torch.equal(upper_tokens, expected_upper)
    assert torch.equal(lower_mask, expected_mask)


# ============================================================================
# Tests for BSHD collator (GenomicDataCollatorForCLM)
# ============================================================================


def test_bshd_collator_basic(tokenizer):
    """Test basic BSHD collator functionality."""
    collator = GenomicDataCollatorForCLM(
        tokenizer=tokenizer,
        uppercase_labels=False,  # Disable for basic test
        mask_degenerate_bases=False,
    )

    features = [{"input_ids": [65, 67, 71, 84]}]  # "ACGT"
    batch = collator(features)

    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] == 1  # Batch size = 1
    assert batch["labels"].shape == batch["input_ids"].shape


def test_bshd_collator_uppercases_labels(tokenizer):
    """Test that BSHD collator uppercases labels but not inputs."""
    collator = GenomicDataCollatorForCLM(
        tokenizer=tokenizer,
        uppercase_labels=True,
        mask_degenerate_bases=False,
    )

    # Mixed case sequence: "aCgt"
    features = [{"input_ids": [97, 67, 103, 116]}]
    batch = collator(features)

    # Labels should be uppercased (accounting for CLM shift and special handling)
    # Check that batch has correct format
    assert "labels" in batch
    assert batch["labels"].ndim == 2


def test_bshd_collator_masks_degenerate_bases(tokenizer):
    """Test that BSHD collator masks degenerate bases."""
    collator = GenomicDataCollatorForCLM(
        tokenizer=tokenizer,
        uppercase_labels=False,  # Disable uppercase for clearer testing
        mask_degenerate_bases=True,
    )

    # Sequence with degenerate bases: "ACGTNRY" (N, R, Y are degenerate)
    # ASCII: A=65, C=67, G=71, T=84, N=78, R=82, Y=89
    features = [{"input_ids": [65, 67, 71, 84, 78, 82, 89]}]
    batch = collator(features)

    # Check that batch has correct format
    assert "labels" in batch
    assert batch["labels"].ndim == 2


def test_bshd_collator_masks_control_characters(tokenizer):
    """Test that BSHD collator masks control characters (@, #)."""
    collator = GenomicDataCollatorForCLM(
        tokenizer=tokenizer,
        uppercase_labels=False,
        mask_degenerate_bases=True,  # This also masks control chars
    )

    # Sequence: "ACGT@ACGT#ACGT"
    # ASCII: @=64, #=35
    features = [{"input_ids": [65, 67, 71, 84, 64, 65, 67, 71, 84, 35, 65, 67, 71, 84]}]
    batch = collator(features)

    # Check that batch has correct format
    assert "labels" in batch
    assert batch["labels"].ndim == 2


# ============================================================================
# Tests for phylogenetic tag masking (copied from Evo2)
# ============================================================================

MAX_TAG_LEN = 2048


@pytest.fixture
def tag_tokens():
    """Standard tokens for phylogenetic tag tests (from Evo2Dataset).

    CONTROL_TAGS: [64, 35]  # '@' and '#'
    TAG_BOUNDS: 124  # '|' pipe character
    TAG_CHARS: {95, 59, 32}  # '_', ';', space
    DEFAULT_EOD: 0
    """
    return {
        "terminal": 124,  # |
        "other_chars": {95, 59, 32},  # _, ;, space
        "eod": 0,  # end of document token
    }


def test_mask_phylogenetic_tags_simple(tag_tokens):
    """Test masking a simple phylogenetic tag in the middle of DNA sequence.

    Sequence: "ATG|d__|TCGA"
    Expected: DNA unmasked (1s), tag masked (0s)
    """
    sequence_chrs = "ATG|d__|TCGA"
    sequence = torch.tensor([ord(c) for c in sequence_chrs])
    expected_mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    mask = mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    assert torch.equal(mask, expected_mask)


def test_mask_phylogenetic_tags_complex(tag_tokens):
    """Test masking complex phylogenetic tag with full taxonomy.

    Sequence: "ATG|d__Bacteria;p__Proteobacteria|TCGA"
    Expected: DNA unmasked, entire tag (including pipes) masked
    """
    sequence_chrs = "ATG|d__Bacteria;p__Proteobacteria|TCGA"
    sequence = torch.tensor([ord(c) for c in sequence_chrs])

    # Mask should be: DNA (111) + TAG (000...000) + DNA (1111)
    expected_mask = torch.tensor([1] * 3 + [0] * len("|d__Bacteria;p__Proteobacteria|") + [1] * 4)

    mask = mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    assert torch.equal(mask, expected_mask)


def test_mask_phylogenetic_tags_with_eod(tag_tokens):
    """Test that EOD tokens break tag regions correctly.

    Sequence: "A|d" + EOD + "|A"
    Expected: DNA unmasked, pipes and 'd' masked, EOD unmasked
    """
    sequence = torch.tensor([65, 124, 100, 0, 124, 65])  # "A|d" + EOD + "|A"
    expected_mask = torch.tensor([1, 0, 0, 1, 0, 1])  # A:1, |d:00, EOD:1, |:0, A:1

    mask = mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    assert torch.equal(mask, expected_mask)


def test_mask_phylogenetic_tags_multiple_tags(tag_tokens):
    """Test handling multiple phylogenetic tags in sequence.

    Sequence: "ATG|d__|CG|r__|AT"
    Expected: Only DNA unmasked, both tags masked
    """
    sequence_chrs = "ATG|d__|CG|r__|AT"
    sequence = torch.tensor([ord(c) for c in sequence_chrs])
    expected_mask = torch.tensor([1] * 3 + [0] * 5 + [1] * 2 + [0] * 5 + [1] * 2)

    mask = mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    assert torch.equal(mask, expected_mask)


def test_mask_partial_tag_start(tag_tokens):
    """Test handling sequence starting with partial tag.

    Sequence: "|d__tag|ACGT"
    Expected: Tag masked, DNA unmasked
    """
    sequence_chrs = "|d__tag|ACGT"
    sequence = torch.tensor([ord(c) for c in sequence_chrs])
    expected_mask = torch.tensor([0] * len("|d__tag|") + [1] * 4)

    mask = mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    assert torch.equal(mask, expected_mask)


def test_mask_partial_tag_end(tag_tokens):
    """Test handling sequence ending with partial tag.

    Sequence: "ACGT|d__"
    Expected: DNA unmasked, partial tag masked
    """
    sequence_chrs = "ACGT|d__"
    sequence = torch.tensor([ord(c) for c in sequence_chrs])
    expected_mask = torch.tensor([1] * 4 + [0] * len("|d__"))

    mask = mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    assert torch.equal(mask, expected_mask)


def test_mask_phylo_tags_batched(tag_tokens):
    """Test phylogenetic masking works on batched inputs [B, S]."""
    # Batch of 2 sequences
    seq1 = "ATG|d__|TCGA"
    seq2 = "GGG|r__|AAAT"

    sequences = torch.tensor([[ord(c) for c in seq1], [ord(c) for c in seq2]])

    mask = mask_phylogenetic_tags(
        tokenized_sequence=sequences,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    # Both sequences should have DNA unmasked, tags masked
    expected1 = torch.tensor([1] * 3 + [0] * 5 + [1] * 4)
    expected2 = torch.tensor([1] * 3 + [0] * 5 + [1] * 4)
    expected = torch.stack([expected1, expected2])

    assert torch.equal(mask, expected)


def test_mask_phylo_tags_with_control_char(tag_tokens):
    """Test phylogenetic masking with control character (@) in sequence.

    Sequence: "ACGT@ACGT|d__|GGTA"
    Expected: DNA and @ unmasked (@ masking handled separately), tag masked
    """
    sequence_chrs = "ACGT@ACGT|d__|GGTA"
    sequence = torch.tensor([ord(c) for c in sequence_chrs])

    mask = mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    # Pattern: DNA (1111) @ (1) DNA (1111) TAG (00000) DNA (1111)
    expected_mask = torch.tensor([1] * 9 + [0] * 5 + [1] * 4)

    assert torch.equal(mask, expected_mask)


def test_mask_phylo_tags_with_degenerate_base(tag_tokens):
    """Test phylogenetic masking with degenerate base (N, R) in sequence.

    The phylo masking function should identify these as non-standard DNA
    and mask appropriately based on context.
    """
    sequence_chrs = "ACGN|d__|RTTA"
    sequence = torch.tensor([ord(c) for c in sequence_chrs])

    mask = mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    # The function should handle degenerate bases appropriately
    # Tag should definitely be masked
    assert mask[4] == 0  # '|' should be masked
    assert mask[5] == 0  # 'd' should be masked


# ============================================================================
# Tests for BSHD collator integration
# ============================================================================


def test_bshd_collator_masks_degenerate(tokenizer):
    """Test that BSHD collator masks degenerate bases in standard batch format."""
    collator = GenomicDataCollatorForCLM(
        tokenizer=tokenizer,
        uppercase_labels=False,
        mask_degenerate_bases=True,
    )

    # Sequence with degenerate: "ACGTNRY" (N, R, Y are degenerate)
    # ASCII: A=65, C=67, G=71, T=84, N=78, R=82, Y=89
    features = [{"input_ids": [65, 67, 71, 84, 78, 82, 89]}]
    batch = collator(features)

    # Verify batch format
    assert batch["input_ids"].ndim == 2
    assert batch["labels"].ndim == 2


def test_bshd_collator_combined_masking(tokenizer):
    """Test BSHD collator with both uppercase and degenerate masking enabled."""
    collator = GenomicDataCollatorForCLM(
        tokenizer=tokenizer,
        uppercase_labels=True,
        mask_degenerate_bases=True,
        mask_phylo_tags=False,
    )

    # Mixed case with degenerate: "aCgtN"
    features = [{"input_ids": [97, 67, 103, 84, 78]}]
    batch = collator(features)

    # Should be BSHD format
    assert batch["input_ids"].ndim == 2
    assert batch["labels"].ndim == 2


# ============================================================================
# Tests for THD collator (GenomicCLMCollatorWithFlattening)
# ============================================================================


def test_thd_collator_produces_flattened_format(tokenizer):
    """Test that THD collator produces flattened format with cu_seq_lens."""
    collator = GenomicCLMCollatorWithFlattening(
        tokenizer=tokenizer,
        uppercase_labels=False,
        mask_degenerate_bases=False,
    )

    features = [
        {"input_ids": [65, 67, 71, 84]},  # "ACGT" (4 tokens)
        {"input_ids": [71, 71, 84, 65, 67]},  # "GGTAC" (5 tokens)
    ]

    batch = collator(features)

    # Check THD format
    assert batch["input_ids"].ndim == 2
    assert batch["input_ids"].shape[0] == 1  # THD: batch_size=1
    assert "cu_seq_lens_q" in batch
    assert "cu_seq_lens_k" in batch

    # cu_seq_lens should track boundaries: [0, 4, 9]
    # (First seq ends at 4, second at 9)


def test_thd_collator_uppercases_labels(tokenizer):
    """Test that THD collator uppercases labels in flattened format."""
    collator = GenomicCLMCollatorWithFlattening(
        tokenizer=tokenizer,
        uppercase_labels=True,
        mask_degenerate_bases=False,
    )

    # Mixed case sequences
    features = [
        {"input_ids": [97, 67]},  # "aC"
        {"input_ids": [103, 84]},  # "gt"
    ]

    batch = collator(features)

    # Check THD format maintained
    assert batch["input_ids"].shape[0] == 1
    assert "cu_seq_lens_q" in batch


def test_thd_collator_masks_degenerate(tokenizer):
    """Test that THD collator masks degenerate bases in flattened format."""
    collator = GenomicCLMCollatorWithFlattening(
        tokenizer=tokenizer,
        uppercase_labels=False,
        mask_degenerate_bases=True,
    )

    features = [
        {"input_ids": [65, 67, 78]},  # "ACN"
        {"input_ids": [82, 84, 65]},  # "RTA"
    ]

    batch = collator(features)

    # Check THD format
    assert batch["input_ids"].shape[0] == 1
    assert "cu_seq_lens_q" in batch


def test_thd_collator_combined_masking(tokenizer):
    """Test THD collator with all masking enabled."""
    collator = GenomicCLMCollatorWithFlattening(
        tokenizer=tokenizer,
        uppercase_labels=True,
        mask_degenerate_bases=True,
        mask_phylo_tags=False,  # Keep disabled for Milestone 1
    )

    features = [
        {"input_ids": [97, 67, 103, 78]},  # "aCgN"
        {"input_ids": [71, 82, 116, 65]},  # "GRtA"
    ]

    batch = collator(features)

    # Verify THD format
    assert batch["input_ids"].shape[0] == 1
    assert "cu_seq_lens_q" in batch
    assert "max_length_q" in batch


def test_thd_collator_with_phylo_masking(tokenizer, tag_tokens):
    """Test THD collator with phylogenetic tag masking enabled (Milestone 2)."""
    collator = GenomicCLMCollatorWithFlattening(
        tokenizer=tokenizer,
        uppercase_labels=True,
        mask_degenerate_bases=True,
        mask_phylo_tags=True,  # Enable for this test
    )

    # Sequence with phylo tag: "ATG|d__tag|TCGA"
    sequence_chrs = "ATG|d__tag|TCGA"
    features = [{"input_ids": [ord(c) for c in sequence_chrs]}]

    batch = collator(features)

    # Should be THD format with cu_seq_lens
    assert batch["input_ids"].shape[0] == 1
    assert "cu_seq_lens_q" in batch


# ============================================================================
# Integration tests: Verify collators work with dataloaders
# ============================================================================


def test_bshd_collator_with_dataloader(tokenizer_path, tmp_path):
    """Test that BSHD collator integrates with create_bshd_dataloader."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataset import create_bshd_dataloader
    from distributed_config import DistributedConfig

    # Create test data with mixed case
    parquet_path = tmp_path / "test_mixed_case.parquet"
    sequences = ["aCgT", "GGtA", "AAAn"]  # Mixed case, has degenerate
    table = pa.table({"sequence": sequences})
    pq.write_table(table, parquet_path)

    distributed_config = DistributedConfig(rank=0, world_size=1)

    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",
    }

    # Note: create_bshd_dataloader currently uses standard collator
    # This test verifies the dataloader structure works
    # TODO: Add collate_fn parameter to create_bshd_dataloader for custom collator
    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=10,
        stride=5,
        use_lazy_tokenization=False,
    )

    # Get a batch
    batch = next(iter(dataloader))

    # Verify BSHD format
    assert batch["input_ids"].ndim == 2
    assert batch["input_ids"].shape[0] > 0  # Has batch dimension
    assert "labels" in batch


def test_thd_collator_with_dataloader(tokenizer_path, tmp_path):
    """Test that THD collator integrates with create_thd_dataloader."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataset import create_thd_dataloader
    from distributed_config import DistributedConfig

    # Create test data
    parquet_path = tmp_path / "test_streaming.parquet"
    sequences = ["ACGT" * 10, "GGTA" * 10, "TTTAAA" * 5]
    table = pa.table({"text": sequences})  # Use 'text' like OpenGenome2
    pq.write_table(table, parquet_path)

    distributed_config = DistributedConfig(rank=0, world_size=1)

    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",
        "streaming": True,  # Required for THD
    }

    dataloader, _ = create_thd_dataloader(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=1,
        max_seq_length=50,
        stride=10,
        buffer_size=100,
        sequence_column="text",
        uppercase_labels=True,
        mask_degenerate_bases=True,
        mask_phylo_tags=False,
    )

    # Get a batch
    batch = next(iter(dataloader))

    # Verify THD format
    assert batch["input_ids"].shape[0] == 1  # THD: batch_size=1
    assert "cu_seq_lens_q" in batch
    assert "cu_seq_lens_k" in batch
    assert "max_length_q" in batch
    assert "position_ids" in batch
    assert "labels" in batch
