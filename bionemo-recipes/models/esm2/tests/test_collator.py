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

from unittest.mock import MagicMock

import pytest
import torch
from transformers import DataCollatorForLanguageModeling

from esm.collator import (
    DataCollatorWithFlattening,
    TokenPackingDataset,
    _split_sample_by_num_tokens,
)


def test_data_collator_with_flattening_basic(tokenizer):
    """Test DataCollatorWithFlattening with input_ids and attention_mask."""
    # Use DataCollatorForLanguageModeling with mlm_probability=0.0 to disable masking
    mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.0)
    collator = DataCollatorWithFlattening(collator=mlm_collator, return_position_ids=True)

    # Create test sequences of different lengths
    features = [
        {"input_ids": [0, 5, 6, 7, 2]},  # 5 tokens
        {"input_ids": [0, 8, 9, 10, 11, 2]},  # 6 tokens
        {"input_ids": [0, 12, 13, 2]},  # 4 tokens
    ]

    # Calculate expected total tokens
    total_tokens = sum(len(feature["input_ids"]) for feature in features)

    # Process batch
    batch = collator(features, return_tensors="pt")

    # Assert total number of tokens is unchanged
    input_ids_tensor = batch["input_ids"]
    assert input_ids_tensor.numel() == total_tokens, f"Expected {total_tokens} tokens, got {input_ids_tensor.numel()}"

    # Assert output shape is [1, total_tokens]
    assert input_ids_tensor.shape == (1, total_tokens), (
        f"Expected shape (1, {total_tokens}), got {input_ids_tensor.shape}"
    )

    # Assert cu_seqlens_q tensor is created properly
    expected_cu_seqlens = torch.tensor([0, 5, 11, 15], dtype=torch.int32)
    torch.testing.assert_close(batch["cu_seq_lens_q"], expected_cu_seqlens)
    torch.testing.assert_close(batch["cu_seq_lens_k"], expected_cu_seqlens)

    # Assert max_length values are correct
    assert batch["max_length_q"] == 6, f"Expected max_length_q=6, got {batch['max_length_q']}"
    assert batch["max_length_k"] == 6, f"Expected max_length_k=6, got {batch['max_length_k']}"

    # Assert position_ids are created properly
    expected_position_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3]], dtype=torch.int64)
    torch.testing.assert_close(batch["position_ids"], expected_position_ids)

    # Assert flattened input_ids matches concatenated original sequences
    expected_input_ids = torch.tensor([[0, 5, 6, 7, 2, 0, 8, 9, 10, 11, 2, 0, 12, 13, 2]], dtype=torch.int64)
    torch.testing.assert_close(input_ids_tensor, expected_input_ids)

    # Assert labels are present (DataCollatorForLanguageModeling always creates them)
    # With mlm_probability=0.0, all labels should be -100 (ignored)
    assert "labels" in batch
    assert (batch["labels"] == -100).all(), "With mlm_probability=0.0, all labels should be -100"


def test_data_collator_with_flattening_with_labels(tokenizer):
    """Test DataCollatorWithFlattening with input_ids, attention_mask, and labels."""
    # Use DataCollatorForLanguageModeling with mlm_probability=0.0 to disable masking
    # Note: DataCollatorForLanguageModeling ignores input labels and creates its own
    mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.0)
    collator = DataCollatorWithFlattening(collator=mlm_collator)

    # Create test sequences (labels will be created by DataCollatorForLanguageModeling)
    features = [
        {"input_ids": [0, 5, 6, 7, 2]},  # 5 tokens
        {"input_ids": [0, 8, 9, 10, 11, 2]},  # 6 tokens
        {"input_ids": [0, 12, 13, 2]},  # 4 tokens
    ]

    # Calculate expected total tokens
    total_tokens = sum(len(feature["input_ids"]) for feature in features)

    # Process batch
    batch = collator(features, return_tensors="pt")

    # Assert total number of tokens is unchanged
    input_ids_tensor = batch["input_ids"]
    labels_tensor = batch["labels"]
    assert input_ids_tensor.numel() == total_tokens, f"Expected {total_tokens} tokens, got {input_ids_tensor.numel()}"
    assert labels_tensor.numel() == total_tokens, f"Expected {total_tokens} label tokens, got {labels_tensor.numel()}"

    # Assert output shapes are [1, total_tokens]
    assert input_ids_tensor.shape == (1, total_tokens), (
        f"Expected input_ids shape (1, {total_tokens}), got {input_ids_tensor.shape}"
    )
    assert labels_tensor.shape == (1, total_tokens), (
        f"Expected labels shape (1, {total_tokens}), got {labels_tensor.shape}"
    )

    # Assert cu_seqlens_q tensor is created properly
    expected_cu_seqlens = torch.tensor([0, 5, 11, 15], dtype=torch.int32)
    torch.testing.assert_close(batch["cu_seq_lens_q"], expected_cu_seqlens)
    torch.testing.assert_close(batch["cu_seq_lens_k"], expected_cu_seqlens)

    # Assert max_length values are correct
    assert batch["max_length_q"] == 6, f"Expected max_length_q=6, got {batch['max_length_q']}"
    assert batch["max_length_k"] == 6, f"Expected max_length_k=6, got {batch['max_length_k']}"

    # Assert flattened input_ids match concatenated original sequences
    expected_input_ids = torch.tensor([[0, 5, 6, 7, 2, 0, 8, 9, 10, 11, 2, 0, 12, 13, 2]], dtype=torch.int64)
    torch.testing.assert_close(input_ids_tensor, expected_input_ids)

    # With mlm_probability=0.0, all labels should be -100 (ignored)
    assert (labels_tensor == -100).all(), "With mlm_probability=0.0, all labels should be -100"

    # Assert that sequence boundaries are properly maintained
    # by checking that token positions match expected values
    seq_lens = [5, 6, 4]  # lengths of the three sequences
    start_idx = 0
    for i, seq_len in enumerate(seq_lens):
        end_idx = start_idx + seq_len
        # Check that the sequence in the flattened tensor matches original
        original_seq = torch.tensor(features[i]["input_ids"], dtype=torch.int64)
        flattened_seq = input_ids_tensor[0, start_idx:end_idx]
        torch.testing.assert_close(flattened_seq, original_seq)
        start_idx = end_idx


def test_data_collator_pads_to_multiple_of(tokenizer):
    """Test DataCollatorWithFlattening with input_ids and attention_mask."""
    # Use DataCollatorForLanguageModeling with mlm_probability=0.0 to disable masking
    mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.0)
    collator = DataCollatorWithFlattening(collator=mlm_collator, pad_to_multiple_of=8, return_position_ids=True)

    # Create test sequences with labels
    features = [
        {"input_ids": [0, 5, 6, 7, 2]},  # 5 tokens
        {"input_ids": [0, 8, 9, 10, 11, 2]},  # 6 tokens
        {"input_ids": [0, 12, 13, 2]},  # 4 tokens
    ]

    # Process batch
    batch = collator(features)

    # Assert total number of tokens is unchanged
    assert batch["input_ids"].numel() == 16, f"Expected 16 tokens, got {batch['input_ids'].numel()}"

    # Assert output shape is [1, 16]
    assert batch["input_ids"].shape == (1, 16), f"Expected shape (1, 16), got {batch['input_ids'].shape}"

    # Assert cu_seqlens_q tensor is created properly
    expected_cu_seqlens = torch.tensor([0, 5, 11, 15, 16], dtype=torch.int32)
    torch.testing.assert_close(batch["cu_seq_lens_q"], expected_cu_seqlens)
    torch.testing.assert_close(batch["cu_seq_lens_k"], expected_cu_seqlens)

    # Assert input_ids are padded with 1
    assert batch["input_ids"][:, -1].item() == 1

    expected_position_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0]], dtype=torch.int64)
    torch.testing.assert_close(batch["position_ids"], expected_position_ids)


def test_mlm_data_collator_with_flattening_basic(tokenizer):
    """Test MLMDataCollatorWithFlattening with basic input_ids and verify labels are created."""
    mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    collator = DataCollatorWithFlattening(collator=mlm_collator, return_position_ids=True)

    # Create test sequences of different lengths
    features = [
        {"input_ids": [0, 5, 6, 7, 2]},  # CLS + amino acids + EOS (5 tokens)
        {"input_ids": [0, 8, 9, 10, 11, 2]},  # CLS + amino acids + EOS (6 tokens)
        {"input_ids": [0, 12, 13, 14, 15, 16, 2]},  # CLS + amino acids + EOS (7 tokens)
    ]

    # Calculate expected total tokens
    total_tokens = sum(len(feature["input_ids"]) for feature in features)

    # Process batch
    batch = collator(features, return_tensors="pt")

    # Assert that input_ids and labels exist and have the same shape
    assert "input_ids" in batch, "input_ids should be present in batch"
    assert "labels" in batch, "labels should be present in batch"

    input_ids_tensor = batch["input_ids"]
    labels_tensor = batch["labels"]

    # Assert both tensors have the same shape
    assert input_ids_tensor.shape == labels_tensor.shape, (
        f"input_ids and labels should have same shape, "
        f"got input_ids: {input_ids_tensor.shape}, labels: {labels_tensor.shape}"
    )

    # Assert total number of tokens is unchanged
    assert input_ids_tensor.numel() == total_tokens, f"Expected {total_tokens} tokens, got {input_ids_tensor.numel()}"
    assert labels_tensor.numel() == total_tokens, f"Expected {total_tokens} label tokens, got {labels_tensor.numel()}"

    # Assert output shape is [1, total_tokens]
    assert input_ids_tensor.shape == (1, total_tokens), (
        f"Expected shape (1, {total_tokens}), got {input_ids_tensor.shape}"
    )

    # Assert cu_seqlens_q tensor is created properly
    expected_cu_seqlens = torch.tensor([0, 5, 11, 18], dtype=torch.int32)
    torch.testing.assert_close(batch["cu_seq_lens_q"], expected_cu_seqlens)
    torch.testing.assert_close(batch["cu_seq_lens_k"], expected_cu_seqlens)

    # Assert max_length values are correct
    assert batch["max_length_q"] == 7, f"Expected max_length_q=7, got {batch['max_length_q']}"
    assert batch["max_length_k"] == 7, f"Expected max_length_k=7, got {batch['max_length_k']}"

    # Assert that Flash Attention metadata is present
    assert "cu_seq_lens_q" in batch, "cu_seq_lens_q should be present for Flash Attention"
    assert "cu_seq_lens_k" in batch, "cu_seq_lens_k should be present for Flash Attention"
    assert "max_length_q" in batch, "max_length_q should be present for Flash Attention"
    assert "max_length_k" in batch, "max_length_k should be present for Flash Attention"

    # Assert that position_ids are created properly
    expected_position_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6]], dtype=torch.int64)
    torch.testing.assert_close(batch["position_ids"], expected_position_ids)


def test_mlm_data_collator_with_flattening_masking(tokenizer, test_proteins):
    """Test MLMDataCollatorWithFlattening with reproducible masking using a seed."""
    # Use a fixed seed for reproducibility
    mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, seed=42)
    collator = DataCollatorWithFlattening(collator=mlm_collator)

    features = [tokenizer(protein) for protein in test_proteins]

    # Calculate expected total tokens
    total_tokens = sum(len(feature["input_ids"]) for feature in features)

    batch = collator(features, return_tensors="pt")

    # Assert shapes match
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape == (1, total_tokens)

    # Create original flattened sequence for comparison
    original_flattened = torch.tensor(
        [[token for sample in features for token in sample["input_ids"]]], dtype=torch.int64
    )

    # Assert that at least one token has been masked (i.e., input differs from original)
    # Since we have 56 total tokens with 15% masking probability, we should get some masking
    num_masked_positions = (batch["labels"] != -100).sum().item()
    assert num_masked_positions > 0, "At least one token should be masked with this much input data"

    # For positions where labels != -100, verify that:
    # 1. The label contains the original token
    # 2. The input_ids at that position might be MASK (4), original token, or random token
    mask_positions = batch["labels"] != -100

    # Check that labels at masked positions contain the original tokens
    original_tokens_at_mask_positions = original_flattened[mask_positions]
    labels_at_mask_positions = batch["labels"][mask_positions]
    torch.testing.assert_close(labels_at_mask_positions, original_tokens_at_mask_positions)

    # Check that non-masked positions have labels = -100
    non_mask_positions = batch["labels"] == -100
    assert non_mask_positions.sum() == total_tokens - num_masked_positions

    # Assert that special tokens (CLS=0, EOS=2) are never masked
    # Find positions of special tokens in the original sequence
    cls_positions = original_flattened == 0
    eos_positions = original_flattened == 2
    special_token_positions = cls_positions | eos_positions

    # Assert that labels at special token positions are -100 (not masked)
    assert (batch["labels"][special_token_positions] == -100).all(), "Special tokens (CLS, EOS) should never be masked"

    # Assert that the attention mask is all ones
    assert (batch["attention_mask"] == 1).all()


def test_mlm_data_collator_with_flattening_pad_to_multiple_of(tokenizer, test_proteins):
    """Test MLMDataCollatorWithFlattening with pad_to_multiple_of."""

    total_tokens = sum(len(tokenizer(protein)["input_ids"]) for protein in test_proteins)
    remainder = -total_tokens % 8
    assert remainder != 0, "Test assumes we need to pad to reach a multiple of 8"

    mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    collator = DataCollatorWithFlattening(collator=mlm_collator, pad_to_multiple_of=8)

    features = [tokenizer(protein) for protein in test_proteins]

    batch = collator(features)

    # Assert total number of tokens is divisible by 8
    assert batch["input_ids"].numel() % 8 == 0, (
        f"Expected {batch['input_ids'].numel()} tokens to be divisible by 8, got {batch['input_ids'].numel()}"
    )

    # Assert output shape is [1, total_tokens]
    assert batch["input_ids"].shape == (1, batch["input_ids"].numel()), (
        f"Expected shape (1, {batch['input_ids'].numel()}), got {batch['input_ids'].shape}"
    )

    # Assert that the last tokens are padding tokens
    assert (batch["input_ids"][:, -remainder:] == tokenizer.pad_token_id).all()

    # Assert that the last labels are masked
    assert (batch["labels"][:, -remainder:] == -100).all()

    # cu_seq_lens is usually len(input) + 1, but we also add one for the mock padding sequence.
    assert len(batch["cu_seq_lens_q"]) == len(test_proteins) + 2
    assert len(batch["cu_seq_lens_k"]) == len(test_proteins) + 2

    # The remainder must be less than the max length of the sequences
    assert batch["max_length_q"] == max(len(feature["input_ids"]) for feature in features)
    assert batch["max_length_k"] == max(len(feature["input_ids"]) for feature in features)

    # Assert that the attention mask is padded with zeros
    assert (batch["attention_mask"][:, -remainder:] == 0).all()
    assert (batch["attention_mask"][:, :-remainder] == 1).all()


def test_mlm_data_collator_with_flattening_bshd_equivalent(tokenizer, test_proteins):
    """Test MLMDataCollatorWithFlattening with bshd_equivalent=True."""
    # Create separate collator instances with the same seed to ensure matching masking
    # The BSHD collator pads to 256
    bshd_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        seed=42,
        pad_to_multiple_of=256,
    )
    thd_collator = DataCollatorWithFlattening(
        collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            seed=42,
            pad_to_multiple_of=256,
        )
    )

    features = [tokenizer(protein) for protein in test_proteins]

    # Do the masking a bunch of times to make sure the random numerics continue to match.
    for _ in range(25):
        thd_batch = thd_collator(features)
        bshd_batch = bshd_collator(features)

        packed_bshd_inputs = bshd_batch["input_ids"][bshd_batch["attention_mask"].to(bool)].unsqueeze(0)
        packed_bshd_labels = bshd_batch["labels"][bshd_batch["attention_mask"].to(bool)].unsqueeze(0)

        # Since we pad out the THD inputs beyond the packed BSHD inputs (for FP8 compatibility), we truncate the THD
        # inputs before comparing.
        torch.testing.assert_close(
            thd_batch["input_ids"][:, : packed_bshd_inputs.shape[1]],
            packed_bshd_inputs,
        )

        torch.testing.assert_close(
            thd_batch["labels"][:, : packed_bshd_labels.shape[1]],
            packed_bshd_labels,
        )


def test_mlm_data_collator_with_flattening_pad_sequences_to_be_divisible_by(tokenizer, test_proteins):
    """Test MLMDataCollatorWithFlattening with pad_sequences_to_be_divisible_by."""
    mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    collator = DataCollatorWithFlattening(collator=mlm_collator, pad_sequences_to_be_divisible_by=16)
    features = [tokenizer(protein) for protein in test_proteins]
    batch = collator(features)
    assert batch["input_ids"].numel() % 16 == 0, (
        f"Expected {batch['input_ids'].numel()} tokens to be divisible by 16, got {batch['input_ids'].numel()}"
    )
    assert batch["input_ids"].shape == (1, batch["input_ids"].numel()), (
        f"Expected shape (1, {batch['input_ids'].numel()}), got {batch['input_ids'].shape}"
    )
    assert (batch["input_ids"][:, -1] == tokenizer.pad_token_id).all()
    assert (batch["labels"][:, -1] == -100).all()


def test_token_packing_dataset():
    """Test that the token packing dataset works."""

    class MockDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            while True:
                yield {"input_ids": torch.arange(torch.randint(1, 100, (1,)).item())}

    dataset = MockDataset()
    token_packing_dataset = TokenPackingDataset(dataset, max_tokens_per_batch=1000)
    for i, batch in enumerate(token_packing_dataset):
        if i == 10:
            break
        total_length = sum([len(sample["input_ids"]) for sample in batch])
        assert 900 <= total_length <= 1000


def test_token_packing_dataset_multiple_epochs():
    """Test that the token packing dataset works over multiple epochs."""

    class MockDataset(torch.utils.data.IterableDataset):
        set_epoch = MagicMock()

        def __init__(self):
            self.data = [{"input_ids": torch.arange(torch.randint(1, 100, (1,)).item())} for _ in range(25)]

        def __iter__(self):
            return iter(self.data)

    dataset = MockDataset()
    token_packing_dataset = TokenPackingDataset(dataset, max_tokens_per_batch=200)
    token_packing_dataset.set_epoch(0)
    MockDataset.set_epoch.assert_called_with(0)
    epoch1 = list(token_packing_dataset)
    token_packing_dataset.set_epoch(1)
    MockDataset.set_epoch.assert_called_with(1)
    epoch2 = list(token_packing_dataset)

    # Make sure each epoch contains some number of samples
    assert len(epoch1) > 0
    assert len(epoch2) > 0


def test_token_packing_dataset_last_sequence_exceeds_max():
    """Test when last sequence would push over token_batch_size - should yield current batch."""

    class MockDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield {"input_ids": torch.arange(40)}  # 40 tokens
            yield {"input_ids": torch.arange(40)}  # 40 tokens (total: 80)
            yield {"input_ids": torch.arange(30)}  # 30 tokens (would exceed 100)

    dataset = MockDataset()
    token_packing_dataset = TokenPackingDataset(dataset, max_tokens_per_batch=100, drop_last=False)
    batches = list(token_packing_dataset)

    # Should have 2 batches: first with 2 sequences (80 tokens), second with 1 sequence (30 tokens)
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert sum(len(sample["input_ids"]) for sample in batches[0]) == 80
    assert len(batches[1]) == 1
    assert sum(len(sample["input_ids"]) for sample in batches[1]) == 30


def test_token_packing_dataset_last_sequence_equals_max():
    """Test when last sequence makes total equal to token_batch_size - should add to batch."""

    class MockDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield {"input_ids": torch.arange(40)}  # 40 tokens
            yield {"input_ids": torch.arange(30)}  # 30 tokens (total: 70)
            yield {"input_ids": torch.arange(30)}  # 30 tokens (total: 100, exactly max)

    dataset = MockDataset()
    token_packing_dataset = TokenPackingDataset(dataset, max_tokens_per_batch=100, drop_last=False)
    batches = list(token_packing_dataset)

    # Should have 1 batch with all 3 sequences totaling exactly 100 tokens
    assert len(batches) == 1
    assert len(batches[0]) == 3
    assert sum(len(sample["input_ids"]) for sample in batches[0]) == 100


def test_token_packing_dataset_last_sequence_less_than_max():
    """Test when last sequence gives less than token_batch_size - should add to batch."""

    class MockDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield {"input_ids": torch.arange(40)}  # 40 tokens
            yield {"input_ids": torch.arange(30)}  # 30 tokens (total: 70)
            yield {"input_ids": torch.arange(20)}  # 20 tokens (total: 90, less than max)

    dataset = MockDataset()
    token_packing_dataset = TokenPackingDataset(dataset, max_tokens_per_batch=100, drop_last=False)
    batches = list(token_packing_dataset)

    # Should have 1 batch with all 3 sequences totaling 90 tokens (less than max)
    assert len(batches) == 1
    assert len(batches[0]) == 3
    assert sum(len(sample["input_ids"]) for sample in batches[0]) == 90


def test__split_sample_by_num_tokens_basic():
    """Test _split_sample_by_num_tokens with basic input_ids."""
    sample = {"input_ids": [0, 5, 6, 7, 8, 9, 2]}
    first, remaining = _split_sample_by_num_tokens(sample, 3)

    assert first["input_ids"] == [0, 5, 6]
    assert remaining["input_ids"] == [7, 8, 9, 2]
    assert len(first["input_ids"]) == 3
    assert len(remaining["input_ids"]) == 4


def test__split_sample_by_num_tokens_with_labels():
    """Test _split_sample_by_num_tokens with input_ids and labels."""
    sample = {"input_ids": [0, 5, 6, 7, 8, 2], "labels": [0, 5, 6, 7, 8, 2]}
    first, remaining = _split_sample_by_num_tokens(sample, 3)

    assert first["input_ids"] == [0, 5, 6]
    assert first["labels"] == [0, 5, 6]
    assert remaining["input_ids"] == [7, 8, 2]
    assert remaining["labels"] == [7, 8, 2]


def test__split_sample_by_num_tokens_with_attention_mask():
    """Test _split_sample_by_num_tokens with input_ids, attention_mask, and labels."""
    sample = {
        "input_ids": [0, 5, 6, 7, 8, 2],
        "attention_mask": [1, 1, 1, 1, 1, 1],
        "labels": [0, 5, 6, 7, 8, 2],
    }
    first, remaining = _split_sample_by_num_tokens(sample, 4)

    assert first["input_ids"] == [0, 5, 6, 7]
    assert first["attention_mask"] == [1, 1, 1, 1]
    assert first["labels"] == [0, 5, 6, 7]
    assert remaining["input_ids"] == [8, 2]
    assert remaining["attention_mask"] == [1, 1]
    assert remaining["labels"] == [8, 2]


def test__split_sample_by_num_tokens_with_token_type_ids():
    """Test _split_sample_by_num_tokens with token_type_ids."""
    sample = {
        "input_ids": [0, 5, 6, 7, 8, 2],
        "token_type_ids": [0, 0, 0, 1, 1, 1],
        "labels": [0, 5, 6, 7, 8, 2],
    }
    first, remaining = _split_sample_by_num_tokens(sample, 3)

    assert first["input_ids"] == [0, 5, 6]
    assert first["token_type_ids"] == [0, 0, 0]
    assert first["labels"] == [0, 5, 6]
    assert remaining["input_ids"] == [7, 8, 2]
    assert remaining["token_type_ids"] == [1, 1, 1]
    assert remaining["labels"] == [7, 8, 2]


def test__split_sample_by_num_tokens_with_token_type():
    """Test _split_sample_by_num_tokens with token_type (alternative name)."""
    sample = {
        "input_ids": [0, 5, 6, 7, 8, 2],
        "token_type": [0, 0, 0, 1, 1, 1],
        "labels": [0, 5, 6, 7, 8, 2],
    }
    first, remaining = _split_sample_by_num_tokens(sample, 3)

    assert first["input_ids"] == [0, 5, 6]
    assert first["token_type"] == [0, 0, 0]
    assert first["labels"] == [0, 5, 6]
    assert remaining["input_ids"] == [7, 8, 2]
    assert remaining["token_type"] == [1, 1, 1]
    assert remaining["labels"] == [7, 8, 2]


def test__split_sample_by_num_tokens_with_tensors():
    """Test _split_sample_by_num_tokens with torch tensors."""
    sample = {
        "input_ids": torch.tensor([0, 5, 6, 7, 8, 2]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1]),
        "labels": torch.tensor([0, 5, 6, 7, 8, 2]),
    }
    first, remaining = _split_sample_by_num_tokens(sample, 3)

    assert torch.equal(first["input_ids"], torch.tensor([0, 5, 6]))
    assert torch.equal(first["attention_mask"], torch.tensor([1, 1, 1]))
    assert torch.equal(first["labels"], torch.tensor([0, 5, 6]))
    assert torch.equal(remaining["input_ids"], torch.tensor([7, 8, 2]))
    assert torch.equal(remaining["attention_mask"], torch.tensor([1, 1, 1]))
    assert torch.equal(remaining["labels"], torch.tensor([7, 8, 2]))


def test__split_sample_by_num_tokens_with_metadata():
    """Test _split_sample_by_num_tokens preserves non-sequence fields."""
    sample = {
        "input_ids": [0, 5, 6, 7, 8, 2],
        "labels": [0, 5, 6, 7, 8, 2],
        "metadata": {"id": 123, "source": "test"},
    }
    first, remaining = _split_sample_by_num_tokens(sample, 3)

    # Sequence fields should be split
    assert first["input_ids"] == [0, 5, 6]
    assert remaining["input_ids"] == [7, 8, 2]

    # Metadata should be copied to both parts
    assert first["metadata"] == {"id": 123, "source": "test"}
    assert remaining["metadata"] == {"id": 123, "source": "test"}


def test__split_sample_by_num_tokens_errors():
    """Test _split_sample_by_num_tokens raises errors for invalid inputs."""
    sample = {"input_ids": [0, 5, 6, 7, 2]}

    # num_tokens >= sample_length should raise ValueError
    with pytest.raises(ValueError, match="num_tokens.*must be less than sample length"):
        _split_sample_by_num_tokens(sample, 5)

    with pytest.raises(ValueError, match="num_tokens.*must be less than sample length"):
        _split_sample_by_num_tokens(sample, 10)

    # num_tokens <= 0 should raise ValueError
    with pytest.raises(ValueError, match="num_tokens.*must be positive"):
        _split_sample_by_num_tokens(sample, 0)

    with pytest.raises(ValueError, match="num_tokens.*must be positive"):
        _split_sample_by_num_tokens(sample, -1)


def test_token_packing_dataset_with_split_samples():
    """Test TokenPackingDataset with split_samples=True ensures exact batch sizes."""

    class MockDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield {"input_ids": torch.arange(40)}  # 40 tokens
            yield {"input_ids": torch.arange(50)}  # 50 tokens
            yield {"input_ids": torch.arange(30)}  # 30 tokens

    dataset = MockDataset()
    token_packing_dataset = TokenPackingDataset(dataset, max_tokens_per_batch=100, split_samples=True, drop_last=False)
    batches = list(token_packing_dataset)

    # First batch should have exactly 100 tokens (40 + 50 + 10 from the 30-token sample)
    assert len(batches) >= 1
    assert sum(len(sample["input_ids"]) for sample in batches[0]) == 100

    # Second batch should start with the remaining 20 tokens from the split sample
    if len(batches) > 1:
        assert sum(len(sample["input_ids"]) for sample in batches[1]) == 20


def test_token_packing_dataset_with_split_samples_exact_fit():
    """Test TokenPackingDataset with split_samples=True when samples exactly fill batches."""

    class MockDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield {"input_ids": torch.arange(50)}  # 50 tokens
            yield {"input_ids": torch.arange(50)}  # 50 tokens (total: 100, exactly max)

    dataset = MockDataset()
    token_packing_dataset = TokenPackingDataset(dataset, max_tokens_per_batch=100, split_samples=True, drop_last=False)
    batches = list(token_packing_dataset)

    # Should have 1 batch with exactly 100 tokens
    assert len(batches) == 1
    assert sum(len(sample["input_ids"]) for sample in batches[0]) == 100


def test_token_packing_dataset_with_split_samples_multiple_fields():
    """Test TokenPackingDataset with split_samples=True handles multiple fields correctly."""

    class MockDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield {
                "input_ids": torch.arange(40),
                "attention_mask": torch.ones(40),
                "labels": torch.arange(40),
            }
            yield {
                "input_ids": torch.arange(50),
                "attention_mask": torch.ones(50),
                "labels": torch.arange(50),
            }
            yield {
                "input_ids": torch.arange(30),
                "attention_mask": torch.ones(30),
                "labels": torch.arange(30),
            }

    dataset = MockDataset()
    token_packing_dataset = TokenPackingDataset(dataset, max_tokens_per_batch=100, split_samples=True, drop_last=False)
    batches = list(token_packing_dataset)

    # First batch should have exactly 100 tokens
    assert len(batches) >= 1
    first_batch_total = sum(len(sample["input_ids"]) for sample in batches[0])
    assert first_batch_total == 100

    # Second batch should have exactly 20 tokens
    second_batch_total = sum(len(sample["input_ids"]) for sample in batches[1])
    assert second_batch_total == 20

    # Verify all fields are present and consistent
    for sample in batches[0]:
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample
        assert len(sample["input_ids"]) == len(sample["attention_mask"])
        assert len(sample["input_ids"]) == len(sample["labels"])
