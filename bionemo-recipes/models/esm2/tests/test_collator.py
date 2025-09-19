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

import torch

from esm.collator import DataCollatorWithFlattening, MLMDataCollatorWithFlattening

# testing
def test_data_collator_with_flattening_basic():
    """Test DataCollatorWithFlattening with input_ids and attention_mask."""
    collator = DataCollatorWithFlattening()

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

    # Assert flattened input_ids matches concatenated original sequences
    expected_input_ids = torch.tensor([[0, 5, 6, 7, 2, 0, 8, 9, 10, 11, 2, 0, 12, 13, 2]], dtype=torch.int64)
    torch.testing.assert_close(input_ids_tensor, expected_input_ids)

    # Assert labels are not present when not provided in input
    assert "labels" not in batch


def test_data_collator_with_flattening_with_labels():
    """Test DataCollatorWithFlattening with input_ids, attention_mask, and labels."""
    collator = DataCollatorWithFlattening()

    # Create test sequences with labels
    features = [
        {"input_ids": [0, 5, 6, 7, 2], "labels": [0, 5, 6, 7, 2]},  # 5 tokens
        {"input_ids": [0, 8, 9, 10, 11, 2], "labels": [0, 8, 9, 10, 11, 2]},  # 6 tokens
        {"input_ids": [0, 12, 13, 2], "labels": [0, 12, 13, 2]},  # 4 tokens
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

    # Assert flattened input_ids and labels match concatenated original sequences
    expected_input_ids = torch.tensor([[0, 5, 6, 7, 2, 0, 8, 9, 10, 11, 2, 0, 12, 13, 2]], dtype=torch.int64)
    expected_labels = torch.tensor([[0, 5, 6, 7, 2, 0, 8, 9, 10, 11, 2, 0, 12, 13, 2]], dtype=torch.int64)

    torch.testing.assert_close(input_ids_tensor, expected_input_ids)
    torch.testing.assert_close(labels_tensor, expected_labels)

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


def test_data_collator_pads_to_multiple_of():
    """Test DataCollatorWithFlattening with input_ids and attention_mask."""
    collator = DataCollatorWithFlattening(pad_to_multiple_of=8, token_pad=1, label_pad=-100)

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


def test_mlm_data_collator_with_flattening_basic(tokenizer):
    """Test MLMDataCollatorWithFlattening with basic input_ids and verify labels are created."""
    collator = MLMDataCollatorWithFlattening(
        tokenizer=tokenizer,
        mlm_probability=0.15,
    )

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


def test_mlm_data_collator_with_flattening_masking(tokenizer, test_proteins):
    """Test MLMDataCollatorWithFlattening with reproducible masking using a seed."""
    # Use a fixed seed for reproducibility
    collator = MLMDataCollatorWithFlattening(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        seed=42,
    )

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

    collator = MLMDataCollatorWithFlattening(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=8,
    )

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
