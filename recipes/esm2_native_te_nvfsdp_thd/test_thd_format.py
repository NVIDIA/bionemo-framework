#!/usr/bin/env python3

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

"""
Unit tests to verify THD format compliance of DataCollatorWithFlattening
"""

import torch
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling, DataCollatorWithFlattening

from dataset import MLMDataCollatorWithFlattening


# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")


def create_test_batch():
    """Create a test batch with variable length sequences"""
    # Create some fake tokenized sequences of different lengths
    # DataCollatorWithFlattening expects input_ids as lists, not tensors
    sequences = [
        {"input_ids": [0, 5, 10, 15, 20, 1], "attention_mask": [1, 1, 1, 1, 1, 1]},
        {"input_ids": [0, 25, 30, 35, 1], "attention_mask": [1, 1, 1, 1, 1]},
        {"input_ids": [0, 40, 45, 50, 55, 60, 65, 1], "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1]},
    ]
    return sequences


def test_thd_format():
    """Test and verify THD format compliance"""
    # Create test data
    batch = create_test_batch()

    # Create the data collator
    data_collator = MLMDataCollatorWithFlattening(
        DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            seed=42,
        ),
        DataCollatorWithFlattening(
            return_flash_attn_kwargs=True,
        ),
    )

    # Process the batch
    sample = data_collator(batch)

    # Verify required keys are present
    required_keys = ["input_ids", "labels", "cu_seq_lens_q"]
    for key in required_keys:
        assert key in sample, f"Required key '{key}' missing from sample"

    # Verify THD format compliance
    input_ids = sample["input_ids"]
    cu_seq_lens = sample["cu_seq_lens_q"]
    position_ids = sample.get("position_ids")
    labels = sample["labels"]

    # Check if it's properly flattened (THD format requirement)
    batch_size, total_length = input_ids.shape
    assert batch_size == 1, f"Batch size should be 1 for flattened format, got {batch_size}"
    assert total_length > 0, f"Total sequence length should be > 0, got {total_length}"

    # Verify Flash Attention metadata is present
    assert cu_seq_lens is not None, "cu_seq_lens_q should be provided for Flash Attention"
    assert len(cu_seq_lens) >= 2, f"cu_seq_lens should have at least 2 elements, got {len(cu_seq_lens)}"

    # Verify sequence boundaries using cu_seq_lens
    num_sequences = len(cu_seq_lens) - 1
    assert num_sequences == len(batch), (
        f"Number of sequences in cu_seq_lens ({num_sequences}) should match batch size ({len(batch)})"
    )

    # Calculate and verify individual sequence lengths
    seq_lengths = []
    for i in range(num_sequences):
        seq_len = cu_seq_lens[i + 1] - cu_seq_lens[i]
        seq_lengths.append(seq_len.item())
        assert seq_len > 0, f"Sequence {i + 1} should have positive length, got {seq_len}"

    # Verify total length matches sum of individual sequences
    total_seq_length = sum(seq_lengths)
    assert total_seq_length == total_length, (
        f"Sum of sequence lengths ({total_seq_length}) should match total length ({total_length})"
    )

    # Verify position_ids reset properly at sequence boundaries if present
    if position_ids is not None:
        position_ids_flat = position_ids.flatten()
        assert position_ids_flat.shape[0] == total_length, "Position IDs should have same length as input_ids"

        for i in range(num_sequences):
            start_idx = cu_seq_lens[i].item()
            end_idx = cu_seq_lens[i + 1].item()
            seq_positions = position_ids_flat[start_idx:end_idx]

            # Check if positions start from 0 and increment properly
            expected_positions = torch.arange(len(seq_positions))
            assert torch.equal(seq_positions, expected_positions), (
                f"Position IDs for sequence {i + 1} should start from 0 and increment by 1"
            )

    # Verify MLM masking is applied
    assert labels.shape == input_ids.shape, "Labels should have same shape as input_ids"
    masked_positions = (labels != -100).sum()
    total_positions = labels.numel()
    masking_ratio = masked_positions.float() / total_positions

    # MLM masking should be approximately 15% (allow some variance)
    assert 0.05 <= masking_ratio <= 0.25, f"MLM masking ratio should be ~15%, got {masking_ratio:.1%}"

    # Verify Flash Attention compatibility
    assert "max_length_q" in sample or "max_length_k" in sample, (
        "Flash Attention metadata (max_length_q or max_length_k) should be present"
    )


def test_thd_format_with_different_batch_sizes():
    """Test THD format with different numbers of sequences"""
    # Test with single sequence
    single_batch = [{"input_ids": [0, 5, 10, 15, 1], "attention_mask": [1, 1, 1, 1, 1]}]

    data_collator = MLMDataCollatorWithFlattening(
        DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, seed=42),
        DataCollatorWithFlattening(return_flash_attn_kwargs=True),
    )

    sample = data_collator(single_batch)

    # Verify single sequence handling
    assert sample["input_ids"].shape[0] == 1, "Should have batch size 1"
    assert len(sample["cu_seq_lens_q"]) == 2, "Single sequence should have cu_seq_lens of length 2"
    assert sample["cu_seq_lens_q"][0] == 0, "First cu_seq_lens should be 0"
    assert sample["cu_seq_lens_q"][1] == sample["input_ids"].shape[1], "Last cu_seq_lens should equal total length"


def test_thd_format_sequence_lengths():
    """Test that sequence lengths are preserved correctly in THD format"""
    batch = create_test_batch()
    original_lengths = [len(seq["input_ids"]) for seq in batch]

    data_collator = MLMDataCollatorWithFlattening(
        DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm_probability=0.0, seed=42
        ),  # No masking for length test
        DataCollatorWithFlattening(return_flash_attn_kwargs=True),
    )

    sample = data_collator(batch)
    cu_seq_lens = sample["cu_seq_lens_q"]

    # Verify each sequence length is preserved
    for i, original_len in enumerate(original_lengths):
        actual_len = cu_seq_lens[i + 1] - cu_seq_lens[i]
        assert actual_len == original_len, f"Sequence {i} length mismatch: expected {original_len}, got {actual_len}"


def test_thd_format_tensor_types():
    """Test that all tensors have correct types and devices"""
    batch = create_test_batch()

    data_collator = MLMDataCollatorWithFlattening(
        DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, seed=42),
        DataCollatorWithFlattening(return_flash_attn_kwargs=True),
    )

    sample = data_collator(batch)

    # Verify tensor types
    assert isinstance(sample["input_ids"], torch.Tensor), "input_ids should be a tensor"
    assert isinstance(sample["labels"], torch.Tensor), "labels should be a tensor"
    assert isinstance(sample["cu_seq_lens_q"], torch.Tensor), "cu_seq_lens_q should be a tensor"

    # Verify dtypes
    assert sample["input_ids"].dtype == torch.long, f"input_ids should be long tensor, got {sample['input_ids'].dtype}"
    assert sample["labels"].dtype == torch.long, f"labels should be long tensor, got {sample['labels'].dtype}"
    assert sample["cu_seq_lens_q"].dtype == torch.int32, (
        f"cu_seq_lens_q should be int32 tensor, got {sample['cu_seq_lens_q'].dtype}"
    )

    # Verify all tensors are on same device
    device = sample["input_ids"].device
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            assert value.device == device, (
                f"All tensors should be on same device, {key} is on {value.device} but input_ids on {device}"
            )


def test_mlm_data_collator_integration():
    """Test that MLMDataCollatorWithFlattening properly integrates both collators"""
    batch = create_test_batch()

    # Test with different MLM probabilities
    for mlm_prob in [0.0, 0.15, 0.3]:
        data_collator = MLMDataCollatorWithFlattening(
            DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_prob, seed=42),
            DataCollatorWithFlattening(return_flash_attn_kwargs=True),
        )

        sample = data_collator(batch)

        # Verify basic structure
        assert "input_ids" in sample, "input_ids should be present"
        assert "labels" in sample, "labels should be present"
        assert "cu_seq_lens_q" in sample, "cu_seq_lens_q should be present"

        # Verify masking behavior
        if mlm_prob == 0.0:
            # No masking - all labels should be -100
            assert (sample["labels"] == -100).all(), "With mlm_probability=0.0, all labels should be -100"
        # TODO: This is a very flaky test with such a small input batch, we should make it larger if we want to ensure a
        # token is masked
        # else: # Some masking should occur masked_count = (sample["labels"] != -100).sum() assert
        #     masked_count > 0, f"With mlm_probability={mlm_prob}, some tokens should be masked"


if __name__ == "__main__":
    test_thd_format()
    test_thd_format_with_different_batch_sizes()
    test_thd_format_sequence_lengths()
    test_thd_format_tensor_types()
    test_mlm_data_collator_integration()
