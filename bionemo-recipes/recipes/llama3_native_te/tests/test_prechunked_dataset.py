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

"""Tests for pre-chunked (globally shuffled shards) dataset tokenization.

The pre-chunked dataset at /data/opengenome2/parquet_split/ contains parquet files where each
row is already a fixed-size window of 8190 bases. Unlike the standard HF streaming pipeline
which chunks long sequences into overlapping windows (max_length=8192, stride=200), these
shards are pre-windowed and globally shuffled.

When max_seq_length=None and stride=None, the tokenizer should:
1. Tokenize each row directly (no windowing/truncation)
2. Add BOS and EOS tokens
3. Result: 8190 bases -> 8190 + BOS + EOS = 8192 tokens
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from dataset import create_bshd_dataloader, create_thd_dataloader, create_tokenized_dataset
from distributed_config import DistributedConfig


BOS_ID = 2
EOS_ID = 0
A_ID = 65  # ASCII 'A'
T_ID = 84  # ASCII 'T'
C_ID = 67  # ASCII 'C'
G_ID = 71  # ASCII 'G'


@pytest.fixture
def prechunked_parquet(tmp_path):
    """Create a parquet file simulating pre-chunked globally shuffled shards.

    Each row is exactly 8190 bases (matching the real /data/opengenome2/parquet_split/ format).
    After tokenization with BOS+EOS, each row should become 8192 tokens.
    """
    sequences = [
        "A" * 8190,
        "T" * 8190,
        "C" * 8190,
        "G" * 8190,
        "ACGT" * 2047 + "AC",  # 8190 bases, mixed
    ]
    table = pa.table({"text": sequences})
    parquet_path = tmp_path / "prechunked_shards.parquet"
    pq.write_table(table, parquet_path)
    return str(parquet_path)


@pytest.fixture
def short_prechunked_parquet(tmp_path):
    """Create a parquet file with short pre-chunked sequences for fast tests."""
    sequences = [
        "ACGT" * 5,  # 20 bases
        "TTTT" * 5,  # 20 bases
        "GGGG" * 5,  # 20 bases
    ]
    table = pa.table({"text": sequences})
    parquet_path = tmp_path / "short_prechunked.parquet"
    pq.write_table(table, parquet_path)
    return str(parquet_path)


def _load_kwargs(parquet_path, streaming=False):
    return {
        "path": "parquet",
        "data_files": parquet_path,
        "split": "train",
        "streaming": streaming,
    }


class TestDirectTokenization:
    """Tests for the no-windowing (pre-chunked) tokenization path."""

    def test_direct_tokenization_adds_bos_eos(self, tokenizer_path, short_prechunked_parquet):
        """Verify that direct tokenization adds BOS and EOS but does not window."""
        dist_config = DistributedConfig(rank=0, world_size=1)

        tokenized_dataset, tokenizer = create_tokenized_dataset(
            distributed_config=dist_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=_load_kwargs(short_prechunked_parquet),
            max_seq_length=None,
            stride=None,
        )

        sample = tokenized_dataset[0]
        tokens = sample["input_ids"]

        assert tokens[0] == BOS_ID, f"First token should be BOS ({BOS_ID}), got {tokens[0]}"
        assert tokens[-1] == EOS_ID, f"Last token should be EOS ({EOS_ID}), got {tokens[-1]}"

    def test_direct_tokenization_preserves_sequence_length(self, tokenizer_path, short_prechunked_parquet):
        """Verify that a 20-base sequence becomes 22 tokens (20 + BOS + EOS)."""
        dist_config = DistributedConfig(rank=0, world_size=1)

        tokenized_dataset, _ = create_tokenized_dataset(
            distributed_config=dist_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=_load_kwargs(short_prechunked_parquet),
            max_seq_length=None,
            stride=None,
        )

        sample = tokenized_dataset[0]
        tokens = sample["input_ids"]
        # 20 bases + BOS + EOS = 22
        assert len(tokens) == 22, f"Expected 22 tokens (20 bases + BOS + EOS), got {len(tokens)}"

    def test_direct_tokenization_no_extra_windows(self, tokenizer_path, short_prechunked_parquet):
        """Verify that 3 input rows produce exactly 3 tokenized samples (no windowing splits)."""
        dist_config = DistributedConfig(rank=0, world_size=1)

        tokenized_dataset, _ = create_tokenized_dataset(
            distributed_config=dist_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=_load_kwargs(short_prechunked_parquet),
            max_seq_length=None,
            stride=None,
        )

        assert len(tokenized_dataset) == 3, f"Expected 3 samples (one per row), got {len(tokenized_dataset)}"

    def test_direct_tokenization_correct_nucleotide_ids(self, tokenizer_path, tmp_path):
        """Verify exact token IDs for a known sequence."""
        parquet_path = tmp_path / "known_seq.parquet"
        table = pa.table({"text": ["ACGT"]})
        pq.write_table(table, parquet_path)

        dist_config = DistributedConfig(rank=0, world_size=1)

        tokenized_dataset, _ = create_tokenized_dataset(
            distributed_config=dist_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=_load_kwargs(str(parquet_path)),
            max_seq_length=None,
            stride=None,
        )

        tokens = tokenized_dataset[0]["input_ids"]
        expected = [BOS_ID, A_ID, C_ID, G_ID, T_ID, EOS_ID]
        assert tokens == expected, f"Expected {expected}, got {tokens}"

    def test_8190_base_shard_becomes_8192_tokens(self, tokenizer_path, prechunked_parquet):
        """The key test: 8190-base pre-chunked shard -> 8192 tokens with BOS+EOS."""
        dist_config = DistributedConfig(rank=0, world_size=1)

        tokenized_dataset, _ = create_tokenized_dataset(
            distributed_config=dist_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=_load_kwargs(prechunked_parquet),
            max_seq_length=None,
            stride=None,
        )

        for i in range(len(tokenized_dataset)):
            tokens = tokenized_dataset[i]["input_ids"]
            assert len(tokens) == 8192, f"Sample {i}: expected 8192 tokens, got {len(tokens)}"
            assert tokens[0] == BOS_ID, f"Sample {i}: first token should be BOS"
            assert tokens[-1] == EOS_ID, f"Sample {i}: last token should be EOS"

    def test_windowed_produces_more_samples_than_direct(self, tokenizer_path, tmp_path):
        """Compare: windowed tokenization creates more samples than direct for long sequences."""
        parquet_path = tmp_path / "long_seq.parquet"
        table = pa.table({"text": ["A" * 3000]})
        pq.write_table(table, parquet_path)

        dist_config = DistributedConfig(rank=0, world_size=1)
        kwargs = _load_kwargs(str(parquet_path))

        # Direct: 1 sequence -> 1 sample
        direct_dataset, _ = create_tokenized_dataset(
            distributed_config=dist_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=kwargs,
            max_seq_length=None,
            stride=None,
        )

        # Windowed: 1 long sequence -> multiple windows
        windowed_dataset, _ = create_tokenized_dataset(
            distributed_config=dist_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=kwargs,
            max_seq_length=1000,
            stride=800,
        )

        assert len(direct_dataset) == 1, "Direct should produce exactly 1 sample"
        assert len(windowed_dataset) > 1, f"Windowed should produce multiple samples, got {len(windowed_dataset)}"


class TestDirectTokenizationStreaming:
    """Tests for the no-windowing path with streaming datasets."""

    def test_streaming_direct_tokenization(self, tokenizer_path, short_prechunked_parquet):
        """Verify direct tokenization works with streaming=True."""
        dist_config = DistributedConfig(rank=0, world_size=1)

        tokenized_dataset, _ = create_tokenized_dataset(
            distributed_config=dist_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=_load_kwargs(short_prechunked_parquet, streaming=True),
            max_seq_length=None,
            stride=None,
        )

        samples = list(tokenized_dataset)
        assert len(samples) == 3, f"Expected 3 samples, got {len(samples)}"

        for i, sample in enumerate(samples):
            tokens = sample["input_ids"]
            assert tokens[0] == BOS_ID, f"Sample {i}: first token should be BOS"
            assert tokens[-1] == EOS_ID, f"Sample {i}: last token should be EOS"
            assert len(tokens) == 22, f"Sample {i}: expected 22 tokens, got {len(tokens)}"

    def test_streaming_8190_shard(self, tokenizer_path, prechunked_parquet):
        """Verify 8190-base shards produce 8192 tokens in streaming mode."""
        dist_config = DistributedConfig(rank=0, world_size=1)

        tokenized_dataset, _ = create_tokenized_dataset(
            distributed_config=dist_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=_load_kwargs(prechunked_parquet, streaming=True),
            max_seq_length=None,
            stride=None,
        )

        count = 0
        for sample in tokenized_dataset:
            tokens = sample["input_ids"]
            assert len(tokens) == 8192, f"Sample {count}: expected 8192 tokens, got {len(tokens)}"
            assert tokens[0] == BOS_ID
            assert tokens[-1] == EOS_ID
            count += 1

        assert count == 5, f"Expected 5 samples, got {count}"


class TestDirectTokenizationDataloaders:
    """Tests for BSHD and THD dataloaders with pre-chunked data."""

    def test_bshd_dataloader_prechunked(self, tokenizer_path, short_prechunked_parquet):
        """Verify BSHD dataloader works with pre-chunked data (no windowing)."""
        dist_config = DistributedConfig(rank=0, world_size=1)

        dataloader, _ = create_bshd_dataloader(
            distributed_config=dist_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=_load_kwargs(short_prechunked_parquet),
            micro_batch_size=2,
            num_workers=0,
            max_seq_length=None,
            stride=None,
        )

        batch = next(iter(dataloader))

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == 2, f"Expected batch size 2, got {batch['input_ids'].shape[0]}"
        # 20 bases + BOS + EOS = 22 tokens per sample
        assert batch["input_ids"].shape[1] == 22, f"Expected seq length 22, got {batch['input_ids'].shape[1]}"

    def test_thd_dataloader_prechunked(self, tokenizer_path, short_prechunked_parquet):
        """Verify THD dataloader works with pre-chunked data using token_micro_batch_size."""
        dist_config = DistributedConfig(rank=0, world_size=1)

        # With pre-chunked data, must use token_micro_batch_size directly
        dataloader, _ = create_thd_dataloader(
            distributed_config=dist_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=_load_kwargs(short_prechunked_parquet, streaming=True),
            token_micro_batch_size=66,  # Fits 3 samples of 22 tokens each
            num_workers=0,
            max_seq_length=None,
            stride=None,
        )

        batch = next(iter(dataloader))

        assert "input_ids" in batch
        assert "cu_seq_lens_q" in batch
        assert "cu_seq_lens_k" in batch
        # THD format: packed into [1, T] shape
        assert batch["input_ids"].ndim == 2
        assert batch["input_ids"].shape[0] == 1

    def test_thd_dataloader_prechunked_requires_token_mbs(self, tokenizer_path, short_prechunked_parquet):
        """Verify THD dataloader raises when using micro_batch_size with max_seq_length=None."""
        dist_config = DistributedConfig(rank=0, world_size=1)

        with pytest.raises(AssertionError, match="max_seq_length must be set"):
            create_thd_dataloader(
                distributed_config=dist_config,
                tokenizer_name_or_path=tokenizer_path,
                load_dataset_kwargs=_load_kwargs(short_prechunked_parquet, streaming=True),
                micro_batch_size=2,  # Can't compute token_mbs without max_seq_length
                num_workers=0,
                max_seq_length=None,
                stride=None,
            )

    def test_bshd_dataloader_8190_shard_batch(self, tokenizer_path, prechunked_parquet):
        """End-to-end: 8190-base shards -> BSHD batch with 8192-length sequences."""
        dist_config = DistributedConfig(rank=0, world_size=1)

        dataloader, _ = create_bshd_dataloader(
            distributed_config=dist_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=_load_kwargs(prechunked_parquet),
            micro_batch_size=2,
            num_workers=0,
            max_seq_length=None,
            stride=None,
        )

        batch = next(iter(dataloader))

        assert batch["input_ids"].shape == (2, 8192), f"Expected shape (2, 8192), got {batch['input_ids'].shape}"
        # Verify BOS/EOS in batch
        assert torch.all(batch["input_ids"][:, 0] == BOS_ID), "All sequences should start with BOS"
        assert torch.all(batch["input_ids"][:, -1] == EOS_ID), "All sequences should end with EOS"
        # All positions should be real (no padding needed since all seqs are same length)
        assert torch.all(batch["attention_mask"] == 1), "All positions should be attended (same-length sequences)"
