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

import tempfile
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from transformers import AutoTokenizer

from dataset import create_bshd_dataloader, create_tokenized_dataset
from distributed_config import DistributedConfig


@pytest.fixture(scope="session")
def tokenizer_path():
    """Get the path to the nucleotide tokenizer."""
    return str(Path(__file__).parent.parent.parent.parent / "models" / "llama3" / "nucleotide_fast_tokenizer")


@pytest.fixture(scope="session")
def tokenizer(tokenizer_path):
    """Load the nucleotide tokenizer."""
    return AutoTokenizer.from_pretrained(tokenizer_path)


@pytest.fixture
def simple_parquet(tmp_path):
    """Create a simple Parquet file with one genomic sequence."""
    parquet_path = tmp_path / "genomic_sequences.parquet"
    
    # Create a minimal dataset with one 1000bp sequence
    sequence = "A" * 1000
    
    table = pa.table({
        "sequence": [sequence],
    })
    
    pq.write_table(table, parquet_path)
    return str(parquet_path)


def test_tokenizer_loads(tokenizer_path):
    """Test that the tokenizer loads correctly."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    assert tokenizer is not None
    assert tokenizer.vocab_size > 0


def test_windowing_creates_multiple_windows_from_long_sequence(tokenizer_path):
    """Test that windowing creates multiple samples from a long sequence using the tokenizer directly."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Create a 5kbp sequence
    sequence = "A" * 5000
    
    # Tokenize with windowing
    result = tokenizer(
        sequence,
        max_length=1000,
        stride=800,  # 800 token overlap
        truncation=True,
        return_overflowing_tokens=True,
        add_special_tokens=True,
    )
    
    # Should create multiple windows
    num_windows = len(result["input_ids"])
    assert num_windows > 1, f"Expected multiple windows, got {num_windows}"
    
    # First and last windows may be shorter, but most should be max_length
    assert len(result["input_ids"][0]) <= 1000


def test_dataset_loads_and_tokenizes_sequence(tokenizer_path, tmp_path):
    """Test that dataset loads and tokenizes a sequence correctly."""
    # Create a Parquet file with a single T sequence
    parquet_path = tmp_path / "genomic_sequences.parquet"
    sequence = "T" * 100
    table = pa.table({"sequence": [sequence]})
    pq.write_table(table, parquet_path)
    
    distributed_config = DistributedConfig(rank=0, world_size=1)
    
    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",  # Explicitly request train split
    }
    
    tokenized_dataset, tokenizer = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=100,
        stride=50,
        buffer_size=10_000,
    )
    
    # Access first sample
    sample = tokenized_dataset[0]
    assert "input_ids" in sample
    assert isinstance(sample["input_ids"], list)
    
    # Check that it contains T tokens (ASCII 84)
    # Remove BOS (2) and EOS (0) tokens
    tokens = sample["input_ids"]
    assert tokens[0] == 2  # BOS
    assert tokens[-1] == 0  # EOS
    # Middle should be all Ts (84)
    assert all(t == 84 for t in tokens[1:-1])


def test_dataloader_produces_correct_batch_structure(tokenizer_path, simple_parquet):
    """Test that the dataloader produces batches with correct structure."""
    distributed_config = DistributedConfig(rank=0, world_size=1)
    
    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": simple_parquet,
        "split": "train",
    }
    
    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=500,
        stride=100,
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    
    # Check batch structure
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    
    # Check batch contains tensors
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert isinstance(batch["attention_mask"], torch.Tensor)
    assert isinstance(batch["labels"], torch.Tensor)
    
    # Check batch size (may vary with lazy tokenization and windowing)
    assert batch["input_ids"].shape[0] >= 1, "Batch should have at least 1 sample"


def test_attention_mask_aligns_with_labels(tokenizer_path, simple_parquet):
    """Test attention_mask correctly identifies real vs padded positions."""
    distributed_config = DistributedConfig(rank=0, world_size=1)
    
    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": simple_parquet,
        "split": "train",
    }
    
    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=500,
        stride=100,
    )
    
    batch = next(iter(dataloader))
    
    # Check that attention_mask is present and valid
    attention_mask = batch["attention_mask"][0]
    
    # Should have some real tokens (attention_mask=1)
    assert torch.sum(attention_mask) > 0, "Should have at least some real tokens"


def test_streaming_dataset_produces_batches(tokenizer_path, simple_parquet):
    """Test that streaming mode works and produces valid batches."""
    distributed_config = DistributedConfig(rank=0, world_size=1)
    
    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": simple_parquet,
        "split": "train",
        "streaming": True,
    }
    
    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=500,
        stride=100,
        buffer_size=10_000,
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    
    # Check batch structure
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    
    # Check batch is tensors
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert batch["input_ids"].shape[0] == 2


def test_windowing_in_dataset_creates_multiple_samples(tokenizer_path, tmp_path):
    """Test that the dataset's windowing creates expected number of samples."""
    # Create a 3kbp sequence
    parquet_path = tmp_path / "genomic_sequences.parquet"
    sequence = "A" * 3000
    table = pa.table({"sequence": [sequence]})
    pq.write_table(table, parquet_path)
    
    distributed_config = DistributedConfig(rank=0, world_size=1)
    
    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",
    }
    
    tokenized_dataset, tokenizer = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=1000,
        stride=800,  # 800 token overlap, so 200 token step
        buffer_size=10_000,
        use_lazy_tokenization=False,  # Use eager tokenization to expand windows
    )
    
    # Count samples
    num_samples = len(tokenized_dataset)
    
    # With 3000bp sequence, max_length=1000, stride=800 (overlap)
    # Should create multiple windows (at least 2)
    assert num_samples >= 2, f"Expected at least 2 windows, got {num_samples}"


def test_lazy_tokenization_returns_batch(tokenizer_path, simple_parquet):
    """Test that lazy tokenization works and returns valid batches."""
    distributed_config = DistributedConfig(rank=0, world_size=1)
    
    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": simple_parquet,
        "split": "train",
        "streaming": False,
    }
    
    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=500,
        stride=100,
        use_lazy_tokenization=True,
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    
    # Verify batch is not None and has correct structure
    assert batch is not None
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert batch["input_ids"].shape[0] >= 1  # At least one sample in batch
