# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""
Unit tests for genomic dataloader
"""

import pytest
import torch
import sqlite3
import tempfile
from pathlib import Path
from omegaconf import DictConfig
from transformers import AutoTokenizer

from sqlite_dataset import create_genomic_dataloader, GenomicSequenceDataset


@pytest.fixture
def tokenizer():
    """Load the nucleotide tokenizer."""
    tokenizer_path = Path(__file__).parent.parent / "nucleotide_tokenizer"
    return AutoTokenizer.from_pretrained(str(tokenizer_path))


@pytest.fixture
def simple_database():
    """Create test database with repeated character sequences."""
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
    db_path = temp_db.name
    temp_db.close()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE sequences (
            contig_id TEXT PRIMARY KEY,
            nt_sequence TEXT NOT NULL,
            length INTEGER NOT NULL
        )
    """)
    
    test_sequences = [
        ("seq_A", "A" * 10000, 10000),
        ("seq_T", "T" * 8000, 8000),
        ("seq_C", "C" * 5000, 5000),
        ("seq_G", "G" * 2000, 2000),
    ]
    
    cursor.executemany(
        "INSERT INTO sequences (contig_id, nt_sequence, length) VALUES (?, ?, ?)",
        test_sequences
    )
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    Path(db_path).unlink()


def test_dataset_index_0_returns_all_as(simple_database, tokenizer):
    """Test dataset[0] returns all As (with seed=42).
    
    Pattern: expected_sequence = [65] * seqlen
    """
    dataset = GenomicSequenceDataset(
        database_path=simple_database,
        seq_length=100,
        tokenizer=tokenizer,
        stride=50,
        min_window_length=50,
        seed=42,
    )
    
    sample = dataset[0]
    nucleotides = sample["input_ids"][1:-1]  # Remove BOS/EOS
    
    # With seed=42, index 0 is all As
    expected_sequence = [65] * len(nucleotides)  # All As
    received_sequence = nucleotides.tolist()
    
    assert received_sequence == expected_sequence


def test_dataset_index_1_returns_all_ts(simple_database, tokenizer):
    """Test dataset[1] returns all Ts (with seed=42).
    
    Pattern: expected_sequence = [84] * 100
    """
    dataset = GenomicSequenceDataset(
        database_path=simple_database,
        seq_length=100,
        tokenizer=tokenizer,
        stride=50,
        min_window_length=50,
        seed=42,
    )
    
    sample = dataset[1]
    nucleotides = sample["input_ids"][1:-1]
    
    # With seed=42, index 1 is all Ts (100 tokens)
    expected_sequence = [84] * 100  # All Ts
    received_sequence = nucleotides.tolist()
    
    assert received_sequence == expected_sequence


def test_dataset_index_3_returns_all_gs(simple_database, tokenizer):
    """Test dataset[3] returns all Gs (with seed=42).
    
    Pattern: expected_sequence = [71] * 100
    """
    dataset = GenomicSequenceDataset(
        database_path=simple_database,
        seq_length=100,
        tokenizer=tokenizer,
        stride=50,
        min_window_length=50,
        seed=42,
    )
    
    sample = dataset[3]
    nucleotides = sample["input_ids"][1:-1]
    
    # With seed=42, index 3 is all Gs (100 tokens)
    expected_sequence = [71] * 100  # All Gs
    received_sequence = nucleotides.tolist()
    
    assert received_sequence == expected_sequence


def test_dataset_index_5_returns_all_cs(simple_database, tokenizer):
    """Test dataset[5] returns all Cs (with seed=42).
    
    Pattern: expected_sequence = [67] * 100
    """
    dataset = GenomicSequenceDataset(
        database_path=simple_database,
        seq_length=100,
        tokenizer=tokenizer,
        stride=50,
        min_window_length=50,
        seed=42,
    )
    
    sample = dataset[5]
    nucleotides = sample["input_ids"][1:-1]
    
    # With seed=42, index 5 is all Cs (100 tokens)
    expected_sequence = [67] * 100  # All Cs
    received_sequence = nucleotides.tolist()
    
    assert received_sequence == expected_sequence


def test_dataloader_returns_batch_with_required_keys(simple_database, tokenizer):
    """Test dataloader batches have input_ids, labels, attention_mask."""
    args = DictConfig({
        "dataset": {
            "database_path": simple_database,
            "seq_length": 1000,
            "batch_size": 2,
            "num_workers": 0,
            "stride": 800,
            "min_window_length": 500,
            "seed": 42,
        }
    })
    
    train_iterator, epoch_len = create_genomic_dataloader(args, tokenizer)
    batch = next(train_iterator)
    
    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch


def test_batch_size_is_correct(simple_database, tokenizer):
    """Test batch size matches configuration."""
    args = DictConfig({
        "dataset": {
            "database_path": simple_database,
            "seq_length": 1000,
            "batch_size": 3,
            "num_workers": 0,
            "stride": 800,
            "min_window_length": 500,
            "seed": 42,
        }
    })
    
    train_iterator, epoch_len = create_genomic_dataloader(args, tokenizer)
    batch = next(train_iterator)
    
    assert batch["input_ids"].shape[0] == 3
    assert batch["labels"].shape[0] == 3
    assert batch["attention_mask"].shape[0] == 3


def test_labels_equal_input_ids_for_causal_lm(simple_database, tokenizer):
    """Test labels equal input_ids for non-padding positions."""
    args = DictConfig({
        "dataset": {
            "database_path": simple_database,
            "seq_length": 1000,
            "batch_size": 2,
            "num_workers": 0,
            "stride": 800,
            "min_window_length": 500,
            "seed": 42,
        }
    })
    
    train_iterator, epoch_len = create_genomic_dataloader(args, tokenizer)
    batch = next(train_iterator)
    
    # For real tokens, labels should equal input_ids
    real_mask = batch["attention_mask"] == 1
    assert torch.all(batch["labels"][real_mask] == batch["input_ids"][real_mask])


def test_padding_positions_are_minus_100(simple_database, tokenizer):
    """Test padding positions are -100 in labels."""
    args = DictConfig({
        "dataset": {
            "database_path": simple_database,
            "seq_length": 1000,
            "batch_size": 2,
            "num_workers": 0,
            "stride": 800,
            "min_window_length": 500,
            "seed": 42,
        }
    })
    
    train_iterator, epoch_len = create_genomic_dataloader(args, tokenizer)
    batch = next(train_iterator)
    
    padding_mask = batch["attention_mask"] == 0
    assert torch.all(batch["labels"][padding_mask] == -100)


def test_epoch_length_equals_total_windows_divided_by_batch_size(simple_database, tokenizer):
    """Test epoch_len is correct for given configuration."""
    args = DictConfig({
        "dataset": {
            "database_path": simple_database,
            "seq_length": 1000,
            "batch_size": 3,
            "num_workers": 0,
            "stride": 800,
            "min_window_length": 500,
            "seed": 42,
        }
    })
    
    train_iterator, epoch_len = create_genomic_dataloader(args, tokenizer)
    
    # Hardcoded: 29 windows total, batch_size=3, drop_last=True -> 9 batches
    assert epoch_len == 9
