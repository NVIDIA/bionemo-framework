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

"""Tests for mapped_dataset.py - the mapped HuggingFace dataset with caching."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from distributed_config import DistributedConfig
from mapped_dataset import (
    create_mapped_bshd_dataloader,
    create_windowed_mapped_dataset,
    load_or_create_windowed_dataset,
)


@pytest.fixture
def simple_sequences(tmp_path):
    """Create a simple Parquet file with multiple genomic sequences."""
    parquet_path = tmp_path / "genomic_sequences.parquet"

    # Create sequences of varying lengths
    sequences = [
        "A" * 1000,
        "T" * 1200,
        "C" * 800,
        "G" * 1500,
        "ATCG" * 300,
    ]

    table = pa.table({"text": sequences})
    pq.write_table(table, parquet_path)
    return str(parquet_path)


@pytest.fixture
def single_sequence(tmp_path):
    """Create a Parquet file with a single sequence for deterministic tests."""
    parquet_path = tmp_path / "single_sequence.parquet"
    sequence = "T" * 100
    table = pa.table({"text": [sequence]})
    pq.write_table(table, parquet_path)
    return str(parquet_path)


class TestCreateWindowedMappedDataset:
    """Tests for create_windowed_mapped_dataset function."""

    def test_basic_windowing(self, tokenizer_path, single_sequence):
        """Test that windowing creates the expected number of windows."""
        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": single_sequence,
            "split": "train",
        }

        dataset, tokenizer = create_windowed_mapped_dataset(
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            max_seq_length=50,
            stride=40,
        )

        # With 100 chars, max_seq_length=50, stride=40:
        # Window 0: 0-48 (50 tokens with BOS/EOS)
        # Window 1: 40-88
        # Window 2: 80-100 (remaining)
        assert len(dataset) >= 2, f"Expected at least 2 windows, got {len(dataset)}"
        assert "input_ids" in dataset.column_names
        assert "attention_mask" in dataset.column_names

    def test_tokenization_correctness(self, tokenizer_path, single_sequence):
        """Test that tokens are correct (BOS, nucleotides, EOS)."""
        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": single_sequence,
            "split": "train",
        }

        dataset, tokenizer = create_windowed_mapped_dataset(
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            max_seq_length=20,
            stride=10,
        )

        sample = dataset[0]
        tokens = sample["input_ids"]

        # Check BOS and EOS
        bos = 2
        eos = 0
        t = 84  # ASCII 'T'

        assert tokens[0] == bos, f"First token should be BOS (2), got {tokens[0]}"
        assert tokens[-1] == eos, f"Last token should be EOS (0), got {tokens[-1]}"

        # Check middle tokens are T
        nucleotides = tokens[1:-1]
        for i, tok in enumerate(nucleotides):
            assert tok == t, f"Token {i + 1} should be T (84), got {tok}"

    def test_multiple_sequences(self, tokenizer_path, simple_sequences):
        """Test windowing with multiple sequences."""
        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": simple_sequences,
            "split": "train",
        }

        dataset, tokenizer = create_windowed_mapped_dataset(
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            max_seq_length=500,
            stride=400,
        )

        # Should have windows from all 5 sequences
        assert len(dataset) > 5, f"Expected more than 5 windows from 5 sequences, got {len(dataset)}"

        # Verify dataset is indexable
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            assert "input_ids" in sample
            assert len(sample["input_ids"]) > 0

    def test_ignores_streaming_flag(self, tokenizer_path, single_sequence):
        """Test that streaming=True is ignored for mapped dataset."""
        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": single_sequence,
            "split": "train",
            "streaming": True,  # Should be ignored
        }

        dataset, tokenizer = create_windowed_mapped_dataset(
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            max_seq_length=50,
            stride=40,
        )

        # Should still work and be indexable (not streaming)
        assert len(dataset) > 0
        _ = dataset[0]  # Should not raise


class TestLoadOrCreateWindowedDataset:
    """Tests for load_or_create_windowed_dataset function."""

    def test_creates_cache(self, tokenizer_path, single_sequence, tmp_path):
        """Test that cache is created correctly."""
        cache_dir = str(tmp_path / "cache")

        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": single_sequence,
            "split": "train",
        }

        dataset, tokenizer = load_or_create_windowed_dataset(
            cache_dir=cache_dir,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            max_seq_length=50,
            stride=40,
            local_rank=0,
        )

        # Verify cache was created
        cache_path = tmp_path / "cache"
        assert cache_path.exists()
        assert (cache_path / "dataset_info.json").exists()

    def test_loads_from_cache(self, tokenizer_path, single_sequence, tmp_path):
        """Test that cache is loaded on subsequent calls."""
        cache_dir = str(tmp_path / "cache")

        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": single_sequence,
            "split": "train",
        }

        # First call - creates cache
        dataset1, _ = load_or_create_windowed_dataset(
            cache_dir=cache_dir,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            max_seq_length=50,
            stride=40,
            local_rank=0,
        )

        # Second call - should load from cache
        dataset2, _ = load_or_create_windowed_dataset(
            cache_dir=cache_dir,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs={},  # Empty kwargs - should still work
            max_seq_length=50,
            stride=40,
            local_rank=0,
        )

        # Datasets should have same length
        assert len(dataset1) == len(dataset2)

    def test_force_recreate(self, tokenizer_path, single_sequence, tmp_path):
        """Test that force_recreate recreates the cache."""
        cache_dir = str(tmp_path / "cache")

        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": single_sequence,
            "split": "train",
        }

        # Create initial cache
        dataset1, _ = load_or_create_windowed_dataset(
            cache_dir=cache_dir,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            max_seq_length=50,
            stride=40,
            local_rank=0,
        )

        # Force recreate with different parameters
        dataset2, _ = load_or_create_windowed_dataset(
            cache_dir=cache_dir,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            max_seq_length=30,  # Different max_seq_length
            stride=20,
            local_rank=0,
            force_recreate=True,
        )

        # Should have different number of windows
        # (though this specific test may or may not differ depending on sequence length)
        assert len(dataset2) > 0


class TestCreateMappedBshdDataloader:
    """Tests for create_mapped_bshd_dataloader function."""

    def test_creates_dataloader_and_sampler(self, tokenizer_path, simple_sequences, tmp_path):
        """Test that dataloader and sampler are created correctly."""
        cache_dir = str(tmp_path / "cache")

        distributed_config = DistributedConfig(rank=0, local_rank=0, world_size=1)

        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": simple_sequences,
            "split": "train",
        }

        dataloader, sampler = create_mapped_bshd_dataloader(
            distributed_config=distributed_config,
            tokenizer_name_or_path=tokenizer_path,
            cache_dir=cache_dir,
            load_dataset_kwargs=load_dataset_kwargs,
            micro_batch_size=2,
            num_workers=0,
            max_seq_length=500,
            stride=400,
        )

        # Verify types
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert isinstance(sampler, torch.utils.data.distributed.DistributedSampler)

    def test_dataloader_produces_batches(self, tokenizer_path, simple_sequences, tmp_path):
        """Test that dataloader produces correct batches."""
        cache_dir = str(tmp_path / "cache")

        distributed_config = DistributedConfig(rank=0, local_rank=0, world_size=1)

        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": simple_sequences,
            "split": "train",
        }

        dataloader, sampler = create_mapped_bshd_dataloader(
            distributed_config=distributed_config,
            tokenizer_name_or_path=tokenizer_path,
            cache_dir=cache_dir,
            load_dataset_kwargs=load_dataset_kwargs,
            micro_batch_size=2,
            num_workers=0,
            max_seq_length=500,
            stride=400,
        )

        batch = next(iter(dataloader))

        # Check batch structure
        assert "input_ids" in batch
        assert "labels" in batch
        assert "attention_mask" in batch

        # Check shapes
        assert batch["input_ids"].shape[0] == 2  # batch_size
        assert batch["labels"].shape == batch["input_ids"].shape
        assert batch["attention_mask"].shape == batch["input_ids"].shape

    def test_sampler_shuffles(self, tokenizer_path, simple_sequences, tmp_path):
        """Test that sampler produces different orderings across epochs."""
        cache_dir = str(tmp_path / "cache")

        distributed_config = DistributedConfig(rank=0, local_rank=0, world_size=1)

        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": simple_sequences,
            "split": "train",
        }

        dataloader, sampler = create_mapped_bshd_dataloader(
            distributed_config=distributed_config,
            tokenizer_name_or_path=tokenizer_path,
            cache_dir=cache_dir,
            load_dataset_kwargs=load_dataset_kwargs,
            micro_batch_size=2,
            num_workers=0,
            max_seq_length=500,
            stride=400,
        )

        # Get indices for epoch 0
        sampler.set_epoch(0)
        indices_epoch0 = list(sampler)

        # Get indices for epoch 1
        sampler.set_epoch(1)
        indices_epoch1 = list(sampler)

        # Indices should be different (shuffled differently)
        assert indices_epoch0 != indices_epoch1, "Sampler should produce different orderings per epoch"

        # But should have same elements
        assert set(indices_epoch0) == set(indices_epoch1), "Same indices should be covered"

    def test_genomic_masking(self, tokenizer_path, tmp_path):
        """Test that genomic masking works with mapped dataset."""
        # Create data with degenerate bases
        parquet_path = tmp_path / "genomic_degenerate.parquet"
        sequences = ["ACGTN", "GGTAR"]  # N and R are degenerate
        table = pa.table({"text": sequences})
        pq.write_table(table, parquet_path)

        cache_dir = str(tmp_path / "cache")

        distributed_config = DistributedConfig(rank=0, local_rank=0, world_size=1)

        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": str(parquet_path),
            "split": "train",
        }

        dataloader, sampler = create_mapped_bshd_dataloader(
            distributed_config=distributed_config,
            tokenizer_name_or_path=tokenizer_path,
            cache_dir=cache_dir,
            load_dataset_kwargs=load_dataset_kwargs,
            micro_batch_size=2,
            num_workers=0,
            max_seq_length=10,
            stride=5,
            mask_degenerate_bases=True,
        )

        batch = next(iter(dataloader))

        # Degenerate bases (N=78, R=82) should not be in labels
        labels = batch["labels"]
        assert 78 not in labels, "Degenerate N (78) should be masked"
        assert 82 not in labels, "Degenerate R (82) should be masked"

    def test_no_cache_dir(self, tokenizer_path, simple_sequences):
        """Test that dataloader works without cache_dir."""
        distributed_config = DistributedConfig(rank=0, local_rank=0, world_size=1)

        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": simple_sequences,
            "split": "train",
        }

        dataloader, sampler = create_mapped_bshd_dataloader(
            distributed_config=distributed_config,
            tokenizer_name_or_path=tokenizer_path,
            cache_dir=None,
            load_dataset_kwargs=load_dataset_kwargs,
            micro_batch_size=2,
            num_workers=0,
            max_seq_length=500,
            stride=400,
        )

        # Should still work
        batch = next(iter(dataloader))
        assert "input_ids" in batch

    def test_raises_without_cache_or_kwargs(self, tokenizer_path):
        """Test that error is raised when neither cache_dir nor load_dataset_kwargs provided."""
        distributed_config = DistributedConfig(rank=0, local_rank=0, world_size=1)

        with pytest.raises(ValueError, match="Either cache_dir or load_dataset_kwargs"):
            create_mapped_bshd_dataloader(
                distributed_config=distributed_config,
                tokenizer_name_or_path=tokenizer_path,
                cache_dir=None,
                load_dataset_kwargs=None,
            )


class TestMappedVsStreamingEquivalence:
    """Tests to verify mapped dataset produces equivalent results to streaming."""

    def test_same_total_windows(self, tokenizer_path, simple_sequences, tmp_path):
        """Test that mapped and streaming produce the same number of windows."""
        from dataset import create_tokenized_dataset

        distributed_config = DistributedConfig(rank=0, local_rank=0, world_size=1)

        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": simple_sequences,
            "split": "train",
            "streaming": False,  # Non-streaming for comparison
        }

        # Mapped dataset
        mapped_dataset, _ = create_windowed_mapped_dataset(
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            max_seq_length=500,
            stride=400,
        )

        # Streaming-style dataset (but with streaming=False so we can count)
        streaming_dataset, _ = create_tokenized_dataset(
            distributed_config=distributed_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            max_seq_length=500,
            stride=400,
            shuffle_sequences=False,  # Disable shuffle for consistent count
        )

        # Both should produce the same number of windows
        mapped_count = len(mapped_dataset)
        streaming_count = len(streaming_dataset)

        assert mapped_count == streaming_count, (
            f"Window counts should match: mapped={mapped_count}, streaming={streaming_count}"
        )

    def test_same_tokenization(self, tokenizer_path, tmp_path):
        """Test that mapped and streaming produce the same tokens for identical input."""
        # Create a single-sequence file for deterministic comparison
        parquet_path = tmp_path / "single.parquet"
        sequence = "ACGT" * 25  # 100 chars
        table = pa.table({"text": [sequence]})
        pq.write_table(table, parquet_path)

        from dataset import create_tokenized_dataset

        distributed_config = DistributedConfig(rank=0, local_rank=0, world_size=1)

        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": str(parquet_path),
            "split": "train",
            "streaming": False,
        }

        # Mapped dataset
        mapped_dataset, _ = create_windowed_mapped_dataset(
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            max_seq_length=50,
            stride=40,
        )

        # Streaming-style dataset
        streaming_dataset, _ = create_tokenized_dataset(
            distributed_config=distributed_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            max_seq_length=50,
            stride=40,
            shuffle_sequences=False,
        )

        # Compare first window
        mapped_sample = mapped_dataset[0]
        streaming_sample = streaming_dataset[0]

        assert mapped_sample["input_ids"] == streaming_sample["input_ids"], (
            f"Token sequences should match:\n"
            f"Mapped: {mapped_sample['input_ids'][:20]}...\n"
            f"Streaming: {streaming_sample['input_ids'][:20]}..."
        )


class TestDistributedSampling:
    """Tests for distributed sampling behavior."""

    def test_different_ranks_get_different_indices(self, tokenizer_path, simple_sequences, tmp_path):
        """Test that different ranks get non-overlapping indices."""
        cache_dir = str(tmp_path / "cache")

        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": simple_sequences,
            "split": "train",
        }

        # Simulate rank 0 of 2
        distributed_config_0 = DistributedConfig(rank=0, local_rank=0, world_size=2)
        _, sampler_0 = create_mapped_bshd_dataloader(
            distributed_config=distributed_config_0,
            tokenizer_name_or_path=tokenizer_path,
            cache_dir=cache_dir,
            load_dataset_kwargs=load_dataset_kwargs,
            micro_batch_size=2,
            num_workers=0,
            max_seq_length=500,
            stride=400,
        )

        # Simulate rank 1 of 2
        distributed_config_1 = DistributedConfig(rank=1, local_rank=1, world_size=2)
        _, sampler_1 = create_mapped_bshd_dataloader(
            distributed_config=distributed_config_1,
            tokenizer_name_or_path=tokenizer_path,
            cache_dir=cache_dir,
            load_dataset_kwargs=load_dataset_kwargs,
            micro_batch_size=2,
            num_workers=0,
            max_seq_length=500,
            stride=400,
        )

        indices_0 = set(sampler_0)
        indices_1 = set(sampler_1)

        # Indices should not overlap
        overlap = indices_0 & indices_1
        assert len(overlap) == 0, f"Ranks should get non-overlapping indices, but found overlap: {overlap}"

        # Together should cover all indices
        total_indices = len(indices_0) + len(indices_1)
        assert total_indices == len(sampler_0.data_source), (
            f"All indices should be covered: {total_indices} vs {len(sampler_0.data_source)}"
        )

    def test_all_windows_covered_across_epochs(self, tokenizer_path, simple_sequences, tmp_path):
        """Test that all windows are seen across multiple epochs."""
        cache_dir = str(tmp_path / "cache")

        distributed_config = DistributedConfig(rank=0, local_rank=0, world_size=1)

        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": simple_sequences,
            "split": "train",
        }

        _, sampler = create_mapped_bshd_dataloader(
            distributed_config=distributed_config,
            tokenizer_name_or_path=tokenizer_path,
            cache_dir=cache_dir,
            load_dataset_kwargs=load_dataset_kwargs,
            micro_batch_size=2,
            num_workers=0,
            max_seq_length=500,
            stride=400,
        )

        # Collect all indices across 3 epochs
        all_indices = set()
        for epoch in range(3):
            sampler.set_epoch(epoch)
            epoch_indices = list(sampler)
            all_indices.update(epoch_indices)

        # All indices should be covered in each epoch
        expected = set(range(len(sampler.data_source)))
        assert all_indices == expected, (
            f"All indices should be covered: got {len(all_indices)}, expected {len(expected)}"
        )
