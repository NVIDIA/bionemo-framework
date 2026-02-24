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

"""Tests for dataloader diagnostics instrumentation.

Verifies that:
  1. DataloaderDiagnostics logs batch stats correctly
  2. Diagnostics integrate with create_bshd_dataloader and create_thd_dataloader
  3. CSV and JSONL output files are created and non-empty
  4. The diagnostics don't break the training data pipeline
"""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from dataloader_diagnostics import DataloaderDiagnostics, EdenDatasetDiagnostics, StreamingDatasetDiagnostics
from dataset import create_bshd_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig


@pytest.fixture
def genomic_parquet(tmp_path):
    """Create a parquet file with multiple genomic sequences for diagnostics testing."""
    parquet_path = tmp_path / "genomic_sequences.parquet"
    sequences = [
        "ACGTACGTACGT" * 100,  # 1200bp
        "TTTTTTTTTTTT" * 80,  # 960bp
        "CCCCCCCCCCCC" * 120,  # 1440bp
        "GGGGGGGGGGGG" * 90,  # 1080bp
        "ATCGATCGATCG" * 110,  # 1320bp
        "NNNACGTNNNNN" * 100,  # 1200bp with Ns
        "acgtacgtacgt" * 100,  # 1200bp lowercase
    ]
    table = pa.table({"text": sequences})
    pq.write_table(table, parquet_path)
    return str(parquet_path)


class TestDataloaderDiagnostics:
    """Tests for the DataloaderDiagnostics class."""

    def test_diagnostics_creates_output_files(self, tmp_path):
        """Test that diagnostics creates CSV and JSONL files."""
        diag = DataloaderDiagnostics(
            rank=0,
            world_size=1,
            log_dir=str(tmp_path),
            tag="test",
            log_every_n_steps=1,
            enabled=True,
        )

        # Create a fake batch
        batch = {
            "input_ids": torch.randint(0, 256, (2, 100)),
            "labels": torch.randint(0, 256, (2, 100)),
            "attention_mask": torch.ones(2, 100, dtype=torch.long),
        }

        diag.log_batch(0, batch)
        diag.log_summary(0)
        diag.close()

        # Verify files exist
        batch_csv = tmp_path / "batch_stats_test_rank0.csv"
        summary_jsonl = tmp_path / "summary_test_rank0.jsonl"
        assert batch_csv.exists(), f"Batch CSV not found at {batch_csv}"
        assert summary_jsonl.exists(), f"Summary JSONL not found at {summary_jsonl}"

        # Verify CSV has content (header + at least 1 data row)
        lines = batch_csv.read_text().strip().split("\n")
        assert len(lines) >= 2, f"Expected at least 2 lines (header + data), got {len(lines)}"

        # Verify JSONL has valid JSON
        summary_lines = summary_jsonl.read_text().strip().split("\n")
        assert len(summary_lines) >= 1
        summary = json.loads(summary_lines[0])
        assert summary["tag"] == "test"
        assert summary["total_batches"] == 1

    def test_diagnostics_tracks_batch_diversity(self, tmp_path):
        """Test that batch hash tracking detects duplicates."""
        diag = DataloaderDiagnostics(
            rank=0,
            world_size=1,
            log_dir=str(tmp_path),
            tag="diversity",
            log_every_n_steps=1,
            enabled=True,
        )

        # Log the same batch twice
        batch = {
            "input_ids": torch.tensor([[65, 67, 71, 84, 65]]),
            "labels": torch.tensor([[65, 67, 71, 84, 65]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        diag.log_batch(0, batch)
        diag.log_batch(1, batch)

        # Log a different batch
        batch2 = {
            "input_ids": torch.tensor([[84, 84, 84, 84, 84]]),
            "labels": torch.tensor([[84, 84, 84, 84, 84]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        diag.log_batch(2, batch2)

        diag.log_summary(2)
        diag.close()

        # Check summary
        summary_jsonl = tmp_path / "summary_diversity_rank0.jsonl"
        summary = json.loads(summary_jsonl.read_text().strip())
        assert summary["total_batches"] == 3
        assert summary["total_duplicate_batches"] == 1  # The repeated batch

    def test_diagnostics_disabled_on_non_zero_rank(self, tmp_path):
        """Test that diagnostics are no-ops on non-zero ranks."""
        diag = DataloaderDiagnostics(
            rank=1,
            world_size=2,
            log_dir=str(tmp_path),
            tag="rank1",
            enabled=True,
        )

        batch = {"input_ids": torch.randint(0, 256, (2, 100))}
        diag.log_batch(0, batch)  # Should be a no-op
        diag.log_summary(0)  # Should be a no-op
        diag.close()

        # No files should be created
        assert not (tmp_path / "batch_stats_rank1_rank1.csv").exists()

    def test_diagnostics_thd_packed_batch(self, tmp_path):
        """Test diagnostics with THD-packed batches (cu_seq_lens)."""
        diag = DataloaderDiagnostics(
            rank=0,
            world_size=1,
            log_dir=str(tmp_path),
            tag="thd",
            log_every_n_steps=1,
            enabled=True,
        )

        # THD batch: 3 sequences packed into 1 tensor
        batch = {
            "input_ids": torch.randint(0, 256, (1, 300)),
            "labels": torch.randint(0, 256, (1, 300)),
            "cu_seq_lens_q": torch.tensor([0, 100, 200, 300], dtype=torch.int32),
        }

        diag.log_batch(0, batch)
        diag.log_summary(0)
        diag.close()

        summary = json.loads((tmp_path / "summary_thd_rank0.jsonl").read_text().strip())
        assert summary["total_batches"] == 1
        assert summary["seq_len_stats"]["mean"] == 100.0


class TestStreamingDatasetDiagnostics:
    """Tests for the StreamingDatasetDiagnostics class."""

    def test_creates_window_order_csv(self, tmp_path):
        """Test that window order CSV is created."""
        diag = StreamingDatasetDiagnostics(
            rank=0,
            log_dir=str(tmp_path),
            tag="streaming_test",
            enabled=True,
        )

        # Log some windows
        for i in range(10):
            diag.log_window(shard_idx=i % 3, seq_len_tokens=100, window_token_hash=f"hash_{i % 5}")

        diag.log_shard_summary(10)
        diag.close()

        csv_file = tmp_path / "window_order_streaming_test_rank0.csv"
        assert csv_file.exists()
        lines = csv_file.read_text().strip().split("\n")
        assert len(lines) >= 2  # header + data


class TestEdenDatasetDiagnostics:
    """Tests for the EdenDatasetDiagnostics class."""

    def test_tracks_access_patterns(self, tmp_path):
        """Test that Eden diagnostics track sequence and sample diversity."""
        diag = EdenDatasetDiagnostics(
            rank=0,
            log_dir=str(tmp_path),
            tag="eden_test",
            enabled=True,
        )

        # Simulate window accesses from different sequences
        diag.log_window(
            window_idx=0, sequence_id="seq_A", sample_id="sample_1", window_in_seq_idx=0, seq_len_tokens=100
        )
        diag.log_window(
            window_idx=5, sequence_id="seq_B", sample_id="sample_2", window_in_seq_idx=0, seq_len_tokens=100
        )
        diag.log_window(
            window_idx=1, sequence_id="seq_A", sample_id="sample_1", window_in_seq_idx=1, seq_len_tokens=100
        )
        diag.log_window(
            window_idx=10, sequence_id="seq_C", sample_id="sample_1", window_in_seq_idx=0, seq_len_tokens=80
        )

        diag.log_summary(4)
        diag.close()

        summary_file = tmp_path / "eden_summary_eden_test_rank0.jsonl"
        assert summary_file.exists()
        summary = json.loads(summary_file.read_text().strip())
        assert summary["unique_sequences"] == 3
        assert summary["unique_samples"] == 2
        assert summary["unique_windows"] == 4


class TestDiagnosticsIntegration:
    """Integration tests: diagnostics with real dataloader creation."""

    def test_bshd_dataloader_with_diagnostics(self, tokenizer_path, genomic_parquet, tmp_path):
        """Test that BSHD dataloader works with diagnostics enabled."""
        distributed_config = DistributedConfig(rank=0, world_size=1)

        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": genomic_parquet,
            "split": "train",
            "streaming": True,
        }

        dataloader, _ = create_bshd_dataloader(
            distributed_config=distributed_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            micro_batch_size=2,
            num_workers=0,
            max_seq_length=256,
            stride=200,
            enable_diagnostics=True,
            diagnostics_log_dir=str(tmp_path),
        )

        # Get batches and log them through batch-level diagnostics
        batch_diag = DataloaderDiagnostics(
            rank=0,
            world_size=1,
            log_dir=str(tmp_path),
            tag="bshd_integration",
            log_every_n_steps=1,
            enabled=True,
        )

        batches = []
        for i, batch in enumerate(dataloader):
            batch_diag.log_batch(i, batch)
            batches.append(batch)
            if i >= 5:
                break

        batch_diag.log_summary(len(batches))
        batch_diag.close()

        # Verify we got batches
        assert len(batches) > 0, "Should have produced at least one batch"

        # Verify diagnostics output exists
        batch_csv = tmp_path / "batch_stats_bshd_integration_rank0.csv"
        assert batch_csv.exists()
        lines = batch_csv.read_text().strip().split("\n")
        assert len(lines) >= 2  # header + at least 1 data row

        # Verify batch structure is still correct (diagnostics didn't break anything)
        for batch in batches:
            assert "input_ids" in batch
            assert "labels" in batch
            assert "attention_mask" in batch
            assert batch["input_ids"].ndim == 2

    def test_thd_dataloader_with_diagnostics(self, tokenizer_path, genomic_parquet, tmp_path):
        """Test that THD dataloader works with diagnostics enabled."""
        distributed_config = DistributedConfig(rank=0, world_size=1)

        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": genomic_parquet,
            "split": "train",
            "streaming": True,
        }

        dataloader, _ = create_thd_dataloader(
            distributed_config=distributed_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            micro_batch_size=1,
            max_seq_length=256,
            stride=200,
            enable_diagnostics=True,
            diagnostics_log_dir=str(tmp_path),
        )

        # Get batches
        batch_diag = DataloaderDiagnostics(
            rank=0,
            world_size=1,
            log_dir=str(tmp_path),
            tag="thd_integration",
            log_every_n_steps=1,
            enabled=True,
        )

        batches = []
        for i, batch in enumerate(dataloader):
            batch_diag.log_batch(i, batch)
            batches.append(batch)
            if i >= 5:
                break

        batch_diag.log_summary(len(batches))
        batch_diag.close()

        assert len(batches) > 0, "Should have produced at least one batch"

        # THD batches should have cu_seq_lens
        for batch in batches:
            assert "input_ids" in batch
            assert "labels" in batch
            assert "cu_seq_lens_q" in batch

    def test_diagnostics_dont_alter_batch_content(self, tokenizer_path, genomic_parquet, tmp_path):
        """Verify that enabling diagnostics doesn't change the batch content."""
        distributed_config = DistributedConfig(rank=0, world_size=1)

        load_dataset_kwargs = {
            "path": "parquet",
            "data_files": genomic_parquet,
            "split": "train",
        }

        # Create dataloader WITHOUT diagnostics
        dl_no_diag, _ = create_bshd_dataloader(
            distributed_config=distributed_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            micro_batch_size=2,
            num_workers=0,
            max_seq_length=256,
            stride=200,
            enable_diagnostics=False,
        )

        # Create dataloader WITH diagnostics
        dl_with_diag, _ = create_bshd_dataloader(
            distributed_config=distributed_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            micro_batch_size=2,
            num_workers=0,
            max_seq_length=256,
            stride=200,
            enable_diagnostics=True,
            diagnostics_log_dir=str(tmp_path),
        )

        # Non-streaming: both should produce identical batches (same DistributedSampler seed)
        batch_no_diag = next(iter(dl_no_diag))
        batch_with_diag = next(iter(dl_with_diag))

        assert batch_no_diag.keys() == batch_with_diag.keys()
        for key in batch_no_diag:
            if isinstance(batch_no_diag[key], torch.Tensor):
                assert torch.equal(batch_no_diag[key], batch_with_diag[key]), (
                    f"Batch content differs for key '{key}' when diagnostics enabled"
                )
