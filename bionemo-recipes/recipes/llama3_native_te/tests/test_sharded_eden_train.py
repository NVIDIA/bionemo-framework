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

"""End-to-end training tests using ShardedEden data with the FSDP2 training loop.

These tests create synthetic genomic SQLite databases, configure training
with use_sharded_eden=true, and verify that:
  - Training runs end-to-end without errors (BSHD and THD)
  - Loss decreases over 200 steps (the model is learning)
  - Loss values are reasonable (> 1.0, i.e., not collapsed)
"""

import gc
import random
import sqlite3
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir


sys.path.insert(0, Path(__file__).parent.parent.as_posix())

from sharded_eden_dataset import compute_num_windows, extract_sample_id
from train_fsdp2 import main as main_fsdp2


# ---------------------------------------------------------------------------
# Fixtures: create synthetic genomic SQLite databases
# ---------------------------------------------------------------------------


def _random_dna(length: int, seed: int = 0) -> str:
    """Generate a deterministic random DNA sequence."""
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def _create_sequence_dbs(base_dir: Path, sequences: list[tuple[str, str]]) -> str:
    """Create per-sample SQLite databases mirroring the BCR Eden layout."""
    db_dir = base_dir / "sequence_dbs"
    db_dir.mkdir(exist_ok=True, parents=True)

    by_sample: dict[str, list[tuple[str, str]]] = {}
    for seq_id, seq in sequences:
        sample_id = extract_sample_id(seq_id)
        by_sample.setdefault(sample_id, []).append((seq_id, seq))

    for sample_id, seqs in by_sample.items():
        sample_dir = db_dir / sample_id
        sample_dir.mkdir(exist_ok=True)
        db_path = sample_dir / f"glm_dataset_{sample_id}.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE sequences (contig_id TEXT PRIMARY KEY, nt_sequence TEXT NOT NULL)")
        conn.executemany("INSERT INTO sequences VALUES (?, ?)", seqs)
        conn.commit()
        conn.close()

    return str(db_dir)


def _create_window_db(
    base_dir: Path,
    sequences: list[tuple[str, str]],
    window_size: int,
    stride: int,
    name: str = "train_windows.db",
) -> str:
    """Create a window mapping database from sequences."""
    db_path = base_dir / name
    conn = sqlite3.connect(str(db_path))

    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
    conn.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        [("window_size", window_size), ("stride", stride), ("window_min_length_threshold", 0)],
    )

    conn.execute(
        "CREATE TABLE window_mappings "
        "(window_idx INTEGER PRIMARY KEY, sequence_id TEXT NOT NULL, window_in_seq_idx INTEGER NOT NULL)"
    )

    global_idx = 0
    total_seqs = 0
    for seq_id, seq in sequences:
        n_win = compute_num_windows(len(seq), window_size, stride)
        for i in range(n_win):
            conn.execute("INSERT INTO window_mappings VALUES (?, ?, ?)", (global_idx, seq_id, i))
            global_idx += 1
        total_seqs += 1

    conn.execute("INSERT OR REPLACE INTO metadata VALUES ('total_windows', ?)", (global_idx,))
    conn.execute("INSERT OR REPLACE INTO metadata VALUES ('distinct_sequences', ?)", (total_seqs,))
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
def eden_data_dir(tmp_path):
    """Create synthetic genomic data in Eden SQLite format.

    Generates 20 sequences of length 512 each (random DNA) across 2 samples.
    With window_size=256 and stride=200, each sequence produces multiple windows,
    giving enough data for 200 training steps with micro_batch_size=1.
    """
    window_size = 256
    stride = 200

    sequences = []
    for i in range(10):
        seq_id = f"BCR__ECT-SAMPLE1__CT1-{i}"
        sequences.append((seq_id, _random_dna(512, seed=i)))
    for i in range(10):
        seq_id = f"BCR__ECT-SAMPLE2__CT1-{i}"
        sequences.append((seq_id, _random_dna(512, seed=100 + i)))

    seq_db_dir = _create_sequence_dbs(tmp_path, sequences)
    train_win_db = _create_window_db(tmp_path, sequences, window_size, stride, "train_windows.db")
    val_win_db = _create_window_db(tmp_path, sequences, window_size, stride, "val_windows.db")

    total_windows = sum(compute_num_windows(len(s), window_size, stride) for _, s in sequences)
    print(
        f"Created Eden test data: {len(sequences)} sequences, {total_windows} windows, "
        f"window_size={window_size}, stride={stride}"
    )

    return {
        "sequence_db_dir": seq_db_dir,
        "train_window_db": train_win_db,
        "val_window_db": val_win_db,
        "window_size": window_size,
        "stride": stride,
        "total_windows": total_windows,
    }


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seeds for reproducibility."""
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sharded_eden_fsdp2_bshd_convergence(tmp_path, recipe_path, eden_data_dir):
    """Test FSDP2 BSHD training with ShardedEden data converges over 200 steps.

    Validates:
    - Training runs end-to-end with use_sharded_eden=true
    - Loss is finite and > 1.0 (not collapsed)
    - Loss decreases from initial value (model is learning)
    """
    tokenizer_path = str(recipe_path / "tokenizers" / "nucleotide_fast_tokenizer")

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                # Model config — small model with nucleotide vocab
                "+config_kwargs.vocab_size=256",
                "config_kwargs.num_hidden_layers=2",
                "config_kwargs.hidden_size=384",
                "config_kwargs.intermediate_size=1536",
                "config_kwargs.num_attention_heads=6",
                "config_kwargs.num_key_value_heads=6",
                "config_kwargs.attn_input_format=bshd",
                # Dataset config
                f"dataset.tokenizer_name_or_path={tokenizer_path}",
                f"dataset.max_seq_length={eden_data_dir['window_size']}",
                "dataset.micro_batch_size=1",
                "dataset.num_workers=0",
                # Enable sharded Eden
                "use_sharded_eden=true",
                f"sharded_eden.sequence_db_dir={eden_data_dir['sequence_db_dir']}",
                f"sharded_eden.train_window_db={eden_data_dir['train_window_db']}",
                f"sharded_eden.stride={eden_data_dir['stride']}",
                # Training config
                "num_train_steps=200",
                "use_torch_compile=false",
                "use_meta_device=false",
                "logger.frequency=10",
            ],
        )

    final_loss = main_fsdp2(config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss is not None, "Training returned None loss"
    assert torch.isfinite(torch.tensor(final_loss)), f"Final loss {final_loss} is not finite"
    assert final_loss > 1.0, f"Final loss {final_loss} is suspiciously low (< 1.0), possible bug"
    assert final_loss < 8.0, f"Final loss {final_loss} is too high (>= 8.0), model isn't learning"
    print(f"BSHD ShardedEden training: final_loss={final_loss:.4f}")


def test_sharded_eden_fsdp2_thd_convergence(tmp_path, recipe_path, eden_data_dir):
    """Test FSDP2 THD training with ShardedEden data converges over 200 steps.

    Validates:
    - THD (sequence packing) path works with ShardedEden data
    - Loss is finite and > 1.0
    - Loss decreases from initial value
    """
    tokenizer_path = str(recipe_path / "tokenizers" / "nucleotide_fast_tokenizer")

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                # Model config — small model with nucleotide vocab
                "+config_kwargs.vocab_size=256",
                "config_kwargs.num_hidden_layers=2",
                "config_kwargs.hidden_size=384",
                "config_kwargs.intermediate_size=1536",
                "config_kwargs.num_attention_heads=6",
                "config_kwargs.num_key_value_heads=6",
                "config_kwargs.attn_input_format=thd",
                # Dataset config
                f"dataset.tokenizer_name_or_path={tokenizer_path}",
                f"dataset.max_seq_length={eden_data_dir['window_size']}",
                "dataset.micro_batch_size=1",
                "dataset.num_workers=0",
                # Enable sharded Eden with THD
                "use_sharded_eden=true",
                "use_sequence_packing=true",
                f"sharded_eden.sequence_db_dir={eden_data_dir['sequence_db_dir']}",
                f"sharded_eden.train_window_db={eden_data_dir['train_window_db']}",
                f"sharded_eden.stride={eden_data_dir['stride']}",
                # Training config
                "num_train_steps=200",
                "use_torch_compile=false",
                "use_meta_device=false",
                "logger.frequency=10",
            ],
        )

    final_loss = main_fsdp2(config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss is not None, "Training returned None loss"
    assert torch.isfinite(torch.tensor(final_loss)), f"Final loss {final_loss} is not finite"
    assert final_loss > 1.0, f"Final loss {final_loss} is suspiciously low (< 1.0), possible bug"
    assert final_loss < 8.0, f"Final loss {final_loss} is too high (>= 8.0), model isn't learning"
    print(f"THD ShardedEden training: final_loss={final_loss:.4f}")
