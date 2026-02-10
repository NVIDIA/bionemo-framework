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

"""Standalone port of John's ShardedEdenDataset for the llama3 native TE recipe.

This is a direct port of ``bionemo.evo2.data.sharded_eden_dataloader.ShardedEdenDataset``
with NeMo / Megatron dependencies removed.  The SQLite query logic, windowing,
connection management, and data retrieval are kept identical to John's code.

The only changes are:
  * NeMo ``TokenizerSpec`` → HuggingFace ``AutoTokenizer``
  * NeMo/Megatron ``dist`` utilities → standard ``torch.distributed``
  * Output format changed from Megatron-style ``{tokens, labels, loss_mask, position_ids}``
    to HF-style ``{input_ids, attention_mask}`` so that the existing BSHD / THD
    collators can handle label creation, padding, and packing.

**No NeMo or Megatron dependencies are required.**
"""

from __future__ import annotations

import csv
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from collator import DataCollatorWithFlattening, TokenPackingDataset
from distributed_config import DistributedConfig
from genomic_dataset import GenomicDataCollator


logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configurable column names (same as John's code)
# -----------------------------------------------------------------------------
SEQUENCE_ID_COLUMN_NAME = "contig_id"
SEQUENCE_COLUMN_NAME = "nt_sequence"


def extract_sample_id(sequence_id: str) -> str:
    """Extract sample ID from sequence ID format: BCR__EXT-SAMPLE1__CT1-1."""
    parts = sequence_id.split("__")[1].split("-")[1:]
    return ".".join(parts)


def compute_num_windows(seq_len: int, window_size: int = 8192, stride: int = 7992) -> int:
    """Helper method to compute number of windows for a sequence."""
    if seq_len < window_size:
        return 1
    else:
        return 1 + (seq_len - window_size) // stride


class ShardedEdenDataset(Dataset):
    """High-performance Dataset that uses SQLite databases for sequence storage and window mapping.

    Direct port of John's ``ShardedEdenDataset`` from
    ``bionemo.evo2.data.sharded_eden_dataloader`` with NeMo/Megatron deps removed.

    Assumes that the window_db_path points to a database pre-computed for a
    specific data split (e.g., train, validation, or test).

    Output format: ``{input_ids: list[int], attention_mask: list[int]}``
    (HF-compatible; the collator handles label creation and padding).
    """

    def __init__(
        self,
        sequence_db_dir: str,
        window_db_path: str,
        tokenizer_name_or_path: str,
        seq_length: int = 8192,
        rc_aug: bool = False,
        stride: int | None = 7992,
        window_min_length_threshold: int | None = None,
        split: str = "train",
        log_windows: bool = False,
        log_dir: str | None = None,
    ) -> None:
        """Initialize the ShardedEdenDataset."""
        super().__init__()
        self.seq_length = seq_length
        self.sequence_db_dir = sequence_db_dir
        self.window_db_path = window_db_path
        self.rc_aug = rc_aug
        self.stride = stride if stride is not None else 7992
        self.window_min_length_threshold = int(window_min_length_threshold) if window_min_length_threshold else 0
        self.split = split
        # Window access logging setup (lazy init in __getitem__)
        self.log_windows = log_windows
        # Remember desired log directory for lazy init in worker processes
        self._log_dir = log_dir

        # --- tokenizer (HF instead of NeMo) --------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._bos_id: int = self.tokenizer.bos_token_id  # type: ignore[assignment]
        self._eos_id: int = self.tokenizer.eos_token_id  # type: ignore[assignment]
        self._pad_id: int = self.tokenizer.pad_token_id  # type: ignore[assignment]

        # Number of special tokens the HF tokenizer adds (BOS + EOS = 2).
        # John's NeMo code uses BOS + SEP + EOS = 3, but our HF tokenizer
        # has no SEP, so we use 2 and adjust eff_len accordingly.
        self._num_special_tokens = 2
        # Effective content length per window (same role as eff_len in John's code)
        self._eff_len = self.seq_length - self._num_special_tokens

        # Create mapping from sample_id to SQLite file path
        # (identical to John's _create_sample_db_mapping)
        self._create_sample_db_mapping()

        # Pre-open all database connections for performance
        # (identical to John's _open_all_sequence_dbs)
        self._open_all_sequence_dbs()

        # Validates metadata and sets up the dataset
        # (identical to John's _validate_and_setup_db)
        self._validate_and_setup_db()

        # Counter for periodic commits if logging is enabled
        if self.log_windows:
            self._log_counter = 0

    # ------------------------------------------------------------------
    # Connection management (ported from John's code)
    # ------------------------------------------------------------------

    def _open_all_sequence_dbs(self):
        """Open all sequence database files ahead of time."""
        self.db_connections: dict[str, sqlite3.Connection] = {}
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Pre-opening {len(self.sample_db_mapping)} sequence database files...")

        for sample_id, db_path in self.sample_db_mapping.items():
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                self.db_connections[sample_id] = conn
            except sqlite3.Error as e:
                logger.error(f"Failed to open database for sample {sample_id} at {db_path}: {e}")
                raise

    def _create_sample_db_mapping(self):
        """Create mapping from sample ID to SQLite file path."""
        self.sample_db_mapping: dict[str, str] = {}

        # Scan the directory for sample SQLite files
        db_dir = Path(self.sequence_db_dir)
        for sample_dir in db_dir.iterdir():
            if sample_dir.is_dir():
                sample_id = sample_dir.name
                db_file = sample_dir / f"glm_dataset_{sample_id}.sqlite"
                if db_file.exists():
                    self.sample_db_mapping[sample_id] = str(db_file)

        if not self.sample_db_mapping:
            raise ValueError(f"No SQLite files found in {self.sequence_db_dir}")

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Found {len(self.sample_db_mapping)} sample SQLite files")

    def _validate_and_setup_db(self):
        """Connects to the window database, validates its metadata, and computes the length of the dataset."""
        self.window_db_conn = sqlite3.connect(f"file:{self.window_db_path}?mode=ro", uri=True)
        cursor = self.window_db_conn.cursor()

        # Validate metadata
        try:
            cursor.execute("SELECT key, value FROM metadata")
            db_meta = dict(cursor.fetchall())

            if "window_size" not in db_meta or "stride" not in db_meta:
                raise ValueError("Database metadata is missing 'window_size' or 'stride' keys.")

            db_window_size = int(db_meta["window_size"])
            db_stride = int(db_meta["stride"])
            db_min_len_raw = db_meta.get("window_min_length_threshold")
            db_min_len = int(db_min_len_raw) if db_min_len_raw is not None else None

            if db_window_size != self.seq_length or db_stride != self.stride:
                raise ValueError(
                    f"Database metadata mismatch! "
                    f"DB created with window_size={db_window_size}, stride={db_stride}. "
                    f"Dataset configured with seq_length={self.seq_length}, stride={self.stride}. "
                    f"Please re-run pre-computation or check your config."
                )

            # Validate presence and value of window_min_length_threshold only if enabled
            if self.window_min_length_threshold and self.window_min_length_threshold > 0:
                if db_min_len is None:
                    raise ValueError(
                        "Database metadata is missing 'window_min_length_threshold'. "
                        "Please re-run the pre-computation script with an updated version to populate this key."
                    )
                if db_min_len != self.window_min_length_threshold:
                    raise ValueError(
                        f"Database metadata mismatch for window_min_length_threshold! "
                        f"DB created with window_min_length_threshold={db_min_len}. "
                        f"Dataset configured with window_min_length_threshold={self.window_min_length_threshold}. "
                        f"Please re-run pre-computation or align the configuration."
                    )
            else:
                # Case: DB is pruned but runtime threshold is not set (> 0)
                if db_min_len is not None and int(db_min_len) > 0:
                    raise ValueError(
                        f"Window DB indicates pruning was applied (window_min_length_threshold={db_min_len}), "
                        "but the current configuration does not set --window-min-length-threshold (> 0). "
                        "Please set the argument to match the DB or use an unpruned database."
                    )
        except sqlite3.OperationalError:
            raise ValueError(
                f"Could not find `metadata` table in {self.window_db_path}. "
                "Please ensure the database was created with a recent version of the pre-computation script."
            )

        # Require modern metadata keys
        if "total_windows" not in db_meta or "distinct_sequences" not in db_meta:
            raise ValueError(
                "Database metadata must contain 'total_windows' and 'distinct_sequences'. "
                "Please re-run the pre-computation script to create an up-to-date window database."
            )

        # Read counts directly from metadata
        self.length = int(db_meta["total_windows"])

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Found {self.length} windows for {self.split} split in {self.window_db_path}.")

        # Distinct sequences directly from metadata
        self.distinct_sequences = int(db_meta["distinct_sequences"])
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Found {self.distinct_sequences} distinct sequences.")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def _get_db_connection(self, sample_id: str) -> sqlite3.Connection:
        """Get a pre-opened database connection for a sample."""
        conn = self.db_connections.get(sample_id)
        if conn is None:
            raise ValueError(f"No pre-opened SQLite connection found for sample {sample_id}")
        return conn

    @staticmethod
    def reverse_complement(seq: str) -> str:
        """Compute reverse complement of a sequence."""
        cmap = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
        return "".join(cmap.get(b, b) for b in reversed(seq))

    def __getitem__(self, idx: int | np.integer) -> dict[str, list[int]]:
        """Get a single item from the dataset.

        Returns ``{input_ids: list[int], attention_mask: list[int]}``
        for compatibility with HF collators.  The collator handles
        padding and label creation.

        The data retrieval logic (window lookup, SUBSTR query, stride
        computation) is identical to John's original code.
        """
        idx = int(idx)
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset with length {self.length}")

        # Step 1: Get the sequence_id and window info from the window DB.
        # (identical to John's code)
        window_cursor = self.window_db_conn.cursor()
        res = window_cursor.execute(
            "SELECT sequence_id, window_in_seq_idx FROM window_mappings WHERE window_idx = ?",
            (idx,),
        ).fetchone()

        if res is None:
            current_dbs = self.window_db_conn.execute("PRAGMA database_list;").fetchall()
            raise IndexError(
                f"Window index {idx} (type {type(idx)}) was not found in the database {current_dbs}, "
                "which is unexpected."
            )

        sequence_id, window_in_seq_idx = res

        # Log window access if enabled (identical to John's code)
        if self.log_windows:
            if not hasattr(self, "_log_writer"):
                self._init_window_logger(self._log_dir)

            try:
                sample_id_for_log = extract_sample_id(sequence_id)
            except Exception:
                sample_id_for_log = "unknown"

            row = [
                idx,
                sequence_id,
                sample_id_for_log,
                int(window_in_seq_idx),
                int(self._rank),
                int(time.time_ns()),
            ]
            self._log_writer.writerow(row)
            self._log_file.flush()

        # Step 2: Get sequence DB connection (identical to John's code)
        if len(self.db_connections) == 1:
            conn = next(iter(self.db_connections.values()))
            cursor = conn.cursor()
            sample_id = None
        else:
            sample_id = extract_sample_id(sequence_id)
            conn = self._get_db_connection(sample_id)
            cursor = conn.cursor()

        # Step 3: Calculate window position (identical to John's code)
        start_pos = window_in_seq_idx * self.stride

        # Step 4: Retrieve the subsequence via SQL SUBSTR
        # (same query as John's code, using self._eff_len instead of eff_len)
        subseq_query = (
            f"SELECT substr({SEQUENCE_COLUMN_NAME}, ?, ?) FROM sequences WHERE {SEQUENCE_ID_COLUMN_NAME} = ?"
        )
        result = cursor.execute(
            subseq_query,
            (start_pos + 1, self._eff_len, sequence_id),
        ).fetchone()

        if result is None or result[0] is None:
            raise ValueError(f"Sequence ID {sequence_id} not found in database for sample {sample_id}")

        seq = result[0].upper()

        # Step 5: Apply reverse complement augmentation if enabled
        # (identical to John's code)
        if self.rc_aug and np.random.default_rng().random() > 0.5:
            seq = self.reverse_complement(seq)

        # Step 6: Tokenize with HF tokenizer
        # John's code: token_ids = header + self.tokenizer.text_to_ids(seq) + footer
        # We use the HF tokenizer which adds BOS/EOS via its post-processor.
        encoding = self.tokenizer(
            seq,
            add_special_tokens=True,
            truncation=True,
            max_length=self.seq_length,
            return_attention_mask=True,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

    # ------------------------------------------------------------------
    # Cleanup (ported from John's code)
    # ------------------------------------------------------------------

    def __del__(self):
        """Close all database connections when the dataset is destroyed."""
        if hasattr(self, "window_db_conn") and self.window_db_conn:
            try:
                self.window_db_conn.close()
            except Exception:
                pass

        if hasattr(self, "db_connections"):
            for conn in self.db_connections.values():
                try:
                    conn.close()
                except Exception:
                    pass

        if hasattr(self, "_log_file") and self._log_file:
            try:
                self._log_file.flush()
            except Exception:
                pass
            try:
                self._log_file.close()
            except Exception:
                pass

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return (
            f"ShardedEdenDataset(windows={self.length}, "
            f"seq_length={self.seq_length}, stride={self.stride}, "
            f"samples={len(self.sample_db_mapping)})"
        )

    # ------------------------------------------------------------------
    # Logging helper methods (ported from John's code)
    # ------------------------------------------------------------------

    def _init_window_logger(self, log_dir: str | None = None):
        """Initialise CSV file for window access logging."""
        import uuid

        rank = dist.get_rank() if dist.is_initialized() else 0
        self._rank = rank
        log_uuid = str(uuid.uuid4())
        base_dir = Path(log_dir) if log_dir else Path(os.getcwd())
        base_dir = base_dir.resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        split_tag = getattr(self, "split", "unknown")
        csv_path = (base_dir / f"window_access_{split_tag}_rank{rank}_{log_uuid[:8]}.csv").resolve()
        if csv_path.exists():
            raise FileExistsError(
                f"File {csv_path} already exists, this should only happen on a uuid conflict "
                "and should be extremely rare"
            )
        self._log_file_path = str(csv_path)

        self._log_file = open(self._log_file_path, mode="a", newline="")
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow(
            [
                "window_idx",
                "sequence_id",
                "sample_id",
                "window_in_seq_idx",
                "rank",
                "access_ts",
            ]
        )

        print(f"Window access logger initialised at {self._log_file_path}")


# ---------------------------------------------------------------------------
# Dataloader factory helpers
# ---------------------------------------------------------------------------


def _make_collator(
    tokenizer: AutoTokenizer,
    *,
    uppercase_labels: bool = False,
    mask_degenerate_bases: bool = False,
    pad_sequences_to_be_divisible_by: int | None = None,
    thd: bool = False,
) -> Any:
    """Build the collator chain for BSHD or THD format."""
    base_mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None if thd else pad_sequences_to_be_divisible_by,
    )

    if thd:
        collator: Any = DataCollatorWithFlattening(
            collator=base_mlm_collator,
            pad_sequences_to_be_divisible_by=pad_sequences_to_be_divisible_by,
        )
    else:
        collator = base_mlm_collator

    if uppercase_labels or mask_degenerate_bases:
        collator = GenomicDataCollator(
            base_collator=collator,
            uppercase_labels=uppercase_labels,
            mask_degenerate_bases=mask_degenerate_bases,
        )

    return collator


def create_sharded_eden_bshd_dataloader(
    dist_config: DistributedConfig,
    sequence_db_dir: str,
    window_db_path: str,
    tokenizer_name_or_path: str,
    seq_length: int = 8192,
    stride: int = 7992,
    micro_batch_size: int = 8,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    seed: int = 42,
    rc_aug: bool = False,
    uppercase_labels: bool = False,
    mask_degenerate_bases: bool = False,
    pad_sequences_to_be_divisible_by: int | None = None,
) -> tuple[DataLoader, DistributedSampler]:
    """Create a BSHD dataloader from a sharded Eden window database.

    Returns ``(DataLoader, DistributedSampler)``.  Call
    ``sampler.set_epoch(epoch)`` at the start of each epoch.
    """
    dataset = ShardedEdenDataset(
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_db_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        seq_length=seq_length,
        stride=stride,
        rc_aug=rc_aug,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist_config.world_size,
        rank=dist_config.rank,
        shuffle=True,
        seed=seed,
    )

    data_collator = _make_collator(
        dataset.tokenizer,
        uppercase_labels=uppercase_labels,
        mask_degenerate_bases=mask_degenerate_bases,
        pad_sequences_to_be_divisible_by=pad_sequences_to_be_divisible_by,
        thd=False,
    )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=micro_batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    logger.info(
        "Created ShardedEden BSHD dataloader: %d windows, batch=%d, world=%d",
        len(dataset),
        micro_batch_size,
        dist_config.world_size,
    )

    return dataloader, sampler


class _ShardedSamplerIterableDataset(torch.utils.data.IterableDataset):
    """Wraps a map-style dataset + DistributedSampler into an IterableDataset.

    Handles multi-worker DataLoader correctly by splitting the sampler
    indices across workers so each worker processes a disjoint subset.
    """

    def __init__(self, dataset: ShardedEdenDataset, sampler: DistributedSampler) -> None:
        super().__init__()
        self.dataset = dataset
        self.sampler = sampler

    def __iter__(self):
        """Yield items, splitting indices across DataLoader workers."""
        worker_info = torch.utils.data.get_worker_info()
        indices = list(self.sampler)

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            indices = indices[worker_id::num_workers]

        for idx in indices:
            yield self.dataset[idx]

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch to the underlying sampler for re-shuffling."""
        self.sampler.set_epoch(epoch)


def create_sharded_eden_thd_dataloader(
    dist_config: DistributedConfig,
    sequence_db_dir: str,
    window_db_path: str,
    tokenizer_name_or_path: str,
    seq_length: int = 8192,
    stride: int = 7992,
    micro_batch_size: int | None = None,
    token_micro_batch_size: int | None = None,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    seed: int = 42,
    rc_aug: bool = False,
    uppercase_labels: bool = False,
    mask_degenerate_bases: bool = False,
    split_samples_in_token_packing: bool = True,
    pad_sequences_to_be_divisible_by: int | None = None,
) -> tuple[DataLoader, _ShardedSamplerIterableDataset]:
    """Create a THD (token-packed) dataloader from a sharded Eden window database.

    Returns ``(DataLoader, iterable_dataset)``.  Call
    ``iterable_dataset.set_epoch(epoch)`` at the start of each epoch.
    """
    dataset = ShardedEdenDataset(
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_db_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        seq_length=seq_length,
        stride=stride,
        rc_aug=rc_aug,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist_config.world_size,
        rank=dist_config.rank,
        shuffle=True,
        seed=seed,
    )

    # Resolve token budget
    if token_micro_batch_size is None:
        assert micro_batch_size is not None, "Provide either micro_batch_size or token_micro_batch_size."
        token_micro_batch_size = micro_batch_size * seq_length
    else:
        assert micro_batch_size is None, "Provide only one of micro_batch_size or token_micro_batch_size."
        assert token_micro_batch_size >= seq_length, "token_micro_batch_size must be >= seq_length."

    iterable_ds = _ShardedSamplerIterableDataset(dataset, sampler)

    data_collator = _make_collator(
        dataset.tokenizer,
        uppercase_labels=uppercase_labels,
        mask_degenerate_bases=mask_degenerate_bases,
        pad_sequences_to_be_divisible_by=pad_sequences_to_be_divisible_by,
        thd=True,
    )

    dataloader = DataLoader(
        TokenPackingDataset(
            iterable_ds,
            max_tokens_per_batch=token_micro_batch_size,
            split_samples=split_samples_in_token_packing,
        ),
        batch_size=None,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    logger.info(
        "Created ShardedEden THD dataloader: %d windows, token_mbs=%d, world=%d",
        len(dataset),
        token_micro_batch_size,
        dist_config.world_size,
    )

    return dataloader, iterable_ds
