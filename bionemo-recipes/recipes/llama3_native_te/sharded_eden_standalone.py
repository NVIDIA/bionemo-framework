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

"""Standalone ShardedEdenDataset without NeMo/Megatron dependencies.

This is a simplified extraction of ShardedEdenDataset from bionemo-evo2 that removes
all NeMo and Megatron dependencies, making it usable with standard PyTorch DataLoader.

Original source: bionemo-evo2/src/bionemo/evo2/data/sharded_eden_dataloader.py
"""

import csv
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, default_collate


logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configurable column names (must match database schema)
# -----------------------------------------------------------------------------
SEQUENCE_ID_COLUMN_NAME = "contig_id"
SEQUENCE_LENGTH_COLUMN_NAME = "length"
SEQUENCE_COLUMN_NAME = "nt_sequence"


def extract_sample_id(sequence_id: str) -> str:
    """Extract sample ID from sequence ID format: BCR__EXT-SAMPLE1__CT1-1."""
    parts = sequence_id.split("__")[1].split("-")[1:]
    return ".".join(parts)


class ShardedEdenDatasetStandalone(Dataset):
    """Standalone ShardedEdenDataset without NeMo/Megatron dependencies.

    This dataset reads genomic sequences from SQLite databases and returns
    tokenized windows for language model training.

    Key features:
    - No NeMo or Megatron imports required
    - Works with any tokenizer that has: bos_id, eos_id, _sep_id, pad_id, text_to_ids()
    - Compatible with standard PyTorch DataLoader and DistributedSampler
    """

    def __init__(
        self,
        tokenizer,
        sequence_db_dir: str,
        window_db_path: str,
        seq_length: int,
        create_attention_mask: bool = False,
        rc_aug: bool = False,
        stride: Optional[int] = 7992,
        window_min_length_threshold: Optional[int] = None,
        use_control_tags: bool = False,
        split: str = "train",
        log_windows: bool = False,
        log_dir: Optional[str] = None,
        skip_stats: bool = True,
        log_tokens: bool = True,
    ) -> None:
        """Initialize the ShardedEdenDataset.

        Args:
            tokenizer: Tokenizer with bos_id, eos_id, _sep_id, pad_id, text_to_ids()
            sequence_db_dir: Directory containing per-sample SQLite databases
            window_db_path: Path to the window mappings database
            seq_length: Sequence length (must match database)
            create_attention_mask: Whether to create attention mask
            rc_aug: Whether to apply reverse complement augmentation
            stride: Stride for windowing (must match database)
            window_min_length_threshold: Minimum window length threshold
            use_control_tags: Whether to use control tags
            split: Data split name (for logging)
            log_windows: Whether to log window access patterns
            log_dir: Directory for window access logs
            skip_stats: Whether to skip computing statistics
            log_tokens: Whether to log tokens (only if log_windows=True)
        """
        super().__init__()
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.sequence_db_dir = sequence_db_dir
        self.window_db_path = window_db_path
        self.create_attention_mask = create_attention_mask
        self.rc_aug = rc_aug
        self.stride = stride if stride is not None else 7992
        self.window_min_length_threshold = int(window_min_length_threshold) if window_min_length_threshold else 0
        self.use_control_tags = use_control_tags
        self.split = split
        self.skip_stats = skip_stats
        self.log_windows = log_windows
        self.log_tokens = log_tokens if log_windows else False
        self._log_dir = log_dir

        # Create mapping from sample_id to SQLite file path
        self._create_sample_db_mapping()

        # Pre-open all database connections for performance
        self._open_all_sequence_dbs()

        # Validates metadata and sets up the dataset
        self._validate_and_setup_db()

        # Prepare control-tag IDs if needed
        if self.use_control_tags:
            self._prepare_control_tags()

        # Attention mask and position ids
        if create_attention_mask:
            self.attention_mask = torch.tril(torch.ones((seq_length, seq_length))).unsqueeze(0) < 0.5

        # Shared position_ids for memory efficiency
        if (
            not hasattr(ShardedEdenDatasetStandalone, "_position_ids")
            or ShardedEdenDatasetStandalone._position_ids.size(0) != seq_length
        ):
            ShardedEdenDatasetStandalone._position_ids = torch.arange(seq_length, dtype=torch.int64)
        self.position_ids = ShardedEdenDatasetStandalone._position_ids

        # Counter for periodic commits if logging is enabled
        if self.log_windows:
            self._log_counter = 0

    def _open_all_sequence_dbs(self):
        """Open all sequence database files ahead of time."""
        self.db_connections = {}
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
        self.sample_db_mapping = {}

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
        """Connects to the window database and validates metadata."""
        self.window_db_conn = sqlite3.connect(f"file:{self.window_db_path}?mode=ro", uri=True)
        cursor = self.window_db_conn.cursor()

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
                    f"Dataset configured with seq_length={self.seq_length}, stride={self.stride}."
                )

            if self.window_min_length_threshold and self.window_min_length_threshold > 0:
                if db_min_len is None:
                    raise ValueError("Database metadata is missing 'window_min_length_threshold'.")
                if db_min_len != self.window_min_length_threshold:
                    raise ValueError(
                        f"Database metadata mismatch for window_min_length_threshold! "
                        f"DB: {db_min_len}, Config: {self.window_min_length_threshold}."
                    )
            elif db_min_len is not None and int(db_min_len) > 0:
                raise ValueError(
                    f"Window DB indicates pruning was applied (threshold={db_min_len}), "
                    "but config does not set --window-min-length-threshold."
                )
        except sqlite3.OperationalError:
            raise ValueError(
                f"Could not find `metadata` table in {self.window_db_path}. "
                "Please ensure the database was created with the pre-computation script."
            )

        if "total_windows" not in db_meta or "distinct_sequences" not in db_meta:
            raise ValueError("Database metadata must contain 'total_windows' and 'distinct_sequences'.")

        self.length = int(db_meta["total_windows"])

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Found {self.length} windows for {self.split} split in {self.window_db_path}.")

        self.distinct_sequences = int(db_meta["distinct_sequences"])
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Found {self.distinct_sequences} distinct sequences.")

    def _prepare_control_tags(self):
        """Prepare control tag IDs for sequences."""
        self.ctrl_ids_map = {}

        cursor = self.window_db_conn.cursor()
        unique_sequence_ids = [row[0] for row in cursor.execute("SELECT DISTINCT sequence_id FROM window_mappings")]

        for seq_id in unique_sequence_ids:
            ctrl_name = seq_id.split("__")[0] if "__" in seq_id else seq_id
            self.ctrl_ids_map[seq_id] = self.tokenizer.text_to_ids(f"<ctrl_{ctrl_name.lower()}>")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def _get_db_connection(self, sample_id: str) -> sqlite3.Connection:
        """Get a pre-opened database connection for a sample."""
        conn = self.db_connections.get(sample_id)
        if conn is None:
            raise ValueError(f"No pre-opened SQLite connection found for sample {sample_id}")
        return conn

    def reverse_complement(self, seq: str) -> str:
        """Compute reverse complement of a sequence."""
        cmap = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
        return "".join(cmap.get(b, b) for b in reversed(seq))

    def __getitem__(self, idx: np.int64) -> dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset with length {self.length}")

        # Get sequence_id and window info from window DB
        window_cursor = self.window_db_conn.cursor()
        res = window_cursor.execute(
            "SELECT sequence_id, window_in_seq_idx FROM window_mappings WHERE window_idx = ?",
            (int(idx),),
        ).fetchone()

        if res is None:
            current_dbs = self.window_db_conn.execute("PRAGMA database_list;").fetchall()
            raise IndexError(f"Window index {idx} not found in database {current_dbs}")

        sequence_id, window_in_seq_idx = res

        # Log window access if enabled
        if self.log_windows:
            if not hasattr(self, "_log_writer"):
                self._init_window_logger(self._log_dir)

            try:
                sample_id_for_log = extract_sample_id(sequence_id)
            except Exception:
                sample_id_for_log = "unknown"

            row = [
                int(idx),
                sequence_id,
                sample_id_for_log,
                int(window_in_seq_idx),
                int(self._rank),
                int(time.time_ns()),
            ]

            if self.log_tokens:
                self._pending_log_tokens = row
            else:
                self._log_writer.writerow(row)
                self._log_file.flush()

        # Get DB connection
        if len(self.db_connections) == 1:
            conn = next(iter(self.db_connections.values()))
            cursor = conn.cursor()
            sample_id = None
        else:
            sample_id = extract_sample_id(sequence_id)
            conn = self._get_db_connection(sample_id)
            cursor = conn.cursor()

        # Calculate window position
        start_pos = window_in_seq_idx * self.stride

        # Build token window
        ctrl_ids = self.ctrl_ids_map.get(sequence_id, []) if self.use_control_tags else []
        bos_id = self.tokenizer.bos_id
        eos_id = self.tokenizer.eos_id
        sep_id = self.tokenizer._sep_id
        pad_id = self.tokenizer.pad_id

        header = [bos_id, *ctrl_ids, sep_id]
        footer = [eos_id]
        special_tokens_count = len(header) + len(footer)
        eff_len = self.seq_length - special_tokens_count

        # Retrieve subsequence from database
        subseq_query = (
            f"SELECT substr({SEQUENCE_COLUMN_NAME}, ?, ?) FROM sequences WHERE {SEQUENCE_ID_COLUMN_NAME} = ?"
        )
        result = cursor.execute(
            subseq_query,
            (start_pos + 1, eff_len, sequence_id),
        ).fetchone()

        if result is None or result[0] is None:
            raise ValueError(f"Sequence ID {sequence_id} not found in database for sample {sample_id}")

        seq = result[0].upper()

        # Apply reverse complement augmentation if enabled
        if self.rc_aug and np.random.default_rng().random() > 0.5:
            seq = self.reverse_complement(seq)

        # Tokenize
        token_ids = header + self.tokenizer.text_to_ids(seq) + footer

        # Pad/trim
        if len(token_ids) < self.seq_length:
            token_ids += [pad_id] * (self.seq_length - len(token_ids))
        else:
            token_ids = token_ids[: self.seq_length]

        tokens = torch.tensor(token_ids, dtype=torch.int64)

        # Create special_ids for loss mask
        flat_ctrl_ids = []
        if isinstance(ctrl_ids, list):
            for item in ctrl_ids:
                if isinstance(item, list):
                    flat_ctrl_ids.extend(item)
                else:
                    flat_ctrl_ids.append(item)

        special_ids_list = [bos_id, eos_id, sep_id, pad_id, *flat_ctrl_ids]
        special_ids = torch.tensor(special_ids_list, dtype=torch.int64)

        # Create labels for next token prediction
        labels = tokens.clone()
        labels[:-1] = tokens[1:]
        labels[-1] = pad_id

        # Create loss mask
        loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        loss_mask[torch.isin(labels, special_ids)] = 0

        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": self.position_ids,
        }
        if self.create_attention_mask:
            batch["attention_mask"] = self.attention_mask

        # Log tokens if enabled
        if self.log_windows and self.log_tokens and hasattr(self, "_pending_log_tokens"):
            first_10 = tokens[:10].tolist()
            self._log_writer.writerow([*self._pending_log_tokens, str(first_10)])
            self._log_file.flush()
            del self._pending_log_tokens

        return batch

    def collate_fn(self, batch):
        """Collate a batch of items into a single dictionary."""
        return default_collate(batch)

    def __del__(self):
        """Close all database connections when the dataset is destroyed."""
        if hasattr(self, "window_db_conn") and self.window_db_conn:
            self.window_db_conn.close()

        if hasattr(self, "db_connections"):
            for conn in self.db_connections.values():
                conn.close()

        if hasattr(self, "_log_file") and self._log_file:
            try:
                self._log_file.flush()
            except Exception:
                pass
            try:
                self._log_file.close()
            except Exception:
                pass

    def _init_window_logger(self, log_dir: Optional[str] = None):
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
            raise FileExistsError(f"File {csv_path} already exists")

        self._log_file_path = str(csv_path)
        self._log_file = open(self._log_file_path, mode="a", newline="")
        self._log_writer = csv.writer(self._log_file)
        header = [
            "window_idx",
            "sequence_id",
            "sample_id",
            "window_in_seq_idx",
            "rank",
            "access_ts",
        ]
        if self.log_tokens:
            header.append("first_10_tokens")
        self._log_writer.writerow(header)

        print(f"Window access logger initialised at {self._log_file_path}")
