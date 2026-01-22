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

"""Convert John's SQLite validation dataset to JSON format for HuggingFace datasets.

This script reads from John's ShardedEdenDataset SQLite format and exports to JSON
that can be loaded with HuggingFace datasets for validation.

Usage:
    python convert_sqlite_to_json.py \
        --sequence-db-dir /data/OG2_database_splits \
        --window-db /data/OG2_database_splits/og2__validation__short.sqlite \
        --output /data/opengenome2/json/validation/validation.json \
        --seq-length 8192 \
        --stride 7992

The output JSON format matches the training data format:
    {"sequence": "ACGTACGT..."}
    {"sequence": "TGCATGCA..."}
    ...
"""

import argparse
import json
import os
import sqlite3
from pathlib import Path

from tqdm import tqdm


def extract_sample_id(sequence_id: str) -> str:
    """Extract sample ID from sequence ID (matches John's logic).

    Example: "BCR__ECT-SAMPLE1__CT1-1" -> "SAMPLE1"
    """
    parts = sequence_id.split("__")[1].split("-")[1:]
    return ".".join(parts)


def get_sample_db_path(sequence_db_dir: str, sample_id: str) -> str:
    """Get the path to a sample's SQLite database."""
    return os.path.join(sequence_db_dir, sample_id, f"glm_dataset_{sample_id}.sqlite")


def main():
    """Convert SQLite validation data to JSON format."""
    parser = argparse.ArgumentParser(description="Convert SQLite validation data to JSON")
    parser.add_argument("--sequence-db-dir", required=True, help="Directory containing sample SQLite files")
    parser.add_argument("--window-db", required=True, help="Path to validation window database")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--seq-length", type=int, default=8192, help="Sequence length (window size)")
    parser.add_argument("--stride", type=int, default=7992, help="Stride between windows")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sequences (for testing)")
    parser.add_argument("--format", choices=["json", "jsonl"], default="jsonl", help="Output format")
    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Connect to window database
    print(f"Loading window database: {args.window_db}")
    window_conn = sqlite3.connect(f"file:{args.window_db}?mode=ro", uri=True)
    window_cursor = window_conn.cursor()

    # Check metadata
    window_cursor.execute("SELECT key, value FROM metadata")
    metadata = dict(window_cursor.fetchall())
    print(f"Window DB metadata: {metadata}")

    db_window_size = int(metadata.get("window_size", args.seq_length))
    db_stride = int(metadata.get("stride", args.stride))

    if db_window_size != args.seq_length or db_stride != args.stride:
        print(f"WARNING: DB created with window_size={db_window_size}, stride={db_stride}")
        print(f"         Using seq_length={args.seq_length}, stride={args.stride}")

    # Count total windows
    window_cursor.execute("SELECT COUNT(*) FROM window_mappings")
    total_windows = window_cursor.fetchone()[0]
    print(f"Total validation windows: {total_windows}")

    if args.limit:
        total_windows = min(total_windows, args.limit)
        print(f"Limiting to {total_windows} windows")

    # Build sample ID -> DB connection cache
    sample_db_cache = {}

    def get_sequence_db(sample_id: str):
        """Get or create connection to sample's sequence database."""
        if sample_id not in sample_db_cache:
            db_path = get_sample_db_path(args.sequence_db_dir, sample_id)
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Sample database not found: {db_path}")
            sample_db_cache[sample_id] = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        return sample_db_cache[sample_id]

    # Fetch all windows
    if args.limit:
        window_cursor.execute(
            "SELECT window_idx, sequence_id, window_in_seq_idx FROM window_mappings ORDER BY window_idx LIMIT ?",
            (args.limit,),
        )
    else:
        window_cursor.execute(
            "SELECT window_idx, sequence_id, window_in_seq_idx FROM window_mappings ORDER BY window_idx"
        )

    windows = window_cursor.fetchall()
    print(f"Fetched {len(windows)} window mappings")

    # Process each window and extract sequences
    sequences = []
    errors = []

    for window_idx, sequence_id, window_in_seq_idx in tqdm(windows, desc="Extracting sequences"):
        try:
            # Get sample ID and database connection
            sample_id = extract_sample_id(sequence_id)
            seq_conn = get_sequence_db(sample_id)
            seq_cursor = seq_conn.cursor()

            # Calculate window position (0-based for Python)
            start_pos = window_in_seq_idx * args.stride

            # Extract subsequence using SQL SUBSTR (1-indexed)
            seq_cursor.execute(
                "SELECT substr(nt_sequence, ?, ?) FROM sequences WHERE contig_id = ?",
                (start_pos + 1, args.seq_length, sequence_id),
            )
            result = seq_cursor.fetchone()

            if result is None or result[0] is None:
                errors.append(f"Sequence not found: {sequence_id}")
                continue

            sequence = result[0].upper()

            # Skip if sequence is too short (padding would be needed)
            if len(sequence) < args.seq_length // 2:
                errors.append(f"Sequence too short: {sequence_id} (len={len(sequence)})")
                continue

            sequences.append({"sequence": sequence})

        except Exception as e:
            errors.append(f"Error processing {sequence_id}: {e}")

    # Close all connections
    window_conn.close()
    for conn in sample_db_cache.values():
        conn.close()

    print(f"\nExtracted {len(sequences)} sequences")
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Write output
    print(f"\nWriting to {args.output}")
    if args.format == "jsonl":
        with open(args.output, "w") as f:
            f.writelines(json.dumps(seq) + "\n" for seq in tqdm(sequences, desc="Writing JSONL"))
    else:
        with open(args.output, "w") as f:
            json.dump(sequences, f)

    print(f"Done! Wrote {len(sequences)} sequences to {args.output}")

    # Print sample statistics
    if sequences:
        lengths = [len(s["sequence"]) for s in sequences[:100]]
        print(
            f"\nSample sequence lengths (first 100): min={min(lengths)}, max={max(lengths)}, avg={sum(lengths) / len(lengths):.0f}"
        )


if __name__ == "__main__":
    main()
