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

"""Explore John's SQLite validation database to understand its structure.

Usage:
    python explore_sqlite_db.py --window-db /data/OG2_database_splits/og2__validation__short.sqlite
    python explore_sqlite_db.py --sequence-db-dir /data/OG2_database_splits --list-samples
"""

import argparse
import os
import sqlite3


def explore_window_db(db_path: str):
    """Explore the window mapping database."""
    print(f"\n{'=' * 60}")
    print(f"WINDOW DATABASE: {db_path}")
    print(f"{'=' * 60}")

    if not os.path.exists(db_path):
        print("ERROR: Database not found!")
        return

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\nTables: {tables}")

    # Check metadata
    if "metadata" in tables:
        cursor.execute("SELECT * FROM metadata")
        metadata = cursor.fetchall()
        print("\nMetadata:")
        for key, value in metadata:
            print(f"  {key}: {value}")

    # Check window_mappings
    if "window_mappings" in tables:
        cursor.execute("SELECT COUNT(*) FROM window_mappings")
        count = cursor.fetchone()[0]
        print(f"\nWindow mappings: {count:,} total")

        # Sample windows
        cursor.execute("SELECT * FROM window_mappings LIMIT 5")
        print("\nSample windows (first 5):")
        for row in cursor.fetchall():
            print(f"  {row}")

        # Get unique sequence IDs count
        cursor.execute("SELECT COUNT(DISTINCT sequence_id) FROM window_mappings")
        unique_seqs = cursor.fetchone()[0]
        print(f"\nUnique sequences: {unique_seqs:,}")

        # Sample sequence IDs
        cursor.execute("SELECT DISTINCT sequence_id FROM window_mappings LIMIT 5")
        print("\nSample sequence IDs:")
        for row in cursor.fetchall():
            print(f"  {row[0]}")

    conn.close()


def list_sample_dbs(sequence_db_dir: str):
    """List all sample SQLite databases in the directory."""
    print(f"\n{'=' * 60}")
    print(f"SEQUENCE DATABASE DIRECTORY: {sequence_db_dir}")
    print(f"{'=' * 60}")

    if not os.path.exists(sequence_db_dir):
        print("ERROR: Directory not found!")
        return

    # Find all sample directories
    sample_dirs = []
    for item in os.listdir(sequence_db_dir):
        item_path = os.path.join(sequence_db_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains a SQLite file
            sqlite_file = os.path.join(item_path, f"glm_dataset_{item}.sqlite")
            if os.path.exists(sqlite_file):
                sample_dirs.append((item, sqlite_file))

    print(f"\nFound {len(sample_dirs)} sample databases")

    # Show first few
    print("\nSample databases (first 10):")
    for sample_id, db_path in sample_dirs[:10]:
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"  {sample_id}: {size_mb:.1f} MB")

    if len(sample_dirs) > 10:
        print(f"  ... and {len(sample_dirs) - 10} more")

    # Explore one sample DB in detail
    if sample_dirs:
        sample_id, db_path = sample_dirs[0]
        print(f"\n{'=' * 60}")
        print(f"SAMPLE DATABASE (first one): {sample_id}")
        print(f"{'=' * 60}")

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # List tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\nTables: {tables}")

        # Check sequences table
        if "sequences" in tables:
            cursor.execute("SELECT COUNT(*) FROM sequences")
            count = cursor.fetchone()[0]
            print(f"\nSequences: {count:,} total")

            # Sample sequences
            cursor.execute("SELECT contig_id, length(nt_sequence) as len FROM sequences LIMIT 5")
            print("\nSample sequences (first 5):")
            for contig_id, seq_len in cursor.fetchall():
                print(f"  {contig_id}: {seq_len:,} bp")

            # Get sequence length stats
            cursor.execute(
                "SELECT MIN(length(nt_sequence)), MAX(length(nt_sequence)), AVG(length(nt_sequence)) FROM sequences"
            )
            min_len, max_len, avg_len = cursor.fetchone()
            print(f"\nSequence length stats: min={min_len:,}, max={max_len:,}, avg={avg_len:,.0f}")

        conn.close()


def main():
    """Explore SQLite databases to understand their structure."""
    parser = argparse.ArgumentParser(description="Explore SQLite databases")
    parser.add_argument("--window-db", help="Path to window mapping database")
    parser.add_argument("--sequence-db-dir", help="Directory containing sample SQLite files")
    parser.add_argument("--sample-db", help="Path to a specific sample database")
    parser.add_argument("--list-samples", action="store_true", help="List all sample databases")
    args = parser.parse_args()

    if args.window_db:
        explore_window_db(args.window_db)

    if args.sequence_db_dir and args.list_samples:
        list_sample_dbs(args.sequence_db_dir)

    if args.sample_db:
        # Explore a specific sample database
        print(f"\n{'=' * 60}")
        print(f"SAMPLE DATABASE: {args.sample_db}")
        print(f"{'=' * 60}")

        conn = sqlite3.connect(f"file:{args.sample_db}?mode=ro", uri=True)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\nTables: {tables}")

        if "sequences" in tables:
            cursor.execute("SELECT COUNT(*) FROM sequences")
            count = cursor.fetchone()[0]
            print(f"\nSequences: {count:,} total")

            cursor.execute("SELECT contig_id, length(nt_sequence) as len FROM sequences LIMIT 10")
            print("\nSample sequences:")
            for contig_id, seq_len in cursor.fetchall():
                print(f"  {contig_id}: {seq_len:,} bp")

        conn.close()

    if not any([args.window_db, args.sequence_db_dir, args.sample_db]):
        parser.print_help()


if __name__ == "__main__":
    main()
