#!/usr/bin/env python3
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

"""Inspect samples from ShardedEden dataloader to understand data structure and metadata.

This script loads a few samples from the sharded eden dataloader and prints:
- Sequence IDs and metadata
- Raw DNA sequences (first portion)
- Tokenized sequences
- Batch structure
- Database metadata

Usage:
    python inspect_sharded_eden_samples.py \
        --window_db_path /data/bcr_eden/OG2_database_splits/og2__train__short.sqlite \
        --sequence_db_dir /data/bcr_eden/OG2_database_splits/ \
        --tokenizer_path ./tokenizers/nucleotide_fast_tokenizer \
        --num_samples 5 \
        --batch_size 2
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# Add parent directory to path to import sharded_eden_dataset
sys.path.insert(0, str(Path(__file__).parent.parent))

from sharded_eden_dataset import ShardedEdenDataset


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)


def inspect_database_metadata(window_db_path: str):
    """Inspect and print metadata from the window database."""
    import sqlite3

    print_separator("Database Metadata")
    conn = sqlite3.connect(window_db_path)
    cursor = conn.cursor()

    # Get metadata
    metadata = {}
    for row in cursor.execute("SELECT key, value FROM metadata"):
        metadata[row[0]] = row[1]

    print("Window Database Metadata:")
    for key, value in sorted(metadata.items()):
        print(f"  {key}: {value}")

    # Get sample of window mappings
    print("\nSample window mappings (first 5):")
    for row in cursor.execute("SELECT window_idx, sequence_id, window_in_seq_idx FROM window_mappings LIMIT 5"):
        print(f"  window_idx={row[0]}, sequence_id={row[1]}, window_in_seq_idx={row[2]}")

    # Count distinct sequences
    cursor.execute("SELECT COUNT(DISTINCT sequence_id) FROM window_mappings")
    distinct_seqs = cursor.fetchone()[0]
    print(f"\nTotal distinct sequences: {distinct_seqs}")

    conn.close()


def decode_tokens(tokenizer, token_ids: list[int], max_tokens: int = 50) -> str:
    """Decode token IDs to string, showing first max_tokens."""
    decoded = tokenizer.decode(token_ids[:max_tokens])
    if len(token_ids) > max_tokens:
        decoded += f" ... ({len(token_ids) - max_tokens} more tokens)"
    return decoded


def inspect_raw_sequence_from_db(dataset, sequence_id: str, start_pos: int, length: int):
    """Inspect raw sequence directly from the database using dataset's connections."""
    from sharded_eden_dataset import SEQUENCE_COLUMN_NAME, SEQUENCE_ID_COLUMN_NAME, extract_sample_id

    try:
        # Use the dataset's existing database connections (more reliable)
        if len(dataset.db_connections) == 1:
            conn = next(iter(dataset.db_connections.values()))
            cursor = conn.cursor()
        else:
            # Try to extract sample_id, but handle different formats gracefully
            try:
                sample_id = extract_sample_id(sequence_id)
                conn = dataset._get_db_connection(sample_id)
            except (ValueError, IndexError, AttributeError):
                # If extract_sample_id fails, try using the first connection
                if len(dataset.db_connections) > 0:
                    conn = next(iter(dataset.db_connections.values()))
                else:
                    return None, "No database connections available"
            cursor = conn.cursor()

        result = cursor.execute(
            f"SELECT substr({SEQUENCE_COLUMN_NAME}, ?, ?) FROM sequences WHERE {SEQUENCE_ID_COLUMN_NAME} = ?",
            (start_pos + 1, length, sequence_id),
        ).fetchone()

        if result and result[0]:
            return result[0].upper(), None
        else:
            return None, "Sequence not found in database"
    except Exception as e:
        return None, f"Error: {e!s}"


def inspect_samples(
    window_db_path: str,
    sequence_db_dir: str,
    tokenizer_path: str,
    num_samples: int = 5,
    batch_size: int = 2,
    seq_length: int = 8192,
    stride: int = 7992,
):
    """Inspect samples from the ShardedEden dataset."""
    print_separator("Loading Dataset")
    print(f"Window DB: {window_db_path}")
    print(f"Sequence DB Dir: {sequence_db_dir}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Seq Length: {seq_length}, Stride: {stride}")

    # Create dataset
    dataset = ShardedEdenDataset(
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_db_path,
        tokenizer_name_or_path=tokenizer_path,
        seq_length=seq_length,
        stride=stride,
        rc_aug=False,  # Disable for reproducible inspection
        pad_in_getitem=True,
    )

    print(f"\nDataset length: {len(dataset)} windows")
    print(f"Distinct sequences: {dataset.distinct_sequences}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for inspection
        num_workers=0,  # Single-threaded for easier debugging
        pin_memory=False,
    )

    print_separator("Inspecting Individual Samples")
    print(
        "NOTE: The llama3_native_te ShardedEdenDataset does NOT use control tags or SEP tokens.\n"
        "It's a simplified port that only uses BOS/EOS (unlike the original Megatron version\n"
        "which uses: [BOS] + [CTRL_IDS] + [SEP] + [DNA_SEQ] + [EOS]).\n"
        "Format here: [BOS] + [DNA_SEQ] + [EOS] + [PAD...]\n"
    )
    # Inspect individual samples
    for i in range(min(num_samples, len(dataset))):
        print(f"\n--- Sample {i} ---")
        sample = dataset[i]

        # Get sequence_id from database for this index
        import sqlite3

        conn = sqlite3.connect(window_db_path)
        cursor = conn.cursor()
        res = cursor.execute(
            "SELECT sequence_id, window_in_seq_idx FROM window_mappings WHERE window_idx = ?",
            (i,),
        ).fetchone()
        conn.close()

        if res:
            sequence_id, window_in_seq_idx = res
            start_pos = window_in_seq_idx * stride
            print(f"Sequence ID: {sequence_id}")
            print(f"Window in sequence index: {window_in_seq_idx}")
            print(f"Start position in sequence: {start_pos}")

            # Get raw sequence from database
            eff_len = seq_length - 2  # BOS + EOS
            raw_seq, error = inspect_raw_sequence_from_db(dataset, sequence_id, start_pos, eff_len)
            if raw_seq:
                print(f"Raw sequence from DB (first 200 chars): {raw_seq[:200]}...")
                print(f"Raw sequence length: {len(raw_seq)}")
                # Check for non-DNA characters (potential metadata)
                dna_chars = set("ATCGN")
                non_dna_chars = [c for c in raw_seq if c.upper() not in dna_chars and not c.isspace()]
                if non_dna_chars:
                    unique_non_dna = sorted(set(non_dna_chars))
                    print(f"⚠️  Non-DNA characters found: {unique_non_dna}")
                    print("   This may indicate metadata in the sequence!")
                else:
                    print("✓ Sequence contains only DNA characters (A, T, C, G, N)")
            elif error:
                print(f"⚠️  Could not retrieve raw sequence: {error}")

        # Decode tokens to see the actual sequence
        token_ids = sample["input_ids"]
        print(f"Token IDs length: {len(token_ids)}")
        print(f"Attention mask length: {len(sample.get('attention_mask', []))}")

        # Decode to see the sequence (this includes BOS/EOS tokens)
        decoded = dataset.tokenizer.decode(token_ids)
        # Show first 200 characters of decoded sequence
        print(f"Decoded sequence (first 200 chars): {decoded[:200]}...")
        if len(decoded) > 200:
            print(f"  ... (total length: {len(decoded)} chars)")

        # Show token IDs (first 30)
        print(f"Token IDs (first 30): {token_ids[:30]}...")
        if len(token_ids) > 30:
            print(f"  ... (total: {len(token_ids)} tokens)")

        # Check for special tokens
        bos_id = dataset.tokenizer.bos_token_id
        eos_id = dataset.tokenizer.eos_token_id
        pad_id = dataset.tokenizer.pad_token_id
        print(f"Special tokens - BOS: {bos_id}, EOS: {eos_id}, PAD: {pad_id}")
        print(f"First token: {token_ids[0]} (BOS={bos_id}, matches: {token_ids[0] == bos_id})")
        last_non_pad = next((t for t in reversed(token_ids) if t != pad_id), None)
        print(
            f"Last non-pad token: {last_non_pad} (EOS={eos_id}, matches: {last_non_pad == eos_id if last_non_pad else False})"
        )

    print_separator("Inspecting Batches")
    print(
        "NOTE: The llama3_native_te ShardedEdenDataset does NOT use control tags or SEP tokens.\n"
        "It's a simplified port that only uses BOS/EOS (unlike the original Megatron version\n"
        "which uses: [BOS] + [CTRL_IDS] + [SEP] + [DNA_SEQ] + [EOS]).\n"
        "Format here: [BOS] + [DNA_SEQ] + [EOS] + [PAD...]\n"
    )

    # Inspect batches
    batch_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_count >= 3:  # Show first 3 batches
            break

        print(f"\n--- Batch {batch_idx} ---")
        print(f"Batch keys: {list(batch.keys())}")

        if "input_ids" in batch:
            input_ids = batch["input_ids"]
            # Handle both list and tensor cases
            if isinstance(input_ids, list):
                print(f"Input IDs type: list, length: {len(input_ids)}")
                # Check if it's a list of lists/tensors or a flat list
                if input_ids and isinstance(input_ids[0], (list, torch.Tensor)):
                    print(f"  This is a list of {len(input_ids)} samples")
                    print(
                        f"  First sample type: {type(input_ids[0])}, length: {len(input_ids[0]) if hasattr(input_ids[0], '__len__') else 'N/A'}"
                    )
                    # Convert first sample to list if it's a tensor
                    if isinstance(input_ids[0], torch.Tensor):
                        first_sample_tokens = input_ids[0].tolist()
                    else:
                        first_sample_tokens = input_ids[0]
                else:
                    # It's a flat list (shouldn't happen with batching, but handle it)
                    print(f"  Flat list of length {len(input_ids)}")
                    first_sample_tokens = input_ids[:100]  # Just show first 100
            elif isinstance(input_ids, torch.Tensor):
                print(f"Input IDs shape: {input_ids.shape}")
                print(f"Input IDs dtype: {input_ids.dtype}")
                first_sample_tokens = input_ids[0].tolist()
            else:
                print(f"Input IDs type: {type(input_ids)}")
                first_sample_tokens = []

            # Show first sample in batch
            if first_sample_tokens:
                print("\nFirst sample in batch:")
                print(f"  Token IDs (first 30): {first_sample_tokens[:30]}...")
                print(f"  Total tokens: {len(first_sample_tokens)}")
                decoded = dataset.tokenizer.decode(first_sample_tokens)
                print(f"  Decoded (first 200 chars): {decoded[:200]}...")

        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"]
            # Handle both list and tensor cases
            if isinstance(attention_mask, list):
                print(f"Attention mask type: list, length: {len(attention_mask)}")
                if attention_mask:
                    if isinstance(attention_mask[0], (list, torch.Tensor)):
                        print(f"  This is a list of {len(attention_mask)} samples")
                        if isinstance(attention_mask[0], torch.Tensor):
                            print(f"  First sample mask sum: {attention_mask[0].sum().item()}")
                        else:
                            print(f"  First sample mask sum: {sum(attention_mask[0])}")
                    else:
                        print(f"  Flat list, sum: {sum(attention_mask)}")
            elif isinstance(attention_mask, torch.Tensor):
                print(f"Attention mask shape: {attention_mask.shape}")
                print(f"Attention mask sum (non-padded tokens): {attention_mask.sum(dim=1).tolist()}")

        batch_count += 1

    print_separator("Summary")
    print(f"✓ Inspected {num_samples} individual samples")
    print(f"✓ Inspected {batch_count} batches (batch_size={batch_size})")
    print(f"✓ Dataset contains {len(dataset)} total windows")
    print(f"✓ Dataset contains {dataset.distinct_sequences} distinct sequences")


def main():
    """Main entry point for the inspection script."""
    parser = argparse.ArgumentParser(
        description="Inspect samples from ShardedEden dataloader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--window_db_path",
        type=str,
        required=True,
        help="Path to the window database SQLite file",
    )
    parser.add_argument(
        "--sequence_db_dir",
        type=str,
        required=True,
        help="Directory containing per-sample sequence SQLite databases",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./tokenizers/nucleotide_fast_tokenizer",
        help="Path to the tokenizer",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of individual samples to inspect",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for dataloader inspection",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=8192,
        help="Sequence length",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=7992,
        help="Stride between windows",
    )

    args = parser.parse_args()

    # Inspect database metadata first
    inspect_database_metadata(args.window_db_path)

    # Inspect samples
    inspect_samples(
        window_db_path=args.window_db_path,
        sequence_db_dir=args.sequence_db_dir,
        tokenizer_path=args.tokenizer_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
