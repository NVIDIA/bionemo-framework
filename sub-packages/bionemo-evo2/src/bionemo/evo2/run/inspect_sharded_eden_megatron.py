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

"""Inspect samples from the original Megatron ShardedEden dataloader.

This script uses the original Megatron implementation which includes:
- Control tags (optional metadata tokens)
- SEP tokens (separator between header and sequence)
- Format: [BOS] + [CTRL_IDS] + [SEP] + [DNA_SEQ] + [EOS]

Usage:
    python inspect_sharded_eden_megatron.py \
        --window-db-path /data/bcr_eden/OG2_database_splits/og2__train__short.sqlite \
        --sequence-db-dir /data/bcr_eden/OG2_database_splits/ \
        --vocab-file /path/to/vocab.json \
        --merges-file /path/to/merges.txt \
        --num-samples 5 \
        --batch-size 2 \
        --use-control-tags
"""

import argparse
import sqlite3
from typing import Optional

from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from torch.utils.data import DataLoader

from bionemo.evo2.data.sharded_eden_dataloader import ShardedEdenDataset
from bionemo.evo2.run.utils import patch_eden_tokenizer


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)


def inspect_database_metadata(window_db_path: str):
    """Inspect and print metadata from the window database."""
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


def inspect_samples(
    window_db_path: str,
    sequence_db_dir: str,
    vocab_file: Optional[str] = None,
    merges_file: Optional[str] = None,
    num_samples: int = 5,
    batch_size: int = 2,
    seq_length: int = 8192,
    stride: int = 7992,
    use_control_tags: bool = False,
    rc_aug: bool = False,
):
    """Inspect samples from the original Megatron ShardedEden dataset."""
    print_separator("Loading Dataset")
    print(f"Window DB: {window_db_path}")
    print(f"Sequence DB Dir: {sequence_db_dir}")
    print(f"Seq Length: {seq_length}, Stride: {stride}")
    print(f"Use Control Tags: {use_control_tags}")

    # Create tokenizer (same as in train.py)
    # The training script uses byte-level tokenizer by default
    if vocab_file is not None or merges_file is not None:
        print("WARNING: vocab_file/merges_file provided, but training script uses byte-level tokenizer.")
        print("Using byte-level tokenizer to match training configuration...")

    tokenizer = get_nmt_tokenizer("byte-level")

    # Patch tokenizer for Eden compatibility (sets _sep_id, etc.)
    patch_eden_tokenizer(tokenizer)

    print("\nTokenizer special tokens:")
    print(f"  BOS ID: {tokenizer.bos_id}")
    print(f"  EOS ID: {tokenizer.eos_id}")
    print(f"  SEP ID: {getattr(tokenizer, '_sep_id', getattr(tokenizer, 'sep_id', None))}")
    print(f"  PAD ID: {tokenizer.pad_id}")

    # Create dataset
    dataset = ShardedEdenDataset(
        tokenizer=tokenizer,
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_db_path,
        seq_length=seq_length,
        stride=stride,
        rc_aug=rc_aug,
        use_control_tags=use_control_tags,
        create_attention_mask=False,
    )

    print(f"\nDataset length: {len(dataset)} windows")
    print(f"Distinct sequences: {dataset.distinct_sequences}")

    if use_control_tags and hasattr(dataset, "ctrl_ids_map"):
        print(f"Control tags map size: {len(dataset.ctrl_ids_map)}")
        # Show a few examples
        sample_ctrl_ids = list(dataset.ctrl_ids_map.items())[:3]
        print("Sample control tag mappings:")
        for seq_id, ctrl_ids in sample_ctrl_ids:
            print(f"  {seq_id}: {ctrl_ids}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for inspection
        num_workers=0,  # Single-threaded for easier debugging
        pin_memory=False,
        collate_fn=dataset.collate_fn,
    )

    print_separator("Inspecting Individual Samples")
    print(
        "The original Megatron version uses:\n"
        "  Format: [BOS] + [CTRL_IDS] + [SEP] + [DNA_SEQ] + [EOS] + [PAD...]\n"
        "  - BOS: Beginning of sequence token\n"
        "  - CTRL_IDS: Optional control tag tokens (metadata about sequence type)\n"
        "  - SEP: Separator token between header and DNA sequence\n"
        "  - DNA_SEQ: The actual DNA sequence tokens\n"
        "  - EOS: End of sequence token\n"
    )

    # Inspect individual samples
    for i in range(min(num_samples, len(dataset))):
        print(f"\n--- Sample {i} ---")
        sample = dataset[i]

        # Get sequence_id from database for this index
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

            # Check for control tags
            if use_control_tags and hasattr(dataset, "ctrl_ids_map"):
                ctrl_ids = dataset.ctrl_ids_map.get(sequence_id, [])
                if ctrl_ids:
                    print(f"Control tag IDs: {ctrl_ids}")
                    # Try to decode them
                    try:
                        ctrl_text = tokenizer.ids_to_text(ctrl_ids)
                        print(f"Control tag text: {ctrl_text}")
                    except Exception:
                        print("  (Could not decode control tags)")

        # Get tokens
        tokens = sample["tokens"]
        labels = sample.get("labels")
        loss_mask = sample.get("loss_mask")

        print(f"Tokens shape: {tokens.shape}")
        print(f"Tokens dtype: {tokens.dtype}")

        # Convert to list for inspection
        token_list = tokens.tolist()

        # Identify special tokens
        bos_id = tokenizer.bos_id
        eos_id = tokenizer.eos_id
        sep_id = getattr(tokenizer, "_sep_id", getattr(tokenizer, "sep_id", None))
        pad_id = tokenizer.pad_id

        print(f"\nSpecial token IDs - BOS: {bos_id}, EOS: {eos_id}, SEP: {sep_id}, PAD: {pad_id}")

        # Find positions of special tokens
        bos_pos = [i for i, t in enumerate(token_list) if t == bos_id]
        sep_pos = [i for i, t in enumerate(token_list) if t == sep_id]
        eos_pos = [i for i, t in enumerate(token_list) if t == eos_id]
        pad_start = next((i for i, t in enumerate(token_list) if t == pad_id), len(token_list))

        print(f"BOS position(s): {bos_pos}")
        print(f"SEP position(s): {sep_pos}")
        print(f"EOS position(s): {eos_pos}")
        print(f"Padding starts at: {pad_start}")

        # Show header (BOS + CTRL + SEP)
        if bos_pos and sep_pos:
            header_end = sep_pos[0]
            header = token_list[: header_end + 1]
            print(f"\nHeader (BOS + CTRL + SEP): {header}")
            if use_control_tags and len(header) > 2:
                ctrl_in_header = header[1:-1]  # Between BOS and SEP
                print(f"  Control IDs in header: {ctrl_in_header}")

        # Show DNA sequence portion
        if sep_pos and eos_pos:
            dna_start = sep_pos[0] + 1
            dna_end = eos_pos[0]
            dna_tokens = token_list[dna_start:dna_end]
            print(f"\nDNA sequence tokens (first 30): {dna_tokens[:30]}...")
            print(f"DNA sequence length: {len(dna_tokens)} tokens")

            # Decode DNA portion
            try:
                dna_text = tokenizer.ids_to_text(dna_tokens[:100])  # First 100 tokens
                print(f"DNA sequence (first 200 chars): {dna_text[:200]}...")
            except Exception as e:
                print(f"Could not decode DNA sequence: {e}")

        # Show footer
        if eos_pos:
            footer_start = eos_pos[0]
            footer = token_list[footer_start:pad_start]
            print(f"\nFooter (EOS + PAD): {footer[:10]}... (showing first 10)")

        # Show loss mask info
        if loss_mask is not None:
            mask_sum = loss_mask.sum().item()
            print(f"\nLoss mask: {mask_sum}/{len(loss_mask)} tokens are included in loss")
            print("  (Special tokens are masked out)")

    print_separator("Inspecting Batches")
    # Inspect batches
    batch_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_count >= 3:  # Show first 3 batches
            break

        print(f"\n--- Batch {batch_idx} ---")
        print(f"Batch keys: {list(batch.keys())}")

        if "tokens" in batch:
            tokens = batch["tokens"]
            print(f"Tokens shape: {tokens.shape}")
            print(f"Tokens dtype: {tokens.dtype}")

            # Show first sample in batch
            print("\nFirst sample in batch:")
            first_sample_tokens = tokens[0].tolist()

            # Find special token positions
            bos_id = tokenizer.bos_id
            sep_id = getattr(tokenizer, "_sep_id", getattr(tokenizer, "sep_id", None))
            eos_id = tokenizer.eos_id

            bos_pos = [i for i, t in enumerate(first_sample_tokens) if t == bos_id]
            sep_pos = [i for i, t in enumerate(first_sample_tokens) if t == sep_id]
            eos_pos = [i for i, t in enumerate(first_sample_tokens) if t == eos_id]

            print(f"  Token IDs (first 30): {first_sample_tokens[:30]}...")
            print(f"  BOS at: {bos_pos[0] if bos_pos else 'not found'}")
            print(f"  SEP at: {sep_pos[0] if sep_pos else 'not found'}")
            print(f"  EOS at: {eos_pos[0] if eos_pos else 'not found'}")

            if sep_pos and eos_pos:
                dna_tokens = first_sample_tokens[sep_pos[0] + 1 : eos_pos[0]]
                try:
                    dna_text = tokenizer.ids_to_text(dna_tokens[:50])
                    print(f"  DNA sequence (first 100 chars): {dna_text[:100]}...")
                except Exception:
                    pass

        if "labels" in batch:
            labels = batch["labels"]
            print(f"Labels shape: {labels.shape}")

        if "loss_mask" in batch:
            loss_mask = batch["loss_mask"]
            print(f"Loss mask shape: {loss_mask.shape}")
            print(f"Loss mask sum per sample: {loss_mask.sum(dim=1).tolist()}")

        if "position_ids" in batch:
            position_ids = batch["position_ids"]
            print(f"Position IDs shape: {position_ids.shape}")

        batch_count += 1

    print_separator("Summary")
    print(f"✓ Inspected {num_samples} individual samples")
    print(f"✓ Inspected {batch_count} batches (batch_size={batch_size})")
    print(f"✓ Dataset contains {len(dataset)} total windows")
    print(f"✓ Dataset contains {dataset.distinct_sequences} distinct sequences")
    if use_control_tags:
        print(f"✓ Control tags enabled with {len(dataset.ctrl_ids_map)} unique mappings")


def main():
    """Main entry point for the inspection script."""
    parser = argparse.ArgumentParser(
        description="Inspect samples from original Megatron ShardedEden dataloader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--window-db-path",
        type=str,
        required=True,
        help="Path to the window database SQLite file",
    )
    parser.add_argument(
        "--sequence-db-dir",
        type=str,
        required=True,
        help="Directory containing per-sample sequence SQLite databases",
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        default=None,
        help="Path to vocabulary file (not used - training script uses byte-level tokenizer)",
    )
    parser.add_argument(
        "--merges-file",
        type=str,
        default=None,
        help="Path to merges file (not used - training script uses byte-level tokenizer)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of individual samples to inspect",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for dataloader inspection",
    )
    parser.add_argument(
        "--seq-length",
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
    parser.add_argument(
        "--use-control-tags",
        action="store_true",
        help="Enable control tags (metadata tokens)",
    )
    parser.add_argument(
        "--rc-aug",
        action="store_true",
        help="Enable reverse complement augmentation",
    )

    args = parser.parse_args()

    # Inspect database metadata first
    inspect_database_metadata(args.window_db_path)

    # Inspect samples
    inspect_samples(
        window_db_path=args.window_db_path,
        sequence_db_dir=args.sequence_db_dir,
        vocab_file=args.vocab_file,
        merges_file=args.merges_file,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        stride=args.stride,
        use_control_tags=args.use_control_tags,
        rc_aug=args.rc_aug,
    )


if __name__ == "__main__":
    main()
