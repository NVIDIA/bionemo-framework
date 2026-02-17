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

"""Create window mappings for hybrid HF dataset approach.

This script creates a lightweight mapping file that stores (sequence_idx, start_position)
for each window. The mapping file is ~4 GB for 238M windows, compared to ~15 TB for
pre-tokenized windows.

During training, the HFWindowedDataset uses this mapping to:
1. Look up which sequence and position a window comes from
2. Fetch the raw text from the Arrow cache (memory-mapped, fast)
3. Slice the window and tokenize on-the-fly

Usage:
    python create_window_mappings.py \
        --data-path /data/opengenome2/json/pretraining_or_both_phases/metagenomes \
        --output /data/opengenome2/cache/metagenome_window_mappings.npy \
        --window-size 8192 \
        --stride 7992

    # Or use existing HF cache:
    python create_window_mappings.py \
        --hf-cache-dir /root/.cache/huggingface/datasets/metagenomes/default/0.0.0/7d4bc899d7a9d4a6 \
        --output /data/opengenome2/cache/metagenome_window_mappings.npy \
        --window-size 8192 \
        --stride 7992
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_num_windows(seq_len: int, window_size: int, stride: int) -> int:
    """Compute number of windows for a sequence.

    Args:
        seq_len: Length of the sequence in characters.
        window_size: Size of each window (effective content, excluding special tokens).
        stride: Stride between windows.

    Returns:
        Number of windows this sequence produces.
    """
    if seq_len <= window_size:
        return 1
    return 1 + (seq_len - window_size) // stride


def create_mappings_from_hf_dataset(
    dataset,
    window_size: int,
    stride: int,
    text_column: str = "text",
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Create window mappings from a HuggingFace dataset.

    This function iterates through the dataset, computes window positions for each
    sequence, and returns a numpy array of (sequence_idx, start_position) pairs.

    Args:
        dataset: HuggingFace Dataset object (must be non-streaming).
        window_size: Size of each window in characters.
        stride: Stride between windows in characters.
        text_column: Name of the column containing text.
        chunk_size: Number of sequences to process before logging progress.

    Returns:
        Numpy array of shape (num_windows, 2) with dtype int32.
        Column 0: sequence_idx, Column 1: start_position.
    """
    num_sequences = len(dataset)
    logger.info(f"Processing {num_sequences:,} sequences...")

    # First pass: count total windows to pre-allocate array
    logger.info("Pass 1/2: Counting windows...")
    total_windows = 0
    window_counts = []

    start_time = time.time()
    for i in range(0, num_sequences, chunk_size):
        end_idx = min(i + chunk_size, num_sequences)
        chunk = dataset[i:end_idx]
        texts = chunk[text_column]

        for text in texts:
            num_win = compute_num_windows(len(text), window_size, stride)
            window_counts.append(num_win)
            total_windows += num_win

        elapsed = time.time() - start_time
        rate = end_idx / elapsed if elapsed > 0 else 0
        eta = (num_sequences - end_idx) / rate if rate > 0 else 0
        logger.info(
            f"  Counted {end_idx:,}/{num_sequences:,} sequences "
            f"({100 * end_idx / num_sequences:.1f}%), "
            f"windows so far: {total_windows:,}, "
            f"rate: {rate:.0f}/s, ETA: {eta / 60:.1f} min"
        )

    logger.info(f"Total windows: {total_windows:,}")
    logger.info(f"Average windows per sequence: {total_windows / num_sequences:.2f}")

    # Second pass: fill the mappings array
    logger.info("Pass 2/2: Creating mappings...")
    mappings = np.zeros((total_windows, 2), dtype=np.int32)

    window_idx = 0
    seq_idx = 0
    start_time = time.time()

    for i in range(0, num_sequences, chunk_size):
        end_idx = min(i + chunk_size, num_sequences)
        chunk = dataset[i:end_idx]
        texts = chunk[text_column]

        for text in texts:
            num_win = window_counts[seq_idx]

            for win_in_seq in range(num_win):
                start_pos = win_in_seq * stride
                mappings[window_idx, 0] = seq_idx
                mappings[window_idx, 1] = start_pos
                window_idx += 1

            seq_idx += 1

        elapsed = time.time() - start_time
        rate = end_idx / elapsed if elapsed > 0 else 0
        eta = (num_sequences - end_idx) / rate if rate > 0 else 0
        logger.info(
            f"  Processed {end_idx:,}/{num_sequences:,} sequences "
            f"({100 * end_idx / num_sequences:.1f}%), "
            f"windows: {window_idx:,}/{total_windows:,}, "
            f"rate: {rate:.0f}/s, ETA: {eta / 60:.1f} min"
        )

    assert window_idx == total_windows, f"Window count mismatch: {window_idx} vs {total_windows}"
    return mappings


def save_mappings(mappings: np.ndarray, output_path: str, metadata: dict | None = None):
    """Save mappings to disk with optional metadata.

    Args:
        mappings: Numpy array of shape (num_windows, 2).
        output_path: Path to save the .npy file.
        metadata: Optional metadata dict to save alongside.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save mappings
    np.save(output_path, mappings)
    logger.info(f"Saved mappings to {output_path}")
    logger.info(f"  Shape: {mappings.shape}")
    logger.info(f"  Size: {mappings.nbytes / 1e9:.2f} GB")

    # Save metadata
    if metadata:
        meta_path = output_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {meta_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create window mappings for hybrid HF dataset approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data-path",
        help="Path to parquet data directory (will load via HF datasets).",
    )
    input_group.add_argument(
        "--hf-cache-dir",
        help="Path to existing HF cache directory (load_from_disk).",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the mappings .npy file.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to process (default: train).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8190,
        help="Window size in characters (default: 8190, which is 8192 - 2 for BOS/EOS).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=7992,
        help="Stride between windows (default: 7992).",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of text column (default: text).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Chunk size for progress logging (default: 100000).",
    )

    args = parser.parse_args()

    # Load dataset
    import datasets

    if args.data_path:
        logger.info(f"Loading dataset from {args.data_path}...")
        dataset = datasets.load_dataset(
            args.data_path,
            split=args.split,
            streaming=False,
        )
    else:
        logger.info(f"Loading dataset from HF cache at {args.hf_cache_dir}...")
        dataset = datasets.load_from_disk(args.hf_cache_dir)
        # If it's a DatasetDict, get the requested split
        if isinstance(dataset, datasets.DatasetDict):
            if args.split in dataset:
                dataset = dataset[args.split]
            else:
                logger.error(f"Split '{args.split}' not found. Available: {list(dataset.keys())}")
                sys.exit(1)

    logger.info(f"Dataset loaded: {len(dataset):,} sequences")

    # Create mappings
    start_time = time.time()
    mappings = create_mappings_from_hf_dataset(
        dataset=dataset,
        window_size=args.window_size,
        stride=args.stride,
        text_column=args.text_column,
        chunk_size=args.chunk_size,
    )
    elapsed = time.time() - start_time

    # Save with metadata
    metadata = {
        "num_sequences": len(dataset),
        "num_windows": len(mappings),
        "window_size": args.window_size,
        "stride": args.stride,
        "text_column": args.text_column,
        "split": args.split,
        "creation_time_seconds": elapsed,
        "data_source": args.data_path or args.hf_cache_dir,
    }

    save_mappings(mappings, args.output, metadata)

    logger.info("=" * 60)
    logger.info("Mapping Creation Complete!")
    logger.info("=" * 60)
    logger.info(f"  Sequences:     {len(dataset):,}")
    logger.info(f"  Windows:       {len(mappings):,}")
    logger.info(f"  Expansion:     {len(mappings) / len(dataset):.2f}x")
    logger.info(f"  File size:     {mappings.nbytes / 1e9:.2f} GB")
    logger.info(f"  Time:          {elapsed / 60:.1f} minutes")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
