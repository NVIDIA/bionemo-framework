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

"""Script to preprocess and cache OpenGenome2/metagenome dataset with windowing.

This script creates a cached HuggingFace dataset with all tokenization and windowing
done upfront. The cached dataset can then be used for training with global shuffling
via DistributedSampler.

Usage:
    python cache_dataset.py \
        --output-dir /data/opengenome2/cache/og2_metagenome_windowed_8192_7992 \
        --tokenizer ./tokenizers/nucleotide_fast_tokenizer \
        --data-path /data/opengenome2/json/pretraining_or_both_phases/metagenomes \
        --max-seq-length 8192 \
        --stride 7992 \
        --num-proc 8

For separate train/val caches:
    # Train split
    python cache_dataset.py \
        --output-dir /data/opengenome2/cache/og2_metagenome_windowed_8192_7992/train \
        --tokenizer ./tokenizers/nucleotide_fast_tokenizer \
        --data-path /data/opengenome2/json/pretraining_or_both_phases/metagenomes \
        --split train \
        --num-proc 8

    # Validation split
    python cache_dataset.py \
        --output-dir /data/opengenome2/cache/og2_metagenome_windowed_8192_7992/val \
        --tokenizer ./tokenizers/nucleotide_fast_tokenizer \
        --data-path /data/opengenome2/json/pretraining_or_both_phases/metagenomes \
        --split validation \
        --num-proc 8
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import datasets

from mapped_dataset import create_windowed_mapped_dataset


logger = logging.getLogger(__name__)


def setup_logging(log_file: str | None = None):
    """Setup logging to both console and optionally a file."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        # Create parent directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Also log to file
        file_handler = logging.FileHandler(log_file, mode="w")
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
    )

    # Also set HuggingFace datasets logging level
    import datasets

    datasets.logging.set_verbosity_info()


def validate_cache(cache_dir: str) -> bool:
    """Validate that a cached dataset is complete and loadable.

    Args:
        cache_dir: Path to the cached dataset directory.

    Returns:
        True if valid, False otherwise.
    """
    cache_path = Path(cache_dir)

    # Check required files exist
    required_files = ["dataset_info.json", "state.json"]
    for fname in required_files:
        if not (cache_path / fname).exists():
            logger.warning(f"Missing required file: {fname}")
            return False

    # Check for arrow files
    arrow_files = list(cache_path.glob("*.arrow")) + list(cache_path.glob("data-*.arrow"))
    if not arrow_files:
        logger.warning("No arrow files found in cache directory")
        return False

    # Try to load the dataset
    try:
        ds = datasets.load_from_disk(cache_dir)
        logger.info(f"Validation passed: {len(ds)} samples loadable")
        return True
    except Exception as e:
        logger.error(f"Failed to load cached dataset: {e}")
        return False


def main():
    """Main entry point for dataset caching script."""
    parser = argparse.ArgumentParser(
        description="Preprocess and cache dataset with windowing for efficient training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Cache OpenGenome2 metagenome data
    python cache_dataset.py \\
        --output-dir /data/cache/og2_metagenome \\
        --tokenizer ./tokenizers/nucleotide_fast_tokenizer \\
        --data-path /data/opengenome2/json/pretraining_or_both_phases/metagenomes

    # With parallel processing
    python cache_dataset.py \\
        --output-dir /data/cache/og2_metagenome \\
        --tokenizer ./tokenizers/nucleotide_fast_tokenizer \\
        --data-path /data/opengenome2/json/pretraining_or_both_phases/metagenomes \\
        --num-proc 16
        """,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for cached dataset. Will be created if it doesn't exist.",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to HuggingFace tokenizer directory or model name.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the parquet data directory or HuggingFace dataset name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to process (default: train).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length for each window (default: 8192).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=7992,
        help="Stride for windowing. Overlap = max_seq_length - stride (default: 7992).",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of the column containing text sequences (default: text).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for tokenization map operation (default: 100). "
        "Keep small since return_overflowing_tokens can expand each input significantly.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes for parallel tokenization. None for single-process.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate an existing cache without creating a new one.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of cache even if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load dataset and show statistics without saving.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file. If provided, logs will be written to both console and file.",
    )

    args = parser.parse_args()

    # Setup logging (must be done before any logging calls)
    setup_logging(args.log_file)

    output_path = Path(args.output_dir)

    # Validation only mode
    if args.validate_only:
        logger.info(f"Validating cache at {args.output_dir}...")
        if validate_cache(args.output_dir):
            logger.info("Cache validation passed!")
            sys.exit(0)
        else:
            logger.error("Cache validation failed!")
            sys.exit(1)

    # Check if cache already exists
    if output_path.exists() and (output_path / "dataset_info.json").exists():
        if not args.force:
            logger.info(f"Cache already exists at {args.output_dir}")
            logger.info("Use --force to recreate, or --validate-only to check integrity")
            sys.exit(0)
        else:
            logger.warning(f"Force flag set, will overwrite existing cache at {args.output_dir}")

    # Build load_dataset_kwargs
    load_dataset_kwargs = {
        "path": args.data_path,
        "split": args.split,
        "streaming": False,  # Must be non-streaming for mapped dataset
    }

    # Warn about memory usage with multiprocessing
    if args.num_proc is not None and args.num_proc > 1:
        logger.warning(
            f"Using {args.num_proc} processes for tokenization. "
            "Memory usage will increase proportionally. "
            "If you encounter OOM or subprocess deaths, try:\n"
            "  1. Reduce --num-proc (try 2-4 instead of 8)\n"
            "  2. Reduce --batch-size (try 50 instead of 100)\n"
            "  3. Use --num-proc 1 (single-process, slower but safer)"
        )

    logger.info("=" * 60)
    logger.info("Dataset Caching Configuration")
    logger.info("=" * 60)
    logger.info(f"  Data path:       {args.data_path}")
    logger.info(f"  Split:           {args.split}")
    logger.info(f"  Output dir:      {args.output_dir}")
    logger.info(f"  Tokenizer:       {args.tokenizer}")
    logger.info(f"  Max seq length:  {args.max_seq_length}")
    logger.info(f"  Stride:          {args.stride}")
    logger.info(f"  Overlap:         {args.max_seq_length - args.stride}")
    logger.info(f"  Text column:     {args.text_column}")
    logger.info(f"  Batch size:      {args.batch_size}")
    logger.info(f"  Num processes:   {args.num_proc or 'single-process'}")
    logger.info(f"  Dry run:         {args.dry_run}")
    logger.info("=" * 60)

    start_time = time.time()

    logger.info("Creating windowed dataset...")
    dataset, _tokenizer = create_windowed_mapped_dataset(
        tokenizer_name_or_path=args.tokenizer,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=args.max_seq_length,
        stride=args.stride,
        text_column=args.text_column,
        tokenize_batch_size=args.batch_size,
        num_proc=args.num_proc,
    )

    processing_time = time.time() - start_time

    # Print statistics
    logger.info("=" * 60)
    logger.info("Dataset Statistics")
    logger.info("=" * 60)
    logger.info(f"  Total windows:   {len(dataset):,}")
    logger.info(f"  Columns:         {dataset.column_names}")
    logger.info(f"  Processing time: {processing_time:.1f}s")

    # Sample a few examples to show stats
    if len(dataset) > 0:
        sample = dataset[0]
        logger.info(f"  Sample length:   {len(sample['input_ids'])} tokens")
        logger.info(f"  Sample tokens:   {sample['input_ids'][:10]}... (first 10)")

    logger.info("=" * 60)

    if args.dry_run:
        logger.info("Dry run complete. Dataset was not saved.")
        return

    # Save to disk
    logger.info(f"Saving dataset to {args.output_dir}...")
    save_start = time.time()

    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(args.output_dir)

    save_time = time.time() - save_start
    total_time = time.time() - start_time

    # Validate the saved dataset
    logger.info("Validating saved dataset...")
    if not validate_cache(args.output_dir):
        logger.error("Validation failed! Cache may be corrupted.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Caching Complete!")
    logger.info("=" * 60)
    logger.info(f"  Output path:     {args.output_dir}")
    logger.info(f"  Total windows:   {len(dataset):,}")
    logger.info(f"  Processing time: {processing_time:.1f}s")
    logger.info(f"  Save time:       {save_time:.1f}s")
    logger.info(f"  Total time:      {total_time:.1f}s")
    logger.info("=" * 60)
    logger.info("")
    logger.info("To use this cache in training:")
    logger.info(f'  cache_dir: "{args.output_dir}"')


if __name__ == "__main__":
    main()
