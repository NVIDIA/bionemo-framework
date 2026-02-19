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

r"""Convert OpenGenome2 JSONL(.gz) files to globally-shuffled, sharded Parquet files.

Why this script exists
======================
The HuggingFace streaming dataloader assigns physical files to ranks.
With 80 original JSONL files and 48 training ranks, each rank gets only
1-2 files — a narrow, biased slice of the full data distribution.  This
causes overfitting (low train loss, high test loss) because each rank's
shuffle buffer only mixes data from its own files.

This script:
1. Reads ALL input JSONL(.gz) files
2. Globally shuffles all sequences (rows) with a reproducible seed
3. Writes them out as N evenly-sized Parquet shards

After resharding, each rank gets ``N / world_size`` files containing
sequences from across the entire dataset — much more representative of
the full distribution.

Input format
============
OpenGenome2 metagenome JSONL files live at::

    /data/opengenome2/json/pretraining_or_both_phases/metagenomes/

Each ``.jsonl.gz`` (or ``.jsonl``) file contains one JSON object per line
with schema::

    {"text": "ATCGATCG...", "record": "chr1:100-200"}   # some files
    {"text": "ATCGATCG..."}                               # other files

Only the ``text`` column is required.  The ``record`` column (metadata) is
inconsistent across shards and is preserved if present but not required.

Output format
=============
Parquet files with a consistent schema: ``{"text": str}`` (one column).
The ``record`` column is dropped for schema consistency across all shards.

Usage
=====
Basic (on a machine with enough RAM for the full dataset)::

    python scripts/reshard_jsonl_to_parquet.py \
        --input-dir /data/opengenome2/json/pretraining_or_both_phases/metagenomes \
        --output-dir /data/opengenome2/parquet/metagenomes_resharded \
        --num-shards 480 \
        --seed 42

Chunked mode (for limited-RAM machines — processes files in chunks)::

    python scripts/reshard_jsonl_to_parquet.py \
        --input-dir /data/opengenome2/json/pretraining_or_both_phases/metagenomes \
        --output-dir /data/opengenome2/parquet/metagenomes_resharded \
        --num-shards 480 \
        --seed 42 \
        --chunk-size 10

Dry run (show file counts and estimated sizes without writing)::

    python scripts/reshard_jsonl_to_parquet.py \
        --input-dir /data/opengenome2/json/pretraining_or_both_phases/metagenomes \
        --output-dir /data/opengenome2/parquet/metagenomes_resharded \
        --num-shards 480 \
        --dry-run

Then update your training config::

    dataset:
      load_dataset_kwargs:
        path: "/data/opengenome2/parquet/metagenomes_resharded"
        split: "train"
        streaming: true

Choosing num_shards
===================
Rule of thumb: ``num_shards = 10 x world_size``.

- 48 GPUs → 480 shards (each rank gets 10 files)
- 64 GPUs → 640 shards
- Must be divisible by world_size for even distribution.

With 480 shards and ~238M sequences, each shard has ~496k sequences.
Because sequences are globally shuffled before sharding, every shard
contains a representative mix of the full dataset.  This means the
shuffle buffer is diverse from step 1, unlike the original files which
may group sequences by biological source.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def discover_input_files(input_dir: str, split: str = "train") -> list[Path]:
    """Find all JSONL/JSONL.gz files for the given split in input_dir.

    Looks for files matching common OpenGenome2 naming patterns:
    - data_metagenomics_train_chunk*.jsonl.gz
    - data_metagenomics_train_chunk*.jsonl
    - *.jsonl.gz / *.jsonl (fallback)

    Args:
        input_dir: Directory containing the JSONL files.
        split: Data split to look for ("train", "valid", "test").

    Returns:
        Sorted list of Path objects for the input files.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Try split-specific patterns first
    patterns = [
        f"data_metagenomics_{split}_chunk*.jsonl.gz",
        f"data_metagenomics_{split}_chunk*.jsonl",
        f"*_{split}_*.jsonl.gz",
        f"*_{split}_*.jsonl",
    ]

    files = []
    for pattern in patterns:
        files = sorted(input_path.glob(pattern))
        if files:
            break

    # Fallback: all jsonl files
    if not files:
        files = sorted(input_path.glob("*.jsonl.gz")) + sorted(input_path.glob("*.jsonl"))

    if not files:
        raise FileNotFoundError(f"No JSONL files found in {input_dir} for split '{split}'")

    return files


def read_jsonl_file_polars(filepath: Path, text_column: str = "text"):
    """Read a single JSONL(.gz) file using Polars and extract the text column.

    Args:
        filepath: Path to the JSONL or JSONL.gz file.
        text_column: Name of the column containing the genomic sequence.

    Returns:
        A polars DataFrame with a single ``text`` column.
    """
    import polars as pl

    logger.info(f"  Reading {filepath.name} ...")

    # Polars handles .gz decompression automatically
    df = pl.read_ndjson(filepath)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {filepath}. Columns: {df.columns}")

    # Keep only the text column for consistent schema
    return df.select(text_column)


def reshard_all_at_once(
    input_files: list[Path],
    output_dir: Path,
    num_shards: int,
    seed: int,
    text_column: str,
    row_group_size: int,
) -> dict:
    """Load all files into memory, globally shuffle, and write shards.

    This is the simplest and most effective approach. It requires enough
    RAM to hold all sequences in memory (typically ~50-100 GB for OG2
    metagenome with ~211M sequences).

    Args:
        input_files: List of input JSONL file paths.
        output_dir: Directory to write Parquet shards.
        num_shards: Number of output Parquet files.
        seed: Random seed for reproducible shuffling.
        text_column: Column name containing the sequence text.
        row_group_size: Parquet row group size.

    Returns:
        Dictionary with stats (total_rows, rows_per_shard, etc.).
    """
    import polars as pl

    # Step 1: Read all files
    logger.info(f"Step 1/3: Reading {len(input_files)} input files...")
    t0 = time.time()

    dfs = []
    for f in input_files:
        df = read_jsonl_file_polars(f, text_column)
        dfs.append(df)
        logger.info(f"    → {len(df):,} sequences")

    combined = pl.concat(dfs)
    total_rows = len(combined)
    t_read = time.time() - t0
    logger.info(f"  Total: {total_rows:,} sequences (read in {t_read:.1f}s)")

    # Step 2: Global shuffle
    logger.info(f"Step 2/3: Globally shuffling {total_rows:,} sequences with seed={seed}...")
    t0 = time.time()

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total_rows)
    # Polars gather (index selection) is very fast
    combined = combined.with_row_index("__idx__")
    combined = combined.filter(pl.col("__idx__").is_in(pl.Series(permutation)).not_())  # dummy to force reorder

    # More direct approach: use numpy permutation to reorder
    # Convert to list of indices and use polars gather
    combined = pl.concat(dfs)  # re-concat (fresh)
    indices = permutation.tolist()
    combined = combined[indices]

    t_shuffle = time.time() - t0
    logger.info(f"  Shuffled in {t_shuffle:.1f}s")

    # Step 3: Write shards
    logger.info(f"Step 3/3: Writing {num_shards} Parquet shards to {output_dir}...")
    t0 = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_per_shard = total_rows // num_shards
    remainder = total_rows % num_shards

    for i in range(num_shards):
        start = i * rows_per_shard + min(i, remainder)
        end = start + rows_per_shard + (1 if i < remainder else 0)
        shard = combined.slice(start, end - start)

        shard_path = output_dir / f"shard_{i:05d}.parquet"
        shard.write_parquet(shard_path, row_group_size=row_group_size)

        if (i + 1) % 50 == 0 or i == num_shards - 1:
            logger.info(f"  Written {i + 1}/{num_shards} shards ({end:,}/{total_rows:,} rows)")

    t_write = time.time() - t0
    logger.info(f"  Wrote {num_shards} shards in {t_write:.1f}s")

    return {
        "total_rows": total_rows,
        "num_shards": num_shards,
        "rows_per_shard": rows_per_shard,
        "remainder": remainder,
        "min_shard_size": rows_per_shard,
        "max_shard_size": rows_per_shard + (1 if remainder > 0 else 0),
    }


def reshard_chunked(
    input_files: list[Path],
    output_dir: Path,
    num_shards: int,
    seed: int,
    text_column: str,
    chunk_size: int,
    row_group_size: int,
) -> dict:
    """Two-pass approach for machines with limited RAM.

    Pass 1: Read all files, count rows, and compute a global permutation.
             Assign each row to a target shard based on the permutation.
    Pass 2: Re-read files in chunks, append each row to its assigned shard.

    This uses ~O(total_rows * 8 bytes) RAM for the permutation array plus
    one chunk of files in memory at a time.

    Args:
        input_files: List of input JSONL file paths.
        output_dir: Directory to write Parquet shards.
        num_shards: Number of output Parquet files.
        seed: Random seed for reproducible shuffling.
        text_column: Column name containing the sequence text.
        chunk_size: Number of input files to process at a time.
        row_group_size: Parquet row group size.

    Returns:
        Dictionary with stats.
    """
    import polars as pl

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: Count total rows per file
    logger.info(f"Pass 1/2: Counting rows in {len(input_files)} files...")
    file_row_counts = []
    total_rows = 0

    for f in input_files:
        # Use scan to count without loading full data
        try:
            count = pl.scan_ndjson(f).select(pl.len()).collect().item()
        except Exception:
            # Fallback: read and count
            df = read_jsonl_file_polars(f, text_column)
            count = len(df)
            del df

        file_row_counts.append(count)
        total_rows += count
        logger.info(f"  {f.name}: {count:,} rows")

    logger.info(f"  Total: {total_rows:,} rows across {len(input_files)} files")

    # Compute global permutation → shard assignment
    logger.info(f"  Computing global permutation (seed={seed})...")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total_rows)

    # Map each global row index to its target shard
    rows_per_shard = total_rows // num_shards
    remainder = total_rows % num_shards

    # shard_assignment[original_row_idx] = target_shard_idx
    shard_assignment = np.empty(total_rows, dtype=np.int32)
    for new_pos, old_pos in enumerate(permutation):
        # Which shard does new_pos fall into?
        # Shards 0..remainder-1 have (rows_per_shard+1) rows, rest have rows_per_shard
        if remainder > 0:
            # First `remainder` shards have one extra row
            threshold = remainder * (rows_per_shard + 1)
            if new_pos < threshold:
                shard_idx = new_pos // (rows_per_shard + 1)
            else:
                shard_idx = remainder + (new_pos - threshold) // rows_per_shard
        else:
            shard_idx = new_pos // rows_per_shard
        shard_assignment[old_pos] = shard_idx

    del permutation  # free memory

    # Pass 2: Re-read files in chunks and distribute rows to shards
    logger.info(f"Pass 2/2: Reading files in chunks of {chunk_size} and writing {num_shards} shards...")

    # Initialize shard buffers (lists of DataFrames per shard)
    shard_buffers: dict[int, list] = {i: [] for i in range(num_shards)}
    shard_row_counts = np.zeros(num_shards, dtype=np.int64)
    global_row_offset = 0

    for chunk_start in range(0, len(input_files), chunk_size):
        chunk_files = input_files[chunk_start : chunk_start + chunk_size]
        chunk_label = f"[files {chunk_start + 1}-{chunk_start + len(chunk_files)}/{len(input_files)}]"
        logger.info(f"  Processing chunk {chunk_label}...")

        for f in chunk_files:
            df = read_jsonl_file_polars(f, text_column)
            n = len(df)

            # Get shard assignments for this file's rows
            assignments = shard_assignment[global_row_offset : global_row_offset + n]

            # Split DataFrame by shard assignment
            df_with_shard = df.with_columns(pl.Series("__shard__", assignments))

            for shard_idx in range(num_shards):
                shard_df = df_with_shard.filter(pl.col("__shard__") == shard_idx).drop("__shard__")
                if len(shard_df) > 0:
                    shard_buffers[shard_idx].append(shard_df)
                    shard_row_counts[shard_idx] += len(shard_df)

            global_row_offset += n

        # Periodically flush large buffers to disk
        for shard_idx in range(num_shards):
            buffer_rows = sum(len(df) for df in shard_buffers[shard_idx])
            if buffer_rows > row_group_size * 2:
                _flush_shard_buffer(shard_buffers[shard_idx], output_dir, shard_idx, row_group_size, append=True)
                shard_buffers[shard_idx] = []

    # Final flush: write remaining buffers
    logger.info("  Flushing remaining buffers...")
    for shard_idx in range(num_shards):
        if shard_buffers[shard_idx]:
            _flush_shard_buffer(shard_buffers[shard_idx], output_dir, shard_idx, row_group_size, append=True)

    del shard_assignment  # free memory

    return {
        "total_rows": total_rows,
        "num_shards": num_shards,
        "rows_per_shard": rows_per_shard,
        "remainder": remainder,
        "min_shard_size": int(shard_row_counts.min()),
        "max_shard_size": int(shard_row_counts.max()),
    }


def _flush_shard_buffer(
    buffer: list,
    output_dir: Path,
    shard_idx: int,
    row_group_size: int,
    append: bool = False,
) -> None:
    """Write a shard buffer to a Parquet file.

    Args:
        buffer: List of DataFrames to concatenate and write.
        output_dir: Output directory.
        shard_idx: Shard index (used in filename).
        row_group_size: Parquet row group size.
        append: If True, append to existing file; if False, overwrite.
    """
    import polars as pl

    if not buffer:
        return

    combined = pl.concat(buffer)
    shard_path = output_dir / f"shard_{shard_idx:05d}.parquet"

    if append and shard_path.exists():
        # Read existing, concat, rewrite
        existing = pl.read_parquet(shard_path)
        combined = pl.concat([existing, combined])

    combined.write_parquet(shard_path, row_group_size=row_group_size)


def dry_run(input_files: list[Path], num_shards: int, world_size: int) -> None:
    """Print stats about the input files without writing anything.

    Args:
        input_files: List of input JSONL file paths.
        num_shards: Proposed number of output shards.
        world_size: Training world size for per-rank estimates.
    """
    logger.info("=" * 70)
    logger.info("DRY RUN — No files will be written")
    logger.info("=" * 70)
    logger.info(f"Input files: {len(input_files)}")

    total_size_mb = sum(f.stat().st_size for f in input_files) / (1024 * 1024)
    logger.info(f"Total input size: {total_size_mb:,.1f} MB")

    for f in input_files[:5]:
        logger.info(f"  {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
    if len(input_files) > 5:
        logger.info(f"  ... and {len(input_files) - 5} more files")

    logger.info(f"\nProposed output shards: {num_shards}")
    logger.info(f"Training world_size: {world_size}")
    files_per_rank = num_shards // world_size
    logger.info(f"Files per rank: {files_per_rank}")
    logger.info(f"Evenly divisible: {'YES ✓' if num_shards % world_size == 0 else 'NO ✗ (ADJUST num_shards!)'}")

    # Estimate sequences per file (rough, based on file sizes)
    avg_file_size_mb = total_size_mb / len(input_files)
    logger.info(f"\nAverage input file size: {avg_file_size_mb:.1f} MB")

    # Try to count one file for estimation
    try:
        import polars as pl

        sample_df = pl.read_ndjson(input_files[0])
        rows_in_first = len(sample_df)
        estimated_total = rows_in_first * len(input_files)
        est_per_shard = estimated_total // num_shards

        logger.info(f"\nEstimated row counts (based on first file having {rows_in_first:,} rows):")
        logger.info(f"  Estimated total sequences: ~{estimated_total:,}")
        logger.info(f"  Estimated sequences per shard: ~{est_per_shard:,}")
        logger.info(f"  Estimated sequences per rank: ~{est_per_shard * files_per_rank:,}")

        # Training timeline estimates
        windows_per_step = 8  # batch_size=1, grad_acc=8
        steps_to_exhaust_shard = est_per_shard // windows_per_step
        logger.info("\n  Training timeline (at 8 windows/optimizer step):")
        logger.info(f"    Steps to exhaust 1 shard: ~{steps_to_exhaust_shard:,}")
        logger.info(f"    Steps to exhaust all rank shards: ~{steps_to_exhaust_shard * files_per_rank:,}")

        del sample_df
    except Exception as e:
        logger.warning(f"Could not estimate row counts: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("To run for real, remove --dry-run")
    logger.info("=" * 70)


def write_metadata(output_dir: Path, stats: dict, args: argparse.Namespace) -> None:
    """Write a metadata JSON file alongside the shards.

    Args:
        output_dir: Output directory.
        stats: Dictionary of resharding statistics.
        args: Command-line arguments used.
    """
    metadata = {
        "script": "reshard_jsonl_to_parquet.py",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "input_dir": str(args.input_dir),
        "split": args.split,
        "seed": args.seed,
        "num_shards": args.num_shards,
        "text_column": args.text_column,
        "chunked_mode": args.chunk_size is not None,
        "stats": stats,
        "usage": {
            "streaming_config": {
                "path": str(output_dir),
                "split": "train",
                "streaming": True,
            },
            "notes": (
                f"With {args.num_shards} shards and world_size=48, "
                f"each rank gets {args.num_shards // 48} files. "
                "Set load_dataset_kwargs.path to the output directory."
            ),
        },
    }

    metadata_path = output_dir / "resharding_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata written to {metadata_path}")


def verify_output(output_dir: Path, num_shards: int, expected_total: int | None = None) -> None:
    """Quick verification of the output shards.

    Args:
        output_dir: Directory containing the Parquet shards.
        num_shards: Expected number of shards.
        expected_total: Expected total number of rows (if known).
    """
    import polars as pl

    logger.info("\nVerifying output...")

    shard_files = sorted(output_dir.glob("shard_*.parquet"))
    assert len(shard_files) == num_shards, f"Expected {num_shards} shards, found {len(shard_files)}"

    total = 0
    sizes = []
    for f in shard_files:
        n = pl.scan_parquet(f).select(pl.len()).collect().item()
        total += n
        sizes.append(n)

    logger.info(f"  Shards: {len(shard_files)}")
    logger.info(f"  Total rows: {total:,}")
    logger.info(f"  Min shard size: {min(sizes):,}")
    logger.info(f"  Max shard size: {max(sizes):,}")
    logger.info(f"  Avg shard size: {total // num_shards:,}")

    if expected_total is not None:
        assert total == expected_total, f"Row count mismatch: expected {expected_total:,}, got {total:,}"
        logger.info(f"  Row count matches expected: {expected_total:,} ✓")

    # Check schema of first and last shard
    first_schema = pl.read_parquet_schema(shard_files[0])
    last_schema = pl.read_parquet_schema(shard_files[-1])
    logger.info(f"  First shard schema: {first_schema}")
    logger.info(f"  Last shard schema: {last_schema}")
    assert first_schema == last_schema, "Schema mismatch between shards!"
    logger.info("  Schema consistent across shards ✓")

    # Spot-check: read first row of first and last shard
    first_row = pl.read_parquet(shard_files[0], n_rows=1)
    last_row = pl.read_parquet(shard_files[-1], n_rows=1)
    logger.info(f"  First shard, first row text length: {len(first_row['text'][0])}")
    logger.info(f"  Last shard, first row text length: {len(last_row['text'][0])}")

    logger.info("  Verification passed ✓")


def main():
    """Entry point for the resharding script."""
    parser = argparse.ArgumentParser(
        description="Convert OpenGenome2 JSONL(.gz) files to globally-shuffled Parquet shards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard resharding (needs ~100GB+ RAM for OG2 metagenome)
  python reshard_jsonl_to_parquet.py \\
      --input-dir /data/opengenome2/json/pretraining_or_both_phases/metagenomes \\
      --output-dir /data/opengenome2/parquet/metagenomes_480shards \\
      --num-shards 480

  # Low-memory mode (processes 10 input files at a time)
  python reshard_jsonl_to_parquet.py \\
      --input-dir /data/opengenome2/json/pretraining_or_both_phases/metagenomes \\
      --output-dir /data/opengenome2/parquet/metagenomes_480shards \\
      --num-shards 480 --chunk-size 10

  # Dry run (check file counts and estimates)
  python reshard_jsonl_to_parquet.py \\
      --input-dir /data/opengenome2/json/pretraining_or_both_phases/metagenomes \\
      --output-dir /data/opengenome2/parquet/metagenomes_480shards \\
      --num-shards 480 --dry-run
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing the input JSONL(.gz) files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write the output Parquet shards.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=480,
        help="Number of output Parquet files. Use 10x your world_size. Default: 480.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible global shuffling. Default: 42.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Data split to process ('train', 'valid', 'test'). Default: 'train'.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name containing the genomic sequence. Default: 'text'.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Process input files in chunks of this size (low-memory mode). "
            "If not set, all files are loaded at once (faster but needs more RAM)."
        ),
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=50_000,
        help="Parquet row group size. Default: 50000.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=48,
        help="Training world size (for dry-run estimates and divisibility check). Default: 48.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats and estimates without writing any files.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the verification step after writing.",
    )

    args = parser.parse_args()

    # Validate
    if args.num_shards % args.world_size != 0:
        logger.warning(
            f"⚠️  num_shards ({args.num_shards}) is not evenly divisible by "
            f"world_size ({args.world_size}). Some ranks will get more files than others. "
            f"Consider using {args.world_size * (args.num_shards // args.world_size + 1)} shards."
        )

    # Discover input files
    input_files = discover_input_files(args.input_dir, args.split)
    logger.info(f"Found {len(input_files)} input files for split '{args.split}'")

    output_dir = Path(args.output_dir)

    # Dry run
    if args.dry_run:
        dry_run(input_files, args.num_shards, args.world_size)
        sys.exit(0)

    # Check output directory
    if output_dir.exists() and any(output_dir.glob("shard_*.parquet")):
        logger.error(f"Output directory {output_dir} already contains shard files. Aborting.")
        logger.error("Delete existing shards or use a different output directory.")
        sys.exit(1)

    # Run resharding
    logger.info("=" * 70)
    logger.info("OpenGenome2 JSONL → Parquet Resharding")
    logger.info("=" * 70)
    logger.info(f"  Input:      {args.input_dir} ({len(input_files)} files)")
    logger.info(f"  Output:     {args.output_dir} ({args.num_shards} shards)")
    logger.info(f"  Seed:       {args.seed}")
    logger.info(f"  Mode:       {'chunked' if args.chunk_size else 'all-at-once'}")
    logger.info(f"  World size: {args.world_size} → {args.num_shards // args.world_size} files/rank")
    logger.info("=" * 70)

    t_total = time.time()

    if args.chunk_size:
        stats = reshard_chunked(
            input_files=input_files,
            output_dir=output_dir,
            num_shards=args.num_shards,
            seed=args.seed,
            text_column=args.text_column,
            chunk_size=args.chunk_size,
            row_group_size=args.row_group_size,
        )
    else:
        stats = reshard_all_at_once(
            input_files=input_files,
            output_dir=output_dir,
            num_shards=args.num_shards,
            seed=args.seed,
            text_column=args.text_column,
            row_group_size=args.row_group_size,
        )

    t_total = time.time() - t_total
    logger.info(f"\nTotal time: {t_total:.1f}s ({t_total / 60:.1f} min)")

    # Write metadata
    write_metadata(output_dir, stats, args)

    # Verify
    if not args.skip_verify:
        verify_output(output_dir, args.num_shards, stats.get("total_rows"))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("DONE! Summary:")
    logger.info(f"  Total sequences: {stats['total_rows']:,}")
    logger.info(f"  Output shards:   {stats['num_shards']}")
    logger.info(f"  Shard sizes:     {stats['min_shard_size']:,} - {stats['max_shard_size']:,} rows")
    logger.info(f"  Files per rank:  {args.num_shards // args.world_size}")
    logger.info("\nTo use in training, update your config:")
    logger.info("  dataset:")
    logger.info("    load_dataset_kwargs:")
    logger.info(f'      path: "{args.output_dir}"')
    logger.info('      split: "train"')
    logger.info("      streaming: true")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
