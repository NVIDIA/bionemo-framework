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

"""Parse John's window logs to extract training data for specific step ranges.

This script reads the window_access_train_rank*.csv files from a production run
and extracts the window_idx values for a given step range (e.g., steps 0-5000).

UNDERSTANDING THE DATA ORDER:
=============================

With MBS=8, Grad Acc=4, TP=4, 48 GPUs:
- DP ranks = 48 / 4 = 12
- Samples per DP rank per optimizer step = 8 * 4 = 32
- Total samples per optimizer step = 32 * 12 = 384

Each DP rank's log is in SEQUENTIAL training order:
- Rows 1-8:   micro-batch 1 of optimizer step 1
- Rows 9-16:  micro-batch 2 of optimizer step 1
- Rows 17-24: micro-batch 3 of optimizer step 1
- Rows 25-32: micro-batch 4 of optimizer step 1 (completes step 1)
- Rows 33-40: micro-batch 1 of optimizer step 2
- ...etc.

To get FULL 384-sample batches, use --interleave-ranks to combine all DP ranks.

USAGE:
======

1. Analyze logs:
   python parse_window_logs.py --log-dir /path/to/window_logs --analyze

2. Extract per-DP-rank (for running on fewer GPUs with same data order):
   python parse_window_logs.py \
       --log-dir /path/to/window_logs \
       --output-dir /data/john_data_5k \
       --sequence-db-dir /data/bcr_eden/OG2_database_splits \
       --window-db-path /data/bcr_eden/OG2_database_splits/og2__train__short.sqlite \
       --tokenizer tokenizers/nucleotide_fast_tokenizer \
       --max-steps 5000 --micro-batch-size 8 --grad-acc-batches 4 --tensor-parallel-size 4

3. Extract FULL batches interleaved (reconstructs exact 384-sample batches):
   python parse_window_logs.py ... --interleave-ranks
"""

import argparse
import csv
import json
import logging
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# Add paths for imports
EVO2_SUBPKG_PATH = Path(__file__).parent.parent.parent.parent.parent / "sub-packages" / "bionemo-evo2" / "src"
sys.path.insert(0, str(EVO2_SUBPKG_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MegatronTokenizerAdapter:
    """Adapts HuggingFace tokenizer to Megatron's tokenizer interface."""

    def __init__(self, hf_tokenizer):
        """Initialize the adapter with a HuggingFace tokenizer."""
        self.hf_tokenizer = hf_tokenizer
        self.bos_id = hf_tokenizer.bos_token_id or 0
        self.eos_id = hf_tokenizer.eos_token_id or 1
        self.pad_id = hf_tokenizer.pad_token_id or hf_tokenizer.eos_token_id or 0
        self._sep_id = hf_tokenizer.sep_token_id or self.eos_id
        self.eod = self.eos_id
        self.vocab_size = hf_tokenizer.vocab_size

    def text_to_ids(self, text: str) -> list[int]:
        """Convert text to token IDs without special tokens."""
        return self.hf_tokenizer.encode(text, add_special_tokens=False)


def analyze_logs(log_dir: str, tensor_parallel_size: int = 1):
    """Analyze the structure of window log files.

    Args:
        log_dir: Directory containing window_access_train_rank*.csv files
        tensor_parallel_size: TP size to identify DP ranks
    """
    pattern = os.path.join(log_dir, "window_access_train_rank*.csv")
    log_files = sorted(glob(pattern))

    if not log_files:
        logger.error(f"No log files found in {log_dir}")
        return

    logger.info(f"Found {len(log_files)} log files")
    logger.info(f"With TP={tensor_parallel_size}, DP ranks = {len(log_files) // tensor_parallel_size}")

    # Analyze first DP rank's file (global rank 0)
    first_file = log_files[0]
    logger.info(f"\nAnalyzing {os.path.basename(first_file)}...")

    with open(first_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.info(f"  Columns: {list(rows[0].keys()) if rows else 'N/A'}")
    logger.info(f"  Total rows: {len(rows)}")

    # Check if logs have tokens
    has_tokens = rows and "first_10_tokens" in rows[0]
    logger.info(f"  Has first_10_tokens column: {has_tokens}")

    if rows:
        logger.info(f"\n  First row: {rows[0]}")
        logger.info(f"  Last row: {rows[-1]}")

        # Show first 10 rows with batch structure
        logger.info("\n  First 16 rows (showing micro-batch structure with MBS=8):")
        for i, row in enumerate(rows[:16]):
            batch_marker = " <-- micro-batch boundary" if (i + 1) % 8 == 0 else ""
            step_marker = " <== OPTIMIZER STEP 1" if i == 31 else ""
            window_idx = row.get("window_idx", row.get("idx", "?"))
            tokens = row.get("first_10_tokens", "N/A")[:30] + "..." if has_tokens else "N/A"
            logger.info(f"    Row {i:3d}: window_idx={window_idx:>12}, tokens={tokens}{batch_marker}{step_marker}")

    # Count rows per DP rank file
    logger.info("\n  Rows per DP rank (every TP-th global rank):")
    total_rows = 0
    dp_rank = 0
    for i, log_file in enumerate(log_files):
        if i % tensor_parallel_size != 0:
            continue  # Skip non-primary TP ranks
        with open(log_file, "r") as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)
            total_rows += count
            if dp_rank < 5:
                logger.info(
                    f"    DP rank {dp_rank} (global {i}): {count} rows = {count // 32} steps (assuming MBS=8, GA=4)"
                )
            dp_rank += 1

    if dp_rank > 5:
        logger.info(f"    ... and {dp_rank - 5} more DP ranks")

    logger.info("\n  SUMMARY:")
    logger.info(f"    Total DP ranks: {dp_rank}")
    logger.info(f"    Rows per DP rank: ~{total_rows // dp_rank}")
    logger.info(f"    Estimated optimizer steps: {(total_rows // dp_rank) // 32} (with MBS=8, GA=4)")


def verify_parsing(
    log_dir: str,
    sequence_db_dir: str,
    window_db_path: str,
    tokenizer_path: str,
    tensor_parallel_size: int = 1,
    num_samples_to_check: int = 10,
    seq_length: int = 8192,
    stride: int = 7992,
):
    """Verify that parsing extracts correct data by comparing with logs.

    This checks:
    1. Window indices are read correctly
    2. Tokens fetched from DB match tokens in log (if logged)

    Args:
        log_dir: Directory containing window_access_train_rank*.csv files
        sequence_db_dir: Directory with per-sample SQLite databases
        window_db_path: Path to window mappings database
        tokenizer_path: Path to HuggingFace tokenizer
        tensor_parallel_size: TP size
        num_samples_to_check: How many samples to verify
        seq_length: Sequence length
        stride: Stride
    """
    from transformers import AutoTokenizer

    pattern = os.path.join(log_dir, "window_access_train_rank0_*.csv")
    log_files = glob(pattern)
    if not log_files:
        logger.error("No rank 0 log file found")
        return

    log_file = log_files[0]
    logger.info(f"Verifying against {os.path.basename(log_file)}")

    # Read log file
    with open(log_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)[:num_samples_to_check]

    has_tokens = "first_10_tokens" in rows[0] if rows else False
    logger.info(f"Checking {len(rows)} samples, has_tokens={has_tokens}")

    # Load tokenizer and dataset
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    tokenizer_adapter = MegatronTokenizerAdapter(hf_tokenizer)

    try:
        from bionemo.evo2.data.sharded_eden_dataloader import ShardedEdenDataset

        logger.info("Loaded ShardedEdenDataset")
    except ImportError as e:
        logger.error(f"Could not import ShardedEdenDataset: {e}")
        return

    dataset = ShardedEdenDataset(
        tokenizer=tokenizer_adapter,
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_db_path,
        seq_length=seq_length,
        stride=stride,
        create_attention_mask=False,
        rc_aug=False,
        skip_stats=True,
    )

    # Verify each sample
    logger.info("\nVerification results:")
    all_match = True
    for i, row in enumerate(rows):
        window_idx = int(row.get("window_idx", row.get("idx")))

        # Fetch from dataset
        sample = dataset[np.int64(window_idx)]
        fetched_tokens = sample["tokens"][:10].tolist()

        # Compare with logged tokens if available
        if has_tokens:
            # Parse logged tokens (format: "[1, 2, 3, ...]")
            logged_tokens_str = row["first_10_tokens"]
            logged_tokens = eval(logged_tokens_str)  # Safe for known format [int, int, ...]

            match = fetched_tokens == logged_tokens
            status = "✓ MATCH" if match else "✗ MISMATCH"
            if not match:
                all_match = False
            logger.info(f"  Row {i}: window_idx={window_idx}")
            logger.info(f"    Logged:  {logged_tokens}")
            logger.info(f"    Fetched: {fetched_tokens}")
            logger.info(f"    {status}")
        else:
            logger.info(f"  Row {i}: window_idx={window_idx}, fetched tokens: {fetched_tokens}")

    if has_tokens:
        if all_match:
            logger.info("\n✓ ALL SAMPLES VERIFIED - Parsing is correct!")
        else:
            logger.error("\n✗ SOME SAMPLES MISMATCHED - Check tokenizer or DB paths")


def extract_window_indices(
    log_dir: str,
    max_steps: int,
    micro_batch_size: int = 8,
    grad_acc_batches: int = 4,
    tensor_parallel_size: int = 1,
    rank: int | None = None,
) -> dict[int, list[int]]:
    """Extract window indices for the first N steps from log files.

    Args:
        log_dir: Directory containing window_access_train_rank*.csv files
        max_steps: Maximum number of training steps to extract
        micro_batch_size: Micro batch size used in training
        grad_acc_batches: Gradient accumulation batches
        tensor_parallel_size: TP size (ranks in same TP group see same data)
        rank: If specified, only extract for this DP rank. Otherwise, extract all DP ranks.

    Returns:
        Dict mapping DP rank -> list of window_idx values
    """
    pattern = os.path.join(log_dir, "window_access_train_rank*.csv")
    log_files = sorted(glob(pattern))

    if not log_files:
        raise FileNotFoundError(f"No log files found matching pattern in {log_dir}")

    # Samples per step per DP rank
    samples_per_step_per_rank = micro_batch_size * grad_acc_batches
    max_samples_per_rank = max_steps * samples_per_step_per_rank

    # With TP, only every TP-th global rank has unique data
    # DP rank 0 = global rank 0, DP rank 1 = global rank TP, etc.
    total_global_ranks = len(log_files)
    num_dp_ranks = total_global_ranks // tensor_parallel_size

    logger.info(f"Extracting window indices for first {max_steps} steps")
    logger.info(f"  tensor_parallel_size = {tensor_parallel_size}")
    logger.info(f"  total_global_ranks = {total_global_ranks}, num_dp_ranks = {num_dp_ranks}")
    logger.info(f"  samples_per_step_per_rank = {micro_batch_size} * {grad_acc_batches} = {samples_per_step_per_rank}")
    logger.info(f"  max_samples_per_rank = {max_samples_per_rank}")

    # Build list of global ranks to process (first rank in each TP group)
    dp_to_global = {dp: dp * tensor_parallel_size for dp in range(num_dp_ranks)}
    logger.info(f"  DP rank -> Global rank mapping: {dp_to_global}")

    result = {}

    for log_file in log_files:
        # Extract rank from filename
        basename = os.path.basename(log_file)
        # Pattern: window_access_train_rank0_dd320078.csv
        parts = basename.replace(".csv", "").split("_")
        file_rank = int(parts[3].replace("rank", ""))

        # Check if this global rank is a DP rank (first in TP group)
        if file_rank % tensor_parallel_size != 0:
            continue  # Skip non-primary TP ranks

        dp_rank = file_rank // tensor_parallel_size

        if rank is not None and dp_rank != rank:
            continue

        logger.info(f"Processing DP rank {dp_rank} (global rank {file_rank}): {basename}")

        window_indices = []
        with open(log_file, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_samples_per_rank:
                    break

                # Get window_idx - column might be 'window_idx' or 'idx'
                if "window_idx" in row:
                    window_indices.append(int(row["window_idx"]))
                elif "idx" in row:
                    window_indices.append(int(row["idx"]))
                else:
                    raise ValueError(f"No window_idx or idx column in {log_file}. Columns: {row.keys()}")

        result[dp_rank] = window_indices
        logger.info(f"  Extracted {len(window_indices)} window indices for DP rank {dp_rank}")

    return result


def interleave_ranks(
    window_indices_by_rank: dict[int, list[int]],
    micro_batch_size: int,
    grad_acc_batches: int,
) -> list[int]:
    """Interleave samples from all DP ranks to reconstruct full global batches.

    This reconstructs the exact order samples appeared in training:
    - Optimizer step 1: [DP0 samples 0-31] + [DP1 samples 0-31] + ... + [DP11 samples 0-31]
    - Optimizer step 2: [DP0 samples 32-63] + [DP1 samples 32-63] + ...
    - etc.

    Args:
        window_indices_by_rank: Dict mapping DP rank -> list of window_idx values
        micro_batch_size: Micro batch size (samples per micro-batch)
        grad_acc_batches: Gradient accumulation batches

    Returns:
        List of window_idx values in global training order
    """
    samples_per_step_per_rank = micro_batch_size * grad_acc_batches
    num_dp_ranks = len(window_indices_by_rank)
    samples_per_global_step = samples_per_step_per_rank * num_dp_ranks

    # Find minimum number of complete steps across all ranks
    min_samples = min(len(indices) for indices in window_indices_by_rank.values())
    num_complete_steps = min_samples // samples_per_step_per_rank

    logger.info(f"Interleaving {num_dp_ranks} DP ranks")
    logger.info(f"  samples_per_step_per_rank = {samples_per_step_per_rank}")
    logger.info(f"  samples_per_global_step = {samples_per_global_step}")
    logger.info(f"  num_complete_steps = {num_complete_steps}")

    interleaved = []
    for step in range(num_complete_steps):
        start_idx = step * samples_per_step_per_rank
        end_idx = start_idx + samples_per_step_per_rank

        # Add samples from each DP rank for this step
        for dp_rank in sorted(window_indices_by_rank.keys()):
            rank_indices = window_indices_by_rank[dp_rank][start_idx:end_idx]
            interleaved.extend(rank_indices)

    logger.info(f"  Total interleaved samples: {len(interleaved)}")
    return interleaved


def convert_to_parquet(
    window_indices_by_rank: dict[int, list[int]],
    sequence_db_dir: str,
    window_db_path: str,
    output_dir: str,
    tokenizer_path: str,
    seq_length: int = 8192,
    stride: int = 7992,
):
    """Convert window indices to parquet files with actual tokens.

    Args:
        window_indices_by_rank: Dict mapping rank -> list of window_idx values
        sequence_db_dir: Directory containing per-sample SQLite databases
        window_db_path: Path to the window mappings database
        output_dir: Directory to save parquet files
        tokenizer_path: Path to HuggingFace tokenizer
        seq_length: Sequence length
        stride: Stride for windowing
    """
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    tokenizer_adapter = MegatronTokenizerAdapter(hf_tokenizer)

    # Import ShardedEdenDataset
    try:
        from bionemo.evo2.data.sharded_eden_dataloader import ShardedEdenDataset

        logger.info("Loaded ShardedEdenDataset from bionemo-evo2")
    except ImportError as e:
        logger.error(f"Could not import ShardedEdenDataset: {e}")
        sys.exit(1)

    # Create dataset
    logger.info("Creating ShardedEdenDataset...")
    dataset = ShardedEdenDataset(
        tokenizer=tokenizer_adapter,
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_db_path,
        seq_length=seq_length,
        stride=stride,
        create_attention_mask=False,
        rc_aug=False,
        skip_stats=True,
    )
    logger.info(f"Dataset has {len(dataset)} windows")

    # Process each rank
    total_samples = 0
    for file_rank, window_indices in sorted(window_indices_by_rank.items()):
        logger.info(f"Processing rank {file_rank} ({len(window_indices)} samples)...")

        all_input_ids = []
        all_window_idx = []

        for window_idx in tqdm(window_indices, desc=f"Rank {file_rank}"):
            sample = dataset[np.int64(window_idx)]
            input_ids = sample["tokens"].tolist()
            all_input_ids.append(input_ids)
            all_window_idx.append(window_idx)

        # Save to parquet
        output_file = os.path.join(output_dir, f"data_rank{file_rank}.parquet")
        table = pa.table({"input_ids": all_input_ids, "window_idx": all_window_idx})
        pq.write_table(table, output_file)
        logger.info(f"Wrote {len(all_input_ids)} samples to {output_file}")
        total_samples += len(all_input_ids)

    # Save metadata
    metadata = {
        "sequence_db_dir": sequence_db_dir,
        "window_db_path": window_db_path,
        "seq_length": seq_length,
        "stride": stride,
        "ranks_processed": list(window_indices_by_rank.keys()),
        "samples_per_rank": {k: len(v) for k, v in window_indices_by_rank.items()},
        "total_samples": total_samples,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nDone! Total: {total_samples} samples across {len(window_indices_by_rank)} ranks")
    logger.info(f"Output: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parse John's window logs and convert to parquet",
    )
    parser.add_argument("--log-dir", required=True, help="Directory with window_access_train_rank*.csv files")
    parser.add_argument("--analyze", action="store_true", help="Just analyze logs, don't convert")
    parser.add_argument(
        "--verify", action="store_true", help="Verify parsing by comparing fetched tokens with logged tokens"
    )

    # Conversion options
    parser.add_argument("--output-dir", help="Directory to save parquet files")
    parser.add_argument("--sequence-db-dir", help="Directory with per-sample SQLite databases")
    parser.add_argument("--window-db-path", help="Path to window mappings database")
    parser.add_argument("--tokenizer", default="tokenizers/nucleotide_fast_tokenizer", help="Tokenizer path")
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=7992)

    # Step range options
    parser.add_argument("--max-steps", type=int, default=5000, help="Max training steps to extract")
    parser.add_argument("--micro-batch-size", type=int, default=8, help="Micro batch size from training")
    parser.add_argument("--grad-acc-batches", type=int, default=4, help="Gradient accumulation batches")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="TP size (ranks in TP group see same data)"
    )
    parser.add_argument("--rank", type=int, default=None, help="Only process this DP rank (default: all)")
    parser.add_argument(
        "--interleave-ranks",
        action="store_true",
        help="Interleave samples from all DP ranks to reconstruct full global batches",
    )

    args = parser.parse_args()

    if args.analyze:
        analyze_logs(args.log_dir, tensor_parallel_size=args.tensor_parallel_size)
        return

    if args.verify:
        if not all([args.sequence_db_dir, args.window_db_path]):
            parser.error("--sequence-db-dir and --window-db-path are required for --verify")
        verify_parsing(
            log_dir=args.log_dir,
            sequence_db_dir=args.sequence_db_dir,
            window_db_path=args.window_db_path,
            tokenizer_path=args.tokenizer,
            tensor_parallel_size=args.tensor_parallel_size,
            seq_length=args.seq_length,
            stride=args.stride,
        )
        return

    # Validate required args for conversion
    if not all([args.output_dir, args.sequence_db_dir, args.window_db_path]):
        parser.error("--output-dir, --sequence-db-dir, and --window-db-path are required for conversion")

    # Extract window indices
    window_indices_by_rank = extract_window_indices(
        log_dir=args.log_dir,
        max_steps=args.max_steps,
        micro_batch_size=args.micro_batch_size,
        grad_acc_batches=args.grad_acc_batches,
        tensor_parallel_size=args.tensor_parallel_size,
        rank=args.rank,
    )

    # Optionally interleave ranks to get full global batches
    if args.interleave_ranks:
        interleaved_indices = interleave_ranks(
            window_indices_by_rank=window_indices_by_rank,
            micro_batch_size=args.micro_batch_size,
            grad_acc_batches=args.grad_acc_batches,
        )
        # Convert to single-rank format for convert_to_parquet
        window_indices_by_rank = {0: interleaved_indices}
        logger.info("Using interleaved indices (full global batches)")

    # Convert to parquet
    convert_to_parquet(
        window_indices_by_rank=window_indices_by_rank,
        sequence_db_dir=args.sequence_db_dir,
        window_db_path=args.window_db_path,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        seq_length=args.seq_length,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
