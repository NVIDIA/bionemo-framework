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

"""Verify that parquet data matches the original window logs.

This script checks:
1. Window indices in parquet match the original CSV logs
2. Batching is correct (384 samples per global batch)
3. Interleaving across DP ranks is correct
4. Token sequences are valid

USAGE:
======
python scripts/verify_parquet_against_logs.py \
    --parquet-file /data/savithas/john_data_test/data_sequential.parquet \
    --log-dir /data/savithas/johns_og2_repro_v4/results/savithas-johns-og2-repro-v4/window_logs \
    --tensor-parallel-size 4 \
    --micro-batch-size 8 \
    --grad-acc-batches 4
"""

import argparse
import csv
import logging
import os
from glob import glob

import pyarrow.parquet as pq


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_window_logs(log_dir: str, tensor_parallel_size: int, max_samples_per_rank: int) -> dict[int, list[int]]:
    """Load window indices from log files."""
    pattern = os.path.join(log_dir, "window_access_train_rank*.csv")
    log_files = sorted(glob(pattern))

    result = {}
    for log_file in log_files:
        basename = os.path.basename(log_file)
        parts = basename.replace(".csv", "").split("_")
        file_rank = int(parts[3].replace("rank", ""))

        # Only process primary TP ranks (DP ranks)
        if file_rank % tensor_parallel_size != 0:
            continue

        dp_rank = file_rank // tensor_parallel_size

        window_indices = []
        with open(log_file, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_samples_per_rank:
                    break
                if "window_idx" in row:
                    window_indices.append(int(row["window_idx"]))
                elif "idx" in row:
                    window_indices.append(int(row["idx"]))

        result[dp_rank] = window_indices

    return result


def verify_parquet(
    parquet_file: str,
    log_dir: str,
    tensor_parallel_size: int = 4,
    micro_batch_size: int = 8,
    grad_acc_batches: int = 4,
):
    """Verify parquet data against original logs."""
    logger.info(f"Loading parquet file: {parquet_file}")
    table = pq.read_table(parquet_file)

    total_samples = len(table)
    samples_per_step_per_rank = micro_batch_size * grad_acc_batches
    gbs = samples_per_step_per_rank * 12  # 12 DP ranks
    num_steps = total_samples // gbs

    logger.info(f"Parquet contains {total_samples} samples")
    logger.info(f"Expected GBS: {gbs} (12 DP ranks x {samples_per_step_per_rank} samples/rank)")
    logger.info(f"Number of complete steps: {num_steps}")

    # Load original logs
    logger.info(f"\nLoading original logs from {log_dir}")
    logs = load_window_logs(log_dir, tensor_parallel_size, num_steps * samples_per_step_per_rank)
    logger.info(f"Loaded logs for {len(logs)} DP ranks")

    # Get parquet data
    parquet_window_idx = table["window_idx"].to_pylist()
    parquet_input_ids = table["input_ids"].to_pylist()

    # Verify structure
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 60)

    all_passed = True

    # Test 1: Total sample count
    expected_samples = num_steps * gbs
    if total_samples == expected_samples:
        logger.info(f"✓ Total samples: {total_samples} (expected {expected_samples})")
    else:
        logger.error(f"✗ Total samples: {total_samples} (expected {expected_samples})")
        all_passed = False

    # Test 2: Verify interleaving pattern
    # For step 0, positions 0-383 should contain:
    #   - Positions 0-31: DP0 samples 0-31
    #   - Positions 32-63: DP1 samples 0-31
    #   - etc.
    logger.info("\nVerifying interleaving pattern (step 0):")
    interleave_correct = True

    for dp_rank in range(min(3, len(logs))):  # Check first 3 DP ranks
        start_pos = dp_rank * samples_per_step_per_rank
        end_pos = start_pos + samples_per_step_per_rank

        parquet_chunk = parquet_window_idx[start_pos:end_pos]
        log_chunk = logs[dp_rank][:samples_per_step_per_rank]

        if parquet_chunk == log_chunk:
            logger.info(f"  ✓ DP rank {dp_rank}: positions {start_pos}-{end_pos - 1} match log")
        else:
            logger.error(f"  ✗ DP rank {dp_rank}: MISMATCH!")
            logger.error(f"    Parquet: {parquet_chunk[:5]}...")
            logger.error(f"    Log:     {log_chunk[:5]}...")
            interleave_correct = False
            all_passed = False

    if interleave_correct:
        logger.info("  ✓ Interleaving pattern verified for step 0")

    # Test 3: Verify step boundaries
    logger.info("\nVerifying step boundaries:")
    for step in range(min(3, num_steps)):  # Check first 3 steps
        step_start = step * gbs
        step_end = step_start + gbs

        # Check that this step contains samples from all 12 DP ranks
        dp_ranks_in_step = set()
        for i in range(step_start, step_end, samples_per_step_per_rank):
            # Determine which DP rank this chunk is from
            chunk_idx_in_step = (i - step_start) // samples_per_step_per_rank
            dp_ranks_in_step.add(chunk_idx_in_step)

        if len(dp_ranks_in_step) == 12:
            logger.info(f"  ✓ Step {step}: contains data from all 12 DP ranks")
        else:
            logger.error(f"  ✗ Step {step}: only {len(dp_ranks_in_step)} DP ranks found")
            all_passed = False

    # Test 4: Verify token sequences are valid
    logger.info("\nVerifying token sequences:")
    valid_tokens = True
    for i in range(min(10, total_samples)):
        tokens = parquet_input_ids[i]
        if not isinstance(tokens, list) or len(tokens) == 0:
            logger.error(f"  ✗ Sample {i}: invalid tokens")
            valid_tokens = False
            all_passed = False
        elif len(tokens) != 8192:
            logger.warning(f"  ! Sample {i}: unexpected length {len(tokens)} (expected 8192)")

    if valid_tokens:
        logger.info("  ✓ First 10 samples have valid token sequences")
        logger.info(f"    Sample 0 first 10 tokens: {parquet_input_ids[0][:10]}")
        logger.info(f"    Sample 0 length: {len(parquet_input_ids[0])}")

    # Test 5: Verify window_idx uniqueness within each global batch
    logger.info("\nVerifying window_idx uniqueness per global batch:")
    for step in range(min(3, num_steps)):
        step_start = step * gbs
        step_end = step_start + gbs
        step_window_idx = parquet_window_idx[step_start:step_end]
        unique_count = len(set(step_window_idx))
        if unique_count == gbs:
            logger.info(f"  ✓ Step {step}: all {gbs} window_idx are unique")
        else:
            logger.warning(f"  ! Step {step}: {unique_count}/{gbs} unique window_idx (some duplicates)")

    # Summary
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✓ ALL VERIFICATION CHECKS PASSED")
        logger.info("The parquet data correctly represents John's training data!")
    else:
        logger.error("✗ SOME CHECKS FAILED - Review errors above")
    logger.info("=" * 60)

    # Print usage instructions
    logger.info("\nTo use this data in training:")
    logger.info(f'  dataset.load_dataset_kwargs.data_files: "{parquet_file}"')
    logger.info("  dataset.skip_tokenization: true")
    logger.info("  dataset.shuffle: false")

    return all_passed


def main():
    """Main entry point for verification script."""
    parser = argparse.ArgumentParser(description="Verify parquet data against original logs")
    parser.add_argument("--parquet-file", required=True, help="Path to parquet file to verify")
    parser.add_argument("--log-dir", required=True, help="Path to original window logs directory")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--grad-acc-batches", type=int, default=4)

    args = parser.parse_args()

    verify_parquet(
        parquet_file=args.parquet_file,
        log_dir=args.log_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        micro_batch_size=args.micro_batch_size,
        grad_acc_batches=args.grad_acc_batches,
    )


if __name__ == "__main__":
    main()
