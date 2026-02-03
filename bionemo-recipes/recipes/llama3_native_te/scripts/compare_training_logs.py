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

"""Compare train_fsdp2 logs with expected parquet data.

This verifies that the HuggingFace DataLoader is serving data in the expected order.

USAGE:
======
python scripts/compare_training_logs.py \
    --training-log-dir /data/savithas/train_fsdp2_logs \
    --parquet-file /data/savithas/john_data_test/data_sequential.parquet \
    --num-gpus 8 \
    --grad-acc-steps 8
"""

import argparse
import ast
import csv
import logging
from glob import glob
from pathlib import Path

import pyarrow.parquet as pq


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Compare training logs with parquet data."""
    parser = argparse.ArgumentParser(description="Compare training logs with parquet data")
    parser.add_argument("--training-log-dir", required=True, help="Directory with training_sequences_rank*.csv")
    parser.add_argument("--parquet-file", required=True, help="Parquet file used for training")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs used")
    parser.add_argument("--grad-acc-steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro batch size")

    args = parser.parse_args()

    # Load parquet data
    logger.info(f"Loading parquet: {args.parquet_file}")
    table = pq.read_table(args.parquet_file)
    parquet_input_ids = table["input_ids"].to_pylist()
    logger.info(f"Parquet has {len(parquet_input_ids)} samples")

    # Load training logs
    log_pattern = str(Path(args.training_log_dir) / "training_sequences_rank*.csv")
    log_files = sorted(glob(log_pattern))
    logger.info(f"Found {len(log_files)} training log files")

    if not log_files:
        logger.error(f"No log files found matching {log_pattern}")
        return

    # Parse logs
    logs_by_rank = {}
    for log_file in log_files:
        rank = int(Path(log_file).stem.split("rank")[1])
        with open(log_file, "r") as f:
            reader = csv.DictReader(f)
            logs_by_rank[rank] = list(reader)
        logger.info(f"  Rank {rank}: {len(logs_by_rank[rank])} entries")

    # Compare
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)

    # With DistributedSampler(shuffle=False), samples are distributed:
    #   Rank 0 gets: samples 0, 8, 16, 24, ... (every num_gpus-th, starting at 0)
    #   Rank 1 gets: samples 1, 9, 17, 25, ... (every num_gpus-th, starting at 1)
    #   etc.
    #
    # Each rank's micro_step 1 = its 1st sample, micro_step 2 = its 2nd sample, etc.
    # So rank i, micro_step m gets parquet sample: i + (m-1) * num_gpus

    gbs = args.num_gpus * args.micro_batch_size * args.grad_acc_steps
    logger.info(f"Configuration: {args.num_gpus} GPUs x MBS={args.micro_batch_size} x GA={args.grad_acc_steps}")
    logger.info(f"Global Batch Size (GBS): {gbs}")
    logger.info(f"Samples per rank per optimizer step: {args.micro_batch_size * args.grad_acc_steps}")

    all_matched = True
    total_comparisons = 0

    for rank in sorted(logs_by_rank.keys()):
        rank_logs = logs_by_rank[rank]
        logger.info(f"\nRank {rank} ({len(rank_logs)} log entries):")

        # Check all entries (or first 10 if many)
        entries_to_check = min(10, len(rank_logs))
        for i, entry in enumerate(rank_logs[:entries_to_check]):
            micro_step = int(entry["micro_step"])  # Per-rank micro step (resets each optimizer step)
            optimizer_step = int(entry["optimizer_step"])
            logged_tokens = ast.literal_eval(entry["first_10_tokens"])

            # Calculate parquet index
            # micro_step is 1-indexed within each optimizer step
            # Total samples seen by this rank = optimizer_step * GA + micro_step
            samples_seen_by_rank = optimizer_step * args.grad_acc_steps + micro_step
            parquet_idx = rank + (samples_seen_by_rank - 1) * args.num_gpus

            if parquet_idx < len(parquet_input_ids):
                expected_tokens = parquet_input_ids[parquet_idx][:10]
                match = logged_tokens == expected_tokens
                status = "✓" if match else "✗"

                if not match:
                    all_matched = False

                logger.info(f"  [{status}] opt_step={optimizer_step}, micro={micro_step} -> parquet[{parquet_idx}]")
                if not match:
                    logger.info(f"      Logged:   {logged_tokens}")
                    logger.info(f"      Expected: {expected_tokens}")

                total_comparisons += 1
            else:
                logger.warning(f"  Entry {i}: parquet_idx={parquet_idx} out of range")

    # Summary
    logger.info("\n" + "=" * 60)
    if all_matched:
        logger.info(f"✓ ALL {total_comparisons} COMPARISONS MATCHED!")
        logger.info("The DataLoader is serving data in the expected order.")
    else:
        logger.error("✗ SOME COMPARISONS FAILED")
        logger.error("Check the mismatches above.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
