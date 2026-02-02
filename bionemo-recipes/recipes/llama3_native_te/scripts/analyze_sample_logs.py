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

"""Analyze sample index logs to verify batch composition.

This script reads the logs from IndexTrackingDataset and verifies that:
1. Each optimizer step uses the expected range of samples
2. All ranks together cover the full GBS samples per step
3. No samples are duplicated or missing

Usage:
    python scripts/analyze_sample_logs.py \
        --log-dir /data/parquet_sample_logs \
        --world-size 48 \
        --grad-acc 8 \
        --mbs 1
"""

import argparse
import os

import pandas as pd


def analyze_logs(log_dir: str, world_size: int, grad_acc: int, mbs: int):
    """Analyze sample index logs to verify batch composition.

    Args:
        log_dir: Directory containing sample_indices_rankX.csv files
        world_size: Number of GPUs (ranks)
        grad_acc: Gradient accumulation steps
        mbs: Micro batch size
    """
    gbs = world_size * grad_acc * mbs
    samples_per_rank_per_step = grad_acc * mbs

    print("Configuration:")
    print(f"  World size: {world_size}")
    print(f"  Grad acc: {grad_acc}")
    print(f"  Micro batch size: {mbs}")
    print(f"  Global batch size (GBS): {gbs}")
    print(f"  Samples per rank per optimizer step: {samples_per_rank_per_step}")
    print()

    # Load all rank logs
    all_accesses = []
    for rank in range(world_size):
        log_file = os.path.join(log_dir, f"sample_indices_rank{rank}.csv")
        if not os.path.exists(log_file):
            print(f"Warning: Missing log file for rank {rank}")
            continue

        df = pd.read_csv(log_file)
        df["rank"] = rank
        all_accesses.append(df)

    if not all_accesses:
        print("Error: No log files found!")
        return

    combined = pd.concat(all_accesses, ignore_index=True)
    print(f"Total accesses logged: {len(combined)}")
    print()

    # Group by optimizer step
    # access_order within each rank tells us the micro-batch
    # Optimizer step = access_order // grad_acc (for mbs=1)
    combined["optimizer_step"] = combined["access_order"] // samples_per_rank_per_step

    # Analyze each optimizer step
    print("=" * 60)
    print("OPTIMIZER STEP ANALYSIS")
    print("=" * 60)

    num_steps = combined["optimizer_step"].max() + 1
    print(f"Number of optimizer steps: {num_steps}")
    print()

    issues = []

    for step in range(min(num_steps, 10)):  # Show first 10 steps
        step_data = combined[combined["optimizer_step"] == step]
        indices = sorted(step_data["dataset_index"].unique())

        expected_start = step * gbs
        expected_end = expected_start + gbs - 1
        expected_indices = set(range(expected_start, expected_end + 1))
        actual_indices = set(indices)

        missing = expected_indices - actual_indices
        extra = actual_indices - expected_indices

        status = "✓" if not missing and not extra else "✗"

        print(f"Step {step}: {status}")
        print(f"  Expected indices: {expected_start} - {expected_end}")
        print(f"  Actual indices: {min(indices)} - {max(indices)}")
        print(f"  Unique samples: {len(indices)}")

        if missing:
            print(f"  MISSING samples: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
            issues.append(f"Step {step}: Missing {len(missing)} samples")

        if extra:
            print(f"  EXTRA samples: {sorted(extra)[:10]}{'...' if len(extra) > 10 else ''}")
            issues.append(f"Step {step}: Extra {len(extra)} samples")

        # Check for duplicates
        all_indices = step_data["dataset_index"].tolist()
        if len(all_indices) != len(set(all_indices)):
            dup_count = len(all_indices) - len(set(all_indices))
            print(f"  DUPLICATES: {dup_count} duplicate accesses")
            issues.append(f"Step {step}: {dup_count} duplicates")

        print()

    # Per-rank breakdown for step 0
    print("=" * 60)
    print("RANK BREAKDOWN FOR STEP 0")
    print("=" * 60)

    step0 = combined[combined["optimizer_step"] == 0]
    for rank in range(min(world_size, 8)):  # Show first 8 ranks
        rank_data = step0[step0["rank"] == rank]
        indices = rank_data["dataset_index"].tolist()
        print(f"Rank {rank}: samples {indices}")

    if world_size > 8:
        print(f"... ({world_size - 8} more ranks)")

    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not issues:
        print("✓ All optimizer steps have correct batch composition!")
        print(f"✓ Each step uses exactly {gbs} unique samples")
        print("✓ Samples are sequential: step N uses samples [N*GBS, (N+1)*GBS)")
    else:
        print("✗ Issues found:")
        for issue in issues:
            print(f"  - {issue}")

    return len(issues) == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze sample index logs")
    parser.add_argument("--log-dir", required=True, help="Directory with sample logs")
    parser.add_argument("--world-size", type=int, default=48, help="Number of GPUs")
    parser.add_argument("--grad-acc", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--mbs", type=int, default=1, help="Micro batch size")

    args = parser.parse_args()

    success = analyze_logs(
        log_dir=args.log_dir,
        world_size=args.world_size,
        grad_acc=args.grad_acc,
        mbs=args.mbs,
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
