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

"""Analyze training logs to verify batch composition and data ordering.

This script reads the logs from the training loop and the parquet dump,
then verifies that:
1. Each optimizer step uses the expected samples in the correct order
2. All ranks together cover the full GBS samples per step
3. The sequence tokens match the parquet dump order

Usage:
    python scripts/analyze_sample_logs.py \
        --log-dir /data/savithas/parquet_sample_logs \
        --parquet-dir /data/savithas/parquet_test \
        --world-size 8 \
        --grad-acc 8 \
        --mbs 1
"""

import argparse
import ast
import os

import pandas as pd


def load_training_logs(log_dir: str, world_size: int) -> pd.DataFrame:
    """Load training sequence logs from all ranks.

    Args:
        log_dir: Directory containing training_sequences_rankX.csv files
        world_size: Number of GPUs (ranks)

    Returns:
        Combined DataFrame with all rank logs
    """
    all_logs = []

    for rank in range(world_size):
        log_file = os.path.join(log_dir, f"training_sequences_rank{rank}.csv")
        if not os.path.exists(log_file):
            print(f"Warning: Missing log file for rank {rank}")
            continue

        df = pd.read_csv(log_file)
        df["rank"] = rank
        all_logs.append(df)

    if not all_logs:
        raise ValueError(f"No log files found in {log_dir}")

    return pd.concat(all_logs, ignore_index=True)


def load_parquet_data(parquet_dir: str) -> list:
    """Load the expected token sequences from parquet files.

    Args:
        parquet_dir: Directory containing parquet files

    Returns:
        List of input_ids lists in dump order
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("Warning: pyarrow not available, skipping parquet comparison")
        return None

    parquet_files = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])

    if not parquet_files:
        print(f"Warning: No parquet files found in {parquet_dir}")
        return None

    all_input_ids = []
    for pf in parquet_files:
        table = pq.read_table(os.path.join(parquet_dir, pf))
        all_input_ids.extend(table.to_pydict()["input_ids"])

    print(f"Loaded {len(all_input_ids)} sequences from parquet files")
    return all_input_ids


def analyze_logs(
    log_dir: str,
    parquet_dir: str | None,
    world_size: int,
    grad_acc: int,
    mbs: int,
):
    """Analyze training logs to verify batch composition.

    Args:
        log_dir: Directory containing training_sequences_rankX.csv files
        parquet_dir: Directory containing parquet files (for comparison)
        world_size: Number of GPUs (ranks)
        grad_acc: Gradient accumulation steps
        mbs: Micro batch size
    """
    gbs = world_size * grad_acc * mbs
    samples_per_rank_per_step = grad_acc * mbs

    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"  World size: {world_size}")
    print(f"  Grad acc: {grad_acc}")
    print(f"  Micro batch size: {mbs}")
    print(f"  Global batch size (GBS): {gbs}")
    print(f"  Samples per rank per optimizer step: {samples_per_rank_per_step}")
    print()

    # Load training logs
    print("Loading training logs...")
    logs = load_training_logs(log_dir, world_size)
    print(f"Total log entries: {len(logs)}")
    print(f"Columns: {list(logs.columns)}")
    print()

    # Load parquet data for comparison
    parquet_data = None
    if parquet_dir:
        print("Loading parquet data...")
        parquet_data = load_parquet_data(parquet_dir)
        print()

    # Analyze by optimizer step
    print("=" * 70)
    print("OPTIMIZER STEP ANALYSIS")
    print("=" * 70)

    num_steps = logs["optimizer_step"].max() + 1
    print(f"Number of optimizer steps: {num_steps}")
    print()

    issues = []

    for step in range(num_steps):
        step_data = logs[logs["optimizer_step"] == step]
        num_samples = len(step_data)

        print(f"Step {step}:")
        print(f"  Total samples across all ranks: {num_samples}")
        print(f"  Expected: {gbs}")

        if num_samples != gbs:
            issues.append(f"Step {step}: Expected {gbs} samples, got {num_samples}")
            print("  ⚠️  MISMATCH!")

        # Show per-rank breakdown
        print("  Per-rank breakdown:")
        for rank in range(world_size):
            rank_data = step_data[step_data["rank"] == rank]
            rank_samples = len(rank_data)
            first_tokens = rank_data["first_10_tokens"].iloc[0] if len(rank_data) > 0 else "N/A"

            # Parse the token string to get actual tokens
            if isinstance(first_tokens, str):
                try:
                    first_tokens = ast.literal_eval(first_tokens)
                except (ValueError, SyntaxError):
                    pass

            expected = samples_per_rank_per_step
            status = "✓" if rank_samples == expected else "✗"
            print(
                f"    Rank {rank}: {rank_samples} samples {status}, first tokens: {first_tokens[:5] if isinstance(first_tokens, list) else first_tokens}"
            )

        print()

    # Compare with parquet order (if available)
    if parquet_data:
        print("=" * 70)
        print("PARQUET ORDER COMPARISON")
        print("=" * 70)

        # For each optimizer step, check if the tokens match expected order
        # With DistributedSampler(shuffle=False), samples should be:
        # Rank 0: indices 0, 8, 16, ... (every world_size-th sample starting at 0)
        # Rank 1: indices 1, 9, 17, ... (every world_size-th sample starting at 1)
        # etc.

        print("Expected DistributedSampler behavior (shuffle=False):")
        print(f"  Each rank gets every {world_size}-th sample")
        print()

        for step in range(min(num_steps, 2)):  # Check first 2 steps
            print(f"Step {step}:")
            step_data = logs[logs["optimizer_step"] == step]

            all_match = True
            for rank in range(world_size):
                rank_data = step_data[step_data["rank"] == rank].sort_values("micro_step")

                for micro_idx, (_, row) in enumerate(rank_data.iterrows()):
                    # Calculate expected parquet index
                    # Step 0: rank 0 gets [0, 8, 16, ...], rank 1 gets [1, 9, 17, ...]
                    # Step 1: continues from where step 0 left off
                    base_idx = step * gbs
                    expected_parquet_idx = base_idx + rank + (micro_idx * world_size)

                    if expected_parquet_idx >= len(parquet_data):
                        print(f"  Warning: Parquet index {expected_parquet_idx} out of range")
                        continue

                    expected_tokens = parquet_data[expected_parquet_idx][:10]

                    actual_tokens = row["first_10_tokens"]
                    if isinstance(actual_tokens, str):
                        try:
                            actual_tokens = ast.literal_eval(actual_tokens)
                        except (ValueError, SyntaxError):
                            actual_tokens = []

                    match = actual_tokens == expected_tokens
                    if not match:
                        all_match = False
                        print(f"  ✗ Rank {rank} micro {micro_idx}: MISMATCH")
                        print(f"      Expected (parquet[{expected_parquet_idx}]): {expected_tokens}")
                        print(f"      Actual: {actual_tokens}")
                        issues.append(f"Step {step} Rank {rank} micro {micro_idx}: Token mismatch")

            if all_match:
                print("  ✓ All tokens match expected parquet order!")
            print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if not issues:
        print("✓ All checks passed!")
        print(f"✓ Each optimizer step processes {gbs} samples correctly")
        if parquet_data:
            print("✓ Token sequences match parquet dump order")
    else:
        print("✗ Issues found:")
        for issue in issues:
            print(f"  - {issue}")

    return len(issues) == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze training sequence logs")
    parser.add_argument("--log-dir", required=True, help="Directory with training logs")
    parser.add_argument("--parquet-dir", default=None, help="Directory with parquet files (optional)")
    parser.add_argument("--world-size", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--grad-acc", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--mbs", type=int, default=1, help="Micro batch size")

    args = parser.parse_args()

    success = analyze_logs(
        log_dir=args.log_dir,
        parquet_dir=args.parquet_dir,
        world_size=args.world_size,
        grad_acc=args.grad_acc,
        mbs=args.mbs,
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
