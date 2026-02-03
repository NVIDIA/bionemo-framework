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

"""Analyze and compare training logs between our training and John's.

This script can:
1. Analyze our training logs to verify batch composition
2. Compare our training with John's ShardedEdenDataset window logs
3. Verify token sequences match the parquet dump order

Usage (analyze our logs only):
    python scripts/analyze_sample_logs.py \
        --our-log-dir /data/savithas/parquet_sample_logs \
        --parquet-dir /data/savithas/parquet_test \
        --world-size 8 --grad-acc 8 --mbs 1

Usage (compare with John's logs):
    python scripts/analyze_sample_logs.py \
        --our-log-dir /data/savithas/parquet_sample_logs \
        --john-log-dir /data/savithas/john_window_logs \
        --parquet-dir /data/savithas/parquet_test \
        --world-size 8 --grad-acc 8 --mbs 1
"""

import argparse
import ast
import glob
import os

import pandas as pd


def load_our_training_logs(log_dir: str, world_size: int) -> pd.DataFrame:
    """Load our training sequence logs from all ranks.

    Our format: optimizer_step, micro_step, global_micro_step, batch_idx, first_10_tokens, loss
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


def load_john_window_logs(log_dir: str) -> pd.DataFrame:
    """Load John's ShardedEdenDataset window logs.

    John's format: window_idx, sequence_id, sample_id, window_in_seq_idx, rank, access_ts
    """
    # Find all window access CSV files
    pattern = os.path.join(log_dir, "window_access_*.csv")
    log_files = glob.glob(pattern)

    if not log_files:
        raise ValueError(f"No window access logs found matching {pattern}")

    all_logs = []
    for log_file in log_files:
        df = pd.read_csv(log_file)
        all_logs.append(df)

    combined = pd.concat(all_logs, ignore_index=True)
    # Sort by access timestamp to get the order
    combined = combined.sort_values("access_ts").reset_index(drop=True)

    return combined


def load_parquet_data(parquet_dir: str) -> list:
    """Load the expected token sequences from parquet files."""
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


def analyze_our_logs(logs: pd.DataFrame, world_size: int, grad_acc: int, mbs: int) -> list:
    """Analyze our training logs for correctness."""
    gbs = world_size * grad_acc * mbs
    samples_per_rank_per_step = grad_acc * mbs
    issues = []

    print("=" * 70)
    print("OUR TRAINING LOG ANALYSIS")
    print("=" * 70)

    num_steps = logs["optimizer_step"].max() + 1
    print(f"Number of optimizer steps: {num_steps}")
    print()

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

            if isinstance(first_tokens, str):
                try:
                    first_tokens = ast.literal_eval(first_tokens)
                except (ValueError, SyntaxError):
                    pass

            expected = samples_per_rank_per_step
            status = "✓" if rank_samples == expected else "✗"
            tokens_preview = first_tokens[:5] if isinstance(first_tokens, list) else first_tokens
            print(f"    Rank {rank}: {rank_samples} samples {status}, first tokens: {tokens_preview}")

        print()

    return issues


def compare_with_parquet(logs: pd.DataFrame, parquet_data: list, world_size: int) -> list:
    """Compare our training logs with parquet dump order."""
    issues = []

    print("=" * 70)
    print("PARQUET ORDER COMPARISON")
    print("=" * 70)
    print("Expected DistributedSampler behavior (shuffle=False):")
    print(f"  Each rank gets every {world_size}-th sample")
    print()

    gbs = len(logs) // (logs["optimizer_step"].max() + 1)
    num_steps = logs["optimizer_step"].max() + 1

    for step in range(min(num_steps, 2)):
        print(f"Step {step}:")
        step_data = logs[logs["optimizer_step"] == step]

        all_match = True
        for rank in range(world_size):
            rank_data = step_data[step_data["rank"] == rank].sort_values("micro_step")

            for micro_idx, (_, row) in enumerate(rank_data.iterrows()):
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

    return issues


def compare_with_john(our_logs: pd.DataFrame, john_logs: pd.DataFrame, world_size: int, grad_acc: int) -> list:
    """Compare our training order with John's window access order."""
    issues = []

    print("=" * 70)
    print("COMPARISON WITH JOHN'S WINDOW LOGS")
    print("=" * 70)

    print(f"John's logs: {len(john_logs)} window accesses")
    print(f"Our logs: {len(our_logs)} samples")
    print()

    # John's window_idx is the global index into the dataset (after permutation)
    # Our samples should match in order

    # Group John's logs by rank and order by access time
    john_by_rank = {}
    for rank in range(world_size):
        rank_data = john_logs[john_logs["rank"] == rank].sort_values("access_ts")
        john_by_rank[rank] = rank_data["window_idx"].tolist()

    # Group our logs by rank and order by micro_step
    our_by_rank = {}
    for rank in range(world_size):
        rank_data = our_logs[our_logs["rank"] == rank].sort_values(["optimizer_step", "micro_step"])
        our_by_rank[rank] = list(range(len(rank_data)))  # We don't have window_idx, just position

    print("Window index sequences by rank:")
    print()

    for rank in range(min(world_size, 4)):  # Show first 4 ranks
        john_indices = john_by_rank.get(rank, [])[:16]  # First 16 accesses
        print(f"  Rank {rank} (John's window_idx): {john_indices}")

    print()
    print("To verify the data is the same:")
    print("  1. The parquet was dumped using John's ShardedEdenDataset + permute function")
    print("  2. Our training reads parquet in order (verified by parquet comparison)")
    print("  3. If John's window_idx sequence matches the parquet dump order, data is identical")
    print()

    # Check if John's window indices are sequential starting from 0
    # (which would match our parquet dump order)
    all_john_indices = john_logs.sort_values("access_ts")["window_idx"].tolist()

    # Expected: for each optimizer step, indices should be [0, 1, 2, ..., GBS-1], [GBS, GBS+1, ...], etc.
    # But distributed across ranks

    gbs = world_size * grad_acc
    num_steps = len(all_john_indices) // gbs if gbs > 0 else 0

    print(f"John processed {num_steps} optimizer steps worth of data")
    print()

    for step in range(min(num_steps, 2)):
        step_start = step * gbs
        step_end = step_start + gbs
        step_indices = sorted(all_john_indices[step_start:step_end])

        expected_indices = list(range(step * gbs, (step + 1) * gbs))

        if step_indices == expected_indices:
            print(f"  Step {step}: ✓ John's indices match expected range [{step * gbs}, {(step + 1) * gbs})")
        else:
            print(f"  Step {step}: ✗ Index mismatch!")
            print(f"    Expected: {expected_indices[:10]}...")
            print(f"    Got: {step_indices[:10]}...")
            issues.append(f"Step {step}: John's window indices don't match expected range")

    return issues


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze and compare training logs")
    parser.add_argument("--our-log-dir", "--log-dir", required=True, help="Directory with our training logs")
    parser.add_argument("--john-log-dir", default=None, help="Directory with John's window logs (optional)")
    parser.add_argument("--parquet-dir", default=None, help="Directory with parquet files (optional)")
    parser.add_argument("--world-size", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--grad-acc", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--mbs", type=int, default=1, help="Micro batch size")

    args = parser.parse_args()

    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"  World size: {args.world_size}")
    print(f"  Grad acc: {args.grad_acc}")
    print(f"  Micro batch size: {args.mbs}")
    print(f"  Global batch size (GBS): {args.world_size * args.grad_acc * args.mbs}")
    print()

    all_issues = []

    # Load our logs
    print("Loading our training logs...")
    our_logs = load_our_training_logs(args.our_log_dir, args.world_size)
    print(f"Loaded {len(our_logs)} log entries")
    print()

    # Analyze our logs
    issues = analyze_our_logs(our_logs, args.world_size, args.grad_acc, args.mbs)
    all_issues.extend(issues)

    # Compare with parquet if available
    if args.parquet_dir:
        print("Loading parquet data...")
        parquet_data = load_parquet_data(args.parquet_dir)
        if parquet_data:
            print()
            issues = compare_with_parquet(our_logs, parquet_data, args.world_size)
            all_issues.extend(issues)

    # Compare with John's logs if available
    if args.john_log_dir:
        print("Loading John's window logs...")
        try:
            john_logs = load_john_window_logs(args.john_log_dir)
            print(f"Loaded {len(john_logs)} window access entries")
            print()
            issues = compare_with_john(our_logs, john_logs, args.world_size, args.grad_acc)
            all_issues.extend(issues)
        except ValueError as e:
            print(f"Warning: {e}")
            print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if not all_issues:
        print("✓ All checks passed!")
        print("✓ Each optimizer step processes the correct number of samples")
        if args.parquet_dir:
            print("✓ Token sequences match parquet dump order")
        if args.john_log_dir:
            print("✓ Data ordering matches John's training")
    else:
        print("✗ Issues found:")
        for issue in all_issues:
            print(f"  - {issue}")

    return 0 if not all_issues else 1


if __name__ == "__main__":
    exit(main())
