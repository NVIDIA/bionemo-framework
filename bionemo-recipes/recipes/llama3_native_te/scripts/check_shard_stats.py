# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Check sequence/window statistics across Parquet shards.

Usage:
    # Check first 5 shards
    python scripts/check_shard_stats.py /data/opengenome2/parquet --num-shards 5

    # Check all shards
    python scripts/check_shard_stats.py /data/opengenome2/parquet

    # Check with different window parameters
    python scripts/check_shard_stats.py /data/opengenome2/parquet --window-size 8192 --stride 7992
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def compute_num_windows(seq_len: int, window_size: int = 8192, stride: int = 7992) -> int:
    """Same formula as eden's ShardedEdenDataset."""
    if seq_len < window_size:
        return 1
    return 1 + (seq_len - window_size) // stride


def main():
    """Check sequence and window statistics across Parquet shards."""
    parser = argparse.ArgumentParser(description="Check sequence/window stats across Parquet shards.")
    parser.add_argument("input_dir", type=str, help="Directory containing Parquet shards.")
    parser.add_argument("--num-shards", type=int, default=None, help="Number of shards to check (default: all).")
    parser.add_argument("--window-size", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=7992)
    parser.add_argument("--text-column", type=str, default="text")
    args = parser.parse_args()

    try:
        import polars as pl

        def read_parquet(f, col):
            return pl.read_parquet(f, columns=[col])[col].str.len_chars().to_list()

    except ImportError:
        import pyarrow.parquet as pq

        def read_parquet(f, col):
            table = pq.read_table(f, columns=[col])
            return [len(s.as_py()) for s in table.column(col)]

    shard_files = sorted(Path(args.input_dir).glob("*.parquet"))
    if not shard_files:
        print(f"No parquet files found in {args.input_dir}")
        sys.exit(1)

    if args.num_shards:
        shard_files = shard_files[: args.num_shards]

    print(f"Checking {len(shard_files)} shards in {args.input_dir}")
    print(f"Window params: window_size={args.window_size}, stride={args.stride}")
    print(f"{'=' * 80}")

    all_shard_seqs = []
    all_shard_windows = []
    all_shard_total_chars = []
    all_seq_lengths = []
    all_windows_per_seq = []

    for i, f in enumerate(shard_files):
        lengths = read_parquet(f, args.text_column)
        n_seqs = len(lengths)
        shard_chars = sum(lengths)
        windows = [compute_num_windows(seq_len, args.window_size, args.stride) for seq_len in lengths]
        total_windows = sum(windows)

        all_shard_seqs.append(n_seqs)
        all_shard_windows.append(total_windows)
        all_shard_total_chars.append(shard_chars)
        all_seq_lengths.extend(lengths)
        all_windows_per_seq.extend(windows)

        # THD estimate: how many packed 8192-token steps does this shard represent?
        thd_steps_this_shard = shard_chars / 8192

        if i < 10 or (i + 1) == len(shard_files):
            print(
                f"  {f.name}: {n_seqs:,} seqs, {total_windows:,} windows, "
                f"total chars: {shard_chars:,}, "
                f"THD steps: {thd_steps_this_shard:,.0f}, "
                f"avg seq len: {np.mean(lengths):,.0f}, "
                f"median seq len: {np.median(lengths):,.0f}"
            )
        elif i == 10:
            print(f"  ... ({len(shard_files) - 10} more shards) ...")
        elif (i + 1) % 50 == 0:
            running_chars = sum(all_shard_total_chars)
            running_windows = sum(all_shard_windows)
            running_thd_steps = running_chars / 8192
            print(
                f"  [{i + 1}/{len(shard_files)}] "
                f"{running_chars:,} chars, {running_windows:,} windows, "
                f"{running_thd_steps:,.0f} THD micro-steps so far"
            )

    print(f"\n{'=' * 80}")
    print(f"SUMMARY ({len(shard_files)} shards)")
    print(f"{'=' * 80}")

    total_seqs = sum(all_shard_seqs)
    total_windows = sum(all_shard_windows)
    total_chars = sum(all_shard_total_chars)
    seq_lengths = np.array(all_seq_lengths)
    windows_per_seq = np.array(all_windows_per_seq)

    print("\nSequences:")
    print(f"  Total: {total_seqs:,}")
    print(f"  Per shard: {total_seqs // len(shard_files):,} avg")
    print(f"  Min shard: {min(all_shard_seqs):,}, Max shard: {max(all_shard_seqs):,}")

    print(f"\nTotal characters: {total_chars:,}")
    print("  (Compare to Peter's 3,652,742,371)")

    print("\nSequence lengths:")
    print(f"  Mean:   {seq_lengths.mean():,.0f} chars")
    print(f"  Median: {np.median(seq_lengths):,.0f} chars")
    print(f"  Min:    {seq_lengths.min():,}")
    print(f"  Max:    {seq_lengths.max():,}")
    print(f"  Std:    {seq_lengths.std():,.0f}")

    print("\nWindows:")
    print(f"  Total: {total_windows:,}")
    print(f"  Per shard: {total_windows // len(shard_files):,} avg")
    print(f"  Min shard: {min(all_shard_windows):,}, Max shard: {max(all_shard_windows):,}")

    print("\nWindows per sequence distribution:")
    from collections import Counter

    dist = Counter(windows_per_seq.tolist())
    for w in sorted(dist.keys())[:10]:
        count = dist[w]
        pct_seqs = 100.0 * count / total_seqs
        total_w = w * count
        pct_wins = 100.0 * total_w / total_windows
        print(f"  {w} window(s): {count:,} seqs ({pct_seqs:.1f}%) = {total_w:,} windows ({pct_wins:.1f}%)")
    if max(dist.keys()) > 10:
        remaining = sum(v for k, v in dist.items() if k > 10)
        remaining_w = sum(k * v for k, v in dist.items() if k > 10)
        print(
            f"  11+ windows: {remaining:,} seqs = {remaining_w:,} windows ({100 * remaining_w / total_windows:.1f}%)"
        )

    # Training estimates
    print(f"\n{'=' * 80}")
    print("TRAINING ESTIMATES")
    print(f"{'=' * 80}")

    avg_window_tokens = min(seq_lengths.mean(), 8192)

    # THD (token packing) estimates — based on characters consumed
    thd_micro_steps_total = total_chars / 8192
    thd_micro_steps_per_rank = thd_micro_steps_total / 48
    thd_optimizer_steps_per_epoch = thd_micro_steps_per_rank / 8  # grad_acc=8

    print("\n  THD (token packing) estimates:")
    print(f"    Total chars: {total_chars:,}")
    print(f"    Total THD micro-steps (all ranks): {thd_micro_steps_total:,.0f}")
    print(f"    THD micro-steps per rank: {thd_micro_steps_per_rank:,.0f}")
    print(f"    Optimizer steps per epoch (grad_acc=8): {thd_optimizer_steps_per_epoch:,.0f}")
    print(f"    Epochs in 180k steps: {180000 / thd_optimizer_steps_per_epoch:.1f}")
    print("    (Compare to Peter's estimate: 1,161 steps per epoch)")

    # Per-shard THD estimates
    avg_chars_per_shard = total_chars / len(shard_files)
    thd_micro_steps_per_shard = avg_chars_per_shard / 8192
    print(f"\n    Chars per shard: {avg_chars_per_shard:,.0f}")
    print(f"    THD micro-steps per shard: {thd_micro_steps_per_shard:,.0f}")
    print(f"    THD optimizer steps per shard (grad_acc=8): {thd_micro_steps_per_shard / 8:,.0f}")

    # Window-based estimates (for reference)
    print("\n  Window-based estimates (for reference):")
    print(f"    Total windows: {total_windows:,}")
    print(f"    Avg window length: ~{avg_window_tokens:,.0f} tokens")
    print(f"    Windows per micro-step (token packing): ~{8192 / avg_window_tokens:.1f}")
    print(f"    Windows per optimizer step per rank: ~{8192 / avg_window_tokens * 8:.0f}")

    per_rank_windows = total_windows / 48
    print(f"    Windows per rank: {per_rank_windows:,.0f}")

    # Buffer estimates
    buffer_size = 50000
    buffer_chars = buffer_size * avg_window_tokens
    per_rank_chars = total_chars / 48
    print("\n  Buffer estimates (50k window buffer):")
    print(f"    Buffer windows: {buffer_size:,}")
    print(f"    Buffer chars: ~{buffer_chars:,.0f}")
    print(f"    Per-rank total chars: {per_rank_chars:,.0f}")
    print(f"    Buffer as % of per-rank chars: {100 * buffer_chars / per_rank_chars:.2f}%")
    print(f"    Buffer as % of per-rank windows: {100 * buffer_size / per_rank_windows:.2f}%")
    thd_buffer_turnover = buffer_size / (8192 / avg_window_tokens * 8)
    print(f"    Buffer turnover: every {thd_buffer_turnover:,.0f} optimizer steps")


if __name__ == "__main__":
    main()
