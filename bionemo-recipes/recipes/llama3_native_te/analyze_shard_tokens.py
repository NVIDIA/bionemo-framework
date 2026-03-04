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

"""Analyze sequence length distributions and simulated BSHD padding for parquet datasets.

Compares two parquet datasets (e.g. parquet2 vs parquet_split) to understand
differences in unpadded token counts per batch.

Usage:
    python analyze_shard_tokens.py \
        --dataset_a /data/opengenome2/parquet2 \
        --dataset_b /data/opengenome2/parquet_split \
        --label_a "parquet2 (unshuffled windows)" \
        --label_b "parquet_split (pre-chunked)" \
        --tokenizer_path ./tokenizers/nucleotide_fast_tokenizer \
        --num_shards 3 \
        --stride 200 \
        --max_seq_length 8192 \
        --grad_acc_steps 8

    Or compare a single dataset:
    python analyze_shard_tokens.py \
        --dataset_a /data/opengenome2/parquet2 \
        --label_a "parquet2" \
        --tokenizer_path ./tokenizers/nucleotide_fast_tokenizer \
        --num_shards 5
"""

import argparse
import glob
import os
import statistics
from collections import Counter

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def load_shard_sequences(shard_path: str, text_column: str = "text", max_rows: int | None = None) -> list[str]:
    """Load raw sequences from a single parquet shard."""
    table = pq.read_table(shard_path, columns=[text_column])
    col = table.column(text_column)
    seqs = [s.as_py() for s in col]
    if max_rows is not None:
        seqs = seqs[:max_rows]
    return seqs


def tokenize_with_windowing(tokenizer, sequences: list[str], max_seq_length: int, stride: int) -> list[list[int]]:
    """Tokenize sequences with windowing (same logic as dataset.py)."""
    result = tokenizer(
        sequences,
        max_length=max_seq_length,
        stride=stride,
        truncation=True,
        return_overflowing_tokens=True,
        add_special_tokens=True,
    )
    return result["input_ids"]


def tokenize_direct(tokenizer, sequences: list[str]) -> list[list[int]]:
    """Tokenize sequences without windowing (same logic as dataset.py for stride=None)."""
    result = tokenizer(
        sequences,
        add_special_tokens=True,
        truncation=False,
    )
    return result["input_ids"]


def compute_stats(lengths: list[int], label: str) -> dict:
    """Compute and print distribution statistics."""
    if not lengths:
        print(f"  [{label}] No data!")
        return {}

    stats = {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "stdev": statistics.stdev(lengths) if len(lengths) > 1 else 0,
    }

    buckets = Counter()
    for length in lengths:
        if length <= 1000:
            buckets["0-1k"] += 1
        elif length <= 2000:
            buckets["1k-2k"] += 1
        elif length <= 4000:
            buckets["2k-4k"] += 1
        elif length <= 6000:
            buckets["4k-6k"] += 1
        elif length <= 8000:
            buckets["6k-8k"] += 1
        else:
            buckets["8k+"] += 1

    print(f"\n  [{label}] Token length distribution:")
    print(f"    Count:  {stats['count']}")
    print(f"    Min:    {stats['min']}")
    print(f"    Max:    {stats['max']}")
    print(f"    Mean:   {stats['mean']:.1f}")
    print(f"    Median: {stats['median']:.1f}")
    print(f"    Stdev:  {stats['stdev']:.1f}")
    print("    Buckets:")
    for bucket in ["0-1k", "1k-2k", "2k-4k", "4k-6k", "6k-8k", "8k+"]:
        cnt = buckets.get(bucket, 0)
        pct = 100.0 * cnt / len(lengths)
        bar = "#" * int(pct / 2)
        print(f"      {bucket:>6s}: {cnt:6d} ({pct:5.1f}%) {bar}")

    return stats


def simulate_bshd_batches(
    token_lengths: list[int], micro_batch_size: int, grad_acc_steps: int, max_seq_length: int, num_batches: int = 100
) -> list[int]:
    """Simulate BSHD batching and compute unpadded tokens per global batch.

    In BSHD mode, each micro-batch pads all sequences to the length of the longest
    sequence in that micro-batch. This simulates that padding effect.
    """
    unpadded_per_global_batch = []
    idx = 0
    for _ in range(num_batches):
        unpadded_in_step = 0
        for _ in range(grad_acc_steps):
            if idx + micro_batch_size > len(token_lengths):
                break
            batch_lengths = token_lengths[idx : idx + micro_batch_size]
            idx += micro_batch_size
            # unpadded = sum of actual token lengths (attention_mask.sum())
            unpadded_in_step += sum(batch_lengths)
        if unpadded_in_step > 0:
            unpadded_per_global_batch.append(unpadded_in_step)

    return unpadded_per_global_batch


def analyze_dataset(
    dataset_path: str,
    label: str,
    tokenizer,
    num_shards: int,
    max_seq_length: int,
    stride: int | None,
    text_column: str,
    micro_batch_size: int,
    grad_acc_steps: int,
    max_rows_per_shard: int | None,
):
    """Analyze a single dataset's token distribution and simulated BSHD batching."""
    shard_files = sorted(os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".parquet"))
    if not shard_files:
        # Fallback: search subdirectories
        shard_files = sorted(glob.glob(os.path.join(dataset_path, "**/*.parquet"), recursive=True))

    if not shard_files:
        print(f"\n  ERROR: No parquet files found in {dataset_path}")
        return None

    print(f"\n{'=' * 80}")
    print(f"  Dataset: {label}")
    print(f"  Path: {dataset_path}")
    print(f"  Total shards found: {len(shard_files)}")
    print(f"  Analyzing first {min(num_shards, len(shard_files))} shards")
    print(f"  Stride: {stride} (windowing {'enabled' if stride is not None else 'disabled'})")
    print(f"{'=' * 80}")

    all_raw_lengths = []
    all_token_lengths = []

    for shard_file in shard_files[:num_shards]:
        shard_name = os.path.basename(shard_file)
        sequences = load_shard_sequences(shard_file, text_column=text_column, max_rows=max_rows_per_shard)
        raw_lengths = [len(s) for s in sequences]
        all_raw_lengths.extend(raw_lengths)

        if stride is not None:
            token_ids_list = tokenize_with_windowing(tokenizer, sequences, max_seq_length, stride)
        else:
            token_ids_list = tokenize_direct(tokenizer, sequences)

        token_lengths = [len(ids) for ids in token_ids_list]
        all_token_lengths.extend(token_lengths)

        print(f"\n  Shard: {shard_name}")
        print(f"    Raw sequences: {len(sequences)}")
        print(
            f"    Raw length (bases): min={min(raw_lengths)}, max={max(raw_lengths)}, mean={statistics.mean(raw_lengths):.0f}"
        )
        print(f"    After tokenization: {len(token_ids_list)} windows/sequences")
        print(
            f"    Token lengths: min={min(token_lengths)}, max={max(token_lengths)}, mean={statistics.mean(token_lengths):.0f}"
        )

        # Show windowing expansion ratio
        if len(token_ids_list) != len(sequences):
            print(
                f"    Windowing expansion: {len(sequences)} seqs -> {len(token_ids_list)} windows ({len(token_ids_list) / len(sequences):.2f}x)"
            )

    print(f"\n  --- Aggregate stats across {min(num_shards, len(shard_files))} shards ---")
    compute_stats(all_raw_lengths, f"{label} raw bases")
    compute_stats(all_token_lengths, f"{label} token lengths")

    # Simulate BSHD batching
    print(f"\n  --- Simulated BSHD Batching (MBS={micro_batch_size}, grad_acc={grad_acc_steps}) ---")
    num_sim_batches = min(200, len(all_token_lengths) // (micro_batch_size * grad_acc_steps))
    if num_sim_batches < 5:
        print(
            f"    Not enough data to simulate batches (need at least {micro_batch_size * grad_acc_steps * 5} windows)"
        )
    else:
        unpadded_counts = simulate_bshd_batches(
            all_token_lengths, micro_batch_size, grad_acc_steps, max_seq_length, num_sim_batches
        )
        if unpadded_counts:
            theoretical_max = micro_batch_size * grad_acc_steps * max_seq_length
            mean_unpadded = statistics.mean(unpadded_counts)
            print(f"    Simulated {len(unpadded_counts)} global batches")
            print(f"    Theoretical max tokens/batch: {theoretical_max}")
            print(
                f"    Unpadded tokens/batch: min={min(unpadded_counts)}, max={max(unpadded_counts)}, mean={mean_unpadded:.0f}"
            )
            print(f"    Padding efficiency: {100 * mean_unpadded / theoretical_max:.1f}%")
            print(f"    Avg unpadded per micro-step: {mean_unpadded / grad_acc_steps:.0f}")

    return {"raw_lengths": all_raw_lengths, "token_lengths": all_token_lengths}


def main():
    """Run the shard token distribution analysis."""
    parser = argparse.ArgumentParser(description="Analyze token distributions in parquet datasets")
    parser.add_argument("--dataset_a", required=True, help="Path to first dataset directory")
    parser.add_argument("--label_a", default="Dataset A", help="Label for first dataset")
    parser.add_argument("--dataset_b", default=None, help="Path to second dataset directory (optional)")
    parser.add_argument("--label_b", default="Dataset B", help="Label for second dataset")
    parser.add_argument("--tokenizer_path", required=True, help="Path to tokenizer")
    parser.add_argument("--num_shards", type=int, default=3, help="Number of shards to analyze per dataset")
    parser.add_argument("--max_seq_length", type=int, default=8192, help="Max sequence length (window size)")
    parser.add_argument("--stride", type=int, default=200, help="Stride for windowing. Use -1 for no windowing.")
    parser.add_argument(
        "--stride_b", type=int, default=None, help="Override stride for dataset B. Use -1 for no windowing."
    )
    parser.add_argument("--text_column", default="text", help="Column name for sequences")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Micro batch size for BSHD simulation")
    parser.add_argument("--grad_acc_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max_rows_per_shard", type=int, default=None, help="Limit rows read per shard (for speed)")
    args = parser.parse_args()

    stride_a = args.stride if args.stride != -1 else None
    if args.stride_b is not None:
        stride_b = args.stride_b if args.stride_b != -1 else None
    else:
        stride_b = stride_a

    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")

    result_a = analyze_dataset(
        dataset_path=args.dataset_a,
        label=args.label_a,
        tokenizer=tokenizer,
        num_shards=args.num_shards,
        max_seq_length=args.max_seq_length,
        stride=stride_a,
        text_column=args.text_column,
        micro_batch_size=args.micro_batch_size,
        grad_acc_steps=args.grad_acc_steps,
        max_rows_per_shard=args.max_rows_per_shard,
    )

    result_b = None
    if args.dataset_b:
        result_b = analyze_dataset(
            dataset_path=args.dataset_b,
            label=args.label_b,
            tokenizer=tokenizer,
            num_shards=args.num_shards,
            max_seq_length=args.max_seq_length,
            stride=stride_b,
            text_column=args.text_column,
            micro_batch_size=args.micro_batch_size,
            grad_acc_steps=args.grad_acc_steps,
            max_rows_per_shard=args.max_rows_per_shard,
        )

    if result_a and result_b:
        print(f"\n{'=' * 80}")
        print("  COMPARISON SUMMARY")
        print(f"{'=' * 80}")
        mean_a = statistics.mean(result_a["token_lengths"])
        mean_b = statistics.mean(result_b["token_lengths"])
        print(f"  {args.label_a}: {len(result_a['token_lengths'])} windows, mean token length = {mean_a:.0f}")
        print(f"  {args.label_b}: {len(result_b['token_lengths'])} windows, mean token length = {mean_b:.0f}")
        print(f"  Difference in mean token length: {mean_b - mean_a:+.0f} ({100 * (mean_b - mean_a) / mean_a:+.1f}%)")

        # If both have BSHD sim data
        theoretical_max = args.micro_batch_size * args.grad_acc_steps * args.max_seq_length
        ratio_a = mean_a / args.max_seq_length
        ratio_b = mean_b / args.max_seq_length
        print("\n  Avg fraction of max_seq_length filled:")
        print(f"    {args.label_a}: {100 * ratio_a:.1f}%")
        print(f"    {args.label_b}: {100 * ratio_b:.1f}%")
        print(f"    Expected unpadded tokens per batch (MBS={args.micro_batch_size}, grad_acc={args.grad_acc_steps}):")
        print(f"      {args.label_a}: ~{int(mean_a * args.micro_batch_size * args.grad_acc_steps)}")
        print(f"      {args.label_b}: ~{int(mean_b * args.micro_batch_size * args.grad_acc_steps)}")
        print(f"    Theoretical max: {theoretical_max}")


if __name__ == "__main__":
    main()
