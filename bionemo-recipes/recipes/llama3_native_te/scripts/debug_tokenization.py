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

"""Debug script to inspect tokenization paths for parquet data.

Run this interactively on a Lepton node to understand what the data looks like
at each stage of the pipeline.

Usage:
    cd /data/savithas/bionemo-framework/bionemo-recipes/recipes/llama3_native_te
    python3 scripts/debug_tokenization.py --data-path /data/opengenome2/parquet_split --num-shards 3
"""

import argparse
from collections import Counter

import datasets
from transformers import AutoTokenizer


def main():
    """Inspect tokenization paths for parquet data."""
    parser = argparse.ArgumentParser(description="Debug tokenization paths")
    parser.add_argument("--data-path", type=str, default="/data/opengenome2/parquet_split")
    parser.add_argument("--tokenizer", type=str, default="./tokenizers/nucleotide_fast_tokenizer")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of raw samples to inspect")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=200)
    args = parser.parse_args()

    # =========================================================================
    # 1. RAW DATA INSPECTION
    # =========================================================================
    print("=" * 80)
    print("1. RAW DATA INSPECTION")
    print("=" * 80)

    ds = datasets.load_dataset(args.data_path, split="train", streaming=True)
    print(f"Dataset: {args.data_path}")
    print(f"Num shards: {ds.num_shards}")

    lengths = []
    length_buckets = Counter()
    for i, sample in enumerate(ds):
        if i >= args.num_samples:
            break
        text = sample["text"]
        length = len(text)
        lengths.append(length)

        # Bucket by range
        if length < 1000:
            length_buckets["<1k"] += 1
        elif length < 5000:
            length_buckets["1k-5k"] += 1
        elif length < 8000:
            length_buckets["5k-8k"] += 1
        elif length <= 8200:
            length_buckets["8k-8.2k (expected)"] += 1
        elif length < 20000:
            length_buckets["8.2k-20k"] += 1
        elif length < 100000:
            length_buckets["20k-100k"] += 1
        else:
            length_buckets["100k+"] += 1

    print(f"\nSampled {len(lengths)} sequences:")
    print(f"  Min length:  {min(lengths)} bases")
    print(f"  Max length:  {max(lengths)} bases")
    print(f"  Mean length: {sum(lengths) / len(lengths):.0f} bases")
    print(f"  Median:      {sorted(lengths)[len(lengths) // 2]} bases")

    print("\nLength distribution:")
    for bucket in ["<1k", "1k-5k", "5k-8k", "8k-8.2k (expected)", "8.2k-20k", "20k-100k", "100k+"]:
        count = length_buckets.get(bucket, 0)
        pct = count / len(lengths) * 100
        bar = "#" * int(pct / 2)
        print(f"  {bucket:>20s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Show some non-8190 sequences
    print("\nExamples of non-8190 sequences:")
    count = 0
    for i, length in enumerate(lengths):
        if length != 8190 and count < 10:
            print(f"  Sample {i}: {length} bases")
            count += 1
    if count == 0:
        print("  (all sampled sequences are exactly 8190 bases)")

    # =========================================================================
    # 2. TOKENIZATION COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. TOKENIZATION COMPARISON (first 20 samples)")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")

    ds2 = datasets.load_dataset(args.data_path, split="train", streaming=True)

    print(
        f"\n{'Sample':>6s} | {'Raw len':>8s} | {'Direct tokens':>14s} | {'Windowed tokens':>15s} | {'Num windows':>11s}"
    )
    print("-" * 70)

    direct_lengths = []
    windowed_lengths = []
    windowed_counts = []

    for i, sample in enumerate(ds2):
        if i >= 20:
            break
        text = sample["text"]
        raw_len = len(text)

        # tokenize_direct path
        direct = tokenizer(text, add_special_tokens=True, truncation=False)
        direct_len = len(direct["input_ids"])
        direct_lengths.append(direct_len)

        # tokenize_with_windowing path
        windowed = tokenizer(
            text,
            max_length=args.max_seq_length,
            stride=args.stride,
            truncation=True,
            return_overflowing_tokens=True,
            add_special_tokens=True,
        )
        # return_overflowing_tokens returns a list of windows
        num_windows = len(windowed["input_ids"])
        window_lens = [len(w) for w in windowed["input_ids"]]
        windowed_lengths.extend(window_lens)
        windowed_counts.append(num_windows)

        marker = "" if raw_len == 8190 else " <-- NOT 8190"
        print(
            f"{i:>6d} | {raw_len:>8d} | {direct_len:>14d} | "
            f"{window_lens[0]:>7d} x {num_windows:<5d}  | {num_windows:>11d}{marker}"
        )

    # =========================================================================
    # 3. TOKEN PACKING SIMULATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. TOKEN PACKING SIMULATION (first 50 samples, max_tokens_per_batch=8192)")
    print("=" * 80)

    max_tokens = args.max_seq_length
    ds3 = datasets.load_dataset(args.data_path, split="train", streaming=True)

    # Simulate tokenize_direct + TokenPackingDataset with split_samples=True
    print("\n--- tokenize_direct + split_samples=True ---")
    batch_num = 0
    current_batch_tokens = 0
    current_batch_seqs = 0
    for i, sample in enumerate(ds3):
        if batch_num >= 10:
            break
        if i >= 50:
            break

        text = sample["text"]
        direct = tokenizer(text, add_special_tokens=True, truncation=False)
        token_len = len(direct["input_ids"])

        remaining = token_len
        while remaining > 0 and batch_num < 10:
            space_left = max_tokens - current_batch_tokens
            if remaining <= space_left:
                current_batch_tokens += remaining
                current_batch_seqs += 1
                remaining = 0
            else:
                # Split: fill current batch, carry remainder
                take = space_left
                remaining -= take
                current_batch_seqs += 1
                print(
                    f"  Batch {batch_num}: {current_batch_tokens + take:>6d} tokens, "
                    f"{current_batch_seqs} seqs (sample {i} split, {remaining} remaining)"
                )
                batch_num += 1
                current_batch_tokens = 0
                current_batch_seqs = 0

            if current_batch_tokens == max_tokens:
                print(f"  Batch {batch_num}: {current_batch_tokens:>6d} tokens, {current_batch_seqs} seqs")
                batch_num += 1
                current_batch_tokens = 0
                current_batch_seqs = 0

    if current_batch_tokens > 0 and batch_num < 10:
        print(f"  Batch {batch_num}: {current_batch_tokens:>6d} tokens, {current_batch_seqs} seqs (partial)")


if __name__ == "__main__":
    main()
