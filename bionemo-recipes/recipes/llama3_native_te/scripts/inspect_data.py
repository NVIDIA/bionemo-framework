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

"""Inspect parquet shard data and tokenized batches for sanity checking.

Run on lepton or wherever the data is mounted:
    python scripts/inspect_data.py --data-path /data/opengenome2/parquet_split --num-batches 3
"""

import argparse

import datasets
from transformers import AutoTokenizer


def main():
    """Load parquet shards and print raw + tokenized batch details for sanity checking."""
    parser = argparse.ArgumentParser(description="Inspect parquet shards and tokenized batches")
    parser.add_argument("--data-path", type=str, default="/data/opengenome2/parquet_split")
    parser.add_argument("--tokenizer", type=str, default="./tokenizers/nucleotide_fast_tokenizer")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--num-batches", type=int, default=3, help="Number of raw examples to inspect")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    # 1. Load raw dataset and inspect
    print("=" * 80)
    print(f"Loading dataset from: {args.data_path}")
    print("=" * 80)
    ds = datasets.load_dataset(path=args.data_path, split=args.split, streaming=True)

    print(f"\nDataset features: {ds.features}")
    print(f"Number of shards: {ds.num_shards}")

    print(f"\n--- First {args.num_batches} raw examples ---")
    for i, example in enumerate(ds):
        if i >= args.num_batches:
            break
        text = example.get("text", "")
        print(f"\nExample {i}:")
        print(f"  Keys: {list(example.keys())}")
        print(f"  Text length (chars): {len(text)}")
        print(f"  Text preview (first 100 chars): {text[:100]}")
        print(f"  Text preview (last 50 chars):  ...{text[-50:]}")
        # Check for degenerate bases
        unique_chars = set(text.upper())
        standard = {"A", "C", "G", "T"}
        degenerate = unique_chars - standard - {"\n", " "}
        if degenerate:
            print(f"  Degenerate bases found: {degenerate}")
        else:
            print("  All bases standard (ACGT only)")

    # 2. Tokenize and inspect
    print("\n" + "=" * 80)
    print(f"Tokenizing with: {args.tokenizer}")
    print(f"  max_seq_length={args.max_seq_length}, stride={args.stride}")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    # Re-load to iterate fresh
    ds = datasets.load_dataset(path=args.data_path, split=args.split, streaming=True)

    print(f"\n--- Tokenized examples (first {args.num_batches}) ---")
    window_counts = []
    for i, example in enumerate(ds):
        if i >= args.num_batches:
            break
        text = example.get("text", "")
        result = tokenizer(
            [text],
            max_length=args.max_seq_length,
            stride=args.stride,
            truncation=True,
            return_overflowing_tokens=True,
            add_special_tokens=True,
        )
        num_windows = len(result["input_ids"])
        window_counts.append(num_windows)

        print(f"\nExample {i}: text_len={len(text)} chars -> {num_windows} window(s)")
        for w, ids in enumerate(result["input_ids"]):
            print(f"  Window {w}: {len(ids)} tokens")
            print(f"    First 10 token IDs: {ids[:10]}")
            print(f"    Last 10 token IDs:  {ids[-10:]}")
            print(f"    Starts with BOS ({tokenizer.bos_token_id}): {ids[0] == tokenizer.bos_token_id}")
            print(f"    Ends with EOS ({tokenizer.eos_token_id}): {ids[-1] == tokenizer.eos_token_id}")
            # Decode a small snippet
            decoded = tokenizer.decode(ids[1:11], skip_special_tokens=False)
            print(f"    Decoded tokens[1:11]: {decoded}")

    # 3. Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Inspected {args.num_batches} examples")
    print(f"Windows per example: {window_counts}")
    if all(w == 1 for w in window_counts):
        print("All examples produce exactly 1 window (pre-chunked data confirmed)")
    else:
        print("WARNING: Some examples produce multiple windows!")


if __name__ == "__main__":
    main()
