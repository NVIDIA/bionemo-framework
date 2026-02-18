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

"""Compare THD batches from HF streaming vs ShardedEden dataloaders.

Prints side-by-side batch statistics so you can verify both paths produce
the same structure and token counts.  No GPU or distributed init required.

Usage (from the llama3_native_te directory):
    python scripts/compare_thd_batches.py \
        --tokenizer ./tokenizers/nucleotide_fast_tokenizer \
        --hf-data-path /data/opengenome2/json/pretraining_or_both_phases/metagenomes/ \
        --eden-sequence-db-dir /data/bcr_eden/OG2_database_splits/ \
        --eden-window-db /data/bcr_eden/OG2_database_splits/og2__train__short.sqlite \
        --num-batches 5
"""

import argparse
import sys


def describe_batch(batch: dict, label: str) -> dict:
    """Print and return key statistics for a single THD batch."""
    input_ids = batch["input_ids"]
    total_tokens = input_ids.numel()

    attn_mask = batch.get("attention_mask")
    if attn_mask is not None:
        unpadded = int(attn_mask.sum().item())
    else:
        unpadded = total_tokens

    cu_seq_lens = batch.get("cu_seq_lens_q")
    if cu_seq_lens is not None:
        num_seqs = len(cu_seq_lens) - 1
        seq_lens = (cu_seq_lens[1:] - cu_seq_lens[:-1]).tolist()
        min_seq = min(seq_lens)
        max_seq = max(seq_lens)
        mean_seq = sum(seq_lens) / len(seq_lens)
    else:
        num_seqs = input_ids.shape[0]
        seq_lens = []
        min_seq = max_seq = mean_seq = input_ids.shape[1] if input_ids.ndim > 1 else total_tokens

    labels = batch.get("labels")
    if labels is not None:
        num_label_ignore = int((labels == -100).sum().item())
        num_label_real = int((labels != -100).sum().item())
    else:
        num_label_ignore = num_label_real = -1

    stats = {
        "total_tokens": total_tokens,
        "unpadded_tokens": unpadded,
        "padding_tokens": total_tokens - unpadded,
        "num_sequences": num_seqs,
        "min_seq_len": min_seq,
        "max_seq_len": max_seq,
        "mean_seq_len": round(mean_seq, 1),
        "labels_real": num_label_real,
        "labels_ignored": num_label_ignore,
        "input_ids_shape": list(input_ids.shape),
    }

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")
    if seq_lens and len(seq_lens) <= 20:
        print(f"  {'per-seq lengths':25s}: {seq_lens}")

    return stats


def create_hf_streaming_thd_dataloader(tokenizer_path, hf_data_path, seq_length, token_mbs, num_workers):
    """Create the HF streaming THD dataloader (single-process, no dist)."""
    sys.path.insert(0, ".")
    from dataset import create_thd_dataloader
    from distributed_config import DistributedConfig

    dist_config = DistributedConfig(rank=0, local_rank=0, world_size=1)
    load_dataset_kwargs = {"path": hf_data_path, "split": "train", "streaming": True}

    dl, _ = create_thd_dataloader(
        distributed_config=dist_config,
        tokenizer_name_or_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        token_micro_batch_size=token_mbs,
        num_workers=num_workers,
        max_seq_length=seq_length,
        stride=200,  # HF stride = overlap in tokens; 200 overlap → step of 7992 (matches Eden)
    )
    return dl


def create_eden_thd_dataloader(tokenizer_path, sequence_db_dir, window_db, seq_length, token_mbs, num_workers):
    """Create the ShardedEden THD dataloader (single-process, no dist)."""
    sys.path.insert(0, ".")
    from distributed_config import DistributedConfig
    from sharded_eden_dataset import create_sharded_eden_thd_dataloader

    dist_config = DistributedConfig(rank=0, local_rank=0, world_size=1)

    dl, _ = create_sharded_eden_thd_dataloader(
        dist_config=dist_config,
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_db,
        tokenizer_name_or_path=tokenizer_path,
        seq_length=seq_length,
        stride=7992,
        token_micro_batch_size=token_mbs,
        num_workers=num_workers,
    )
    return dl


def main():
    """Compare THD batches from HF streaming vs ShardedEden dataloaders."""
    parser = argparse.ArgumentParser(description="Compare THD batches from HF streaming vs ShardedEden")
    parser.add_argument("--tokenizer", required=True, help="Path to HF tokenizer")
    parser.add_argument("--hf-data-path", default=None, help="Path to HF streaming data (skip if not available)")
    parser.add_argument("--eden-sequence-db-dir", default=None, help="Path to Eden sequence DB dir")
    parser.add_argument("--eden-window-db", default=None, help="Path to Eden window DB")
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Equivalent MBS for token budget")
    parser.add_argument("--num-batches", type=int, default=5, help="Number of batches to inspect")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 = main process)")
    args = parser.parse_args()

    token_mbs = args.micro_batch_size * args.seq_length
    print(f"\nToken budget per batch: {token_mbs} (mbs={args.micro_batch_size} x seq={args.seq_length})")
    print(f"Inspecting {args.num_batches} batches from each dataloader\n")

    # ---- HF Streaming THD ----
    if args.hf_data_path:
        print("\n" + "#" * 60)
        print("# HF STREAMING THD DATALOADER")
        print("#" * 60)
        hf_dl = create_hf_streaming_thd_dataloader(
            args.tokenizer, args.hf_data_path, args.seq_length, token_mbs, args.num_workers
        )
        hf_stats = []
        for i, batch in enumerate(hf_dl):
            if i >= args.num_batches:
                break
            stats = describe_batch(batch, f"HF Streaming THD - Batch {i}")
            hf_stats.append(stats)
    else:
        print("\nSkipping HF streaming (--hf-data-path not provided)")
        hf_stats = []

    # ---- ShardedEden THD ----
    if args.eden_sequence_db_dir and args.eden_window_db:
        print("\n" + "#" * 60)
        print("# SHARDED EDEN THD DATALOADER")
        print("#" * 60)
        eden_dl = create_eden_thd_dataloader(
            args.tokenizer,
            args.eden_sequence_db_dir,
            args.eden_window_db,
            args.seq_length,
            token_mbs,
            args.num_workers,
        )
        eden_stats = []
        for i, batch in enumerate(eden_dl):
            if i >= args.num_batches:
                break
            stats = describe_batch(batch, f"ShardedEden THD - Batch {i}")
            eden_stats.append(stats)
    else:
        print("\nSkipping ShardedEden (--eden-sequence-db-dir / --eden-window-db not provided)")
        eden_stats = []

    # ---- Summary comparison ----
    if hf_stats and eden_stats:
        print("\n" + "#" * 60)
        print("# SIDE-BY-SIDE SUMMARY")
        print("#" * 60)
        header = f"{'Metric':30s} | {'HF Streaming':>15s} | {'ShardedEden':>15s} | {'Match?':>6s}"
        print(header)
        print("-" * len(header))

        for key in [
            "total_tokens",
            "unpadded_tokens",
            "padding_tokens",
            "num_sequences",
            "min_seq_len",
            "max_seq_len",
        ]:
            hf_vals = [s[key] for s in hf_stats]
            eden_vals = [s[key] for s in eden_stats]
            hf_avg = sum(hf_vals) / len(hf_vals)
            eden_avg = sum(eden_vals) / len(eden_vals)
            match = "YES" if abs(hf_avg - eden_avg) < 1 else "no"
            print(f"  {key:28s} | {hf_avg:15.1f} | {eden_avg:15.1f} | {match:>6s}")

        print()
        print("Key things to check:")
        print("  1. total_tokens should equal token_micro_batch_size for both (with split_samples=True)")
        print("  2. unpadded_tokens should equal total_tokens for both (THD = no padding)")
        print("  3. padding_tokens should be 0 for both")
        print("  4. num_sequences will differ (different data, different window lengths)")
        print("  5. input_ids shape should be [1, token_micro_batch_size] for both")


if __name__ == "__main__":
    main()
