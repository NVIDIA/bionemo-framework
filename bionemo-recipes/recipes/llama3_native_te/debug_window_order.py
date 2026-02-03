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

"""Debug script to verify window ordering in HuggingFace streaming datasets.

This script helps you understand whether windows from a single sequence are yielded
sequentially (A0, A1, A2, B0, B1, ...) or shuffled independently (A0, B0, A1, C0, ...).

Key insight: With streaming datasets, the shuffle happens at the SEQUENCE level before
windowing. After windowing via return_overflowing_tokens, windows from the same sequence
should be yielded consecutively (unless further shuffled).

Usage:
    # Run on 8 GPUs (adjust path to your data)
    torchrun --nproc_per_node=8 debug_window_order.py

    # Single GPU test
    python debug_window_order.py
"""

import logging
import os
from collections import Counter
from dataclasses import dataclass

import datasets
import datasets.distributed
import torch
from transformers import AutoTokenizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DebugConfig:
    """Configuration for debug run."""

    # Data path (JSON streaming format)
    data_path: str = "/data/opengenome2/json/pretraining_or_both_phases/metagenomes"
    tokenizer_path: str = "./tokenizers/nucleotide_fast_tokenizer"

    # Windowing params
    max_seq_length: int = 8192
    stride: int = 7992  # Match your config

    # Shuffle params
    shuffle: bool = True
    buffer_size: int = 50_000

    # How many samples to inspect
    num_batches_to_log: int = 100
    batch_size: int = 8  # micro batch size


def get_distributed_info():
    """Get distributed training info."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def create_debug_dataset_with_tracking(config: DebugConfig, rank: int, world_size: int):
    """Create dataset with window tracking metadata.

    The key is to preserve overflow_to_sample_mapping which tells us
    which original sequence each window came from.
    """
    # Load streaming dataset
    dataset = datasets.load_dataset(
        path=config.data_path,
        split="train",
        streaming=True,
    )

    # Shard by rank
    if world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=rank)
        logger.info(f"[Rank {rank}] Sharded dataset for {world_size} GPUs")

    # Shuffle (at sequence level, before windowing!)
    if config.shuffle:
        dataset = dataset.shuffle(seed=42, buffer_size=config.buffer_size)
        logger.info(f"[Rank {rank}] Shuffle enabled with buffer_size={config.buffer_size}")
    else:
        logger.info(f"[Rank {rank}] Shuffle DISABLED - sequences in original order")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

    # Track sequence IDs for debugging
    # We'll use a counter to assign IDs to each source sequence in the batch
    batch_seq_counter = [0]  # Mutable to track across map calls

    def tokenize_with_tracking(examples):
        """Tokenize with windowing and track which sequence each window comes from."""
        batch_start_id = batch_seq_counter[0]

        result = tokenizer(
            examples["text"],
            max_length=config.max_seq_length,
            stride=config.stride,
            truncation=True,
            return_overflowing_tokens=True,
            add_special_tokens=True,
        )

        # overflow_to_sample_mapping tells us which input sequence each window came from
        # e.g., [0, 0, 0, 1, 1, 2, 2, 2, 2] means windows 0-2 are from seq 0, 3-4 from seq 1, etc.
        sample_mapping = result.get("overflow_to_sample_mapping", list(range(len(result["input_ids"]))))

        # Create global sequence IDs for tracking
        # This helps track sequences across batches
        global_seq_ids = [batch_start_id + local_id for local_id in sample_mapping]

        # Track window index within each sequence
        window_indices = []
        current_seq = -1
        current_window = 0
        for local_id in sample_mapping:
            if local_id != current_seq:
                current_seq = local_id
                current_window = 0
            window_indices.append(current_window)
            current_window += 1

        batch_seq_counter[0] += len(set(sample_mapping))

        # Add tracking info
        result["_source_seq_id"] = global_seq_ids
        result["_window_idx"] = window_indices
        result["_first_4_tokens"] = [ids[:4] for ids in result["input_ids"]]

        return result

    # Apply tokenization with tracking
    tokenized = dataset.select_columns(["text"]).map(
        tokenize_with_tracking,
        batched=True,
        batch_size=config.batch_size,
        remove_columns=["text"],
    )

    return tokenized, tokenizer


def analyze_window_order(samples: list[dict], rank: int) -> dict:
    """Analyze whether windows from same sequence appear consecutively.

    Returns stats about window ordering patterns.
    """
    if not samples:
        return {}

    # Extract sequence IDs and window indices
    seq_ids = [s["_source_seq_id"] for s in samples]
    window_idxs = [s["_window_idx"] for s in samples]
    first_tokens = [s["_first_4_tokens"] for s in samples]

    # Check for consecutive windows from same sequence
    consecutive_count = 0
    switch_count = 0
    prev_seq = None

    for i, seq_id in enumerate(seq_ids):
        if prev_seq is not None:
            if seq_id == prev_seq:
                consecutive_count += 1
            else:
                switch_count += 1
        prev_seq = seq_id

    # Count windows per sequence
    seq_counter = Counter(seq_ids)
    seqs_with_multiple_windows = sum(1 for c in seq_counter.values() if c > 1)

    # Log detailed trace of first N samples
    logger.info(f"\n{'=' * 80}")
    logger.info(f"[Rank {rank}] WINDOW ORDER ANALYSIS (first {len(samples)} samples)")
    logger.info(f"{'=' * 80}")

    # Show first 30 samples in detail
    logger.info("\nSample trace (idx | seq_id | window_idx | first_4_tokens):")
    for i, (sid, widx, ftok) in enumerate(zip(seq_ids[:30], window_idxs[:30], first_tokens[:30])):
        marker = "←NEW SEQ" if i > 0 and sid != seq_ids[i - 1] else ""
        logger.info(f"  {i:3d} | seq={sid:4d} | win={widx:2d} | tokens={ftok} {marker}")

    # Summary stats
    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY:")
    logger.info(f"  Total samples analyzed: {len(samples)}")
    logger.info(f"  Unique sequences seen: {len(seq_counter)}")
    logger.info(f"  Sequences with multiple windows: {seqs_with_multiple_windows}")
    logger.info(f"  Consecutive same-seq pairs: {consecutive_count}")
    logger.info(f"  Sequence switches: {switch_count}")
    logger.info(f"  Consecutive ratio: {consecutive_count / (consecutive_count + switch_count + 1e-9):.2%}")

    # Windows per sequence distribution
    window_counts = list(seq_counter.values())
    logger.info("\n  Windows per sequence:")
    logger.info(
        f"    min: {min(window_counts)}, max: {max(window_counts)}, avg: {sum(window_counts) / len(window_counts):.1f}"
    )

    # Pattern detection
    if consecutive_count > switch_count * 2:
        logger.info("\n  ✓ PATTERN: Windows appear MOSTLY SEQUENTIAL within sequences")
        logger.info("    (consecutive pairs >> switches)")
    else:
        logger.info("\n  ✗ PATTERN: Windows appear SHUFFLED across sequences")
        logger.info("    (similar number of consecutive pairs and switches)")

    logger.info(f"{'=' * 80}\n")

    return {
        "total_samples": len(samples),
        "unique_sequences": len(seq_counter),
        "consecutive_pairs": consecutive_count,
        "switch_count": switch_count,
        "consecutive_ratio": consecutive_count / (consecutive_count + switch_count + 1e-9),
    }


def main():
    """Run window order debugging."""
    rank, world_size, local_rank = get_distributed_info()

    # Initialize distributed if multi-GPU
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")

    config = DebugConfig()

    logger.info(f"[Rank {rank}/{world_size}] Starting window order debug")
    logger.info(f"  Data path: {config.data_path}")
    logger.info(f"  max_seq_length: {config.max_seq_length}")
    logger.info(f"  stride: {config.stride}")
    logger.info(f"  shuffle: {config.shuffle}")
    logger.info(f"  buffer_size: {config.buffer_size}")

    # Create dataset with tracking
    dataset, tokenizer = create_debug_dataset_with_tracking(config, rank, world_size)

    # Collect samples
    samples = []
    for i, sample in enumerate(dataset):
        samples.append(sample)
        if i >= config.num_batches_to_log * config.batch_size:
            break

    # Analyze window ordering
    _ = analyze_window_order(samples, rank)

    # Log tokens if you want to verify the actual content
    if rank == 0 and len(samples) > 0:
        logger.info("\n[Rank 0] First 5 samples decoded:")
        for i, s in enumerate(samples[:5]):
            tokens = s["input_ids"][:20]
            decoded = tokenizer.decode(tokens)
            logger.info(f"  Sample {i}: {decoded[:50]}...")

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
