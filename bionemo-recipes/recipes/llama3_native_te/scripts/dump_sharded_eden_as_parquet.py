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

"""Dump data from John's ShardedEdenDataset as Parquet files for HuggingFace datasets.

This script saves data in a format that can be loaded directly by the existing
HuggingFace dataset pipeline, using the existing collator and DistributedSampler.

The key difference from dump_sharded_eden_data.py:
- Outputs Parquet files with `input_ids` column (pre-tokenized)
- Can be loaded with `datasets.load_dataset("parquet", data_files=...)`
- Uses existing collator logic for batching
- Uses existing DistributedSampler for sharding

This approach is SAFER because it reuses the battle-tested HF pipeline.

USAGE:
======

1. Dump the data:
   python scripts/dump_sharded_eden_as_parquet.py \
       --sequence-db-dir /data/bcr_eden/OG2_database_splits/opengenome2-metagenome \
       --window-db-path /data/bcr_eden/OG2_database_splits/og2__train__short.sqlite \
       --output-dir /data/sharded_eden_parquet \
       --num-samples 3072000 \
       --seed 42

2. Train using the parquet files:
   python train_fsdp2.py --config-name L2_og2_parquet

   With config:
       dataset:
           load_dataset_kwargs:
               path: parquet
               data_files: /data/sharded_eden_parquet/*.parquet
               split: train
               streaming: false  # CRITICAL: disable streaming
           shuffle: false  # CRITICAL: preserve order
           skip_windowing: true  # Data is already windowed
           skip_tokenization: true  # Data is already tokenized
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# Add paths for imports
EVO2_RECIPE_PATH = Path(__file__).parent.parent.parent / "evo2_megatron" / "src"
EVO2_SUBPKG_PATH = Path(__file__).parent.parent.parent.parent.parent / "sub-packages" / "bionemo-evo2" / "src"
CORE_SUBPKG_PATH = Path(__file__).parent.parent.parent.parent.parent / "sub-packages" / "bionemo-core" / "src"
sys.path.insert(0, str(EVO2_RECIPE_PATH))
sys.path.insert(0, str(EVO2_SUBPKG_PATH))
sys.path.insert(0, str(CORE_SUBPKG_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def permute(index: int, length: int, seed: int) -> int:
    """Index into a permuted array with constant space and time complexity.

    This is a copy of bionemo.core.data.permute.permute() for standalone use.
    """
    import warnings

    if length <= 1:
        raise ValueError("The length of the permuted range must be greater than 1.")

    if index not in range(length):
        raise ValueError("The index to permute must be in the range [0, l).")

    if seed < 0:
        raise ValueError("The permutation seed must be greater than or equal to 0.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        w = length - 1
        w |= w >> 1
        w |= w >> 2
        w |= w >> 4
        w |= w >> 8
        w |= w >> 16

        while True:
            index ^= seed
            index *= 0xE170893D
            index ^= seed >> 16
            index ^= (index & w) >> 4
            index ^= seed >> 8
            index *= 0x0929EB3F
            index ^= seed >> 23
            index ^= (index & w) >> 1
            index *= 1 | seed >> 27
            index *= 0x6935FA69
            index ^= (index & w) >> 11
            index *= 0x74DCB303
            index ^= (index & w) >> 2
            index *= 0x9E501CC3
            index ^= (index & w) >> 2
            index *= 0xC860A3DF
            index &= w
            if index < length:
                break

    return (index + seed) % length


class MegatronTokenizerAdapter:
    """Adapts HuggingFace tokenizer to Megatron's tokenizer interface."""

    def __init__(self, hf_tokenizer):
        """Initialize the adapter with a HuggingFace tokenizer."""
        self.hf_tokenizer = hf_tokenizer
        self.bos_id = hf_tokenizer.bos_token_id or 0
        self.eos_id = hf_tokenizer.eos_token_id or 1
        self.pad_id = hf_tokenizer.pad_token_id or hf_tokenizer.eos_token_id or 0
        self._sep_id = hf_tokenizer.sep_token_id or self.eos_id
        self.eod = self.eos_id
        self.vocab_size = hf_tokenizer.vocab_size

    def text_to_ids(self, text: str) -> list[int]:
        """Convert text to token IDs without special tokens."""
        return self.hf_tokenizer.encode(text, add_special_tokens=False)


def compute_epoch_seeds(base_seed: int, num_epochs: int) -> np.ndarray:
    """Compute epoch seeds exactly like MultiEpochDatasetResampler does."""
    rng = np.random.default_rng(base_seed)
    return rng.integers(0, np.iinfo(np.int32).max, size=num_epochs)


def global_index_to_permuted_local_index(
    global_idx: int, dataset_len: int, epoch_seeds: np.ndarray, shuffle: bool = True
) -> tuple[int, int]:
    """Convert global index to (epoch, permuted_idx) exactly like MultiEpochDatasetResampler."""
    epoch = global_idx // dataset_len
    idx = global_idx % dataset_len
    if shuffle:
        idx = permute(idx, dataset_len, int(epoch_seeds[epoch]))
    return epoch, idx


def dump_as_parquet(
    sequence_db_dir: str,
    window_db_path: str,
    output_dir: str,
    tokenizer_path: str,
    num_samples: int,
    seq_length: int = 8192,
    stride: int = 7992,
    seed: int = 42,
    shuffle: bool = True,
    samples_per_file: int = 10000,
):
    """Dump samples from ShardedEdenDataset as Parquet files.

    Creates parquet files with `input_ids` column that can be loaded
    by HuggingFace datasets.

    Args:
        sequence_db_dir: Directory containing per-sample SQLite databases
        window_db_path: Path to the window mappings database
        output_dir: Directory to save parquet files
        tokenizer_path: Path to HuggingFace tokenizer
        num_samples: Total number of samples to dump
        seq_length: Sequence length
        stride: Stride for windowing
        seed: Random seed (must match John's dataset_seed)
        shuffle: Whether to apply shuffling (should be True)
        samples_per_file: Number of samples per parquet file
    """
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    tokenizer_adapter = MegatronTokenizerAdapter(hf_tokenizer)

    # Import ShardedEdenDataset
    try:
        from bionemo.evo2.data.sharded_eden_dataset_provider import ShardedEdenDataset

        logger.info("Loaded ShardedEdenDataset from evo2_megatron recipe")
    except ImportError:
        try:
            from bionemo.evo2.data.sharded_eden_dataloader import ShardedEdenDataset

            logger.info("Loaded ShardedEdenDataset from bionemo-evo2 sub-package")
        except ImportError as e:
            logger.error(f"Could not import ShardedEdenDataset: {e}")
            sys.exit(1)

    # Create dataset
    logger.info("Creating ShardedEdenDataset:")
    logger.info(f"  sequence_db_dir: {sequence_db_dir}")
    logger.info(f"  window_db_path: {window_db_path}")

    dataset = ShardedEdenDataset(
        tokenizer=tokenizer_adapter,
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_db_path,
        seq_length=seq_length,
        stride=stride,
        create_attention_mask=False,
        rc_aug=False,
        skip_stats=True,
    )

    dataset_len = len(dataset)
    logger.info(f"Dataset has {dataset_len} windows")

    # Compute epoch seeds
    num_epochs = math.ceil(num_samples / dataset_len)
    epoch_seeds = compute_epoch_seeds(seed, num_epochs)
    logger.info(f"Need {num_samples} samples across {num_epochs} epochs")

    # Save metadata
    metadata = {
        "sequence_db_dir": sequence_db_dir,
        "window_db_path": window_db_path,
        "seq_length": seq_length,
        "stride": stride,
        "seed": seed,
        "shuffle": shuffle,
        "num_samples": num_samples,
        "total_windows": dataset_len,
        "samples_per_file": samples_per_file,
        "pad_token_id": int(hf_tokenizer.pad_token_id),
        "bos_token_id": int(hf_tokenizer.bos_token_id or 0),
        "eos_token_id": int(hf_tokenizer.eos_token_id or 1),
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Dump samples in parquet files
    logger.info(f"Dumping {num_samples} samples to parquet files...")

    all_input_ids = []
    all_window_idx = []  # Store window_idx for comparison with John's logs
    file_idx = 0
    parquet_files = []

    for global_idx in tqdm(range(num_samples), desc="Processing samples"):
        # Get permuted index exactly like MultiEpochDatasetResampler
        epoch, permuted_idx = global_index_to_permuted_local_index(global_idx, dataset_len, epoch_seeds, shuffle)

        # Get sample from dataset
        sample = dataset[np.int64(permuted_idx)]

        # The ShardedEdenDataset returns 'tokens' which is the input_ids
        # We need to convert to a list for parquet storage
        input_ids = sample["tokens"].tolist()
        all_input_ids.append(input_ids)
        all_window_idx.append(int(permuted_idx))  # This is John's window_idx!

        # Write parquet file when we have enough samples
        if len(all_input_ids) >= samples_per_file:
            output_file = os.path.join(output_dir, f"data_{file_idx:06d}.parquet")
            table = pa.table({"input_ids": all_input_ids, "window_idx": all_window_idx})
            pq.write_table(table, output_file)
            parquet_files.append(f"data_{file_idx:06d}.parquet")
            logger.info(f"Wrote {len(all_input_ids)} samples to {output_file}")
            all_input_ids = []
            all_window_idx = []
            file_idx += 1

    # Write remaining samples
    if all_input_ids:
        output_file = os.path.join(output_dir, f"data_{file_idx:06d}.parquet")
        table = pa.table({"input_ids": all_input_ids, "window_idx": all_window_idx})
        pq.write_table(table, output_file)
        parquet_files.append(f"data_{file_idx:06d}.parquet")
        logger.info(f"Wrote {len(all_input_ids)} samples to {output_file}")

    # Save index
    with open(os.path.join(output_dir, "index.json"), "w") as f:
        json.dump({"files": parquet_files, "num_samples": num_samples}, f, indent=2)

    logger.info(f"\nDone! Created {len(parquet_files)} parquet files in {output_dir}")
    logger.info(f"Total samples: {num_samples}")
    logger.info("\nTo use in training:")
    logger.info('  load_dataset_kwargs.path: "parquet"')
    logger.info(f'  load_dataset_kwargs.data_files: "{output_dir}/*.parquet"')
    logger.info("  load_dataset_kwargs.streaming: false")
    logger.info("  shuffle: false")
    logger.info("  skip_tokenization: true")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dump ShardedEdenDataset to Parquet files for HuggingFace datasets",
    )
    parser.add_argument("--sequence-db-dir", required=True)
    parser.add_argument("--window-db-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="tokenizers/nucleotide_fast_tokenizer")
    parser.add_argument("--num-samples", type=int, required=True, help="Total samples to dump")
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=7992)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-file", type=int, default=10000)
    parser.add_argument("--no-shuffle", action="store_true")

    args = parser.parse_args()

    dump_as_parquet(
        sequence_db_dir=args.sequence_db_dir,
        window_db_path=args.window_db_path,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        stride=args.stride,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        samples_per_file=args.samples_per_file,
    )


if __name__ == "__main__":
    main()
