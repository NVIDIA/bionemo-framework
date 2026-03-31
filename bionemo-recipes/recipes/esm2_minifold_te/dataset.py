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

"""Dataset for protein structure prediction training.

Provides:
- StructureDataset: loads protein structures from parquet/PDB files
- SyntheticStructureDataset: generates random data for testing
- create_dataloader: creates a DataLoader for training
"""

import logging

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)


class SyntheticStructureDataset(Dataset):
    """Generates synthetic protein structure data for testing.

    Each sample contains:
        input_ids: Random ESM-2 token IDs (L,)
        attention_mask: All ones (L,)
        mask: All ones (L,) as float
        coords: Random Ca coordinates (L, 3)
    """

    def __init__(self, num_samples: int = 1000, max_seq_length: int = 128, seed: int = 42):
        self.num_samples = num_samples
        self.max_seq_length = max_seq_length
        self.rng = torch.Generator().manual_seed(seed)

        # ESM-2 special tokens: 0=cls, 1=pad, 2=eos, 3=unk, then 4-23 are AA tokens
        self.vocab_start = 4
        self.vocab_end = 24  # 20 amino acid tokens

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random sequence length (at least 16 residues)
        seq_len = torch.randint(16, self.max_seq_length - 2, (1,), generator=self.rng).item()

        # Generate random tokens: [CLS] + AA tokens + [EOS] + padding
        tokens = torch.randint(self.vocab_start, self.vocab_end, (seq_len,), generator=self.rng)
        input_ids = torch.zeros(self.max_seq_length, dtype=torch.long)
        input_ids[0] = 0  # CLS
        input_ids[1 : seq_len + 1] = tokens
        input_ids[seq_len + 1] = 2  # EOS

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = torch.zeros(self.max_seq_length, dtype=torch.long)
        attention_mask[: seq_len + 2] = 1  # CLS + tokens + EOS

        # Residue mask: 1 for real residues (excludes special tokens)
        mask = torch.zeros(self.max_seq_length, dtype=torch.float)
        mask[: seq_len + 2] = 1.0

        # Random Ca coordinates (Angstroms)
        coords = torch.randn(self.max_seq_length, 3) * 10.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mask": mask,
            "coords": coords,
        }


class ParquetStructureDataset(Dataset):
    """Loads protein structures from a parquet file.

    Expected columns:
        sequence: str - amino acid sequence
        coords: list[list[float]] - Ca coordinates (N, 3)
    """

    def __init__(self, parquet_path: str, tokenizer, max_seq_length: int = 256):
        import pandas as pd

        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = row["sequence"]

        # Tokenize
        encoded = self.tokenizer(
            sequence,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Mask is same as attention_mask but as float
        mask = attention_mask.float()

        # Coordinates: pad to max_seq_length
        coords_raw = torch.tensor(row["coords"], dtype=torch.float32)
        coords = torch.zeros(self.max_seq_length, 3)
        seq_len = min(len(coords_raw), self.max_seq_length)
        coords[:seq_len] = coords_raw[:seq_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mask": mask,
            "coords": coords,
        }


def create_dataloader(
    dist_config: DistributedConfig,
    micro_batch_size: int = 2,
    max_seq_length: int = 128,
    num_workers: int = 0,
    dataset_type: str = "synthetic",
    parquet_path: str | None = None,
    tokenizer_name: str | None = None,
    num_samples: int = 1000,
    **kwargs,
):
    """Create a DataLoader for structure prediction training.

    Args:
        dist_config: Distributed training configuration.
        micro_batch_size: Batch size per GPU.
        max_seq_length: Maximum sequence length.
        num_workers: Number of DataLoader workers.
        dataset_type: "synthetic" or "parquet".
        parquet_path: Path to parquet file (required if dataset_type="parquet").
        tokenizer_name: HuggingFace tokenizer name (required if dataset_type="parquet").
        num_samples: Number of synthetic samples.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        Tuple of (DataLoader, DistributedSampler).
    """
    if dataset_type == "synthetic":
        dataset = SyntheticStructureDataset(
            num_samples=num_samples,
            max_seq_length=max_seq_length,
        )
    elif dataset_type == "parquet":
        from transformers import EsmTokenizer

        tokenizer = EsmTokenizer.from_pretrained(tokenizer_name)
        dataset = ParquetStructureDataset(
            parquet_path=parquet_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist_config.world_size,
        rank=dist_config.rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader, sampler
