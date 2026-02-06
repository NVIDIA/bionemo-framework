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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Dataset and dataloader for reading pre-dumped tensor files from Megatron training.

This module enables exact data matching between Megatron and HuggingFace training by:
1. Reading tensor files dumped from Megatron's gpt_data_step
2. Converting Megatron format to HuggingFace format
3. Preserving exact data order for debugging

Usage:
    # In John's Megatron training, set env vars to dump:
    export DUMP_BATCHES=1
    export DUMP_BATCHES_DIR=/data/megatron_batches

    # Then in HuggingFace training, read the dumped tensors:
    dataloader = create_tensor_dataloader(
        tensor_dir="/data/megatron_batches",
        ...
    )
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)


class DumpedTensorDataset(Dataset):
    """Dataset that reads pre-saved tensor batches from disk.

    Each tensor file contains a single sample with:
    - tokens: [seq_length] - input token IDs
    - labels: [seq_length] - target labels (already shifted by Megatron)
    - loss_mask: [seq_length] - 1 for positions to include in loss, 0 otherwise
    - position_ids: [seq_length] - position indices (optional)

    The dataset converts to HuggingFace format:
    - input_ids: same as tokens
    - labels: converted with -100 for ignored positions
    - attention_mask: created from padding detection

    When rank is specified, only reads files for that rank: rank_{rank}_sample_*.pt
    This matches the per-rank dump format from Megatron.
    """

    def __init__(
        self,
        tensor_dir: str,
        pad_id: int = 1,
        convert_labels: bool = True,
        rank: int | None = None,
    ):
        """Initialize the dataset.

        Args:
            tensor_dir: Directory containing tensor files
            pad_id: Padding token ID (used for attention_mask creation)
            convert_labels: Whether to convert labels to HF format (-100 for ignored)
            rank: If specified, only load files for this rank (rank_{rank}_sample_*.pt).
                  If None, loads all sample_*.pt files (legacy format).
        """
        self.tensor_dir = Path(tensor_dir)
        self.pad_id = pad_id
        self.convert_labels = convert_labels
        self.rank = rank

        # Discover tensor files (per-rank or all)
        if rank is not None:
            file_pattern = f"rank_{rank}_sample_*.pt"
        else:
            file_pattern = "sample_*.pt"

        self.tensor_files = sorted(self.tensor_dir.glob(file_pattern))

        if len(self.tensor_files) == 0:
            raise ValueError(f"No tensor files found in {tensor_dir} matching {file_pattern}")

        logger.info(f"DumpedTensorDataset: Found {len(self.tensor_files)} files for rank={rank} in {tensor_dir}")

        # Log first few files for verification
        for i, f in enumerate(self.tensor_files[:3]):
            logger.info(f"  File {i}: {f.name}")

    def __len__(self):
        """Return number of samples."""
        return len(self.tensor_files)

    def __getitem__(self, idx):
        """Load a tensor file and convert to HuggingFace format."""
        tensor_path = self.tensor_files[idx]
        data = torch.load(tensor_path, map_location="cpu")

        tokens = data["tokens"]
        labels = data["labels"]
        loss_mask = data.get("loss_mask")
        position_ids = data.get("position_ids")

        # Convert labels to HuggingFace format if requested
        if self.convert_labels and loss_mask is not None:
            # Clone to avoid modifying cached data
            labels = labels.clone()
            # Set ignored positions to -100 (HF's ignore index)
            labels[loss_mask == 0] = -100

        # Create attention mask in HuggingFace extended format
        # 0 = attend, -10000 = masked (padding)
        padding_mask = tokens == self.pad_id
        attention_mask = padding_mask.float() * -10000.0

        result = {
            "input_ids": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # Include loss_mask for logging/debugging
        if loss_mask is not None:
            result["loss_mask"] = loss_mask

        # Include position_ids if available
        if position_ids is not None:
            result["position_ids"] = position_ids

        return result


def create_tensor_dataloader(
    distributed_config: DistributedConfig,
    tensor_dir: str,
    micro_batch_size: int = 1,
    grad_acc_steps: int = 1,
    pad_id: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    seed: int = 42,
    log_sequences: bool = False,
    log_dir: str | None = None,
):
    """Create a dataloader that reads pre-dumped tensor files.

    Each rank reads only its own files: rank_{rank}_sample_*.pt
    This matches the per-rank dump format from Megatron, ensuring exact data matching.

    Args:
        distributed_config: Distributed training configuration
        tensor_dir: Directory containing tensor files from Megatron dump
        micro_batch_size: Batch size per GPU
        grad_acc_steps: Gradient accumulation steps (for logging info)
        pad_id: Padding token ID
        num_workers: DataLoader workers
        shuffle: Whether to shuffle (default False to preserve Megatron order)
        seed: Random seed for shuffling
        log_sequences: Whether to log sequences (for debugging)
        log_dir: Directory for sequence logs

    Returns:
        Tuple of (dataloader, dataset) - Note: no sampler needed since each rank has its own files
    """
    # Each rank creates a dataset with ONLY its files
    # This ensures exact matching with Megatron's per-rank data
    dataset = DumpedTensorDataset(
        tensor_dir=tensor_dir,
        pad_id=pad_id,
        convert_labels=True,
        rank=distributed_config.rank,  # Only load files for this rank
    )

    # No DistributedSampler needed - each rank already has its own files!
    # Just use a regular DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=shuffle,  # Usually False to preserve dump order
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # Calculate effective batch info
    gbs = distributed_config.world_size * micro_batch_size * grad_acc_steps

    logger.info("Created per-rank tensor DataLoader:")
    logger.info(f"  Tensor dir: {tensor_dir}")
    logger.info(f"  Rank {distributed_config.rank}: {len(dataset)} samples")
    logger.info(f"  Batches per GPU: {len(dataloader)}")
    logger.info(f"  MBS={micro_batch_size}, GA={grad_acc_steps}, GPUs={distributed_config.world_size} → GBS={gbs}")
    logger.info(f"  Shuffle: {shuffle}")

    if log_sequences and log_dir:
        logger.info(f"  Sequence logging enabled: {log_dir}")

    # Return dataset instead of sampler (no sampler needed)
    return dataloader, dataset
