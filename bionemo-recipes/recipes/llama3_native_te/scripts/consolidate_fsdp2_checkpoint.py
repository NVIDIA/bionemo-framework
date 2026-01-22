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

"""Consolidate FSDP2 distributed checkpoint shards into a single file.

FSDP2 saves checkpoints as distributed .distcp files. This script consolidates
them into a single model.safetensors file by reading all shards and reconstructing
the full tensors.

Skips _extra_state (Transformer Engine FP8 state) which isn't needed for inference.

Usage:
    python consolidate_fsdp2_checkpoint.py \
        --checkpoint-dir /path/to/step_5000 \
        --output-path /path/to/consolidated/model.safetensors

Note: Does NOT require torchrun - runs as a regular Python script.
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def consolidate_sharded_checkpoint(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Read all shards from a DCP checkpoint and reconstruct full tensors.

    This bypasses dcp.load() which has strict sharding requirements.
    Instead, we read the metadata and manually reconstruct each tensor
    from its shards across all .distcp files.

    Args:
        checkpoint_dir: Path to checkpoint directory containing .distcp files

    Returns:
        Dictionary mapping tensor names to full (unsharded) tensors
    """
    reader = FileSystemReader(str(checkpoint_dir))
    metadata = reader.read_metadata()

    logger.info(f"Checkpoint contains {len(metadata.state_dict_metadata)} entries")

    # Filter to only model weights (skip optimizer, scheduler, _extra_state)
    model_keys = []
    for key in metadata.state_dict_metadata.keys():
        # Skip non-model entries
        if "optim" in key or "scheduler" in key:
            continue
        # Skip TE extra state (FP8 scaling factors - not needed for inference)
        if "_extra_state" in key:
            continue
        # Skip scalar metadata
        if key in ["app.step", "app.epoch", "step", "epoch"]:
            continue
        model_keys.append(key)

    logger.info(f"Found {len(model_keys)} model weight tensors to consolidate")

    # Build consolidated state dict
    consolidated = {}

    for key in model_keys:
        tensor_meta = metadata.state_dict_metadata[key]

        if not isinstance(tensor_meta, TensorStorageMetadata):
            logger.debug(f"Skipping non-tensor {key}: {type(tensor_meta)}")
            continue

        # Get full tensor shape and dtype
        full_shape = tensor_meta.size
        # Default to bfloat16 for model weights
        dtype = torch.bfloat16 if tensor_meta.properties.dtype is None else tensor_meta.properties.dtype

        # Create empty tensor to fill with shards
        full_tensor = torch.zeros(full_shape, dtype=dtype)

        # Read all chunks/shards for this tensor
        for chunk in tensor_meta.chunks:
            # Each chunk has offsets and sizes
            offsets = chunk.offsets
            sizes = chunk.sizes

            # Read this chunk from the checkpoint
            # FileSystemReader.read_data expects a list of (fqn, storage_meta) tuples
            try:
                # Build slice for where this chunk goes in the full tensor
                slices = tuple(slice(o, o + s) for o, s in zip(offsets, sizes))

                # Read the chunk data - we need to use the internal reader
                # The chunk data is stored in files named by the storage key
                storage_key = chunk.storage_key if hasattr(chunk, "storage_key") else None

                if storage_key:
                    # Find which .distcp file contains this chunk
                    for storage_md in metadata.storage_data:
                        if storage_md.storage_key == storage_key:
                            # Read from file
                            chunk_data = reader._read_item(storage_md)
                            full_tensor[slices] = chunk_data
                            break
            except Exception as e:
                logger.warning(f"Failed to read chunk for {key}: {e}")
                continue

        # Clean up the key name (remove "app.model." prefix)
        clean_key = key
        if clean_key.startswith("app.model."):
            clean_key = clean_key[len("app.model.") :]
        elif clean_key.startswith("app."):
            clean_key = clean_key[len("app.") :]

        consolidated[clean_key] = full_tensor
        logger.debug(f"Consolidated {clean_key}: {full_tensor.shape}")

    return consolidated


def consolidate_via_torch_load(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Alternative approach: directly load .distcp files with torch.load.

    Each .distcp file contains a pickled dict of tensor shards.
    We merge all shards to reconstruct full tensors.
    """
    distcp_files = sorted(checkpoint_dir.glob("*.distcp"))
    logger.info(f"Loading {len(distcp_files)} .distcp files...")

    # Collect all shards by tensor name
    shards_by_key: dict[str, list] = defaultdict(list)

    for distcp_file in distcp_files:
        try:
            # Load the shard file
            shard_data = torch.load(distcp_file, map_location="cpu", weights_only=False)

            # Each file contains partial tensors
            if isinstance(shard_data, dict):
                for key, value in shard_data.items():
                    # Skip non-model entries
                    if "optim" in key or "scheduler" in key:
                        continue
                    if "_extra_state" in key:
                        continue
                    if key in ["step", "epoch"]:
                        if key == "step":
                            logger.info(f"Checkpoint step: {value}")
                        continue

                    if isinstance(value, torch.Tensor):
                        shards_by_key[key].append(value)
        except Exception as e:
            logger.warning(f"Failed to load {distcp_file}: {e}")

    logger.info(f"Found shards for {len(shards_by_key)} unique tensors")

    # Merge shards (for FSDP, tensors are sharded along dim 0)
    consolidated = {}
    for key, shards in shards_by_key.items():
        if len(shards) == 1:
            full_tensor = shards[0]
        else:
            # FSDP2 shards on dim 0, concatenate
            try:
                full_tensor = torch.cat(shards, dim=0)
            except Exception:
                # If cat fails, just use first shard
                logger.warning(f"Failed to merge shards for {key}, using first shard")
                full_tensor = shards[0]

        # Clean up key name
        clean_key = key
        if clean_key.startswith("app.model."):
            clean_key = clean_key[len("app.model.") :]
        elif clean_key.startswith("app."):
            clean_key = clean_key[len("app.") :]

        consolidated[clean_key] = full_tensor

    return consolidated


def main():
    """Consolidate FSDP2 checkpoint shards into a single file."""
    parser = argparse.ArgumentParser(description="Consolidate FSDP2 distributed checkpoint")
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True, help="Path to checkpoint directory with .distcp files"
    )
    parser.add_argument("--output-path", type=str, required=True, help="Output path for consolidated checkpoint")
    parser.add_argument("--format", choices=["safetensors", "pt"], default="safetensors", help="Output format")
    parser.add_argument(
        "--method",
        choices=["metadata", "direct"],
        default="direct",
        help="Method: 'metadata' uses DCP metadata, 'direct' loads .distcp files directly",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_dir)
    output_path = Path(args.output_path)

    # Verify checkpoint exists
    distcp_files = list(checkpoint_path.glob("*.distcp"))
    if not distcp_files:
        raise FileNotFoundError(f"No .distcp files found in {checkpoint_path}")
    logger.info(f"Found {len(distcp_files)} .distcp shard files")

    # Consolidate using selected method
    if args.method == "metadata":
        logger.info("Using metadata-based consolidation...")
        consolidated = consolidate_sharded_checkpoint(checkpoint_path)
    else:
        logger.info("Using direct .distcp file loading...")
        consolidated = consolidate_via_torch_load(checkpoint_path)

    if not consolidated:
        raise RuntimeError("No tensors were consolidated. Check the checkpoint format.")

    logger.info(f"Consolidated {len(consolidated)} tensors")

    # Save consolidated checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "safetensors":
        from safetensors.torch import save_file

        # Convert to contiguous tensors
        clean_state_dict = {}
        for k, v in consolidated.items():
            if isinstance(v, torch.Tensor):
                clean_state_dict[k] = v.contiguous()

        save_file(clean_state_dict, output_path)
        logger.info(f"Saved consolidated checkpoint to {output_path} (safetensors format)")
    else:
        torch.save(consolidated, output_path)
        logger.info(f"Saved consolidated checkpoint to {output_path} (pytorch format)")

    # Print checkpoint info
    num_params = sum(p.numel() for p in consolidated.values() if isinstance(p, torch.Tensor))
    size_mb = sum(p.numel() * p.element_size() for p in consolidated.values() if isinstance(p, torch.Tensor)) / 1024**2
    logger.info(f"Checkpoint contains {num_params:,} parameters ({size_mb:.1f} MB)")

    # List some keys for verification
    logger.info("Sample keys in consolidated checkpoint:")
    for i, key in enumerate(sorted(consolidated.keys())[:10]):
        tensor = consolidated[key]
        logger.info(f"  {key}: {tensor.shape} {tensor.dtype}")
    if len(consolidated) > 10:
        logger.info(f"  ... and {len(consolidated) - 10} more")


if __name__ == "__main__":
    main()
