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
them into a single model.safetensors file using PyTorch's DCP APIs.

Based on: https://docs.ray.io/en/master/train/user-guides/fsdp.html

Usage:
    torchrun --nproc_per_node=1 consolidate_fsdp2_checkpoint.py \
        --checkpoint-dir /path/to/step_5000 \
        --output-path /path/to/consolidated/model.safetensors \
        --config-name L2_og2_metagenome_7b
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modeling_llama_te import NVLlamaConfig, NVLlamaModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_name: str) -> dict:
    """Load hydra config from YAML file."""
    from omegaconf import OmegaConf

    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "hydra_config" / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    logger.info(f"Loaded config from {config_path}")
    return OmegaConf.load(config_path)


class ModelOnlyState:
    """Stateful wrapper that only loads model weights, skipping optimizer/scheduler/extra_state."""

    def __init__(self, model: torch.nn.Module):
        """Initialize with model."""
        self.model = model
        self.epoch = 0
        self.step = 0

    def state_dict(self):
        """Return state dict for DCP."""
        # Get sharded state dict from the FSDP model
        model_sd = dict(self.model.named_parameters())

        return {
            "model": model_sd,
            "optim": {},
            "scheduler": {},
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict):
        """Load state dict, only model weights."""
        self.epoch = state_dict.get("epoch", 0)
        # Model weights are loaded automatically by DCP into the FSDP module


def main():
    """Consolidate FSDP2 checkpoint using DCP with get_model_state_dict."""
    parser = argparse.ArgumentParser(description="Consolidate FSDP2 distributed checkpoint")
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True, help="Path to checkpoint directory with .distcp files"
    )
    parser.add_argument("--output-path", type=str, required=True, help="Output path for consolidated checkpoint")
    parser.add_argument("--config-name", type=str, required=True, help="Name of the hydra config (without .yaml)")
    parser.add_argument("--format", choices=["safetensors", "pt"], default="safetensors", help="Output format")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_dir)
    output_path = Path(args.output_path)

    # Verify checkpoint exists
    distcp_files = list(checkpoint_path.glob("*.distcp"))
    if not distcp_files:
        raise FileNotFoundError(f"No .distcp files found in {checkpoint_path}")
    logger.info(f"Found {len(distcp_files)} .distcp shard files")

    # Initialize distributed (required for FSDP2 and DCP)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    logger.info(f"Rank {rank}/{world_size}, local_rank={local_rank}")

    # Create device mesh for FSDP2
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

    # Load config
    cfg = load_config(args.config_name)

    # Create model config
    model_cfg = cfg.get("model", cfg)
    llama_config = NVLlamaConfig(
        vocab_size=model_cfg.get("vocab_size", 256),
        num_hidden_layers=model_cfg.get("num_hidden_layers", 32),
        hidden_size=model_cfg.get("hidden_size", 4096),
        intermediate_size=model_cfg.get("intermediate_size", 14336),
        num_attention_heads=model_cfg.get("num_attention_heads", 32),
        num_key_value_heads=model_cfg.get("num_key_value_heads", 32),
        max_position_embeddings=model_cfg.get("max_position_embeddings", 8192),
        rope_theta=model_cfg.get("rope_theta", 500000),
        initializer_range=model_cfg.get("initializer_range", 0.02),
        attn_input_format=model_cfg.get("attn_input_format", "thd"),
        rope_scaling=model_cfg.get("rope_scaling", None),
    )
    logger.info(f"Creating model with config: {llama_config.__dict__}")

    # Create model on meta device first to save memory
    logger.info("Creating model on meta device...")
    with torch.device("meta"):
        model = NVLlamaModel(llama_config)

    # Move to real device with empty tensors
    model.to_empty(device=device)

    # Apply FSDP2 using fully_shard (the new API)
    # Use reshard_after_forward=False to keep full parameters
    logger.info("Applying FSDP2 sharding...")
    fully_shard(model, mesh=mesh, reshard_after_forward=False)

    # Load the distributed checkpoint
    # DCP handles resharding automatically from 48 ranks to our current world_size
    logger.info(f"Loading distributed checkpoint from {checkpoint_path}...")
    logger.info("DCP will automatically reshard from original 48 ranks to current setup")

    try:
        # Create state dict structure matching what was saved
        # The checkpoint was saved with {"app": AppState} structure
        app_state = ModelOnlyState(model)
        state_dict = {"app": app_state}

        # Load with DCP - it handles resharding
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=str(checkpoint_path),
        )
        logger.info(f"Successfully loaded checkpoint (epoch {app_state.epoch})")

    except Exception as e:
        logger.error(f"DCP load failed: {e}")
        logger.info("This may be due to _extra_state mismatch from Transformer Engine.")
        logger.info("Trying alternative approach...")

        # Alternative: Load directly into model state dict
        # First get the current model's state dict structure
        model_sd = dict(model.named_parameters())

        # Create a flat state dict for loading
        flat_sd = {"app.model." + k: v for k, v in model_sd.items()}

        try:
            dcp.load(
                state_dict=flat_sd,
                checkpoint_id=str(checkpoint_path),
            )
            logger.info("Alternative load succeeded")
        except Exception as e2:
            logger.error(f"Alternative load also failed: {e2}")
            raise RuntimeError(f"Could not load checkpoint: {e}, {e2}")

    # Extract full state dict using get_model_state_dict
    # This all-gathers the sharded parameters to rank 0
    logger.info("Extracting full state dict (all-gathering to rank 0)...")
    full_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=True,  # Reconstruct full model
            cpu_offload=True,  # Move to CPU to save GPU memory
        ),
    )

    logger.info(f"Extracted {len(full_state_dict)} tensors")

    # Save consolidated checkpoint (rank 0 only)
    if rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.format == "safetensors":
            from safetensors.torch import save_file

            # Convert to contiguous tensors
            clean_state_dict = {}
            for k, v in full_state_dict.items():
                if isinstance(v, torch.Tensor):
                    clean_state_dict[k] = v.contiguous().cpu()

            save_file(clean_state_dict, output_path)
            logger.info(f"Saved consolidated checkpoint to {output_path} (safetensors format)")
        else:
            torch.save(full_state_dict, output_path)
            logger.info(f"Saved consolidated checkpoint to {output_path} (pytorch format)")

        # Print checkpoint info
        num_params = sum(p.numel() for p in full_state_dict.values() if isinstance(p, torch.Tensor))
        size_mb = (
            sum(p.numel() * p.element_size() for p in full_state_dict.values() if isinstance(p, torch.Tensor))
            / 1024
            / 1024
        )
        logger.info(f"Checkpoint contains {num_params:,} parameters ({size_mb:.1f} MB)")

        # List some keys for verification
        logger.info("Sample keys in consolidated checkpoint:")
        for i, key in enumerate(sorted(full_state_dict.keys())[:10]):
            tensor = full_state_dict[key]
            logger.info(f"  {key}: {tensor.shape} {tensor.dtype}")

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
