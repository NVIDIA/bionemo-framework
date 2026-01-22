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

FSDP2 saves checkpoints as multiple .distcp files (one per rank). This script
consolidates them into a single model.safetensors file that can be loaded on
a single GPU.

The script must be run with torchrun to initialize distributed environment:
    torchrun --nproc_per_node=1 consolidate_fsdp2_checkpoint.py \
        --checkpoint-dir /path/to/step_5000 \
        --output-path /path/to/consolidated/model.safetensors \
        --config-name L2_og2_metagenome_7b

Alternatively, on multiple GPUs to speed up loading:
    torchrun --nproc_per_node=8 consolidate_fsdp2_checkpoint.py ...
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy


# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelOnlyAppState:
    """A minimal AppState that only loads model weights, skipping optimizer/scheduler.

    This matches the {"app": {...}} checkpoint structure but only restores model weights.
    """

    def __init__(self, model: torch.nn.Module):
        """Initialize with just a model."""
        self.model = model
        self.step = 0
        self.epoch = 0

    def state_dict(self):
        """Get state dict structure matching what was saved."""
        with FullyShardedDataParallel.state_dict_type(
            self.model,
            FullyShardedDataParallel.StateDictType.SHARDED_STATE_DICT,
        ):
            model_state_dict = self.model.state_dict()

        return {
            "model": model_state_dict,
            "optim": {},  # Placeholder
            "scheduler": {},  # Placeholder
            "step": self.step,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict: dict):
        """Load only the model weights from state dict."""
        with FullyShardedDataParallel.state_dict_type(
            self.model,
            FullyShardedDataParallel.StateDictType.SHARDED_STATE_DICT,
        ):
            self.model.load_state_dict(state_dict["model"])

        self.step = state_dict.get("step", 0)
        self.epoch = state_dict.get("epoch", 0)


def main():
    """Consolidate FSDP2 checkpoint shards."""
    parser = argparse.ArgumentParser(description="Consolidate FSDP2 distributed checkpoint")
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True, help="Path to checkpoint directory with .distcp files"
    )
    parser.add_argument("--output-path", type=str, required=True, help="Output path for consolidated checkpoint")
    parser.add_argument(
        "--config-name", type=str, default="L2_og2_metagenome_7b", help="Config name for model architecture"
    )
    parser.add_argument("--format", choices=["safetensors", "pt"], default="safetensors", help="Output format")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_dir)
    output_path = Path(args.output_path)

    # Verify checkpoint exists
    distcp_files = list(checkpoint_path.glob("*.distcp"))
    if not distcp_files:
        raise FileNotFoundError(f"No .distcp files found in {checkpoint_path}")

    logger.info(f"Found {len(distcp_files)} .distcp shard files")

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    logger.info(f"Rank {rank}/{world_size}, local_rank={local_rank}")

    # Create device mesh
    init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

    # Load config
    config_path = Path(__file__).parent.parent / "hydra_config" / f"{args.config_name}.yaml"
    if config_path.exists():
        config = OmegaConf.load(config_path)
        if rank == 0:
            logger.info(f"Loaded config from {config_path}")
    else:
        raise FileNotFoundError(f"Config not found at {config_path}")

    # Create model config
    model_config_kwargs = OmegaConf.to_container(config.get("config_kwargs", {}), resolve=True)
    model_config_kwargs.setdefault("vocab_size", 256)
    model_config_kwargs.setdefault("attn_input_format", "bshd")

    if rank == 0:
        logger.info(f"Creating model with config: {model_config_kwargs}")
    model_config = NVLlamaConfig.from_pretrained(
        config.get("config_name_or_path", "meta-llama/Llama-3.1-8B"),
        **model_config_kwargs,
    )

    # Create model on GPU
    if rank == 0:
        logger.info("Creating model...")
    model = NVLlamaForCausalLM(model_config).to(f"cuda:{local_rank}")

    # Wrap with FSDP using NO_SHARD to replicate model for simpler extraction
    model = FullyShardedDataParallel(model, sharding_strategy=ShardingStrategy.NO_SHARD, device_id=local_rank)

    # Build state dict matching checkpoint structure ({"app": AppState})
    # Use ModelOnlyAppState which only loads model weights, skipping optimizer/scheduler
    app_state = ModelOnlyAppState(model=model)
    state_dict = {"app": app_state}

    # Load distributed checkpoint
    if rank == 0:
        logger.info(f"Loading distributed checkpoint from {checkpoint_path}...")
    try:
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=str(checkpoint_path),
        )
        if rank == 0:
            logger.info("Successfully loaded distributed checkpoint!")
    except Exception as e:
        logger.error(f"Failed to load distributed checkpoint: {e}")
        logger.error("Make sure you're running with torchrun --nproc_per_node=1")
        raise

    # Extract model state dict (only on rank 0)
    if rank == 0:
        # Get the underlying model state dict
        model_state_dict = {}

        # Get the unwrapped model
        unwrapped_model = model.module if hasattr(model, "module") else model

        for name, param in unwrapped_model.named_parameters():
            model_state_dict[name] = param.detach().cpu().contiguous()

        # Also get buffers
        for name, buffer in unwrapped_model.named_buffers():
            model_state_dict[name] = buffer.detach().cpu().contiguous()

        # Save consolidated checkpoint
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.format == "safetensors":
            from safetensors.torch import save_file

            save_file(model_state_dict, output_path)
            logger.info(f"Saved consolidated checkpoint to {output_path} (safetensors format)")
        else:
            torch.save(model_state_dict, output_path)
            logger.info(f"Saved consolidated checkpoint to {output_path} (pytorch format)")

        # Print checkpoint info
        num_params = sum(p.numel() for p in model_state_dict.values())
        size_mb = sum(p.numel() * p.element_size() for p in model_state_dict.values()) / (1024 * 1024)
        logger.info(f"Checkpoint contains {num_params:,} parameters ({size_mb:.1f} MB)")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
