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

import logging
import os
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import transformers
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)

# ============================================================================
# Helper functions
# ============================================================================


def get_latest_checkpoint(ckpt_path: str | os.PathLike) -> tuple[str, int]:
    """Get the latest checkpoint path and step number."""
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint directory does not exist: {ckpt_path}")

    checkpoint_files = [f for f in os.listdir(ckpt_path) if f.startswith("step_")]
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {ckpt_path}")

    latest = max(checkpoint_files, key=lambda x: int(Path(x).stem.split("_")[1]))
    step = int(Path(latest).stem.split("_")[1])
    return os.path.join(ckpt_path, latest), step


def should_save_checkpoint(step: int, save_every_n_steps: int) -> bool:
    """Determine if a checkpoint should be saved."""
    if save_every_n_steps > 0 and step % save_every_n_steps == 0 and step > 0:
        return True
    return False


# ============================================================================
# DDP Checkpointing
# ============================================================================


def load_checkpoint_ddp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    """Load DDP checkpoint."""
    checkpoint_path, step = get_latest_checkpoint(ckpt_path)
    # checkpoint_path already includes .pt extension from get_latest_checkpoint

    checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{dist_config.local_rank}", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    # Increment the step by one to avoid re-running the previous step.
    step = checkpoint["step"] + 1

    if dist_config.is_main_process():
        logger.info(f"Loaded DDP checkpoint from step {step}")

    return model, optimizer, scheduler, step


def save_checkpoint_ddp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    step: int,
    dist_config: DistributedConfig,
) -> None:
    """Save DDP checkpoint - only on main process."""
    if not dist_config.is_main_process():
        return

    checkpoint_path = os.path.join(ckpt_path, f"step_{step}.pt")
    os.makedirs(ckpt_path, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        checkpoint_path,
    )
    logger.info(f"Saved DDP checkpoint to {checkpoint_path}")


def save_final_model_ddp(
    model: torch.nn.Module,
    save_directory: str | os.PathLike,
    dist_config: DistributedConfig,
) -> None:
    """Save final model for DDP - only on main process."""
    if not dist_config.is_main_process():
        return

    # Unwrap model if wrapped
    underlying_model: transformers.PreTrainedModel = model.module if hasattr(model, "module") else model  # type: ignore

    os.makedirs(save_directory, exist_ok=True)
    underlying_model.save_pretrained(save_directory, state_dict=underlying_model.state_dict(), safe_serialization=True)
    logger.info(f"Saved final DDP model to {save_directory}")


# ============================================================================
# mFSDP Checkpointing
# ============================================================================


def load_checkpoint_mfsdp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    """Load mFSDP distributed checkpoint.

    Args:
        model: The model to load.
        optimizer: The optimizer to load.
        scheduler: The LR scheduler to load.
        ckpt_path: The directory containing checkpoints.
        dist_config: The distributed configuration.

    Returns:
        Tuple of (model, optimizer, scheduler, step).
    """
    checkpoint_path, step = get_latest_checkpoint(ckpt_path)

    ckpt_state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metadata": {},
    }
    torch.distributed.checkpoint.load(state_dict=ckpt_state_dict, checkpoint_id=str(checkpoint_path))

    model.load_state_dict(ckpt_state_dict["model"])
    optimizer.load_state_dict(ckpt_state_dict["optimizer"])
    scheduler.load_state_dict(ckpt_state_dict["scheduler"])

    # Get step from metadata if available, otherwise from filename
    metadata_step = ckpt_state_dict.get("metadata", {}).get("step")
    if metadata_step is not None:
        step = metadata_step

    # Ensure all ranks have completed loading before proceeding
    torch.distributed.barrier()

    logger.info(f"Loaded mFSDP checkpoint from step {step}")
    return model, optimizer, scheduler, step


def save_checkpoint_mfsdp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    step: int,
) -> None:
    """Save mFSDP distributed checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        scheduler: The LR scheduler to save.
        ckpt_path: The directory to save the checkpoint.
        step: The step number to save the checkpoint.
    """
    checkpoint_path = os.path.join(ckpt_path, f"step_{step}")
    os.makedirs(ckpt_path, exist_ok=True)

    # Save model, optimizer, scheduler state, and metadata
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metadata": {"step": step},
    }

    torch.distributed.checkpoint.save(
        state_dict,
        checkpoint_id=checkpoint_path,
    )
    logger.info(f"Saved mFSDP checkpoint to {checkpoint_path}")


def save_final_model_mfsdp(
    model: torch.nn.Module,
    save_directory: str | os.PathLike,
    dist_config: DistributedConfig,
) -> None:
    """Save final model for mFSDP - requires parameter gathering on all ranks."""
    # Parameter gathering must happen on ALL processes
    logger.info("Starting mFSDP parameter gathering...")
    model._replace_param_with_raw_if_needed()
    model.all_gather_pipeline.all_gather_params(list(model.module.parameters()))

    for param in model.module.parameters():
        bucket_id = model.param_and_grad_buffer.param_to_param_group[param]
        model.all_gather_pipeline.wait_bucket_ready(bucket_id)

    logger.info("mFSDP parameter gathering completed")

    # Only main process saves the model
    if not dist_config.is_main_process():
        return

    os.makedirs(save_directory, exist_ok=True)
    model.module.save_pretrained(save_directory, state_dict=model.module.state_dict(), safe_serialization=True)
    logger.info(f"Saved final mFSDP model to {save_directory}")


# ============================================================================
# FSDP2 Checkpointing
# ============================================================================


def _load_distributed_checkpoint_fsdp2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_dirs: list[str],
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    # Load distributed checkpoint (newer format)
    # Find the latest checkpoint directory
    latest_step = max(int(d.split("_")[1]) for d in checkpoint_dirs)
    checkpoint_dir = os.path.join(ckpt_path, f"step_{latest_step}")

    # Initialize empty state dicts for loading
    model_state_dict = {}
    optimizer_state_dict = {}
    scheduler_state_dict = {}
    metadata = {}

    state_dict = {
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
        "scheduler": scheduler_state_dict,
        "metadata": metadata,
    }

    # All ranks participate in distributed load
    dcp.load(
        state_dict=state_dict,
        checkpoint_id=checkpoint_dir,
    )

    step = metadata.get("step", latest_step)

    # Set the loaded state dicts (sharded format)
    set_model_state_dict(
        model=model,
        model_state_dict=model_state_dict,
        options=StateDictOptions(
            full_state_dict=False,  # Sharded format
        ),
    )

    set_optimizer_state_dict(
        model=model,
        optimizers=optimizer,
        optim_state_dict=optimizer_state_dict,
        options=StateDictOptions(
            full_state_dict=False,  # Sharded format
        ),
    )

    # Load scheduler state if available
    scheduler.load_state_dict(scheduler_state_dict)

    logger.info(f"Loaded distributed FSDP2 checkpoint from step {step}")
    return model, optimizer, scheduler, step


def _load_legacy_checkpoint_fsdp2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    # Load legacy checkpoint (older format)
    checkpoint_path, step = get_latest_checkpoint(ckpt_path)

    # Only rank 0 loads the checkpoint file from disk
    if dist_config.is_main_process():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = checkpoint["model"]
        optimizer_state = checkpoint["optimizer"]
        scheduler_state = checkpoint["scheduler"]
        step = checkpoint.get("step", step)
    else:
        model_state = {}
        optimizer_state = {}
        scheduler_state = {}
        step = 0

    # Broadcast step number to all ranks
    step_tensor = torch.tensor([step], dtype=torch.int64, device=f"cuda:{dist_config.local_rank}")
    torch.distributed.broadcast(step_tensor, src=0)
    step = step_tensor.item()

    # ALL ranks call set_model_state_dict - DCP handles broadcasting from rank 0
    set_model_state_dict(
        model=model,
        model_state_dict=model_state,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
        ),
    )

    # ALL ranks MUST call set_optimizer_state_dict - DCP handles broadcasting from rank 0
    set_optimizer_state_dict(
        model=model,
        optimizers=optimizer,
        optim_state_dict=optimizer_state,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
        ),
    )

    # Load scheduler state if available (only on rank 0, then broadcast happens automatically)
    if dist_config.is_main_process() and scheduler_state:
        scheduler.load_state_dict(scheduler_state)

    # Ensure all ranks have completed loading before proceeding
    torch.distributed.barrier()

    logger.info(f"Loaded legacy FSDP2 checkpoint from step {step}")
    return model, optimizer, scheduler, int(step)


def load_checkpoint_fsdp2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    """Load FSDP2 checkpoint - distributed or legacy format.

    Automatically detects checkpoint format and loads appropriately.
    Both model and optimizer states are properly restored across all ranks.
    """
    # Check for distributed checkpoint directories (step_X folders)
    checkpoint_dirs = [
        d for d in os.listdir(ckpt_path) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_path, d))
    ]

    if checkpoint_dirs:
        return _load_distributed_checkpoint_fsdp2(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dirs=checkpoint_dirs,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
        )

    # Check for legacy checkpoint files (step_X.pt files)
    checkpoint_files = [f for f in os.listdir(ckpt_path) if f.startswith("step_") and f.endswith(".pt")]

    if checkpoint_files:
        return _load_legacy_checkpoint_fsdp2(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
        )

    logger.info("No FSDP2 checkpoints found")
    return model, optimizer, scheduler, 0


def _save_distributed_checkpoint_fsdp2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    step: int,
) -> None:
    # Distributed checkpoint - each rank saves its own shard
    checkpoint_dir = os.path.join(ckpt_path, f"step_{step}")

    # Get sharded state dicts (each rank has its portion)
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=False,  # Keep sharded
            cpu_offload=True,
        ),
    )

    optimizer_state_dict = get_optimizer_state_dict(
        model=model,
        optimizers=optimizer,
        options=StateDictOptions(
            full_state_dict=False,  # Keep sharded
            cpu_offload=True,
        ),
    )

    # Save metadata and scheduler state separately (step number, scheduler)
    scheduler_state_dict = scheduler.state_dict()
    metadata = {"step": step}

    # All ranks participate in distributed save
    state_dict = {
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
        "scheduler": scheduler_state_dict,
        "metadata": metadata,
    }

    dcp.save(
        state_dict=state_dict,
        checkpoint_id=checkpoint_dir,
    )

    logger.info(f"Saved distributed FSDP2 checkpoint to {checkpoint_dir}")


def _save_legacy_checkpoint_fsdp2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    step: int,
    dist_config: DistributedConfig,
) -> None:
    # Legacy path: gather full state dict and save on rank 0 (can OOM for large models)
    # ALL ranks must call get_model_state_dict for the collective communication
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )

    # ALL ranks must call get_optimizer_state_dict for the collective communication
    optimizer_state_dict = get_optimizer_state_dict(
        model=model,
        optimizers=optimizer,  # Note: parameter name is 'optimizers' even for single optimizer
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )

    # Only rank 0 saves the checkpoint
    if not dist_config.is_main_process():
        return

    checkpoint_path = os.path.join(ckpt_path, f"step_{step}.pt")
    os.makedirs(ckpt_path, exist_ok=True)

    torch.save(
        {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        checkpoint_path,
    )
    logger.info(f"Saved FSDP2 checkpoint to {checkpoint_path}")


def save_checkpoint_fsdp2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    step: int,
    dist_config: DistributedConfig,
    use_distributed_checkpoint: bool = True,  # Default to distributed for scalability
) -> None:
    """Save FSDP2 checkpoint - distributed sharded or gathered based on flag.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        scheduler: The LR scheduler to save.
        ckpt_path: The directory to save the checkpoint.
        step: The step number to save the checkpoint.
        dist_config: The distributed configuration.
        use_distributed_checkpoint: If True, use distributed checkpoint (each rank saves its shard).
                                  If False, gather full state and save on rank 0 (for small models).
    """
    if use_distributed_checkpoint:
        _save_distributed_checkpoint_fsdp2(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            step=step,
        )

    else:
        _save_legacy_checkpoint_fsdp2(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            step=step,
            dist_config=dist_config,
        )


def save_final_model_fsdp2(
    model: torch.nn.Module,
    save_directory: str | os.PathLike,
    dist_config: DistributedConfig,
) -> None:
    """Save final model for FSDP2 - gather on all ranks, save on main."""
    # ALL ranks must participate in gathering
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )

    # Only main process saves
    if not dist_config.is_main_process():
        return

    os.makedirs(save_directory, exist_ok=True)

    # Save just the weights using safetensors

    save_file(model_state_dict, os.path.join(save_directory, "model.safetensors"))

    # Save the config
    underlying_model = model.module if hasattr(model, "module") else model
    if hasattr(underlying_model, "config"):
        underlying_model.config.save_pretrained(save_directory)

    logger.info(f"Saved final FSDP2 model to {save_directory} (weights + config only)")
