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
from typing import Any, Dict, Tuple

import torch
import torch.distributed.checkpoint
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)


# ============================================================================
# Helper functions
# ============================================================================


def get_latest_checkpoint(ckpt_dir: str) -> Tuple[str, int]:
    """Get the latest checkpoint path and step number."""
    if not os.path.exists(ckpt_dir):
        raise ValueError(f"Checkpoint directory does not exist: {ckpt_dir}")

    checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.startswith("step_")]
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {ckpt_dir}")

    latest = max(checkpoint_files, key=lambda x: int(Path(x).stem.split("_")[1]))
    step = int(Path(latest).stem.split("_")[1])
    return os.path.join(ckpt_dir, latest), step


# ============================================================================
# DDP Checkpointing
# ============================================================================


def load_checkpoint_ddp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: str,
    dist_config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    """Load DDP checkpoint."""
    try:
        checkpoint_path, step = get_latest_checkpoint(ckpt_dir)
        # checkpoint_path already includes .pt extension from get_latest_checkpoint

        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{dist_config.local_rank}", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        # Get step from checkpoint if available, otherwise from filename
        step = checkpoint.get("step", step)

        if dist_config.is_main_process():
            logger.info(f"Loaded DDP checkpoint from step {step}")

        return model, optimizer, step
    except Exception as e:
        logger.error(f"Failed to load DDP checkpoint: {e}")
        return model, optimizer, 0


def save_checkpoint_ddp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: str,
    step: int,
    dist_config: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Save DDP checkpoint - only on main process."""
    if not dist_config.is_main_process():
        return

    checkpoint_path = os.path.join(ckpt_dir, f"step_{step}.pt")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, checkpoint_path)
    logger.info(f"Saved DDP checkpoint to {checkpoint_path}")


def save_final_model_ddp(
    model: torch.nn.Module,
    save_directory: str,
    dist_config: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Save final model for DDP - only on main process."""
    if not dist_config.is_main_process():
        return

    # Unwrap model if wrapped
    underlying_model = model.module if hasattr(model, "module") else model

    os.makedirs(save_directory, exist_ok=True)
    underlying_model.save_pretrained(save_directory, state_dict=underlying_model.state_dict(), safe_serialization=True)
    logger.info(f"Saved final DDP model to {save_directory}")


# ============================================================================
# mFSDP Checkpointing
# ============================================================================


def load_checkpoint_mfsdp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: str,
    logger: logging.Logger,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    """Load mFSDP distributed checkpoint."""
    try:
        checkpoint_path, step = get_latest_checkpoint(ckpt_dir)

        ckpt_state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.distributed.checkpoint.load(state_dict=ckpt_state_dict, checkpoint_id=str(checkpoint_path))

        model.load_state_dict(ckpt_state_dict["model"])
        optimizer.load_state_dict(ckpt_state_dict["optimizer"])

        # Ensure all ranks have completed loading before proceeding
        torch.distributed.barrier()

        logger.info(f"Loaded mFSDP checkpoint from step {step}")
        return model, optimizer, step
    except Exception as e:
        logger.error(f"Failed to load mFSDP checkpoint: {e}")
        return model, optimizer, 0


def save_checkpoint_mfsdp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: str,
    step: int,
    logger: logging.Logger,
) -> None:
    """Save mFSDP distributed checkpoint."""
    checkpoint_path = os.path.join(ckpt_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    # TODO: Do teh gather.
    torch.distributed.checkpoint.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        checkpoint_id=checkpoint_path,
    )
    logger.info(f"Saved mFSDP checkpoint to {checkpoint_path}")


def save_final_model_mfsdp(
    model: torch.nn.Module,
    save_directory: str,
    dist_config: Dict[str, Any],
    logger: logging.Logger,
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


def load_checkpoint_fsdp2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: str,
    dist_config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    """Load FSDP2 checkpoint - load on main, broadcast to all.

    Both model and optimizer states are properly restored across all ranks
    using distributed checkpoint APIs to avoid state divergence.
    """
    try:
        checkpoint_path, step = get_latest_checkpoint(ckpt_dir)
        # checkpoint_path already includes .pt extension from get_latest_checkpoint

        # Only rank 0 loads the checkpoint file from disk
        if dist_config.is_main_process():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model_state = checkpoint["model"]
            optimizer_state = checkpoint["optimizer"]
            step = checkpoint.get("step", step)
        else:
            model_state = {}
            optimizer_state = {}
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
        # This ensures optimizer state is properly synchronized across all ranks
        set_optimizer_state_dict(
            model=model,
            optimizer=optimizer,
            optim_state_dict=optimizer_state,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )

        # Ensure all ranks have completed loading before proceeding
        torch.distributed.barrier()

        logger.info(f"Loaded FSDP2 checkpoint from step {step} (model and optimizer restored on all ranks)")
        return model, optimizer, step
    except Exception as e:
        logger.error(f"Failed to load FSDP2 checkpoint: {e}")
        return model, optimizer, 0


def save_checkpoint_fsdp2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: str,
    step: int,
    dist_config: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Save FSDP2 checkpoint - gather on all ranks, save on main."""
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

    checkpoint_path = os.path.join(ckpt_dir, f"step_{step}.pt")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save({"model": model_state_dict, "optimizer": optimizer_state_dict, "step": step}, checkpoint_path)
    logger.info(f"Saved FSDP2 checkpoint to {checkpoint_path}")


def save_final_model_fsdp2(
    model: torch.nn.Module,
    save_directory: str,
    dist_config: Dict[str, Any],
    logger: logging.Logger,
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
