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
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import transformers
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader

from dataset import load_dataloader, save_dataloader
from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)

# ============================================================================
# Helper functions
# ============================================================================


def get_latest_checkpoint(ckpt_path: str | os.PathLike) -> tuple[Path | None, int]:
    """Get the latest checkpoint path and step number.

    Returns:
        Tuple of (checkpoint path, step number).
        If no checkpoint files are found, returns (None, 0).
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return None, 0

    checkpoints = [f for f in ckpt_path.iterdir() if f.name.startswith("step_")]

    if not checkpoints:
        return None, 0

    latest = max(checkpoints, key=lambda x: int(Path(x).stem.split("_")[1]))
    step = int(Path(latest).stem.split("_")[1])
    return latest, step


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
    dataloader: StatefulDataLoader | None = None,
) -> tuple[
    torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int, StatefulDataLoader | None, int
]:
    """Load DDP checkpoint."""
    checkpoint_path, _ = get_latest_checkpoint(ckpt_path)

    if not checkpoint_path:
        logger.info("No DDP checkpoint found, starting from scratch")
        return model, optimizer, scheduler, 0, dataloader, 0

    checkpoint = torch.load(
        checkpoint_path / "checkpoint.pt", map_location=f"cuda:{dist_config.local_rank}", weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    dataloader_num_workers = checkpoint.get("dataloader_num_workers", None)
    step = checkpoint.get("step", None)
    epoch = checkpoint.get("epoch", None)

    if dist_config.is_main_process():
        logger.info(f"Loaded DDP checkpoint from step {step}")

    if dataloader is not None:
        load_dataloader(
            dataloader=dataloader,
            ckpt_path=checkpoint_path,
            dist_config=dist_config,
            num_workers=dataloader_num_workers,
        )
        logger.info(f"Loaded DDP dataloader from step {step} from {checkpoint_path}")

    # Increment the step by one to avoid re-running the previous step.
    step += 1

    return model, optimizer, scheduler, step, dataloader, epoch


def save_checkpoint_ddp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    step: int,
    epoch: int,
    dist_config: DistributedConfig,
    dataloader: StatefulDataLoader | None = None,
) -> None:
    """Saves the Dataloader state and the DDP checkpoint."""
    ckpt_path = Path(ckpt_path)
    checkpoint_path = ckpt_path / f"step_{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if dataloader is not None:
        save_dataloader(
            dataloader=dataloader,
            ckpt_path=checkpoint_path,
            dist_config=dist_config,
        )
        logger.info(f"Saved DDP dataloader to {checkpoint_path}")

    if not dist_config.is_main_process():
        return

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "dataloader_num_workers": dataloader.num_workers if dataloader is not None else None,
        },
        checkpoint_path / "checkpoint.pt",
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
    dist_config: DistributedConfig,
    dataloader: StatefulDataLoader | None = None,
) -> tuple[
    torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int, StatefulDataLoader | None, int
]:
    """Load mFSDP distributed checkpoint.

    Args:
        model: The model to load.
        optimizer: The optimizer to load.
        scheduler: The LR scheduler to load.
        ckpt_path: The directory containing checkpoints.
        dist_config: The distributed configuration.
        dataloader: The dataloader to load.

    Returns:
        Tuple of (model, optimizer, scheduler, step).
    """
    checkpoint_path, step = get_latest_checkpoint(ckpt_path)
    if not checkpoint_path:
        logger.info("No mFSDP checkpoint found, starting from scratch")
        return model, optimizer, scheduler, 0, dataloader, 0

    ckpt_state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metadata": {
            "step": step,  # Initialize with current step from filename
            "epoch": 0,  # Initialize with default epoch
            "dataloader_num_workers": dataloader.num_workers
            if dataloader is not None
            else 1,  # Initialize with current or default
        },
    }
    torch.distributed.checkpoint.load(state_dict=ckpt_state_dict, checkpoint_id=checkpoint_path)

    model.load_state_dict(ckpt_state_dict["model"])
    optimizer.load_state_dict(ckpt_state_dict["optimizer"])
    scheduler.load_state_dict(ckpt_state_dict["scheduler"])

    # Get step from metadata if available, otherwise from filename
    step = ckpt_state_dict.get("metadata").get("step")
    epoch = ckpt_state_dict.get("metadata").get("epoch")
    dataloader_num_workers = ckpt_state_dict.get("metadata").get("dataloader_num_workers")

    if dataloader is not None:
        load_dataloader(
            dataloader=dataloader,
            ckpt_path=checkpoint_path,
            dist_config=dist_config,
            num_workers=dataloader_num_workers,
        )

    # Increment the step by one to avoid re-running the previous step.
    step += 1

    # Ensure all ranks have completed loading before proceeding
    torch.distributed.barrier()

    logger.info(f"Loaded mFSDP checkpoint from step {step}")
    return model, optimizer, scheduler, step, dataloader, epoch


def save_checkpoint_mfsdp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    step: int,
    dist_config: DistributedConfig,
    dataloader: StatefulDataLoader | None = None,
    epoch: int = 0,
) -> None:
    """Save mFSDP distributed checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        scheduler: The LR scheduler to save.
        ckpt_path: The directory to save the checkpoint.
        step: The step number to save the checkpoint.
        dist_config: The distributed configuration.
        dataloader: The dataloader to save.
        epoch: The epoch number to save the checkpoint.
    """
    ckpt_path = Path(ckpt_path)
    checkpoint_path = ckpt_path / f"step_{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if dataloader is not None:
        save_dataloader(
            dataloader=dataloader,
            ckpt_path=checkpoint_path,
            dist_config=dist_config,
        )
        logger.info(f"Saved mFSDP dataloader to {checkpoint_path}")

    # Save model, optimizer, scheduler state, and metadata
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metadata": {
            "step": step,
            "epoch": epoch,
            "dataloader_num_workers": dataloader.num_workers if dataloader is not None else None,
        },
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

    from megatron_fsdp.uneven_dtensor import gather_uneven_dtensor_to_full_tensor

    unsharded_state_dict = {
        # Gather all parameters to CPU, and remove the "module." prefix from the Megatron-FSDP class wrapper.
        k.removeprefix("module."): gather_uneven_dtensor_to_full_tensor(
            v, target_device=torch.device("cpu")
        ).to_local()
        if isinstance(v, torch.distributed.tensor.DTensor)
        else v
        for k, v in model.state_dict().items()
    }

    # Only main process saves the model
    if not dist_config.is_main_process():
        return

    os.makedirs(save_directory, exist_ok=True)
    model.module.save_pretrained(save_directory, state_dict=unsharded_state_dict, safe_serialization=True)
    logger.info(f"Saved final mFSDP model to {save_directory}")


# ============================================================================
# FSDP2 Checkpointing
# ============================================================================


@dataclass
class AppState(Stateful):
    """AppState for FSDP2 checkpoint.

    Adapted from https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    step: int = 0
    dataloader_num_workers: int | None = None
    epoch: int = 0
    state_dict_options: StateDictOptions = field(
        default_factory=lambda: StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
        )
    )

    def state_dict(self):
        """Get the state dict for the model, optimizer, scheduler, and step."""
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer, options=self.state_dict_options
        )
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "dataloader_num_workers": self.dataloader_num_workers,
        }

    def load_state_dict(self, state_dict: dict):
        """Load the state dict for the model, optimizer, scheduler, and step."""
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
            options=self.state_dict_options,
        )
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]
        self.dataloader_num_workers = state_dict["dataloader_num_workers"]


def load_checkpoint_fsdp2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
    dataloader: StatefulDataLoader | None = None,
) -> tuple[
    torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int, StatefulDataLoader | None, int
]:
    """Load FSDP2 checkpoint.

    Args:
        model: The model to load.
        optimizer: The optimizer to load.
        scheduler: The LR scheduler to load.
        ckpt_path: The directory containing checkpoints.
        dist_config: The distributed configuration.
        dataloader: The dataloader to load.
    """
    checkpoint_path, _ = get_latest_checkpoint(ckpt_path)
    if not checkpoint_path:
        logger.info("No FSDP2 checkpoint found, starting from scratch")
        return model, optimizer, scheduler, 0, dataloader, 0

    app_state = AppState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_num_workers=dataloader.num_workers if dataloader is not None else None,
    )

    state_dict = {"app": app_state}
    dcp.load(state_dict, checkpoint_id=checkpoint_path)

    # Note: Is this how i do it?
    dataloader_num_workers = app_state.dataloader_num_workers
    epoch = app_state.epoch

    if dataloader is not None:
        load_dataloader(
            dataloader=dataloader,
            ckpt_path=checkpoint_path,
            dist_config=dist_config,
            num_workers=dataloader_num_workers,
        )
    # Increment the step by one to avoid re-running the previous step.
    step = app_state.step + 1

    logger.info(f"Loaded distributed FSDP2 checkpoint from step {app_state.step}")
    return model, optimizer, scheduler, step, dataloader, epoch


def save_checkpoint_fsdp2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    step: int,
    epoch: int,
    dist_config: DistributedConfig,
    dataloader: StatefulDataLoader | None = None,
) -> None:
    """Save FSDP2 checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        scheduler: The LR scheduler to save.
        ckpt_path: The directory to save the checkpoint.
        step: The step number to save the checkpoint.
        epoch: The epoch number to save the checkpoint.
        dist_config: The distributed configuration.
        dataloader: The dataloader to save.
    """
    ckpt_path = Path(ckpt_path)
    checkpoint_path = ckpt_path / f"step_{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if dataloader is not None:
        save_dataloader(
            dataloader=dataloader,
            ckpt_path=checkpoint_path,
            dist_config=dist_config,
        )
        logger.info(f"Saved FSDP2 dataloader to {ckpt_path}")

    state_dict = {
        "app": AppState(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            dataloader_num_workers=dataloader.num_workers if dataloader is not None else None,
            epoch=epoch,
        )
    }
    dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_path)
    logger.info(f"Saved distributed FSDP2 checkpoint to {checkpoint_path}")


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
