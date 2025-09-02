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
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from megatron_fsdp import fully_shard
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from imagenet_dataset import ImageNetDataset, infinite_dataloader
from imagenet_utils import transforms_imagenet_eval, transforms_imagenet_train
from vit import VisionTransformer


_logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config", config_name="vit_base_patch16_224")
def main(cfg) -> None:
    """Distributed Setup"""
    # Initialize distributed training environment.
    torch.distributed.init_process_group()

    # Associate all future device operations in the current process
    # with a uniquely-indexed local device, e.g. "cuda:0" on Rank 0.
    local_rank = int(os.getenv("LOCAL_RANK", torch.distributed.get_rank()))
    torch.cuda.set_device(local_rank)

    # Initialize DeviceMesh. Validate parallelism sizes.
    if (
        cfg.distributed.dp_inter * cfg.distributed.dp_shard * cfg.distributed.cp * cfg.distributed.tp
        != torch.distributed.get_world_size()
    ):
        raise ValueError(
            f"Invalid parallelism sizes: dp_inter({cfg.distributed.dp_inter}) * dp_shard({cfg.distributed.dp_shard}) * cp({cfg.distributed.cp}) * tp({cfg.distributed.tp}) != world_size({torch.distributed.get_world_size()})"
        )
    device_mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        mesh_shape=(
            cfg.distributed.dp_inter,
            cfg.distributed.dp_shard,
            cfg.distributed.cp,
            cfg.distributed.tp,
        ),
        mesh_dim_names=("dp_inter", "dp_shard", "cp", "tp"),
    )

    # Sub-meshes (possibly) required for Megatron-FSDP.
    # WARNING: These have a tendency to be deleted by Torch. Save references
    # or pass them to all classes or functions that use them.
    # DP: Only relevant when using HSDP, where we need the flattened DP group for data parallelism. (Otherwise, just pass dp_shard.)
    device_mesh[("dp_inter", "dp_shard")]._flatten("dp")
    # DP-Shard-CP: Only required if using CP. Otherwise, just pass dp_shard to FSDP.
    device_mesh[("dp_shard", "cp")]._flatten("dp_cp_shard")
    # HSDP (DP-CP): Only required if using HSDP. Otherwise, don't pass hybrid_fsdp_group to Megatron-FSDP.
    device_mesh[("dp_inter", "dp_shard", "cp")]._flatten("hsdp")

    """
    Profiling
    """
    if cfg.profiling.torch_memory_profile:
        # Start Torch memory profiling.
        torch.cuda.memory._record_memory_history(**cfg.profiling.torch_memory_profile_kwargs)
        torch_memory_profiler_snapshot = None

    if cfg.profiling.wandb and torch.distributed.get_rank() == 0:
        # Initialize WandB on main process.
        wandb.init(
            **cfg.profiling.wandb_kwargs,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )

    """
    Model
    """
    with (
        # Meta Device Initialization
        torch.device("meta") if cfg.fsdp.init_model_with_meta_device else nullcontext()
    ):
        vit_kwargs = dict(cfg.model.vit)
        if cfg.fsdp.init_model_with_meta_device:
            vit_kwargs["weight_init"] = None
        if cfg.model.transformer_engine:
            from transformer_engine.pytorch import TransformerLayer

            vit_kwargs["block_fn"] = TransformerLayer
            vit_kwargs["micro_batch_size"] = cfg.dataset.train.batch_size
            vit_kwargs["tp_group"] = device_mesh["tp"].get_group()
            vit_kwargs["tp_size"] = device_mesh["tp"].size()
        model = VisionTransformer(**vit_kwargs)
        if cfg.model.channels_last:
            model.to(memory_format=torch.channels_last)

    # Create optimizer.
    optimizer = AdamW(model.parameters(), **cfg.optimizer)

    # Initialize Megatron-FSDP.
    model, optimizer = fully_shard(
        # Torch (Root) Module
        model,
        # Torch Optimizer
        optimizer=optimizer,
        # ZeRO Sharding Strategy: None (0) -> Optim (1) -> Grad (2) -> Weights (3)
        zero_dp_strategy=cfg.fsdp.zero_dp_strategy,
        # FSDP "Unit Modules" - The sub-modules of the model that you want to shard!
        fsdp_unit_modules=cfg.fsdp.fsdp_unit_modules,
        # Use Hybrid FSDP (HSDP).
        use_hybrid_fsdp=cfg.fsdp.use_hybrid_fsdp,
        # Inter / Outer DP Sharding Strategy: None (0) -> Optim (1) -> Grad (2) -> Weights (3)
        # Note: This adds a second stage of sharding that generalizes DP-Replicate. Think of it
        # like an extra stage of NCCL divide-and-conquer when using all-gather or reduce-scatter.
        # Currently, this does not fully-shard the gradients and weights, only the optimizer state,
        # so the memory will be only marginally better than sharding on only DP-Shard.
        outer_dp_sharding_strategy=cfg.fsdp.outer_dp_sharding_strategy,
        # Megatron-FSDP Device Mesh / Distributed Environment
        device_mesh=device_mesh,
        # Always required to use Megatron-FSDP. What we shard on.
        dp_shard_dim="dp_cp_shard",
        # Required if using HSDP. The second / intermediate set of data-parallel process groups.
        dp_inter_dim="dp_inter",
        # Required if using TP, either from TransformerEngine (TP=1) / Megatron or DTensor-based TP.
        tp_dim="tp",
        # Required if using HSDP. Created by flattening everything we shard on, e.g. DP-CP.
        hybrid_fsdp_group=device_mesh["hsdp"].get_group(),
        # Load the model on device in shards to avoid OOM. Requires device("meta")-init for model.
        init_model_with_meta_device=cfg.fsdp.init_model_with_meta_device,
        # Reduce gradients in FP32.
        grad_reduce_in_fp32=cfg.fsdp.grad_reduce_in_fp32,
        # Store distributed optimization state in FP32.
        preserve_fp32_weights=cfg.fsdp.preserve_fp32_weights,
        # Sync gradients each step. Allows for gradient transformations after backward pass, but
        # deactivates compute-communication overlap going into the subsequent training step.
        sync_grads_each_step=True,
        # Preprocess state dict for DCP checkpointing. Required for Torch Distributed Checkpoint.
        preproc_state_dict_for_dcp_ckpt=True,
    )

    # Auto-Resume: Load latest model and optimizer checkpoints.
    latest_step_idx = 0
    if cfg.training.checkpoint.path and Path(cfg.training.checkpoint.path).exists():
        # Get latest checkpoint sub-directory, which should ONLY contain Torch DCP checkpoint sub-directories.
        subdirs = [x.absolute() for x in Path(cfg.training.checkpoint.path).iterdir() if x.is_dir()]
        if len(subdirs) > 0:
            # We expect a checkpoint named as: step_<step_idx>_loss_<loss_value>.
            # Get the latest step, the directory with the most recent modification time.
            opt_metric_coeff = 1 if cfg.training.checkpoint.resume_from_metric == "+" else -1
            latest_subdir = max(
                subdirs,
                key=lambda x: (
                    opt_metric_coeff * float(x.name.split("_")[3])
                    if cfg.training.checkpoint.resume_from_metric
                    else 0,
                    x.stat().st_mtime,
                ),
            )
            # Track latest step to continue training from.
            latest_step_idx = int(latest_subdir.name.split("_")[1])
            # Load model and optimizer checkpoints.
            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.distributed.checkpoint.load(state_dict, checkpoint_id=latest_subdir)
            model.load_state_dict(state_dict["model"])
            optimizer.load_state_dict(state_dict["optimizer"])
            if torch.distributed.get_rank() == 0:
                _logger.info(f"Loaded latest model and optimizer checkpoints from: {latest_subdir}")

    """
    Dataset
    """
    # Training
    imagenet_train_ds = ImageNetDataset(
        root=cfg.dataset.train.root,
        class_map=cfg.dataset.train.class_map,
        transform=transforms_imagenet_train(**cfg.dataset.train.transform_kwargs),
        class_filter=cfg.dataset.train.class_filter,
    )
    train_sampler = DistributedSampler(
        imagenet_train_ds,
        # Send distinct samples to all DP ranks only!
        num_replicas=device_mesh["dp"].size(),
        rank=device_mesh["dp"].get_local_rank(),
        shuffle=cfg.dataset.train.shuffle,
        seed=cfg.random.seed,
    )
    train_dataloader = DataLoader(
        imagenet_train_ds,
        batch_size=cfg.dataset.train.batch_size,
        sampler=train_sampler,
        num_workers=cfg.dataset.num_workers,
        # IMPORTANT: persistent_workers=True is required for Megatron-FSDP and
        # Torch DCP, because CUDA/NCCL and Dataloader kill each others workers!
        # Alternatively, you can set num_workers=0.
        persistent_workers=True,
    )
    if torch.distributed.get_rank() == 0:
        _logger.info(f"Training Dataset Size: {len(imagenet_train_ds)}")

    # Validation
    imagenet_val_ds = ImageNetDataset(
        root=cfg.dataset.val.root,
        class_map=cfg.dataset.val.class_map,
        label_map=cfg.dataset.val.label_map,
        transform=transforms_imagenet_eval(**cfg.dataset.val.transform_kwargs),
        class_filter=cfg.dataset.val.class_filter,
    )
    val_sampler = DistributedSampler(
        imagenet_val_ds,
        # Send distinct samples to all DP ranks only!
        num_replicas=device_mesh["dp"].size(),
        rank=device_mesh["dp"].get_local_rank(),
        shuffle=cfg.dataset.val.shuffle,
        seed=cfg.random.seed,
    )
    val_dataloader = DataLoader(
        imagenet_val_ds,
        batch_size=cfg.dataset.val.batch_size,
        sampler=val_sampler,
        num_workers=cfg.dataset.num_workers,
        # IMPORTANT: persistent_workers=True is required for Megatron-FSDP and
        # Torch DCP, because CUDA/NCCL and Dataloader kill each others workers!
        # Alternatively, you can set num_workers=0.
        persistent_workers=True,
    )
    if torch.distributed.get_rank() == 0:
        _logger.info(f"Validation Dataset Size: {len(imagenet_val_ds)}")

    """
    Training Utilities
    """
    # Loss Function
    loss_fn = torch.nn.CrossEntropyLoss().to(device=torch.device(f"cuda:{torch.cuda.current_device()}"))

    # LR Scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.steps)

    """
    Training Loop
    """

    if torch.distributed.get_rank() == 0:
        progress_bar = tqdm(range(cfg.training.steps), desc="Model Training", disable=False, initial=latest_step_idx)

    # Training Loop
    t_start = time.perf_counter()
    dataset_size = len(imagenet_train_ds)
    global_batch_size = cfg.dataset.train.batch_size * device_mesh["dp"].size()
    steps_per_epoch = math.ceil(dataset_size / global_batch_size)
    for batch_idx, (input, target) in enumerate(
        infinite_dataloader(train_dataloader, train_sampler), start=latest_step_idx
    ):
        # Skip to latest step.
        data_load_time = time.perf_counter() - t_start

        # Set training mode.
        model.train()

        # Match model input shape.
        if cfg.model.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # Move input and target to GPU, which is set by torch.cuda.set_device.
        input = input.cuda()
        target = target.cuda()

        # Model Forward Pass
        output = model(input)
        loss = loss_fn(output, target)
        loss_value = loss.detach().item()

        # Model Backward Pass
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

        # Step Optimizer and LR Scheduler
        optimizer.step()
        lr_scheduler.step()

        # Validation
        if batch_idx % cfg.training.val_interval == 0 and batch_idx > 0:
            model.eval()
            with torch.inference_mode():
                loss_sum = 0
                batch_count = 0
                for input, target in val_dataloader:
                    # Forward Pass
                    input = input.cuda()
                    target = target.cuda()
                    output = model(input)
                    loss = loss_fn(output, target)
                    # Reduce loss (for logging ONLY). If not using CP, sufficient to reduce across DP instead of HSDP.
                    torch.distributed.all_reduce(
                        loss,
                        op=torch.distributed.ReduceOp.AVG,
                        group=device_mesh["hsdp"].get_group(),
                    )
                    loss_sum += loss.detach().item()
                    batch_count += 1

            # Normalize summed loss by distributed size and number of batches.
            normalized_loss = loss_sum / batch_count
            if torch.distributed.get_rank() == 0:
                # Log validation loss.
                _logger.info(f"Validation Loss: {normalized_loss:.3f}")
                if cfg.profiling.wandb:
                    wandb.log({"val/loss": normalized_loss})

            # Save validated checkpoint.
            if cfg.training.checkpoint.path:
                # Create checkpoint sub-directory.
                ckpt_dir = Path(cfg.training.checkpoint.path) / f"step_{batch_idx}_loss_{normalized_loss:.3f}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                # Save model and optimizer checkpoints.
                state_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.distributed.checkpoint.save(state_dict, checkpoint_id=ckpt_dir)
                # Relax checkpoint permissions, which may be helpful when saving checkpoints in a container owned by root.
                mode = 0o777
                for dirpath, _, filenames in os.walk(ckpt_dir):
                    # Change current directory perms.
                    os.chmod(dirpath, mode)
                    for filename in filenames:
                        # Change file perms.
                        file_path = Path(dirpath) / filename
                        os.chmod(file_path, mode)
                if torch.distributed.get_rank() == 0:
                    _logger.info(f"Saved validated checkpoint to: {ckpt_dir}")

        # Log metrics to logger and wandb on main process.
        if torch.distributed.get_rank() == 0 and batch_idx % cfg.training.log_interval == 0:
            # Measure step time.
            t_end = time.perf_counter()
            step_time = t_end - t_start
            # Compute average learning rate.
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            # Log metrics to STDOUT.
            _logger.info(
                f"Train: [Epoch {batch_idx * global_batch_size // dataset_size} / Step {(batch_idx % steps_per_epoch) + 1:>4d}/{steps_per_epoch} "
                f"({100.0 * ((batch_idx % steps_per_epoch) + 1) / steps_per_epoch:>3.0f}%)]  "
                f"Loss: {loss_value:#.3g}  "
                f"Time: {step_time:.3f}s ({global_batch_size / step_time:>7.2f} samples/sec)  "
                f"Memory: {torch.cuda.memory.max_memory_reserved() / 1024**3} GB   "
                f"LR: {lr:.3e}  "
                f"Data Load Time: {data_load_time:.3f}s"
            )
            # Log metrics to WandB.
            if cfg.profiling.wandb:
                wandb.log(
                    {
                        "train/loss": loss_value,
                        "train/global_step": batch_idx,
                        "train/learning_rate": lr,
                        "train/grad_norm": total_norm,
                        "train/epoch": batch_idx * global_batch_size / dataset_size,
                        "train/step_time": step_time,
                    }
                )

            # Update Torch profiler snapshot.
            if cfg.profiling.torch_memory_profile:
                torch_memory_profiler_snapshot = torch.cuda.memory._snapshot()

            progress_bar.update(1)

        # Reset timer.
        t_start = time.perf_counter()

        # Terminate if completed training steps.
        if batch_idx >= cfg.training.steps:
            break

    # Dump memory profiler snapshot.
    # TODO(@cspades): Migrate to the new Torch profiler!
    if cfg.profiling.torch_memory_profile:
        from pickle import dump

        with open(
            # Path will only exist when using @hydra.main()!
            Path(HydraConfig.get().runtime.output_dir) / "torch_memory_profiler_snapshot.pickle",
            "wb",
        ) as f:
            dump(torch_memory_profiler_snapshot, f)

    if cfg.profiling.wandb and torch.distributed.get_rank() == 0:
        wandb.finish()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
