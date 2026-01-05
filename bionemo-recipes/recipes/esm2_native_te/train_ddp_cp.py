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
from pathlib import Path

import hydra
import torch
import transformer_engine.pytorch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from transformer_engine.common.recipe import Format
from transformers import AutoConfig, AutoModelForMaskedLM

from checkpoint import load_checkpoint_ddp, save_checkpoint_ddp, save_final_model_ddp, should_save_checkpoint
from dataset import create_cp_dataloader
from distributed_config import DistributedConfig
from perf_logger import PerfLogger
from scheduler import get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)


@hydra.main(config_path="hydra_config", config_name="L0_sanity_cp", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train ESM-2 with TE layers using DDP.

    Returns:
        float: The loss value for the final batch.
    """
    # Initialize the distributed configuration, including creating the distributed process group.
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    # Validate that world_size is divisible by cp_size
    if dist_config.world_size % args.cp_size != 0:
        raise ValueError(
            f"world_size ({dist_config.world_size}) must be divisible by cp_size ({args.cp_size}). "
            f"Set cp_size to a divisor of world_size."
        )

    # Calculate DDP size (number of data parallel replicas)
    ddp_size = dist_config.world_size // args.cp_size

    logger.info(
        f"Creating device mesh: world_size={dist_config.world_size}, ddp_size={ddp_size}, cp_size={args.cp_size}"
    )

    # Create a device mesh for DDP and CP.
    # The mesh is organized as [DDP_dimension, CP_dimension] where:
    # - DDP dimension: number of data parallel replicas (world_size // cp_size)
    # - CP dimension: context parallel size
    # Total ranks = ddp_size * cp_size = world_size
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(ddp_size, args.cp_size),
        mesh_dim_names=("ddp", "cp"),
    )

    # Create an FP8 recipe -- this is only used if FP8 is enabled in the config.
    fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
        fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
    )

    # Create an empty ESM-2 model with a masked language model head, e.g. "nvidia/esm2_t6_8M_UR50D".
    # Note: token_dropout is set to False because it's not compatible with context parallelism.
    config = AutoConfig.from_pretrained(
        args.model_tag, trust_remote_code=True, token_dropout=False, dtype=torch.bfloat16
    )
    # If we're using sequence packing with TE layers, we need to pass the `attn_input_format` argument.
    if args.use_sequence_packing:
        config.attn_input_format = "thd"

    # Optionally use transformer engine to initialize only fp8 versions of weights by setting
    # `fp8_config.fp8_model_init_kwargs.enabled` to `True`, as opposed to using the default where both bfloat16 and fp8
    # versions of weights are kept.
    with transformer_engine.pytorch.fp8_model_init(recipe=fp8_recipe, **args.fp8_config.fp8_model_init_kwargs):
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    logger.info("Initialized Model:\n%s", model)

    # Create optimizer.
    optimizer = AdamW(model.parameters(), **args.adamw_kwargs)
    scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    model = model.to(device=device)
    group_fsdp_cp = device_mesh[("ddp", "cp")]._flatten("dp_cp").get_group()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_config.local_rank],
        output_device=dist_config.local_rank,
        process_group=group_fsdp_cp,
    )

    if args.cp_size > 1:
        for i, transformer_layer in enumerate(model.module.esm.encoder.layers):
            logger.debug(f"Rank {dist_config.rank}: Setting CP group for layer {i}")
            transformer_layer.set_context_parallel_group(
                device_mesh["cp"].get_group(),
                torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()),
                torch.cuda.Stream(),
            )

    # Context Parallelism requires THD Sequence Packing.
    assert args.use_sequence_packing, "Context Parallelism requires THD Sequence Packing."

    train_dataloader, dataset_or_sampler = create_cp_dataloader(
        dist_config,
        cp_mesh=device_mesh["cp"],
        **args.dataset,
    )

    if args.use_torch_compile:
        # If we're using torch.compile, we need to do this before loading the checkpoint to ensure key consistency.
        model = torch.compile(model)

    # If we're resuming from a checkpoint, load it and set the start step. Otherwise, start from step 0.
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_ddp" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_ddp(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
            dataloader=train_dataloader,
        )
    else:
        start_step = 0
        epoch = 0

    perf_logger = PerfLogger(dist_config, args)

    # Training loop
    step = start_step
    while step < args.num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa PLW2901

            # Forward pass with mixed precision.
            with transformer_engine.pytorch.fp8_autocast(enabled=args.fp8_config.enabled, fp8_recipe=fp8_recipe):
                outputs = model(**batch)

            # Backward pass.
            loss = outputs.loss
            loss.backward()

            # Compute and clip gradient norms.
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

            # Step optimizer.
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            perf_logger.log_step(
                step=step,
                batch=batch,
                outputs=outputs,
                grad_norm=total_norm,
                lr=optimizer.param_groups[0]["lr"],
            )

            if ckpt_path and should_save_checkpoint(step, args.checkpoint.save_every_n_steps):
                save_checkpoint_ddp(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ckpt_path=ckpt_path,
                    step=step,
                    epoch=epoch,
                    dist_config=dist_config,
                    dataloader=train_dataloader if args.dataset.use_stateful_dataloader else None,
                    max_checkpoints=args.checkpoint.max_checkpoints,
                )

            step += 1
            if step >= args.num_train_steps:
                break

        # Dataloader exhausted, incrementing epoch
        epoch += 1
        dataset_or_sampler.set_epoch(epoch)

    # Save final model to a .safetensors file.
    if args.checkpoint.save_final_model and ckpt_path:
        save_final_model_ddp(
            model=model,
            save_directory=ckpt_path / "final_model",
            dist_config=dist_config,
        )

    # Clean up distributed training
    perf_logger.finish()
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
