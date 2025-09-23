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
from dataset import create_dataloader
from distributed_config import DistributedConfig
from perf_logger import PerfLogger
from scheduler import get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train ESM-2 with TE layers using ddp.

    Returns:
        float: The loss value for the final batch.
    """
    # Initialize the distributed configuration, including creating the distributed process group.
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(dist_config.local_rank)

    # Create a device mesh for DDP. While this isn't strictly necessary, it mirrors the device mesh we create for FSDP2
    # and MFSDP.
    device = torch.device(f"cuda:{dist_config.local_rank}")
    device_mesh = init_device_mesh("cuda", mesh_shape=(dist_config.world_size,), mesh_dim_names=("ddp",))

    # Create an empty ESM-2 model with a masked language model head.
    config = AutoConfig.from_pretrained(args.model_tag, trust_remote_code=True, dtype=torch.bfloat16)
    # If we're using sequence packing with TE layers, we need to pass the `attn_input_format` argument.
    if args.dataset.use_sequence_packing:
        config.attn_input_format = "thd"
    model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    # The huggingface model has a contact head that we don't use in masked language pre-training, so we delete it to
    # avoid errors with unused parameters.
    try:
        del model.esm.contact_head
    except AttributeError:
        pass

    # Create optimizer.
    optimizer = AdamW(model.parameters(), **args.adamw_kwargs)
    scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    model = model.to(device=device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_config.local_rank],
        output_device=dist_config.local_rank,
        device_mesh=device_mesh["ddp"],
    )

    # Create an FP8 recipe
    if args.fp8_config.enabled:
        fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
            fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
        )
    else:
        fp8_recipe = None

    # Create a dataloader that just infinitely loops over the dataset.
    train_iterator = create_dataloader(dist_config, **args.dataset)

    # If we're resuming from a checkpoint, load it and set the start step. Otherwise, start from step 0.
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_ddp" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        model, optimizer, scheduler, start_step = load_checkpoint_ddp(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
        )
    else:
        start_step = 0

    perf_logger = PerfLogger(dist_config, args)

    # Training loop.
    model.train()
    for step in range(start_step, args.num_train_steps):
        # Get batch.
        batch = next(train_iterator)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass with mixed precision.
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
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

        if ckpt_path and should_save_checkpoint(step, args.checkpoint.save_every_n_steps):
            save_checkpoint_ddp(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ckpt_path=ckpt_path,
                step=step,
                dist_config=dist_config,
            )

        perf_logger.log_step(step, loss.detach().item(), total_norm, optimizer.param_groups[0]["lr"])

    # Save final model using save_pretrained
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
