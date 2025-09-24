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
import sys
import time

import hydra
import torch
import transformer_engine.pytorch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from tqdm import tqdm
from transformer_engine.common.recipe import Format
from transformers import AutoConfig, AutoModelForMaskedLM

from checkpoint import load_checkpoint_ddp, save_checkpoint_ddp, save_final_model_ddp
from dataset import create_dataloader
from distributed_config import DistributedConfig
from scheduler import get_linear_schedule_with_warmup
from utils import get_batch_on_this_cp_rank


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:  # noqa: C901
    """Train ESM-2 with TE layers using ddp.

    Returns:
        float: The loss value for the final batch.
    """
    # Initialize the distributed configuration, including creating the distributed process group.

    # Get the script name without extension and add it to checkpoint directory
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    ckpt_dir = os.path.join(args.checkpoint.ckpt_dir, script_name)
    logger.info(f"Checkpoint directory: {ckpt_dir}")
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.context_parallel is None:
        raise ValueError("Context parallel is required for this recipe.")

    cp_size = args.context_parallel
    dp_size = args.data_parallel
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(dist_config.local_rank)

    # Create a device mesh for DDP. While this isn't strictly necessary, it mirrors the device mesh we create for FSDP2
    # and MFSDP.
    device = torch.device(f"cuda:{dist_config.local_rank}")

    device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(dp_size, cp_size, 1),
            mesh_dim_names=("dp", "cp", "tp"), # TODO: Remove the TP part.
        )

    if dist_config.is_main_process():
        wandb.init(**args.wandb_init_args, config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True))

    # Create an empty ESM-2 model with a masked language model head.
    config = AutoConfig.from_pretrained(args.model_tag, trust_remote_code=True, dtype=torch.bfloat16)
    config.attn_mask_type = "no_mask"
    config.max_position_embeddings = args.dataset.max_seq_length
    # If we're using sequence packing with TE layers, we need to pass the `attn_input_format` argument.
    if args.dataset.use_sequence_packing:
        config.attn_input_format = "thd"
    model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
    #TODO copy file over and then hack it.

    try:
        del model.esm.contact_head
    except AttributeError:
        pass

    # Log model and number of parameters on main process.
    if dist_config.is_main_process():
        logger.info("model:\n%s", model)
        logger.info(f"total number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer.
    # Convert OmegaConf to regular dict to avoid serialization issues later
    adamw_kwargs = OmegaConf.to_container(args.adamw_kwargs, resolve=True)
    optimizer = AdamW(model.parameters(), **adamw_kwargs)
    lr_scheduler_kwargs = OmegaConf.to_container(args.lr_scheduler_kwargs, resolve=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, **lr_scheduler_kwargs)

    model = model.to(device=device)
    group_fsdp_cp = device_mesh[("dp", "cp")]._flatten("dp_cp").get_group()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_config.local_rank],
        output_device=dist_config.local_rank,
        process_group=group_fsdp_cp,
    )

    cp_group = device_mesh["cp"].get_group()
    if cp_size > 1:
        for i, transformer_layer in enumerate(model.module.esm.encoder.layers):
            transformer_layer.set_context_parallel_group(
                cp_group,
                torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()),
                torch.cuda.Stream()
            )
    # TODO: IDK if I need this barrier
    torch.distributed.barrier()

    # Create an FP8 recipe
    if args.fp8_config.enabled:
        fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
            fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
        )
    else:
        fp8_recipe = None

    # Create a dataloader that just infinitely loops over the dataset.
    train_iterator = create_dataloader(dist_config, **args.dataset)

    # Load checkpoint if it exists and resume is enabled
    start_step = 0
    if args.checkpoint.resume_from_checkpoint:
        model, optimizer, start_step = load_checkpoint_ddp(
            model=model,
            optimizer=optimizer,
            ckpt_dir=ckpt_dir,
            dist_config=dist_config,
            logger=logger,
        )
        # Increment start_step to avoid re-running the checkpointed step
        start_step = min(start_step + 1, args.num_train_steps)
        try:
            scheduler.last_epoch = start_step - 1
        except Exception:
            for _ in range(start_step):
                scheduler.step()

    # Training loop.
    model.train()
    if dist_config.is_main_process():
        progress_bar = tqdm(range(args.num_train_steps), desc="Training", disable=False)
    previous_step_time = time.perf_counter()
    loss_value = None
    for step in range(start_step, args.num_train_steps):
        # Get batch.
        batch = next(train_iterator)
        print(f"Batch keys: {list(batch.keys())}")
        input_ids, labels, attention_mask = get_batch_on_this_cp_rank(
            cu_seqlens_padded=None,
            input_ids_padded=batch['input_ids'],
            labels_padded=batch['labels'],
            position_ids_padded=batch['attention_mask'], # TODO HACK IT
            cp_group=cp_group,
            qvk_format="bshd"
        )
        batch['input_ids'] = input_ids
        batch['labels'] = labels
        batch['attention_mask'] = attention_mask

        
        batch = {k: v.to(device, non_blocking=True).contiguous() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

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

        if (
            args.checkpoint.save_every_n_steps > 0 and step % args.checkpoint.save_every_n_steps == 0 and step > 0
        ):  # Skip step 0
            save_checkpoint_ddp(
                model=model,
                optimizer=optimizer,
                ckpt_dir=ckpt_dir,
                step=step,
                dist_config=dist_config,
                logger=logger,
            )

        # Log metrics to logger and wandb on main process.
        if dist_config.is_main_process():
            loss_value = loss.detach().item()
            current_time = time.perf_counter()
            step_time = current_time - previous_step_time
            previous_step_time = current_time

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Step %d loss: %f, grad_norm: %f, lr: %f",
                step,
                loss_value,
                total_norm,
                current_lr,
            )
            wandb.log(
                {
                    "train/loss": loss_value,
                    "train/global_step": step,
                    "train/learning_rate": current_lr,
                    "train/grad_norm": total_norm,
                    "train/step_time": step_time,
                }
            )

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss_value})

    # Save final model using save_pretrained
    final_model_dir = os.path.join(ckpt_dir, "final_model")
    save_final_model_ddp(
        model=model,
        save_directory=final_model_dir,
        dist_config=dist_config,
        logger=logger,
    )

    # Clean up distributed training
    if dist_config.is_main_process():
        wandb.finish()

    torch.distributed.destroy_process_group()

    return loss_value


if __name__ == "__main__":
    main()
