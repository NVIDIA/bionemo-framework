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
import transformers
import wandb
from megatron_fsdp.fully_shard import fully_shard
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from tqdm import tqdm
from transformer_engine.common.recipe import Format
from transformers import AutoConfig, AutoModelForMaskedLM

from checkpoint import load_checkpoint_mfsdp, save_checkpoint_mfsdp, save_final_model_mfsdp
from dataset import create_dataloader
from distributed_config import DistributedConfig
from scheduler import get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:  # noqa: C901
    """Train ESM-2 with TE layers using mfsdp.

    Model names are valid ESM-2 model sizes, e.g.:
    - "esm2_t6_8M_UR50D"
    - "esm2_t36_3B_UR50D"
    - "esm2_t48_15B_UR50D"

    Returns:
        float: The loss value for the final batch.
    """
    # Get the script name without extension and add it to checkpoint directory
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    ckpt_dir = os.path.join(args.checkpoint.ckpt_dir, script_name)
    logger.info(f"Checkpoint directory: {ckpt_dir}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize the distributed configuration, including creating the distributed process group.
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(dist_config.local_rank)

    # Create a device mesh for FSDP.
    # We have to create a dummy mesh dimension for context parallel and tensor parallel for things
    # to work correctly with mfsdp.
    device = torch.device(f"cuda:{dist_config.local_rank}")
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size, 1, 1),
        mesh_dim_names=("fsdp", "cp", "tp"),
    )

    if dist_config.is_main_process():
        wandb.init(**args.wandb_init_args, config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True))

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

    # Log model and number of parameters on main process.
    if dist_config.is_main_process():
        logger.info("model:\n%s", model)
        logger.info(f"total number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer.
    # Convert OmegaConf to regular dict to avoid serialization issues later
    adamw_kwargs = OmegaConf.to_container(args.adamw_kwargs, resolve=True)
    optimizer = AdamW(model.parameters(), **adamw_kwargs)

    # Wrap model in megatron-fsdp
    model, optimizer = fully_shard(
        module=model,
        optimizer=optimizer,
        fsdp_unit_modules=[
            transformer_engine.pytorch.TransformerLayer,
            transformer_engine.pytorch.LayerNorm,
            transformer_engine.pytorch.LayerNormLinear,
            transformers.models.esm.modeling_esm.EsmLayer,
        ],
        device_mesh=device_mesh,
        dp_shard_dim="fsdp",
        tp_dim="tp",
        **args.fully_shard_kwargs,
    )

    # This is important; the LR scheduler modifies optimizer.step(), so this needs to get created
    # after the optimizer gets wrapped in FSDP. Here we use a warmup and linear decay scheduler.
    lr_scheduler_kwargs = OmegaConf.to_container(args.lr_scheduler_kwargs, resolve=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, **lr_scheduler_kwargs)

    # Create a dataloader that just infinitely loops over the dataset.
    train_iterator = create_dataloader(dist_config, **args.dataset)

    # Create an FP8 recipe
    if args.fp8_config.enabled:
        fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
            fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
        )
        logger.info("Training with FP8: %s", fp8_recipe)
    else:
        fp8_recipe = None

    # Training loop.
    model.train()
    if dist_config.is_main_process():
        progress_bar = tqdm(range(args.num_train_steps), desc="Training", disable=False)

    # Load checkpoint if it exists and resume is enabled
    start_step = 0
    if args.checkpoint.resume_from_checkpoint:
        model, optimizer, start_step = load_checkpoint_mfsdp(
            model=model,
            optimizer=optimizer,
            ckpt_dir=ckpt_dir,
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
    previous_step_time = time.perf_counter()
    loss_value = None
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

        if (
            args.checkpoint.save_every_n_steps > 0 and step % args.checkpoint.save_every_n_steps == 0 and step > 0
        ):  # Skip step 0
            # For mfsdp, always use distributed checkpointing
            save_checkpoint_mfsdp(
                model=model,
                optimizer=optimizer,
                ckpt_dir=ckpt_dir,
                step=step,
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

    final_model_dir = os.path.join(ckpt_dir, "final_model")
    save_final_model_mfsdp(
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
