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
import time
from dataclasses import dataclass, field

import hydra
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM

from dataset import create_dataloader
from scheduler import get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class DistributedConfig:
    """Class to track distributed ranks."""

    rank: int = field(default_factory=dist.get_rank)
    local_rank: int = field(default_factory=lambda: int(os.environ["LOCAL_RANK"]))
    world_size: int = field(default_factory=dist.get_world_size)

    def is_main_process(self) -> bool:
        """This is the global rank 0 process, to be used for wandb logging, etc."""
        return self.rank == 0


@hydra.main(config_path="hydra_config", config_name="L0_sanity.yaml", version_base="1.2")
def main(args: DictConfig):
    """Train ESM-2 with TE layers using nvFSDP.

    Model names are valid ESM-2 model sizes, e.g.:
    - "esm2_t6_8M_UR50D"
    - "esm2_t36_3B_UR50D"
    - "esm2_t48_15B_UR50D"
    """
    # Initialize distributed training and create a device mesh for FSDP.
    # We have to create a dummy mesh dimension for context parallel and tensor parallel for things
    # to work correctly with nvFSDP.
    dist.init_process_group(backend="nccl")
    dist_config = DistributedConfig()
    torch.cuda.set_device(dist_config.local_rank)
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size, 1, 1),
        mesh_dim_names=("fsdp", "cp", "tp"),
    )
    device = torch.device(f"cuda:{dist_config.local_rank}")
    logger.info("Initialized distributed training: %s", dist_config)

    if dist_config.is_main_process():
        wandb.init(**args.wandb_init_args, config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True))

    # Create an empty ESM-2 model with a masked language model head.
    if "facebook" in args.model_name:
        config = AutoConfig.from_pretrained(args.model_name, dtype=torch.bfloat16)
        model = AutoModelForMaskedLM.from_config(config, attn_implementation="flash_attention_2")
        del model.esm.contact_head

    else:
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True, dtype=torch.bfloat16)
        config.max_seq_length = args.max_seq_length
        config.micro_batch_size = args.micro_batch_size
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    # Log model and number of parameters on main process.
    if dist_config.is_main_process():
        logger.info("model:\n%s", model)
        logger.info(f"total number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer.
    optimizer = AdamW(model.parameters(), **args.adamw_kwargs)
    scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    model = model.to(device=device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_config.local_rank],
        output_device=dist_config.local_rank,
        device_mesh=device_mesh["fsdp"],
    )

    # Training loop.
    model.train()
    if dist_config.is_main_process():
        progress_bar = tqdm(range(args.num_train_steps), desc="Training", disable=False)

    # Create a dataloader that just infinitely loops over the dataset.
    train_iterator, epoch_len = create_dataloader(
        args.data_path,
        args.micro_batch_size,
        max_length=args.max_seq_length,
    )

    # Training loop.
    previous_step_time = time.perf_counter()
    for step in range(args.num_train_steps):
        # Get batch.
        batch = next(train_iterator)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass with mixed precision.
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
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

        # Log metrics to logger and wandb on main process.
        if dist_config.is_main_process():
            current_time = time.perf_counter()
            step_time = current_time - previous_step_time
            previous_step_time = current_time

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Step %d loss: %f, grad_norm: %f, lr: %f",
                step,
                loss.detach().item(),
                total_norm,
                current_lr,
            )
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/global_step": step,
                    "train/learning_rate": current_lr,
                    "train/grad_norm": total_norm,
                    "train/epoch": step / epoch_len,
                    "train/step_time": step_time,
                }
            )

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

    # Clean up distributed training
    if dist_config.is_main_process():
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
