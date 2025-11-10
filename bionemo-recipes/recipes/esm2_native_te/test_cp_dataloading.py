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
from dataset import create_bshd_dataloader, create_thd_dataloader, CPAwareDataloader
from distributed_config import DistributedConfig
from perf_logger import PerfLogger
from scheduler import get_linear_schedule_with_warmup
from utils import DummyDataloader


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        f"Creating device mesh: world_size={dist_config.world_size}, "
        f"ddp_size={ddp_size}, cp_size={args.cp_size}"
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
    config = AutoConfig.from_pretrained(args.model_tag, trust_remote_code=True, dtype=torch.bfloat16)
    # If we're using sequence packing with TE layers, we need to pass the `attn_input_format` argument.
    if args.use_sequence_packing:
        config.attn_input_format = "thd"

    # Optionally use transformer engine to initialize only fp8 versions of weights by setting
    # `fp8_config.fp8_model_init_kwargs.enabled` to `True`, as opposed to using the default where both bfloat16 and fp8
    # versions of weights are kept.
    with transformer_engine.pytorch.fp8_model_init(recipe=fp8_recipe, **args.fp8_config.fp8_model_init_kwargs):
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    logger.info("Initialized Model:\n%s", model)

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
    group_fsdp_cp = device_mesh[("ddp", "cp")]._flatten("dp_cp").get_group()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_config.local_rank],
        output_device=dist_config.local_rank,
        process_group=group_fsdp_cp,
    )
    cp_group = device_mesh["cp"].get_group()
    cp_rank = device_mesh.get_local_rank("cp")

    if args.cp_size > 1:
        for i, transformer_layer in enumerate(model.module.esm.encoder.layers):
            logger.debug(f"Rank {dist_config.rank}: Setting CP group for layer {i}")
            transformer_layer.set_context_parallel_group(
                cp_group,
                torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()),
                torch.cuda.Stream()
            )

    torch.distributed.barrier()

    
    # If we're using sequence packing, create a THD dataloader, otherwise create a BSHD dataloader.
    train_dataloader, dataset_or_sampler = (
        create_thd_dataloader(dist_config, **args.dataset)
        if args.use_sequence_packing
        else create_bshd_dataloader(dist_config, **args.dataset)
    )
    # Make a dummy dataloader that just loads the mock data

    dummy_dataloader = DummyDataloader(cp_size=args.cp_size)
    train_dataloader = CPAwareDataloader(dummy_dataloader, dist_config, cp_group, cp_rank, max_seq_length=args.dataset.max_seq_length)

    sample = next(iter(train_dataloader))

    # Now print out the (1) CP rank, (2) Global rank and (3) sample batch.
    print(f"CP rank: {cp_rank}")
    print(f"Global rank: {dist_config.rank}")
    print(f"Sample batch: {sample}")
    # Clean up distributed training
    torch.distributed.destroy_process_group()

    return None


if __name__ == "__main__":
    main()
