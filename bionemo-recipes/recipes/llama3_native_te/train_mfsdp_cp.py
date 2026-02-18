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

"""Megatron-FSDP with Context Parallelism training script for Llama 3 with TransformerEngine.

Combines Megatron-FSDP with Context Parallelism (CP), where each sequence is split across multiple
GPUs along the sequence dimension. This is useful for training with very long sequences that do not
fit into a single GPU's memory even with FSDP alone. Only supports TE-accelerated models
(NVLlamaForCausalLM).

For standard FSDP2 training without context parallelism, use ``train_fsdp2.py`` instead.
For FSDP2 with context parallelism, use ``train_fsdp2_cp.py`` instead.
"""

import gc
import logging
from pathlib import Path

import hydra
import nvtx
import torch
import transformer_engine.pytorch
from megatron_fsdp.fully_shard import fully_shard as mfsdp_fully_shard
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from transformer_engine.common.recipe import Format

from checkpoint import (
    load_checkpoint_mfsdp,
    save_checkpoint_mfsdp,
    save_final_model_mfsdp,
    should_save_checkpoint,
)
from collator import ContextParallelDataLoaderWrapper, DataCollatorForContextParallel
from dataset import create_bshd_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM
from perf_logger import PerfLogger
from scheduler import get_cosine_annealing_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity_cp", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train Llama3 with TE layers using Megatron-FSDP with Context Parallelism.

    Returns:
        float: The loss value for the final batch.
    """
    # --- Distributed Setup ---
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    # Create a 3D device mesh (dp, cp, tp) where tp is a dummy dimension of size 1 required by mfsdp.
    dp_size = dist_config.world_size // args.cp_size
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dp_size, args.cp_size, 1),
        mesh_dim_names=("dp", "cp", "tp"),
    )
    logger.info("Created device mesh: %s", device_mesh)

    # --- Model Configuration ---
    fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
        fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
    )

    # --- Model Initialization ---
    config = NVLlamaConfig.from_pretrained(args.config_name_or_path, dtype=torch.bfloat16, **args.config_kwargs)

    # mfsdp does not support tied weight parameters. If tie_word_embeddings is enabled, we need to untie them so that
    # lm_head.weight and embed_tokens.weight are separate parameters for the mfsdp optimizer buffer.
    if config.tie_word_embeddings:
        logger.warning(
            "Megatron-FSDP does not support tied weight parameters. Setting tie_word_embeddings=False. "
            "This means lm_head.weight will be a separate parameter from embed_tokens.weight."
        )
        config.tie_word_embeddings = False

    # Optionally use transformer engine to initialize only fp8 versions of weights by setting
    # `fp8_config.quantized_model_init_kwargs.enabled` to `True`, as opposed to using the default where both bfloat16
    # and fp8 versions of weights are kept.
    # NOTE: Meta device initialization for mfsdp is handled by the `init_model_with_meta_device` kwarg in
    # fully_shard_kwargs, so we do NOT use `torch.device("meta")` here (unlike train_fsdp2_cp.py).
    with transformer_engine.pytorch.quantized_model_init(
        recipe=fp8_recipe, **args.fp8_config.quantized_model_init_kwargs
    ):
        model = NVLlamaForCausalLM(config)

    logger.info("Initialized Model:\n%s", model)

    # --- Optimizer (created before mfsdp wrapping, will be wrapped by fully_shard) ---
    # Convert OmegaConf to regular dict to avoid serialization issues (BIONEMO-2873).
    optimizer = AdamW(model.parameters(), **OmegaConf.to_container(args.adamw_kwargs, resolve=True))  # type: ignore

    # --- Distributed Wrapping (Megatron-FSDP + CP) ---
    model, optimizer = mfsdp_fully_shard(
        module=model,
        optimizer=optimizer,
        fsdp_unit_modules=[
            transformer_engine.pytorch.TransformerLayer,
            transformer_engine.pytorch.LayerNorm,
            transformer_engine.pytorch.LayerNormLinear,
        ],
        device_mesh=device_mesh,
        dp_shard_dim="dp",
        tp_dim="tp",
        **args.fully_shard_kwargs,
    )

    # Attach the CP group to each transformer layer.
    for layer in model.module.model.layers:
        layer.set_context_parallel_group(
            device_mesh["cp"].get_group(),
            torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()),
            torch.cuda.Stream(),
        )

    # --- Scheduler (must be created after mfsdp wrapping since fully_shard modifies the optimizer) ---
    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    if args.use_torch_compile:
        logger.warning(
            "BIONEMO-2977: Using torch.compile with mfsdp is currently not supported. `use_torch_compile` was set to "
            "true, but will be ignored."
        )

    # --- Data Loading ---
    # Create the context-aware dataloader.
    if args.dataset.get("pad_sequences_to_be_divisible_by", None) is None:
        # The dual chunk algorithm gives each CP rank 2 chunks from each sequence, so we need each sequence to be
        # divisible by cp_mesh.size() * 2.
        logger.info("pad_sequences_to_be_divisible_by is not provided, using cp_mesh.size() * 2")
        OmegaConf.update(args, "dataset.pad_sequences_to_be_divisible_by", device_mesh["cp"].size() * 2)

    # We only create the dataloader on rank 0, which is responsible for loading data for all CP (and eventually TP)
    # ranks. This ensures that the data remains synchronized, even if we're using a non-deterministic data pipeline.
    if device_mesh["cp"].get_local_rank() == 0:
        if args.use_sequence_packing:
            train_dataloader, dataset_or_sampler = create_thd_dataloader(dist_config, **args.dataset)
        else:
            train_dataloader, dataset_or_sampler = create_bshd_dataloader(dist_config, **args.dataset)

        train_dataloader.collate_fn = DataCollatorForContextParallel(
            collator=train_dataloader.collate_fn,
            device_mesh=device_mesh,
            qkv_format=args.config_kwargs.attn_input_format,
            is_causal_lm=True,
        )

    else:
        train_dataloader = None
        dataset_or_sampler = None

    # On all ranks, we create a ContextParallelDataLoaderWrapper that broadcasts the data from cp rank 0.
    train_dataloader = ContextParallelDataLoaderWrapper(train_dataloader, device_mesh["cp"])

    # --- Checkpoint Resume ---
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_mfsdp" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        logger.info("Attempting to load checkpoint from %s", ckpt_path)
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_mfsdp(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
            dataloader=train_dataloader,
        )
        logger.info("Checkpoint loaded, resuming from step %s, epoch %s", start_step, epoch)
    else:
        logger.info("No checkpoint to load, starting from scratch")
        start_step = 0
        epoch = 0

    perf_logger = PerfLogger(dist_config, args)

    gc.collect()
    torch.cuda.empty_cache()

    # --- Training Loop ---
    logger.info("Starting training loop from step %s to %s", start_step, args.num_train_steps)
    step = start_step
    micro_step = 0  # Gradient accumulation step counter
    while step < args.num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

            micro_step += 1

            # Forward pass with mixed precision.
            with nvtx.annotate("Forward pass", color="green"):
                with transformer_engine.pytorch.autocast(enabled=args.fp8_config.enabled, recipe=fp8_recipe):
                    outputs = model(**batch)

            # Backward pass - scale loss by grad_acc_steps for proper gradient averaging
            loss = outputs.loss / args.grad_acc_steps

            with nvtx.annotate("Backward pass", color="red"):
                loss.backward()

            # Log microbatch step data for accumulation metrics
            perf_logger.log_micro_step(step=step, batch=batch, outputs=outputs)

            # The end of a "full" step (i.e. after possibly multiple gradient accumulation steps).
            if micro_step % args.grad_acc_steps == 0:
                micro_step = 0

                # Compute and clip gradient norms.
                # NOTE: grad clipping with mfsdp has been reported to cause hangs in some configurations.
                # If you experience hangs, try commenting out the clip_grad_norm_ call.
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Step optimizer.
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                perf_logger.log_step(
                    step=step,
                    grad_norm=total_norm,
                    lr=optimizer.param_groups[0]["lr"],
                )

                if ckpt_path and should_save_checkpoint(step, args.checkpoint.save_every_n_steps):
                    save_checkpoint_mfsdp(
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
        if dataset_or_sampler is not None:  # The dataset only exists on rank 0
            dataset_or_sampler.set_epoch(epoch)

    # --- Cleanup ---
    if args.checkpoint.save_final_model and ckpt_path:
        save_final_model_mfsdp(
            model=model,
            save_directory=ckpt_path / "final_model",
            dist_config=dist_config,
        )

    perf_logger.finish()
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
