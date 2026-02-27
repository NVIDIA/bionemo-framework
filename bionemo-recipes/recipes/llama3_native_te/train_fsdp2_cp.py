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

"""FSDP2 with Context Parallelism training script for Llama 3 with TransformerEngine.

Combines Fully Sharded Data Parallel v2 with Context Parallelism (CP), where each sequence is
split across multiple GPUs along the sequence dimension. This is useful for training with very long
sequences that do not fit into a single GPU's memory even with FSDP2 alone. Only supports
TE-accelerated models (NVLlamaForCausalLM).

For standard FSDP2 training without context parallelism, use ``train_fsdp2.py`` instead.
"""

import gc
import logging
from contextlib import nullcontext
from pathlib import Path

import hydra
import nvtx
import torch
import transformer_engine.pytorch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.optim import AdamW
from transformer_engine.common.recipe import Format

from checkpoint import (
    _ckpt_futures,
    load_checkpoint_fsdp2,
    save_checkpoint_fsdp2,
    save_final_model_fsdp2,
    should_save_checkpoint,
)
from collator import ContextParallelDataLoaderWrapper, DataCollatorForContextParallel
from dataset import create_bshd_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM
from perf_logger import PerfLogger
from scheduler import get_cosine_annealing_schedule_with_warmup
from train_fsdp2 import get_parameter_groups_with_weight_decay, run_validation, set_seed


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity_cp", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train Llama3 with TE layers using FSDP2 with Context Parallelism.

    Returns:
        float: The loss value for the final batch.
    """
    # --- Distributed Setup ---
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    # Set random seeds (same seed on ALL ranks for FSDP2/DTensor)
    seed = getattr(args, "seed", 42)
    set_seed(seed)

    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size // args.cp_size, args.cp_size),
        mesh_dim_names=("dp", "cp"),
    )
    logger.info("Created device mesh: %s", device_mesh)

    # --- Model Configuration ---
    fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
        fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
    )

    # Determine dtype for model initialization
    use_fp32_master_weights = getattr(args, "use_fp32_master_weights", False)
    model_dtype = torch.float32 if use_fp32_master_weights else torch.bfloat16

    if use_fp32_master_weights:
        logger.info("FP32 master weights enabled: model init in FP32")

    # --- Model Initialization ---
    config_kwargs = OmegaConf.to_container(args.config_kwargs, resolve=True) if args.config_kwargs else {}

    # Normalize rope_scaling: transformers >=5.0 expects "rope_type" instead of "type"
    if "rope_scaling" in config_kwargs and isinstance(config_kwargs["rope_scaling"], dict):
        rs = config_kwargs["rope_scaling"]
        if "type" in rs and "rope_type" not in rs:
            rs["rope_type"] = rs.pop("type")

    # Handle Spike-No-More embedding initialization
    if getattr(args, "spike_no_more_embedding_init", False):
        config_kwargs["embedding_init_std"] = 1.0
        config_kwargs["tie_word_embeddings"] = False
        logger.info("Spike-No-More enabled: embedding_init_std=1.0, tie_word_embeddings=False")

    # Handle Megatron-style scaled initialization
    if getattr(args, "use_megatron_scaled_init", False):
        config_kwargs["use_megatron_scaled_init"] = True
        logger.info("Megatron scaled init enabled: proj/fc2 use std/sqrt(2*num_layers)")

    config = NVLlamaConfig.from_pretrained(args.config_name_or_path, dtype=model_dtype, **config_kwargs)

    # Optionally use transformer engine to initialize only fp8 versions of weights by setting
    # `fp8_config.quantized_model_init_kwargs.enabled` to `True`, as opposed to using the default where both bfloat16
    # and fp8 versions of weights are kept.
    with (
        torch.device("meta") if args.use_meta_device else nullcontext(),
        transformer_engine.pytorch.quantized_model_init(
            recipe=fp8_recipe, **args.fp8_config.quantized_model_init_kwargs
        ),
    ):
        model = NVLlamaForCausalLM(config)

    logger.info("Initialized Model:\n%s", model)

    # --- Distributed Wrapping (FSDP2 + CP) ---
    cp_dp_mesh = device_mesh["dp", "cp"]._flatten(mesh_dim_name="dp_shard_cp")

    # Create MixedPrecisionPolicy for FSDP when using FP32 master weights
    mp_policy = None
    if use_fp32_master_weights:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
        )
        logger.info("MixedPrecisionPolicy: param_dtype=bf16, reduce_dtype=fp32, output_dtype=bf16")

    # Shard the transformer layers with FSDP. For Llama3, the transformer stack is in model.model.layers.
    # Each decoder layer should be individually sharded before sharding the full model.
    if mp_policy is not None:
        for layer in model.model.layers:
            fully_shard(layer, mesh=cp_dp_mesh, mp_policy=mp_policy)
        fully_shard(model, mesh=cp_dp_mesh, mp_policy=mp_policy)
    else:
        for layer in model.model.layers:
            fully_shard(layer, mesh=cp_dp_mesh)
        fully_shard(model, mesh=cp_dp_mesh)

    # Attach the CP group to the model.
    for layer in model.model.layers:
        layer.set_context_parallel_group(
            device_mesh["cp"].get_group(),
            torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()),
            torch.cuda.Stream(),
        )

    if args.use_meta_device:
        # TE layers require special handling to initialize the weights from the meta device.
        model.init_empty_weights()

    # --- Optimizer & Scheduler ---
    adamw_kwargs = OmegaConf.to_container(args.adamw_kwargs, resolve=True)
    use_wd_grouping = getattr(args, "use_weight_decay_grouping", True)

    if use_wd_grouping:
        weight_decay = adamw_kwargs.pop("weight_decay", 0.1)
        skip_embedding_wd = getattr(args, "skip_embedding_weight_decay", False)
        param_groups = get_parameter_groups_with_weight_decay(
            model=model,
            weight_decay=weight_decay,
            skip_embeddings=skip_embedding_wd,
        )
        optimizer = AdamW(param_groups, **adamw_kwargs)  # type: ignore
        logger.info(f"Weight decay grouping enabled: wd={weight_decay}, skip_embeddings={skip_embedding_wd}")
    else:
        optimizer = AdamW(model.parameters(), **adamw_kwargs)  # type: ignore
        logger.info(f"Weight decay grouping disabled: wd={adamw_kwargs.get('weight_decay', 0.1)} for all params")

    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    if args.use_torch_compile:
        # If we're using torch.compile, we need to do this before loading the checkpoint to ensure key consistency.
        model = torch.compile(model)

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
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_fsdp2" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        logger.info("Attempting to load checkpoint from %s", ckpt_path)
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_fsdp2(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
            dataloader=train_dataloader,
            process_group=cp_dp_mesh.get_group(),
        )
        logger.info("Checkpoint loaded, resuming from step %s, epoch %s", start_step, epoch)
    else:
        logger.info("No checkpoint to load, starting from scratch")
        start_step = 0
        epoch = 0

    perf_logger = PerfLogger(dist_config, args)

    # Setup validation if enabled
    val_config = getattr(args, "validation", None)
    val_enabled = val_config is not None and getattr(val_config, "enabled", False)
    val_dataloader = None

    if val_enabled:
        val_data_path = getattr(val_config, "data_path", None)
        if val_data_path:
            logger.info(f"Setting up validation dataloader from {val_data_path}")
            val_dataset_kwargs = OmegaConf.to_container(args.dataset, resolve=True)
            val_dataset_kwargs["load_dataset_kwargs"] = {
                "path": "json",
                "data_files": val_data_path,
                "split": "train",
                "streaming": True,
            }
            val_dataset_kwargs["use_stateful_dataloader"] = False
            val_dataset_kwargs["num_workers"] = 0
            if hasattr(val_config, "micro_batch_size") and val_config.micro_batch_size is not None:
                val_dataset_kwargs["micro_batch_size"] = val_config.micro_batch_size

            if args.use_sequence_packing:
                val_dataloader, _ = create_thd_dataloader(dist_config, **val_dataset_kwargs)
            else:
                val_dataloader, _ = create_bshd_dataloader(dist_config, **val_dataset_kwargs)

            logger.info(
                f"Validation enabled: every {val_config.eval_interval} steps, {val_config.num_batches} batches"
            )
        else:
            logger.warning("Validation enabled but no data_path specified, skipping validation")
            val_enabled = False

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
                    save_checkpoint_fsdp2(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        ckpt_path=ckpt_path,
                        step=step,
                        epoch=epoch,
                        dist_config=dist_config,
                        dataloader=train_dataloader if args.dataset.use_stateful_dataloader else None,
                        process_group=cp_dp_mesh.get_group(),
                        max_checkpoints=args.checkpoint.max_checkpoints,
                        async_save=args.checkpoint.async_save,
                    )

                # Run validation at specified interval
                if val_enabled and val_dataloader is not None and step > 0 and step % val_config.eval_interval == 0:
                    try:
                        val_metrics = run_validation(
                            model=model,
                            val_dataloader=val_dataloader,
                            num_batches=val_config.num_batches,
                            device=device,
                            dist_config=dist_config,
                        )
                        if dist_config.rank == 0:
                            logger.info(
                                f"[Step {step}] Validation: loss={val_metrics['val_loss']:.4f} "
                                f"(megatron={val_metrics.get('val_loss_megatron', 0):.4f}), "
                                f"ppl={val_metrics['val_ppl']:.2f}, tokens={val_metrics['val_tokens']:,}"
                            )
                        perf_logger.log_validation(step, val_metrics)
                    except Exception as e:
                        logger.error(f"Validation failed at step {step}: {e}")
                        torch.distributed.barrier()

                step += 1
                if step >= args.num_train_steps:
                    break

        # Dataloader exhausted, incrementing epoch
        epoch += 1
        if dataset_or_sampler is not None:  # The dataset only exists on rank 0
            dataset_or_sampler.set_epoch(epoch)

    # --- Cleanup ---
    if args.checkpoint.save_final_model and ckpt_path:
        save_final_model_fsdp2(
            model=model,
            save_directory=ckpt_path / "final_model",
            dist_config=dist_config,
        )

    # Make sure we don't have any outstanding checkpoint save futures.
    if args.checkpoint.async_save and "fsdp2" in _ckpt_futures and _ckpt_futures["fsdp2"] is not None:
        _ckpt_futures["fsdp2"].result()

    perf_logger.finish()
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
