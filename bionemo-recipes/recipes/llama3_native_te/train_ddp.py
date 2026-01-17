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
from contextlib import nullcontext
from pathlib import Path

import hydra
import nvdlfw_inspect.api as debug_api
import torch
import transformer_engine
import transformer_engine.pytorch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from transformer_engine.common.recipe import Format
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from checkpoint import load_checkpoint_ddp, save_checkpoint_ddp, save_final_model_ddp, should_save_checkpoint
from dataset import create_bshd_dataloader, create_bshd_packed_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM
from perf_logger import PerfLogger
from scheduler import get_cosine_annealing_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train Llama3 with TE layers using DDP for genomic sequences.

    Returns:
        float: The loss value for the final batch.
    """
    # Initialize the distributed configuration, including creating the distributed process group.
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    # TE Debug feature logging
    if args.fp8_stats_config.enabled and not args.fp8_config.enabled:
        raise ValueError(
            "fp8_stats_config.enabled is true but fp8_config.enabled is false, please set fp8_config.enabled to true in the config if you wish to collect FP8 stats"
        )

    if args.fp8_stats_config.enabled:
        fp8_stats_file = args.fp8_stats_config.fp8_stats_file
        fp8_log_dir = Path(args.fp8_stats_config.fp8_log_dir) / f"rank_{dist_config.rank}"
        fp8_log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Logging FP8 stats to {fp8_log_dir}")
        te_features_dir = str(Path(transformer_engine.__file__).parent / "debug" / "features")
        debug_api.initialize(
            config_file=fp8_stats_file,
            feature_dirs=[te_features_dir],
            log_dir=fp8_log_dir,
            default_logging_enabled=True,
        )

    # Create a device mesh for DDP. While this isn't strictly necessary, it mirrors the device mesh we create for FSDP2
    # and MFSDP.
    device_mesh = init_device_mesh("cuda", mesh_shape=(dist_config.world_size,), mesh_dim_names=("dp",))

    # Create an FP8 recipe -- this is only used if FP8 is enabled in the config.
    fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
        fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
    )

    if args.use_te:
        config_class = NVLlamaConfig
        model_class = NVLlamaForCausalLM
    else:
        config_class = LlamaConfig
        model_class = LlamaForCausalLM

    # Create an empty Llama3 model with a causal language model head, e.g. "meta-llama/Meta-Llama-3-8B".
    config = config_class.from_pretrained(args.config_name_or_path, dtype=torch.bfloat16, **args.config_kwargs)

    # Optionally use transformer engine to initialize only fp8 versions of weights by setting
    # `fp8_config.fp8_model_init_kwargs.enabled` to `True`, as opposed to using the default where both bfloat16 and fp8
    # versions of weights are kept.
    with transformer_engine.pytorch.fp8_model_init(recipe=fp8_recipe, **args.fp8_config.fp8_model_init_kwargs):
        model = model_class(config)

    logger.info("Initialized Model:\n%s", model)

    # Create optimizer.
    optimizer = AdamW(model.parameters(), **args.adamw_kwargs)
    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    if args.fp8_stats_config.enabled:
        debug_api.infer_and_assign_layer_names(model)

    model = model.to(device=device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_config.local_rank],
        output_device=dist_config.local_rank,
        device_mesh=device_mesh["dp"],
    )

    if args.use_sequence_packing:
        if args.config_kwargs.attn_input_format == "bshd":
            # BSHD with full packing (cross-boundary attention, no cu_seqlens)
            train_dataloader, dataset_or_sampler = create_bshd_packed_dataloader(dist_config, **args.dataset)
        else:
            # THD with packing (respects boundaries via cu_seqlens)
            train_dataloader, dataset_or_sampler = create_thd_dataloader(dist_config, **args.dataset)
    else:
        # Standard BSHD with windowing (no packing)
        train_dataloader, dataset_or_sampler = create_bshd_dataloader(dist_config, **args.dataset)

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
    micro_step = 0  # Gradient accumulation step counter
    while step < args.num_train_steps:
        for batch in train_dataloader:
            print(batch["input_ids"].shape)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa PLW2901

            micro_step += 1
            # Use no_sync to prevent gradient synchronization until the last microbatch
            with model.no_sync() if micro_step % args.grad_acc_steps != 0 else nullcontext():
                # Forward pass with mixed precision.
                with transformer_engine.pytorch.fp8_autocast(enabled=args.fp8_config.enabled, fp8_recipe=fp8_recipe):
                    outputs = model(**batch)

                # Backward pass - scale loss by grad_acc_steps for proper gradient averaging
                loss = outputs.loss / args.grad_acc_steps
                loss.backward()

                # Log microbatch step data for accumulation metrics
                perf_logger.log_micro_step(batch=batch, outputs=outputs)
            if args.fp8_stats_config.enabled:
                debug_api.step()

            # Gradient accumulation - only step optimizer after accumulating gradients
            if micro_step % args.grad_acc_steps == 0:
                micro_step = 0

                # Compute and clip gradient norms.
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

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
    if args.fp8_stats_config.enabled:
        debug_api.end_debug()
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
