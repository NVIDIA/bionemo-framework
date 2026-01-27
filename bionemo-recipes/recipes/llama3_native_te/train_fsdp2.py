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

import gc
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import nvdlfw_inspect.api as debug_api
import torch
import transformer_engine
import transformer_engine.pytorch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.optim import AdamW
from torchdata.stateful_dataloader import StatefulDataLoader
from transformer_engine.common.recipe import Format
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from checkpoint import (
    _ckpt_futures,
    load_checkpoint_fsdp2,
    save_checkpoint_fsdp2,
    save_final_model_fsdp2,
    should_save_checkpoint,
)
from dataset import create_bshd_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig
from fp8_debugging import initialize_fp8_debugging
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM
from perf_logger import PerfLogger
from scheduler import get_cosine_annealing_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train Llama3 with TE layers using FSDP2.

    Returns:
        float: The loss value for the final batch.
    """
    # Initialize the distributed configuration, including creating the distributed process group.
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    # TE Debug feature logging - MUST be done BEFORE FSDP wrapping
    if args.fp8_stats_config.enabled:
        initialize_fp8_debugging(dist_config, **args.fp8_stats_config, fp8_enabled=args.fp8_config.enabled)

    # Create a device mesh for FSDP.
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

    # Determine dtype for model initialization
    # When use_fp32_master_weights=True, we create the model in FP32 and use MixedPrecisionPolicy
    # to cast to BF16 for forward/backward. This matches Megatron's main_params_dtype=torch.float32
    use_fp32_master_weights = getattr(args, "use_fp32_master_weights", False)
    model_dtype = torch.float32 if use_fp32_master_weights else torch.bfloat16

    if use_fp32_master_weights:
        logger.info("FP32 master weights enabled: model init in FP32, compute in BF16")

    # Create an empty Llama3 model with a causal language model head, e.g. "meta-llama/Meta-Llama-3-8B".
    # Convert DictConfig to regular dict to avoid JSON serialization issues in transformers logging
    config_kwargs: dict[str, Any] = {}
    if args.config_kwargs:
        config_kwargs_raw = (
            OmegaConf.to_container(args.config_kwargs, resolve=True)
            if isinstance(args.config_kwargs, DictConfig)
            else args.config_kwargs
        )
        if isinstance(config_kwargs_raw, dict):
            config_kwargs = {str(key): value for key, value in config_kwargs_raw.items()}
        else:
            raise TypeError("config_kwargs must be a dict when provided.")
    rope_scaling = config_kwargs.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        if "rope_type" not in rope_scaling and "type" in rope_scaling:
            rope_scaling = {**rope_scaling, "rope_type": rope_scaling.pop("type")}
        elif "rope_type" not in rope_scaling:
            rope_scaling = {**rope_scaling, "rope_type": "default"}
        config_kwargs["rope_scaling"] = rope_scaling

    # Handle Spike-No-More embedding initialization (https://arxiv.org/abs/2312.16903)
    # When enabled, embeddings are initialized with std=1.0 instead of 0.02 to prevent loss spikes.
    if getattr(args, "spike_no_more_embedding_init", False):
        config_kwargs["embedding_init_std"] = 1.0
        config_kwargs["tie_word_embeddings"] = False  # Must not share embeddings with output weights
        logger.info("Spike-No-More enabled: embedding_init_std=1.0, tie_word_embeddings=False")

    # Handle Megatron-style scaled initialization for residual output layers
    # When enabled, proj and fc2 use std/sqrt(2*num_layers) instead of std
    if getattr(args, "use_megatron_scaled_init", False):
        config_kwargs["use_megatron_scaled_init"] = True
        logger.info("Megatron scaled init enabled: proj/fc2 use std/sqrt(2*num_layers)")

    config = config_class.from_pretrained(args.config_name_or_path, dtype=model_dtype, **config_kwargs)
    if getattr(config, "rope_theta", None) is None:
        rope_type = None
        if isinstance(config_kwargs.get("rope_scaling"), dict):
            rope_type = config_kwargs["rope_scaling"].get("rope_type")
        config.rope_theta = 500000.0 if rope_type == "llama3" else 10000.0
        logger.warning("config.rope_theta was None; defaulting to %s", config.rope_theta)

    # Optionally use transformer engine to initialize only fp8 versions of weights by setting
    # `fp8_config.fp8_model_init_kwargs.enabled` to `True`, as opposed to using the default where both bfloat16 and fp8
    # versions of weights are kept.
    with (
        torch.device("meta") if args.use_meta_device else nullcontext(),
        transformer_engine.pytorch.fp8_model_init(recipe=fp8_recipe, **args.fp8_config.fp8_model_init_kwargs),
    ):
        model = model_class(config)

    logger.info("Initialized Model:\n%s", model)

    # Create MixedPrecisionPolicy for FSDP when using FP32 master weights
    # This casts FP32 master weights to BF16 for forward/backward, then back to FP32 for optimizer
    mp_policy = None
    if use_fp32_master_weights:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,  # Cast params to BF16 for forward/backward compute
            reduce_dtype=torch.float32,  # Accumulate gradients in FP32 for precision
            output_dtype=torch.bfloat16,  # Output activations in BF16
            cast_forward_inputs=False,  # Do not cast inputs to param_dtype (BF16)
        )
        logger.info(
            "MixedPrecisionPolicy: param_dtype=bf16, reduce_dtype=fp32, output_dtype=bf16, cast_forward_inputs=False"
        )

    # Shard the transformer layers with FSDP. For Llama3, the transformer stack is in model.model.layers.
    # Each decoder layer should be individually sharded before sharding the full model.
    if mp_policy is None:
        for layer in model.model.layers:
            fully_shard(layer, mesh=device_mesh["dp"])
        fully_shard(model, mesh=device_mesh["dp"])
    else:
        for layer in model.model.layers:
            fully_shard(layer, mesh=device_mesh["dp"], mp_policy=mp_policy)
        fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)

    # If we're using meta device, we need to move sharded weights to the cuda device and initialize the parameters.
    if args.use_meta_device and isinstance(model, NVLlamaForCausalLM):
        # TE requires a special method to initialize the weights from the meta device.
        model.init_empty_weights()

    elif args.use_meta_device and isinstance(model, LlamaForCausalLM):
        model.to_empty(device=device)
        model.apply(model._init_weights)

    # Assign names to layers so debug API can identify them
    if args.fp8_stats_config.enabled:
        debug_api.infer_and_assign_layer_names(model)

    # Create optimizer. Convert OmegaConf to regular dict to avoid serialization issues (BIONEMO-2873).
    optimizer = AdamW(model.parameters(), **OmegaConf.to_container(args.adamw_kwargs, resolve=True))  # type: ignore
    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    if args.use_sequence_packing:
        train_dataloader, dataset_or_sampler = create_thd_dataloader(dist_config, **args.dataset)
    else:
        train_dataloader, dataset_or_sampler = create_bshd_dataloader(dist_config, **args.dataset)

    if args.use_torch_compile:
        # If we're using torch.compile, we need to do this before loading the checkpoint to ensure key consistency.
        model = torch.compile(model)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("torch.compile returned a non-Module object.")

    # If we're resuming from a checkpoint, load it and set the start step. Otherwise, start from step 0.
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_fsdp2" if args.checkpoint.ckpt_dir else None
    stateful_dataloader: StatefulDataLoader | None = None
    if args.dataset.use_stateful_dataloader:
        if not isinstance(train_dataloader, StatefulDataLoader):
            raise TypeError("Expected StatefulDataLoader when use_stateful_dataloader=True.")
        stateful_dataloader = train_dataloader

    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        logger.info(f"Attempting to load checkpoint from {ckpt_path}")
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_fsdp2(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
            dataloader=stateful_dataloader,
            process_group=device_mesh.get_group("dp"),
        )
        logger.info(f"Checkpoint loaded, resuming from step {start_step}, epoch {epoch}")
    else:
        logger.info("No checkpoint to load, starting from scratch")
        start_step = 0
        epoch = 0

    perf_logger = PerfLogger(dist_config, args)

    gc.collect()
    torch.cuda.empty_cache()

    # Training loop
    logger.info(f"Starting training loop from step {start_step} to {args.num_train_steps}")
    step = start_step
    micro_step = 0  # Gradient accumulation step counter

    # Create autocast context for FP32 master weights (casts compute to BF16)
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_fp32_master_weights else nullcontext()

    if train_dataloader is None:
        raise RuntimeError("Expected train_dataloader to be initialized before training.")

    while step < args.num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

            micro_step += 1

            # Forward pass with mixed precision.
            with autocast_ctx:
                with transformer_engine.pytorch.fp8_autocast(enabled=args.fp8_config.enabled, fp8_recipe=fp8_recipe):
                    outputs = model(**batch)

            # Backward pass - scale loss by grad_acc_steps for proper gradient averaging
            loss = outputs.loss / args.grad_acc_steps
            loss.backward()

            # Log microbatch step data for accumulation metrics
            perf_logger.log_micro_step(batch=batch, outputs=outputs)

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
                    save_checkpoint_fsdp2(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        ckpt_path=ckpt_path,
                        step=step,
                        epoch=epoch,
                        dist_config=dist_config,
                        dataloader=stateful_dataloader,
                        process_group=device_mesh.get_group("dp"),
                        max_checkpoints=args.checkpoint.max_checkpoints,
                        async_save=args.checkpoint.async_save,
                    )

                step += 1
                if step >= args.num_train_steps:
                    break

        # Dataloader exhausted, incrementing epoch
        epoch += 1
        set_epoch = getattr(dataset_or_sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    # Save final model to a .safetensors file.
    if args.checkpoint.save_final_model and ckpt_path:
        save_final_model_fsdp2(
            model=model,
            save_directory=ckpt_path / "final_model",
            dist_config=dist_config,
        )

    # Make sure we don't have any outstanding checkpoint save futures.
    if args.checkpoint.async_save and "fsdp2" in _ckpt_futures and _ckpt_futures["fsdp2"] is not None:
        _ckpt_futures["fsdp2"].result()

    # Clean up distributed training
    perf_logger.finish()
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
