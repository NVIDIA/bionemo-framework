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
import torch.cuda.nvtx as nvtx
import transformer_engine
import transformer_engine.pytorch

from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.optim import AdamW

from transformer_engine.common.recipe import Format
from transformers import AutoConfig, AutoModelForMaskedLM

from modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM

# This import seems to be needed with meta device init and AutoModel.from_config
from transformers.models.esm.modeling_esm import EsmForMaskedLM  # noqa: F401

from checkpoint import load_checkpoint_fsdp2, save_checkpoint_fsdp2, save_final_model_fsdp2, should_save_checkpoint
from dataset import create_bshd_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig
from perf_logger import PerfLogger
from quantization import initialize_quant_stats_logging, resolve_quantization_layers
from scheduler import get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train ESM-2 with TE layers using fsdp2.

    Returns:
        float: The loss value for the final batch.
    """
    # Initialize the distributed configuration, including creating the distributed process group.
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    # Load model config early so we know the number of layers for auto-populating layer lists.
    config = NVEsmConfig.from_pretrained(
        args.model_tag, dtype=torch.float32 if args.use_fp32_master_weights else torch.bfloat16
    )
    num_layers = config.num_hidden_layers

    # Resolve layer-wise quantization assignments.
    quant_layers = resolve_quantization_layers(
        num_layers=num_layers,
        fp8_enabled=args.fp8_config.enabled,
        fp4_enabled=args.fp4_config.enabled,
        fp8_layers=OmegaConf.to_container(args.fp8_layers, resolve=True) if args.fp8_layers is not None else None,
        fp4_layers=OmegaConf.to_container(args.fp4_layers, resolve=True) if args.fp4_layers is not None else None,
    )
    fp8_layers = quant_layers.fp8_layers_0indexed
    fp4_layers = quant_layers.fp4_layers_0indexed

    if args.quant_stats_config.enabled:
        initialize_quant_stats_logging(
            quant_stats_file=args.quant_stats_config.quant_stats_file,
            quant_log_dir=args.quant_stats_config.quant_log_dir,
            rank=dist_config.rank,
            quant_layers=quant_layers,
        )

    # Create a device mesh for FSDP.
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size,),
        mesh_dim_names=("dp",),
    )

    # Create an FP8 recipe -- this is only used if FP8 is enabled in the config.
    if args.fp8_config.enabled:
        fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
            fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
        )
    
    if args.fp4_config.enabled:
        fp4_recipe = hydra.utils.get_class(args.fp4_config.fp4_recipe)(
            fp4_format=Format[args.fp4_config.fp4_format], **args.fp4_config.fp4_recipe_kwargs
        )

    # If we're using sequence packing with TE layers, we need to pass the `attn_input_format` argument.
    if args.use_sequence_packing:
        config.attn_input_format = "thd"

    # Optionally use transformer engine to initialize only fp8 versions of weights by setting
    # `fp8_config.fp8_model_init_kwargs.enabled` to `True`, as opposed to using the default where both bfloat16 and fp8
    # versions of weights are kept.
    with (
        torch.device("meta") if args.use_meta_device else nullcontext(),
    ):
        # model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
        model = NVEsmForMaskedLM(config)

    logger.info("Initialized Model:\n%s", model)

    # We call the transformer stack "layers" in our TE models, but it's called "layer" in the original ESM-2 models.
    transformer_stack = model.esm.encoder.layers if hasattr(model.esm.encoder, "layers") else model.esm.encoder.layer

    mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,    # Cast params to BF16 for forward/backward
    reduce_dtype=torch.float32,   # Gradient reductions in FP32
    output_dtype=torch.bfloat16,   # Forward output dtype
    )
    if args.use_fp32_master_weights:
        for layer in transformer_stack:
            fully_shard(layer, mesh=device_mesh["dp"], mp_policy=mp_policy)
        fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)
    else:
        for layer in transformer_stack:
            fully_shard(layer, mesh=device_mesh["dp"])
        fully_shard(model, mesh=device_mesh["dp"])
    # Create a layer map for the transformer stack.
    layer_number_quantized_recipe_map = {}
    fp8_layers_set = set(fp8_layers) if fp8_layers else set()
    fp4_layers_set = set(fp4_layers) if fp4_layers else set()
    for layer_number, layer in enumerate(transformer_stack):
        if layer_number in fp8_layers_set:
            layer_number_quantized_recipe_map[layer_number] = fp8_recipe
        elif layer_number in fp4_layers_set:
            layer_number_quantized_recipe_map[layer_number] = fp4_recipe
        else:
            layer_number_quantized_recipe_map[layer_number] = None

    model.esm.encoder.layer_number_quantized_recipe_map = layer_number_quantized_recipe_map
    # If we're using meta device, we need to move sharded weights to the cuda device and initialize the parameters.
    # Note, this should happen before we create the optimizer.
    if args.use_meta_device:
        if hasattr(model, "init_empty_weights"):
            # TE layers require special handling to initialize the weights from the meta device.
            model.init_empty_weights()
        else:
            model.to_empty(device=device)
            model.apply(model._init_weights)

    # Assign names to layers so debug API can identify them
    if args.quant_stats_config.enabled:
        debug_api.infer_and_assign_layer_names(model)

    # Create optimizer. Convert OmegaConf to regular dict to avoid serialization issues (BIONEMO-2873).
    optimizer = AdamW(model.parameters(), **OmegaConf.to_container(args.adamw_kwargs, resolve=True))  # type: ignore
    # Note: Got an error about mixed torch.Tensor and DTensor here, so using AdamW instead.
    scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    # If we're using sequence packing, create a THD dataloader, otherwise create a BSHD dataloader.
    train_dataloader, dataset_or_sampler = (
        create_thd_dataloader(dist_config, **args.dataset)
        if args.use_sequence_packing
        else create_bshd_dataloader(dist_config, **args.dataset)
    )

    if args.use_torch_compile:
        # If we're using torch.compile, we need to do this before loading the checkpoint to ensure key consistency.
        model = torch.compile(model)

    # If we're resuming from a checkpoint, load it and set the start step. Otherwise, start from step 0.
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_fsdp2" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_fsdp2(
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

    # Nsight Systems profiling setup.
    nsys_enabled = args.nsys_profiling.enabled
    nsys_start_step = args.nsys_profiling.start_step if nsys_enabled else -1
    nsys_end_step = args.nsys_profiling.end_step if nsys_enabled else -1
    nsys_ranks = set(OmegaConf.to_container(args.nsys_profiling.ranks, resolve=True)) if nsys_enabled else set()
    nsys_profiling_active = False

    if nsys_enabled and dist_config.rank in nsys_ranks:
        logger.info(
            f"Nsight profiling enabled for rank {dist_config.rank}: "
            f"will capture steps [{nsys_start_step}, {nsys_end_step})"
        )

    # Training loop
    step = start_step
    while step < args.num_train_steps:
        for batch in train_dataloader:
            # --- Nsys: start profiler at the configured step ---
            if nsys_enabled and step == nsys_start_step and dist_config.rank in nsys_ranks:
                logger.info(f"[Rank {dist_config.rank}] Starting nsys capture at step {step}")
                torch.cuda.cudart().cudaProfilerStart()
                nsys_profiling_active = True

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901
            
            # --- Forward pass ---
            nvtx.range_push(f"step_{step}")
            nvtx.range_push("forward")
            with transformer_engine.pytorch.autocast(enabled=args.fp8_config.enabled, recipe=fp8_recipe if args.fp8_config.enabled else None):
                outputs = model(**batch)
            nvtx.range_pop()  # forward

            # --- Backward pass ---
            nvtx.range_push("backward")
            loss = outputs.loss
            loss.backward()
            nvtx.range_pop()  # backward

            # --- Grad clip ---
            nvtx.range_push("clip_grad_norm")
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            nvtx.range_pop()  # clip_grad_norm

            # --- Optimizer step ---
            nvtx.range_push("optimizer_step")
            optimizer.step()
            scheduler.step()
            nvtx.range_pop()  # optimizer_step

            if args.quant_stats_config.enabled:
                debug_api.step()

            optimizer.zero_grad()
            nvtx.range_pop()  # step_N

            perf_logger.log_step(
                step=step,
                batch=batch,
                outputs=outputs,
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
                    max_checkpoints=args.checkpoint.max_checkpoints,
                )

            # --- Nsys: stop profiler at the configured step ---
            if nsys_profiling_active and step >= nsys_end_step:
                logger.info(f"[Rank {dist_config.rank}] Stopping nsys capture at step {step}")
                torch.cuda.cudart().cudaProfilerStop()
                nsys_profiling_active = False

            step += 1
            if step >= args.num_train_steps:
                break

        # Dataloader exhausted, incrementing epoch
        epoch += 1
        dataset_or_sampler.set_epoch(epoch)

    # Save final model to a .safetensors file.
    if args.checkpoint.save_final_model and ckpt_path:
        save_final_model_fsdp2(
            model=model,
            save_directory=ckpt_path / "final_model",
            dist_config=dist_config,
        )

    # Ensure nsys profiler is stopped if training ended before end_step.
    if nsys_profiling_active:
        logger.info(f"[Rank {dist_config.rank}] Stopping nsys capture at end of training (step {step})")
        torch.cuda.cudart().cudaProfilerStop()
        nsys_profiling_active = False

    # Clean up distributed training
    perf_logger.finish()
    if args.quant_stats_config.enabled:
        debug_api.end_debug()
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
