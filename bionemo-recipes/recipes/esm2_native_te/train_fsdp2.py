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
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
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

    if args.quant_stats_config.enabled:
        quant_stats_file = args.quant_stats_config.quant_stats_file
        quant_log_dir = Path(args.quant_stats_config.quant_log_dir) / f"rank_{dist_config.rank}"
        quant_log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Logging quant stats to {quant_log_dir}")
        te_features_dir = str(Path(transformer_engine.__file__).parent / "debug" / "features")
        debug_api.initialize(
            config_file=quant_stats_file,
            feature_dirs=[te_features_dir],
            log_dir=quant_log_dir,
            default_logging_enabled=True,
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
    elif args.fp4_config.enabled:
        fp4_recipe = hydra.utils.get_class(args.fp4_config.fp4_recipe)(
            fp4_format=Format[args.fp4_config.fp4_format], **args.fp4_config.fp4_recipe_kwargs
        )
    else:
        print("No FP8 or FP4 config enabled, using default bfloat16")
    # Create an empty ESM-2 model with a masked language model head, e.g. "nvidia/esm2_t6_8M_UR50D".
    bf16_layers = OmegaConf.to_container(args.bf16_layers, resolve=True) if args.bf16_layers is not None and args.fp4_config.enabled else None
    config = NVEsmConfig.from_pretrained(args.model_tag, dtype=torch.bfloat16, bf16_layers=bf16_layers)
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

    for layer in transformer_stack:
        fully_shard(layer, mesh=device_mesh["dp"])
    fully_shard(model, mesh=device_mesh["dp"])

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

    # Training loop
    step = start_step
    while step < args.num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

            fp_context = transformer_engine.pytorch.fp8_autocast(enabled=args.fp8_config.enabled, fp8_recipe=fp8_recipe) if args.fp8_config.enabled else nullcontext()
            fp_context = transformer_engine.pytorch.autocast(enabled=True, recipe=fp4_recipe) if args.fp4_config.enabled else fp_context
            # Note: FOr NVFP4 it looks like its just autocast? https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#FP8-autocasting
            # Forward pass with mixed precision.
            with fp_context:
                outputs = model(**batch)

            # Backward pass.
            loss = outputs.loss
            loss.backward()

            # Compute and clip gradient norms.
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

            # Step optimizer.
            optimizer.step()
            scheduler.step()

            if args.quant_stats_config.enabled:
                debug_api.step()

            optimizer.zero_grad()

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

    # Clean up distributed training
    perf_logger.finish()
    if args.quant_stats_config.enabled:
        debug_api.end_debug()
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
