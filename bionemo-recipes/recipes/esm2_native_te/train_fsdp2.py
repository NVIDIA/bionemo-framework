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
import tempfile
from contextlib import nullcontext
from pathlib import Path

import hydra
import nvdlfw_inspect.api as debug_api
import torch
import transformer_engine
import transformer_engine.pytorch
import yaml

from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.optim import AdamW

from transformer_engine.pytorch.optimizers import FusedAdam
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


def generate_layer_regex(layer_numbers: list[int] | None) -> str:
    """Generate a regex pattern to match specific layer numbers (1-indexed).
    
    Args:
        layer_numbers: List of layer numbers (1-indexed, as shown in logs).
                       If empty or None, returns a pattern that matches nothing.
        
    Returns:
        Regex pattern string for matching those layers' linear sublayers.
    """
    if not layer_numbers:
        # Return a pattern that matches nothing (non-existent layer)
        return r"model\.esm\.encoder\.layers\.DISABLED_NO_LAYERS_SPECIFIED"
    # Use alternation for arbitrary layer lists: (1|2|3|4|5)
    layer_pattern = "|".join(str(n) for n in sorted(layer_numbers))
    return rf"model\.esm\.encoder\.layers\.({layer_pattern})\..*(layernorm_qkv|proj|fc1|fc2)"


def update_quant_stats_config(
    config_file: str,
    fp4_layers: list[int] | None,
    fp8_layers: list[int] | None,
) -> str:
    """Update the quant stats YAML config with layer-specific regex patterns.
    
    Args:
        config_file: Path to the original YAML config file.
        fp4_layers: List of layer numbers for FP4 (1-indexed).
        fp8_layers: List of layer numbers for FP8 (1-indexed).
        
    Returns:
        Path to the updated config file (may be a temp file).
        
    Raises:
        ValueError: If fp4_layers and fp8_layers have overlapping layer numbers.
    """
    # Check for overlapping layers
    fp4_set = set(fp4_layers) if fp4_layers else set()
    fp8_set = set(fp8_layers) if fp8_layers else set()
    overlap = fp4_set & fp8_set
    if overlap:
        raise ValueError(
            f"fp4_layers and fp8_layers cannot have overlapping layer numbers. "
            f"Found overlap: {sorted(overlap)}"
        )
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Update FP4 section if it exists (always update, even if empty to disable matching)
    if "example_fp4_tensor_stat_collection" in config:
        fp4_regex = generate_layer_regex(fp4_layers)
        config["example_fp4_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"] = fp4_regex
        if fp4_layers:
            logger.info(f"Updated FP4 layer regex to match layers: {fp4_layers}")
        else:
            logger.info("FP4 layers empty - regex set to match nothing")
    
    # Update FP8 section if it exists (always update, even if empty to disable matching)
    if "example_fp8_tensor_stat_collection" in config:
        fp8_regex = generate_layer_regex(fp8_layers)
        config["example_fp8_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"] = fp8_regex
        if fp8_layers:
            logger.info(f"Updated FP8 layer regex to match layers: {fp8_layers}")
        else:
            logger.info("FP8 layers empty - regex set to match nothing")
    
    # Write to a temp file to avoid modifying the original
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, temp_file, default_flow_style=False)
    temp_file.close()
    
    # Log the updated config for visibility
    config_str = yaml.dump(config, default_flow_style=False)
    logger.info(f"Created updated quant stats config at: {temp_file.name}")
    logger.info(f"Updated quant stats config contents:\n{config_str}")
    
    return temp_file.name


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

    # Parse layer lists first (1-indexed from args, used for both quant stats and internal recipe mapping)
    fp8_layers_1indexed = OmegaConf.to_container(args.fp8_layers, resolve=True) if args.fp8_layers is not None and args.fp8_config.enabled else None
    fp4_layers_1indexed = OmegaConf.to_container(args.fp4_layers, resolve=True) if args.fp4_layers is not None and args.fp4_config.enabled else None
    
    # Convert to 0-indexed for internal use (use 'is not None' to handle empty lists correctly)
    fp8_layers = [layer - 1 for layer in fp8_layers_1indexed] if fp8_layers_1indexed is not None else None
    fp4_layers = [layer - 1 for layer in fp4_layers_1indexed] if fp4_layers_1indexed is not None else None

    if args.quant_stats_config.enabled:
        quant_stats_file = args.quant_stats_config.quant_stats_file
        
        # Update the quant stats config with layer-specific regex patterns (using 1-indexed layer numbers)
        quant_stats_file = update_quant_stats_config(
            config_file=quant_stats_file,
            fp4_layers=fp4_layers_1indexed,
            fp8_layers=fp8_layers_1indexed,
        )
        
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
    
    if args.fp4_config.enabled:
        fp4_recipe = hydra.utils.get_class(args.fp4_config.fp4_recipe)(
            fp4_format=Format[args.fp4_config.fp4_format], **args.fp4_config.fp4_recipe_kwargs
        )

    config = NVEsmConfig.from_pretrained(args.model_tag, dtype=torch.float32)
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
    for layer in transformer_stack:
        fully_shard(layer, mesh=device_mesh["dp"], mp_policy=mp_policy)
    fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)

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
    # optimizer = FusedAdam(model.parameters(),
    #     lr=4e-4,
    #     betas=(0.9, 0.98),
    #     eps=1e-8,
    #     weight_decay=0.01,
    #     master_weights=True,
    #     master_weight_dtype=torch.float32,
    #     )
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

    # Training loop
    step = start_step
    while step < args.num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901
            
            # Use an outer FP8 recipe.
            with transformer_engine.pytorch.autocast(enabled=args.fp8_config.enabled, recipe=fp8_recipe):
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
