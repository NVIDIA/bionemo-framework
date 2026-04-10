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

"""Fully Sharded Data Parallel v2 (FSDP2) training script for Llama 3 with TransformerEngine.

Model weights and optimizer states are sharded across GPUs, allowing training of models that exceed
the memory of a single GPU. Supports both TE-accelerated (NVLlamaForCausalLM) and standard
HuggingFace (LlamaForCausalLM) models.

For very long sequences, use ``train_fsdp2_cp.py`` which adds Context Parallelism on top of FSDP2.
"""

import gc
import logging
from contextlib import nullcontext
from pathlib import Path

import hydra
import nvdlfw_inspect.api as debug_api
import torch
import transformer_engine.pytorch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.optim import AdamW
from transformer_engine.common.recipe import Format
from transformer_engine.pytorch.optimizers import FusedAdam
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
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM
from perf_logger import PerfLogger
from quantization import initialize_quant_stats_logging, resolve_layer_precision
from scheduler import get_cosine_annealing_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DTensorFusedAdam(FusedAdam):
    """FusedAdam with DTensor-compatible state initialization.

    TE's FusedAdam uses ``torch.empty(param.shape, ...)`` to create optimizer states, which produces
    a regular Tensor even when ``param`` is a DTensor (FSDP2). This causes a crash on ``copy_()``
    between the regular-Tensor state and the DTensor param. This subclass fixes the issue by using
    ``torch.empty_like`` / ``torch.zeros_like`` which preserve DTensor sharding.

    See: https://github.com/NVIDIA/TransformerEngine/blob/main/examples/pytorch/quantized_model_init/fully_shard.py
    """

    def _initialize_state(self, param, state_name, zero_buffer, store_param_remainders=False):
        dtype = self.name_to_dtype_map[state_name]
        if store_param_remainders:
            data = torch.zeros_like(param, dtype=torch.int16)
        else:
            data = torch.empty_like(param, dtype=dtype)
        if zero_buffer:
            data.zero_()

        if dtype == torch.uint8:
            # FP8 quantizer path — delegate to parent (only used with QuantizedTensors).
            super()._initialize_state(param, state_name, zero_buffer, store_param_remainders)
            return

        self.state[param][state_name] = data

        # Create scale if necessary.
        if dtype != torch.float32:
            if param not in self._scales:
                self._scales[param] = {}
            self._scales[param][state_name] = torch.ones([1], dtype=torch.float32, device=param.device)


def _init_master_weights_from_high_precision(optimizer: "DTensorFusedAdam", model: torch.nn.Module) -> None:
    """Initialize optimizer master weights from high-precision init values.

    When quantized_model_init is used with preserve_high_precision_init_val=True, each FP8 parameter
    stores the original BF16 init values in CPU memory. This function copies those into the optimizer's
    FP32 master weight states, then frees the CPU copies. Without this, master weights would be
    initialized from dequantized FP8 values, introducing quantization noise at initialization.
    """
    count = 0
    for param in model.parameters():
        if hasattr(param, "get_high_precision_init_val"):
            hp_val = param.get_high_precision_init_val()
            if hp_val is not None:
                # Trigger optimizer state initialization if not yet done
                if param not in optimizer.state or "master_param" not in optimizer.state[param]:
                    optimizer.initialize_state(param, store_param_remainders=False)
                optimizer.set_scaled_state(param, "master_param", hp_val.to(param.device).float())
                param.clear_high_precision_init_val()
                count += 1
    if count > 0:
        logger.info("Initialized %d master weight(s) from high-precision init values", count)
    else:
        logger.info(
            "No parameters with high-precision init values found (quantized_model_init may not have been used)"
        )


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train Llama3 with TE layers using FSDP2.

    Returns:
        float: The loss value for the final batch.
    """
    # --- Distributed Setup ---
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    device_mesh = init_device_mesh("cuda", mesh_shape=(dist_config.world_size,), mesh_dim_names=("dp",))

    if args.use_te:
        config_class = NVLlamaConfig
        model_class = NVLlamaForCausalLM
    else:
        config_class = LlamaConfig
        model_class = LlamaForCausalLM

    # --- Model Configuration ---
    # FusedAdam maintains its own FP32 master weights, so model stays BF16.
    # MixedPrecisionPolicy path inits in FP32 and casts to BF16 via FSDP.
    use_fp32_mp = args.use_fp32_master_weights and not args.use_fp32_master_weights_fused
    config = config_class.from_pretrained(
        args.config_name_or_path,
        dtype=torch.float32 if use_fp32_mp else torch.bfloat16,
        **args.config_kwargs,
    )

    # Resolve layer-wise quantization assignments and store on config.
    layer_precision = resolve_layer_precision(
        num_layers=config.num_hidden_layers,
        fp8_enabled=args.fp8_config.enabled,
        fp4_enabled=args.fp4_config.enabled,
        fp8_layers=OmegaConf.to_container(args.fp8_layers, resolve=True) if args.fp8_layers is not None else None,
        fp4_layers=OmegaConf.to_container(args.fp4_layers, resolve=True) if args.fp4_layers is not None else None,
    )
    config.layer_precision = layer_precision

    if args.quant_stats_config.enabled:
        initialize_quant_stats_logging(
            quant_stats_file=args.quant_stats_config.quant_stats_file,
            quant_log_dir=args.quant_stats_config.quant_log_dir,
            rank=dist_config.rank,
            layer_precision=layer_precision,
        )

    # Create quantization recipes -- these are only used if FP8/FP4 is enabled in the config.
    fp8_recipe = None
    fp4_recipe = None
    if args.fp8_config.enabled:
        fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
            fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
        )
    if args.fp4_config.enabled:
        fp4_recipe = hydra.utils.get_class(args.fp4_config.fp4_recipe)(
            fp4_format=Format[args.fp4_config.fp4_format], **args.fp4_config.fp4_recipe_kwargs
        )

    if args.fp8_config.quantized_model_init_kwargs.get("enabled", False) and not (
        args.fp8_config.enabled or args.fp4_config.enabled
    ):
        raise ValueError(
            "fp8_config.quantized_model_init_kwargs.enabled=true requires fp8_config.enabled=true or "
            "fp4_config.enabled=true. Enable at least one quantization format to use quantized model initialization."
        )

    # --- Model Initialization ---
    # Optionally use transformer engine to initialize only fp8 versions of weights by setting
    # `fp8_config.quantized_model_init_kwargs.enabled` to `True`, as opposed to using the default where both bfloat16
    # and fp8 versions of weights are kept.
    with (
        torch.device("meta") if args.use_meta_device else nullcontext(),
        transformer_engine.pytorch.quantized_model_init(
            recipe=fp8_recipe, **args.fp8_config.quantized_model_init_kwargs
        ),
    ):
        model = (
            model_class(config, fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)
            if model_class is NVLlamaForCausalLM
            else model_class(config)
        )

    logger.info("Initialized Model:\n%s", model)

    # --- Distributed Wrapping (FSDP2) ---
    if use_fp32_mp:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=False,
        )
    else:
        mp_policy = MixedPrecisionPolicy()

    # Each decoder layer should be individually sharded before sharding the full model.
    for layer in model.model.layers:
        fully_shard(layer, mesh=device_mesh["dp"], mp_policy=mp_policy)
    fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)

    # Attach quantization recipes to the model (layer precision is already on config).
    if isinstance(model, NVLlamaForCausalLM):
        model.model.set_recipes(fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)

    # If we're using meta device, we need to move sharded weights to the cuda device and initialize the parameters.
    if args.use_meta_device:
        if args.use_te:
            # TE requires a special method to initialize the weights from the meta device.
            model.init_empty_weights()
        else:
            model.to_empty(device=device)
            model.apply(model._init_weights)

    # Assign names to layers so debug API can identify them
    if args.quant_stats_config.enabled:
        debug_api.infer_and_assign_layer_names(model)

    # --- Optimizer & Scheduler ---
    # Convert OmegaConf to regular dict to avoid serialization issues (BIONEMO-2873).
    adamw_kwargs = OmegaConf.to_container(args.adamw_kwargs, resolve=True)
    if args.use_fp32_master_weights_fused:
        # TE FusedAdam maintains FP32 master copies of BF16 params internally.
        # 'fused' kwarg is not used by TE's FusedAdam (it's always fused).
        adamw_kwargs.pop("fused", None)
        optimizer = DTensorFusedAdam(model.parameters(), master_weights=True, **adamw_kwargs)  # type: ignore
        logger.info("Using TE FusedAdam with FP32 master weights")

        # When using quantized_model_init with preserve_high_precision_init_val=True,
        # initialize FP32 master weights from the original high-precision values instead of
        # from dequantized FP8 values. This avoids quantization noise in initialization.
        # See: https://github.com/NVIDIA/TransformerEngine/blob/main/examples/pytorch/quantized_model_init/fully_shard.py
        if args.fp8_config.quantized_model_init_kwargs.get("preserve_high_precision_init_val", False):
            _init_master_weights_from_high_precision(optimizer, model)
    else:
        optimizer = AdamW(model.parameters(), **adamw_kwargs)  # type: ignore
    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    if args.use_torch_compile:
        # If we're using torch.compile, we need to do this before loading the checkpoint to ensure key consistency.
        model = torch.compile(model)

    # --- Data Loading ---
    if args.use_sequence_packing:
        train_dataloader, dataset_or_sampler = create_thd_dataloader(dist_config, **args.dataset)
    else:
        train_dataloader, dataset_or_sampler = create_bshd_dataloader(dist_config, **args.dataset)

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
            process_group=device_mesh.get_group("dp"),
        )
        logger.info("Checkpoint loaded, resuming from step %s, epoch %s", start_step, epoch)
    else:
        logger.info("No checkpoint to load, starting from scratch")
        start_step = 0
        epoch = 0

    perf_logger = PerfLogger(dist_config, args, start_step=start_step)

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

            # Forward pass - quantization autocast is handled inside the model via set_recipes().
            outputs = model(**batch)

            # Backward pass - scale loss by grad_acc_steps for proper gradient averaging
            loss = outputs.loss / args.grad_acc_steps
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
                        process_group=device_mesh.get_group("dp"),
                        max_checkpoints=args.checkpoint.max_checkpoints,
                        async_save=args.checkpoint.async_save,
                    )

                step += 1
                if step >= args.num_train_steps:
                    break

        # Dataloader exhausted, incrementing epoch
        epoch += 1
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
