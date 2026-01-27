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
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import nvdlfw_inspect.api as debug_api
import torch
import transformer_engine
import transformer_engine.pytorch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.optim import AdamW
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


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    val_dataloader,
    num_batches: int,
    device: torch.device,
    dist_config: DistributedConfig,
    fp8_config: DictConfig,
    fp8_recipe,
    autocast_ctx,
) -> dict:
    """Run validation and compute loss metrics.

    Args:
        model: The model to evaluate.
        val_dataloader: DataLoader for validation data.
        num_batches: Number of batches to evaluate.
        device: Device to run on.
        dist_config: Distributed config for logging.
        fp8_config: FP8 configuration.
        fp8_recipe: FP8 recipe for autocast.
        autocast_ctx: Autocast context manager.

    Returns:
        Dictionary with val_loss and val_ppl.
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_evaluated = 0

    for batch in val_dataloader:
        if num_evaluated >= num_batches:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

        # Forward pass with same precision as training
        with autocast_ctx:
            with transformer_engine.pytorch.fp8_autocast(enabled=fp8_config.enabled, fp8_recipe=fp8_recipe):
                outputs = model(**batch)

        # Get loss from model output
        loss = outputs.loss
        if loss is not None:
            total_loss += loss.item()
            # Count valid tokens (non-padding labels)
            labels = batch.get("labels", None)
            if labels is not None:
                num_tokens = (labels != -100).sum().item()
            else:
                num_tokens = batch["input_ids"].numel()
            total_tokens += num_tokens
        num_evaluated += 1

    # Aggregate across ranks
    loss_tensor = torch.tensor([total_loss, float(total_tokens), float(num_evaluated)], device=device)
    torch.distributed.all_reduce(loss_tensor)
    global_loss = loss_tensor[0].item()
    global_tokens = int(loss_tensor[1].item())
    global_batches = int(loss_tensor[2].item())

    # Compute average loss (HF-style mean across batches)
    avg_loss = global_loss / max(global_batches, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    model.train()

    return {
        "val_loss": avg_loss,
        "val_ppl": perplexity,
        "val_tokens": global_tokens,
        "val_batches": global_batches,
    }


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    For FSDP2/DTensor, ALL ranks must use the SAME seed to ensure weights
    are initialized identically before sharding.

    Note: Data parallelism is handled by dataset sharding (each rank gets
    different data via dataset.shard()), not by different random seeds.

    Args:
        seed: Random seed (same on all ranks).
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set seed to {seed} (same on all ranks for FSDP2)")


def _to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a DTensor to a local tensor, or return the tensor as-is if it's already local.

    Args:
        tensor: A tensor that may be a DTensor (from FSDP2).

    Returns:
        The local tensor data.
    """
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return cast(torch.Tensor, to_local())
    return tensor


@torch.no_grad()
def log_init_stats(model: torch.nn.Module, dist_config: DistributedConfig) -> dict[str, dict[str, float]]:
    """Log initialization statistics for key model layers.

    Logs mean/std of weights for embeddings, lm_head, and sample projection layers.
    Useful for debugging initialization differences between frameworks.

    Note: With FSDP2, this logs statistics of the LOCAL shard on each rank.
    The stats will differ across ranks since each rank holds a different shard.

    Args:
        model: The initialized model.
        dist_config: Distributed config for rank checking.

    Returns:
        Dictionary mapping layer names to their init statistics.
    """
    stats = {}

    for name, param in model.named_parameters():
        # Log key layers for debugging initialization
        keys_to_log = [
            "embed_tokens",
            "lm_head",
            "o_proj",
            ".proj.",
            "q_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            ".fc1.",
            ".fc2.",
            "layernorm",
            "rmsnorm",
            "input_layernorm",
            "post_attention_layernorm",
            "norm",
        ]
        if not any(key in name.lower() for key in keys_to_log):
            continue

        # For FSDP2, convert DTensor to local tensor
        try:
            data = _to_local_tensor(param.data).float()
        except Exception:
            continue

        # Skip empty tensors
        if data.numel() == 0:
            continue

        # Compute stats
        try:
            mean_val = data.mean().item()
            var_val = torch.mean(torch.pow(data - mean_val, 2)).item()
            std_val = var_val**0.5
            min_val = data.min().item()
            max_val = data.max().item()
        except RuntimeError:
            continue

        layer_stats = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
        }
        stats[name] = layer_stats

        if dist_config.rank == 0:
            logger.info(
                f"Init stats (local shard) - {name}: mean={layer_stats['mean']:.4f}, "
                f"std={layer_stats['std']:.4f}, range=[{layer_stats['min']:.4f}, {layer_stats['max']:.4f}]"
            )

    return stats


def get_parameter_groups_with_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    skip_embeddings: bool = False,
) -> list[dict]:
    """Create parameter groups with proper weight decay filtering.

    Follows Megatron convention:
    - Skip weight decay on bias terms
    - Skip weight decay on 1D parameters (LayerNorm/RMSNorm weights)
    - Optionally skip weight decay on embedding layers

    Args:
        model: The model to get parameter groups from.
        weight_decay: The weight decay value for parameters that should have decay.
        skip_embeddings: Whether to skip weight decay on embedding layers.

    Returns:
        List of parameter group dicts for the optimizer.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Skip weight decay on:
        # 1. Bias terms (name ends with 'bias')
        # 2. 1D parameters (LayerNorm/RMSNorm weights)
        # 3. Embedding layers (when skip_embeddings=True)
        should_skip_decay = name.endswith(".bias") or param.dim() == 1 or (skip_embeddings and "embed" in name.lower())

        if should_skip_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    logger.info(
        f"Weight decay groups: {len(decay_params)} params with decay, {len(no_decay_params)} params without decay"
    )

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


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

    # Set random seeds (same seed on ALL ranks for FSDP2/DTensor)
    # This is CRITICAL - all ranks must have identical weights before sharding
    # Data parallelism is handled by dataset sharding, not by different seeds
    seed = getattr(args, "seed", 42)  # Default to 42 if not specified
    set_seed(seed)

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
        config_kwargs = cast(dict[str, Any], OmegaConf.to_container(args.config_kwargs, resolve=True))
    rope_scaling = config_kwargs.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        if "rope_type" not in rope_scaling and "type" in rope_scaling:
            rope_scaling = {**rope_scaling, "rope_type": rope_scaling.pop("type")}
        elif "rope_type" not in rope_scaling:
            rope_scaling = {**rope_scaling, "rope_type": "default"}
        config_kwargs["rope_scaling"] = rope_scaling
    rope_theta = config_kwargs.get("rope_theta")
    if rope_theta is None and isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type")
        if rope_type == "llama3":
            rope_theta = 500000.0
    if rope_theta is not None:
        config_kwargs["rope_theta"] = rope_theta

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
            cast_forward_inputs=False,  # Let autocast decide op-level dtypes
        )
        logger.info(
            "MixedPrecisionPolicy: param_dtype=bf16, reduce_dtype=fp32, output_dtype=bf16, cast_forward_inputs=False"
        )

        # Shard the transformer layers with FSDP. For Llama3, the transformer stack is in model.model.layers.
        # Each decoder layer should be individually sharded before sharding the full model.
        for layer in model.model.layers:
            fully_shard(layer, mesh=device_mesh["dp"], mp_policy=mp_policy)
        fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)
    else:
        # Shard the transformer layers with FSDP. For Llama3, the transformer stack is in model.model.layers.
        # Each decoder layer should be individually sharded before sharding the full model.
        for layer in model.model.layers:
            fully_shard(layer, mesh=device_mesh["dp"])
        fully_shard(model, mesh=device_mesh["dp"])

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

    # Log initialization statistics if enabled (useful for debugging Spike-No-More and scaled init)
    if getattr(args, "log_init_stats", False):
        log_init_stats(model, dist_config)

    # Log initial parameter dtype before optimizer setup for debugging mixed precision behavior.
    first_param = next(model.parameters(), None)
    if first_param is not None:
        logger.info("Model param dtype before optimizer: %s", first_param.dtype)

    # Create optimizer. Convert OmegaConf to regular dict to avoid serialization issues (BIONEMO-2873).
    adamw_kwargs = cast(dict[str, Any], OmegaConf.to_container(args.adamw_kwargs, resolve=True))

    # Check if we should use weight decay grouping (skip decay on bias and 1D params)
    # Default to True to match Megatron convention
    use_wd_grouping = getattr(args, "use_weight_decay_grouping", True)

    if use_wd_grouping:
        # Megatron-style: skip weight decay on bias and 1D params (LayerNorm)
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
        # Original behavior: same weight decay for all params
        optimizer = AdamW(model.parameters(), **adamw_kwargs)  # type: ignore
        logger.info(f"Weight decay grouping disabled: wd={adamw_kwargs.get('weight_decay', 0.1)} for all params")

    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    if args.use_sequence_packing:
        train_dataloader, dataset_or_sampler = create_thd_dataloader(dist_config, **args.dataset)
    else:
        train_dataloader, dataset_or_sampler = create_bshd_dataloader(dist_config, **args.dataset)

    if args.use_torch_compile:
        # If we're using torch.compile, we need to do this before loading the checkpoint to ensure key consistency.
        model = torch.compile(model)
        model = cast(torch.nn.Module, model)

    # If we're resuming from a checkpoint, load it and set the start step. Otherwise, start from step 0.
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_fsdp2" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        logger.info(f"Attempting to load checkpoint from {ckpt_path}")
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_fsdp2(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
            dataloader=train_dataloader,  # type: ignore[arg-type]
            process_group=device_mesh.get_group("dp"),
        )
        logger.info(f"Checkpoint loaded, resuming from step {start_step}, epoch {epoch}")
    else:
        logger.info("No checkpoint to load, starting from scratch")
        start_step = 0
        epoch = 0

    perf_logger = PerfLogger(dist_config, args)

    # Setup validation if enabled
    val_config = cast(DictConfig | None, getattr(args, "validation", None))
    val_enabled = val_config is not None and getattr(val_config, "enabled", False)
    val_config_enabled = cast(DictConfig | None, val_config) if val_enabled else None
    val_dataloader = None

    if val_config_enabled is not None:
        val_data_path = getattr(val_config_enabled, "data_path", None)
        if val_data_path:
            logger.info(f"Setting up validation dataloader from {val_data_path}")

            # Create validation dataloader config - copy training config and override data path
            val_dataset_kwargs = cast(dict[str, Any], OmegaConf.to_container(args.dataset, resolve=True))

            # Override load_dataset_kwargs for validation data
            val_dataset_kwargs["load_dataset_kwargs"] = {
                "path": "json",
                "data_files": val_data_path,
                "split": "train",
                "streaming": True,
            }

            # Don't use stateful dataloader for validation
            val_dataset_kwargs["use_stateful_dataloader"] = False

            # Optionally override validation batch size
            if hasattr(val_config_enabled, "micro_batch_size") and val_config_enabled.micro_batch_size is not None:
                val_dataset_kwargs["micro_batch_size"] = val_config_enabled.micro_batch_size

            # Use same data format as training
            if args.use_sequence_packing:
                val_dataloader, _ = create_thd_dataloader(dist_config, **val_dataset_kwargs)  # type: ignore[arg-type]
            else:
                val_dataloader, _ = create_bshd_dataloader(dist_config, **val_dataset_kwargs)  # type: ignore[arg-type]

            logger.info(
                f"Validation enabled: every {val_config_enabled.eval_interval} steps, "
                f"{val_config_enabled.num_batches} batches"
            )
        else:
            logger.warning("Validation enabled but no data_path specified, skipping validation")
            val_enabled = False

    gc.collect()
    torch.cuda.empty_cache()

    # Training loop
    logger.info(f"Starting training loop from step {start_step} to {args.num_train_steps}")
    step = start_step
    micro_step = 0  # Gradient accumulation step counter

    # Create autocast context for FP32 master weights (casts compute to BF16).
    # Allow override via config for debugging (default: enabled when use_fp32_master_weights=True).
    use_autocast = getattr(args, "use_autocast", use_fp32_master_weights)
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
    if use_fp32_master_weights:
        logger.info("FP32 master weights: use_autocast=%s", use_autocast)

    if train_dataloader is None:
        raise RuntimeError("Expected train_dataloader to be initialized before training.")

    logged_dtypes = False
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

            if not logged_dtypes:
                grad_param = next((p for p in model.parameters() if p.grad is not None), None)
                if grad_param is not None and grad_param.grad is not None:
                    logger.info(
                        "Dtypes after first backward: param=%s grad=%s loss=%s",
                        grad_param.dtype,
                        grad_param.grad.dtype,
                        outputs.loss.dtype,
                    )
                logged_dtypes = True

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
                        dataloader=train_dataloader if args.dataset.use_stateful_dataloader else None,  # type: ignore[arg-type]
                        process_group=device_mesh.get_group("dp"),
                        max_checkpoints=args.checkpoint.max_checkpoints,
                        async_save=args.checkpoint.async_save,
                    )

                # Run validation at specified interval
                if (
                    val_config_enabled is not None
                    and val_dataloader is not None
                    and step > 0
                    and step % val_config_enabled.eval_interval == 0
                ):
                    val_metrics = run_validation(
                        model=model,
                        val_dataloader=val_dataloader,
                        num_batches=val_config_enabled.num_batches,
                        device=device,
                        dist_config=dist_config,
                        fp8_config=args.fp8_config,
                        fp8_recipe=fp8_recipe,
                        autocast_ctx=autocast_ctx,
                    )
                    if dist_config.rank == 0:
                        logger.info(
                            f"[Step {step}] Validation: loss={val_metrics['val_loss']:.4f}, "
                            f"ppl={val_metrics['val_ppl']:.2f}, tokens={val_metrics['val_tokens']:,}"
                        )
                    perf_logger.log_validation(step, val_metrics)

                step += 1
                if step >= args.num_train_steps:
                    break

        # Dataloader exhausted, incrementing epoch
        epoch += 1
        if hasattr(dataset_or_sampler, "set_epoch"):
            cast(Any, dataset_or_sampler).set_epoch(epoch)

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
