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

import hydra
import numpy as np
import torch
import transformer_engine.pytorch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.optim import AdamW
from transformer_engine.common.recipe import Format
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from checkpoint import load_checkpoint_fsdp2, save_checkpoint_fsdp2, save_final_model_fsdp2, should_save_checkpoint
from dataset import create_bshd_dataloader, create_bshd_packed_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM
from perf_logger import PerfLogger
from scheduler import get_cosine_annealing_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_seed(seed: int, rank: int = 0) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch (CPU and CUDA).
    For distributed training, each rank gets a unique seed based on the base seed + rank.

    Args:
        seed: Base random seed.
        rank: Distributed rank (added to seed for per-rank uniqueness).
    """
    effective_seed = seed + rank
    random.seed(effective_seed)
    # Set numpy legacy RNG - needed for compatibility with HuggingFace datasets and other libs
    np.random.seed(effective_seed)  # noqa: NPY002
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)
    # For full reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set seed to {effective_seed} (base={seed}, rank={rank})")


def clip_grad_norm_fsdp2(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    process_group=None,
) -> torch.Tensor:
    """Clip gradient norm for FSDP2 models with proper global norm computation.

    Unlike torch.nn.utils.clip_grad_norm_, this function properly computes the
    global gradient norm across all FSDP2 shards by performing an all-reduce
    of the squared local norms before computing the total norm.

    This is critical for FSDP2 because gradients are sharded as DTensors across
    ranks. Using the standard clip_grad_norm_ would only compute the local shard
    norm, leading to incorrect clipping behavior.

    Args:
        parameters: Iterable of parameters to clip gradients for.
        max_norm: Maximum norm value for clipping.
        norm_type: Type of norm to use (default: 2.0 for L2 norm).
        process_group: The process group for all-reduce. If None, uses default group.

    Returns:
        The total (global) gradient norm before clipping.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return torch.tensor(0.0)

    # Get device from the first gradient's underlying local tensor
    # For DTensors, we need to handle them specially
    first_grad = parameters[0].grad
    if hasattr(first_grad, "_local_tensor"):
        # DTensor - get the local tensor's device
        device = first_grad._local_tensor.device
        dtype = first_grad._local_tensor.dtype
    else:
        device = first_grad.device
        dtype = first_grad.dtype

    if norm_type == float("inf"):
        # For inf norm, need to find max across all ranks
        local_max = 0.0
        for p in parameters:
            grad = p.grad
            if hasattr(grad, "_local_tensor"):
                grad = grad._local_tensor
            local_max = max(local_max, grad.abs().max().item())
        total_norm = torch.tensor(local_max, device=device, dtype=dtype)
        torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=process_group)
    else:
        # Compute sum of squared norms locally using the underlying local tensors
        local_norm_sq = 0.0
        for p in parameters:
            grad = p.grad
            # For DTensors, access the local tensor shard directly
            if hasattr(grad, "_local_tensor"):
                grad = grad._local_tensor
            local_norm_sq += grad.norm(norm_type).pow(norm_type).item()

        # Create a regular tensor for all-reduce (not a DTensor)
        local_norm_sq_tensor = torch.tensor(local_norm_sq, device=device, dtype=dtype)
        # All-reduce to get global sum of squared norms
        torch.distributed.all_reduce(local_norm_sq_tensor, op=torch.distributed.ReduceOp.SUM, group=process_group)
        # Compute final norm
        total_norm = local_norm_sq_tensor.pow(1.0 / norm_type)

    # Clip gradients - need to handle DTensors specially
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0).item()

    for p in parameters:
        grad = p.grad
        if hasattr(grad, "_local_tensor"):
            # DTensor - modify the underlying local tensor
            grad._local_tensor.mul_(clip_coef_clamped)
        else:
            grad.detach().mul_(clip_coef_clamped)

    return total_norm


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
        skip_embeddings: Whether to skip weight decay on embedding layers. Default False to match John's setup.

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

    # Log counts for debugging
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

    # Set random seeds for reproducibility (matching John's --seed 42)
    seed = getattr(args, "seed", 42)  # Default to 42 if not specified
    set_seed(seed, dist_config.global_rank)

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

    # Create an empty Llama3 model with a causal language model head, e.g. "meta-llama/Meta-Llama-3-8B".
    # Convert config_kwargs to regular dict to avoid JSON serialization issues with nested DictConfig
    config_kwargs_dict = OmegaConf.to_container(args.config_kwargs, resolve=True)

    # Handle Spike-No-More embedding initialization (https://arxiv.org/abs/2312.16903)
    # When enabled, embeddings are initialized with std=1.0 instead of 0.02 to prevent loss spikes.
    if getattr(args, "spike_no_more_embedding_init", False):
        config_kwargs_dict["embedding_init_std"] = 1.0
        config_kwargs_dict["tie_word_embeddings"] = False  # Must not share embeddings with output weights
        logger.info("Spike-No-More enabled: embedding_init_std=1.0, tie_word_embeddings=False")

    # Handle Megatron-style scaled initialization for residual output layers
    # When enabled, proj and fc2 use std/sqrt(2*num_layers) instead of std
    if getattr(args, "use_megatron_scaled_init", False):
        config_kwargs_dict["use_megatron_scaled_init"] = True
        logger.info("Megatron scaled init enabled: proj/fc2 use std/sqrt(2*num_layers)")

    config = config_class.from_pretrained(args.config_name_or_path, dtype=torch.bfloat16, **config_kwargs_dict)

    # Optionally use transformer engine to initialize only fp8 versions of weights by setting
    # `fp8_config.fp8_model_init_kwargs.enabled` to `True`, as opposed to using the default where both bfloat16 and fp8
    # versions of weights are kept.
    with (
        torch.device("meta") if args.use_meta_device else nullcontext(),
        transformer_engine.pytorch.fp8_model_init(recipe=fp8_recipe, **args.fp8_config.fp8_model_init_kwargs),
    ):
        model = model_class(config)

    logger.info("Initialized Model:\n%s", model)

    # Shard the transformer layers with FSDP. For Llama3, the transformer stack is in model.model.layers.
    # Each decoder layer should be individually sharded before sharding the full model.
    for layer in model.model.layers:
        fully_shard(layer, mesh=device_mesh["dp"])
    fully_shard(model, mesh=device_mesh["dp"])

    if args.use_meta_device:
        model.to_empty(device=device)
        model.apply(model._init_weights)

    # Log initialization stats (for debugging, matching Megatron's TEV tracking)
    # Disabled: DTensor operations don't work well with FSDP2 sharding
    # if dist_config.rank == 0:
    #     for name, param in model.named_parameters():
    #         if "embed" in name.lower() or "lm_head" in name.lower():
    #             logger.info(f"Init stats - {name}")

    # Create optimizer. Convert OmegaConf to regular dict to avoid serialization issues (BIONEMO-2873).
    adamw_kwargs = OmegaConf.to_container(args.adamw_kwargs, resolve=True)

    # Check if we should use weight decay grouping (skip decay on bias and 1D params)
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
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_fsdp2" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        logger.info(f"Attempting to load checkpoint from {ckpt_path}")
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_fsdp2(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
            dataloader=train_dataloader,
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
    micro_step = 0
    while step < args.num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

            micro_step += 1

            # Forward pass with mixed precision.
            # Note: FP8 is selectively applied inside the model (first/last layers stay in bf16)
            outputs = model(**batch, fp8_enabled=args.fp8_config.enabled, fp8_recipe=fp8_recipe)

            # Backward pass - scale loss by grad_acc_steps for proper gradient averaging
            # Note: FSDP2 averages gradients across ALL ranks, while Megatron with TP only
            # averages across DP ranks. This can cause gradient magnitude differences.
            # Use loss_scale to compensate (e.g., set to 4-16 to match Megatron gradient norms).
            loss_scale = getattr(args, "loss_scale", 1.0)
            loss = outputs.loss * loss_scale / args.grad_acc_steps
            loss.backward()

            # Log microbatch step data for accumulation metrics
            perf_logger.log_micro_step(batch=batch, outputs=outputs)

            # Gradient accumulation - only step optimizer after accumulating gradients
            if micro_step % args.grad_acc_steps == 0:
                micro_step = 0

                # Compute and clip gradient norms.
                if args.get("use_distributed_grad_clip", False):
                    # FSDP2-aware global norm: all-reduces squared norms across shards before clipping.
                    # This matches Megatron's global gradient norm computation.
                    total_norm = clip_grad_norm_fsdp2(
                        model.parameters(),
                        max_norm=1.0,
                        norm_type=2.0,
                        process_group=device_mesh.get_group("dp"),
                    ).item()
                else:
                    # Standard PyTorch clip (may only compute local shard norm for FSDP2)
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

    # Save final model to a .safetensors file.
    if args.checkpoint.save_final_model and ckpt_path:
        save_final_model_fsdp2(
            model=model,
            save_directory=ckpt_path / "final_model",
            dist_config=dist_config,
        )

    # Clean up distributed training
    perf_logger.finish()
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
