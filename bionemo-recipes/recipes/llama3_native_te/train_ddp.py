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
import random
from contextlib import nullcontext
from pathlib import Path

import hydra
import numpy as np
import torch
import transformer_engine.pytorch
from omegaconf import DictConfig, OmegaConf
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
    np.random.seed(effective_seed)  # noqa: NPY002
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set seed to {effective_seed} (base={seed}, rank={rank})")


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
    """Train Llama3 with TE layers using DDP for genomic sequences.

    Returns:
        float: The loss value for the final batch.
    """
    # Initialize the distributed configuration, including creating the distributed process group.
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    # Set random seeds for reproducibility
    seed = getattr(args, "seed", 42)
    set_seed(seed, dist_config.rank)

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
    # Convert config_kwargs to regular dict to avoid JSON serialization issues with nested DictConfig
    config_kwargs_dict = OmegaConf.to_container(args.config_kwargs, resolve=True)

    # Handle Spike-No-More embedding initialization (https://arxiv.org/abs/2312.16903)
    if getattr(args, "spike_no_more_embedding_init", False):
        config_kwargs_dict["embedding_init_std"] = 1.0
        config_kwargs_dict["tie_word_embeddings"] = False
        logger.info("Spike-No-More enabled: embedding_init_std=1.0, tie_word_embeddings=False")

    # Handle Megatron-style scaled initialization for residual output layers
    if getattr(args, "use_megatron_scaled_init", False):
        config_kwargs_dict["use_scaled_init"] = True
        logger.info("Megatron scaled init enabled for proj/fc2 layers")

    config = config_class.from_pretrained(args.config_name_or_path, dtype=torch.bfloat16, **config_kwargs_dict)

    # Optionally use transformer engine to initialize only fp8 versions of weights by setting
    # `fp8_config.fp8_model_init_kwargs.enabled` to `True`, as opposed to using the default where both bfloat16 and fp8
    # versions of weights are kept.
    with transformer_engine.pytorch.fp8_model_init(recipe=fp8_recipe, **args.fp8_config.fp8_model_init_kwargs):
        model = model_class(config)

    # Enable gradient checkpointing if requested (trades compute for memory)
    if getattr(args, "use_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    logger.info("Initialized Model:\n%s", model)

    # Create optimizer with optional weight decay grouping
    adamw_kwargs = OmegaConf.to_container(args.adamw_kwargs, resolve=True)
    if getattr(args, "use_weight_decay_grouping", False):
        weight_decay = adamw_kwargs.pop("weight_decay", 0.1)
        skip_embeddings = getattr(args, "skip_embedding_weight_decay", False)
        param_groups = get_parameter_groups_with_weight_decay(model, weight_decay, skip_embeddings)
        optimizer = AdamW(param_groups, **adamw_kwargs)
    else:
        optimizer = AdamW(model.parameters(), **adamw_kwargs)
    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    model = model.to(device=device)

    # === DEBUG INITIALIZATION BREAKPOINT ===
    # Set debug_init=true in config or command line to enable
    if getattr(args, "debug_init", False) and dist_config.rank == 0:
        import math

        logger.info("=" * 60)
        logger.info("DEBUG INIT: Entering initialization debug mode")
        logger.info("=" * 60)

        # Helper function to run all init checks
        def debug_init_checks():
            print("\n" + "=" * 60)
            print("CHECK 1: Meta device / dtype / device status")
            print("=" * 60)
            p = next(model.parameters())
            print(f"  is_meta: {p.is_meta}")
            print(f"  device: {p.device}")
            print(f"  dtype: {p.dtype}")
            print(f"  requires_grad: {p.requires_grad}")

            print("\n" + "=" * 60)
            print("CHECK 2: Fan-in/Fan-out for fused QKV layers")
            print("=" * 60)
            for name, param in model.named_parameters():
                if "layernorm_qkv.weight" in name and param.ndim == 2:
                    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(param)
                    print(f"\n  {name}")
                    print(f"    shape: {list(param.shape)}")
                    print(f"    fan_in: {fan_in}, fan_out: {fan_out}")
                    print(f"    std: {param.std().item():.6f}")
                    break

            print("\n" + "=" * 60)
            print("CHECK 3: Q/K/V per-projection variance (fused QKV)")
            print("=" * 60)
            for name, param in model.named_parameters():
                if "layernorm_qkv.weight" in name and param.ndim == 2:
                    # Chunk into Q, K, V along the output dimension
                    w_q, w_k, w_v = param.chunk(3, dim=-1)
                    print(f"\n  {name} breakdown:")
                    print(f"    Full tensor std: {param.std().item():.6f}")
                    print(f"    W_q std: {w_q.std().item():.6f}")
                    print(f"    W_k std: {w_k.std().item():.6f}")
                    print(f"    W_v std: {w_v.std().item():.6f}")
                    ratio = max(w_q.std().item(), w_k.std().item(), w_v.std().item()) / min(
                        w_q.std().item(), w_k.std().item(), w_v.std().item()
                    )
                    if ratio > 1.1:
                        print(f"    WARNING: Q/K/V stds differ by {ratio:.2f}x!")
                    else:
                        print(f"    OK: Q/K/V stds are consistent (ratio={ratio:.3f})")
                    break

            print("\n" + "=" * 60)
            print("CHECK 4: Embedding and output projection quantiles")
            print("=" * 60)
            for name, param in model.named_parameters():
                if "embed_tokens" in name:
                    q = torch.quantile(param.float().flatten(), torch.tensor([0.01, 0.5, 0.99], device=param.device))
                    print("\n  embed_tokens:")
                    print(f"    quantiles [1%, 50%, 99%]: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}]")
                    print(f"    first 5 values: {param.flatten()[:5].tolist()}")
                    # For std=1.0, expect 1% quantile ~ -2.33, 99% ~ +2.33
                    expected_99 = 2.33 * param.std().item()
                    print(f"    expected 99% for Gaussian: +/-{expected_99:.2f}")
                    break

            print("\n" + "=" * 60)
            print("CHECK 5: Scaled init layers (o_proj, fc2)")
            print("=" * 60)
            num_layers = len(model.model.layers)
            expected_scaled_std = 0.02 / math.sqrt(2.0 * num_layers)
            print(f"  num_layers: {num_layers}")
            print(f"  expected scaled std: {expected_scaled_std:.6f}")
            for name, param in model.named_parameters():
                if "layers.0.self_attention.proj" in name:
                    print(f"\n  {name}:")
                    print(f"    actual std: {param.std().item():.6f}")
                    print(f"    expected: {expected_scaled_std:.6f}")
                    ratio = param.std().item() / expected_scaled_std
                    if 0.8 < ratio < 1.2:
                        print("    OK: Within 20% of expected")
                    else:
                        print(f"    WARNING: {ratio:.2f}x off from expected!")
                if "layers.0.layernorm_mlp.fc2" in name:
                    print(f"\n  {name}:")
                    print(f"    actual std: {param.std().item():.6f}")
                    print(f"    expected: {expected_scaled_std:.6f}")
                    ratio = param.std().item() / expected_scaled_std
                    if 0.8 < ratio < 1.2:
                        print("    OK: Within 20% of expected")
                    else:
                        print(f"    WARNING: {ratio:.2f}x off from expected!")

            print("\n" + "=" * 60)
            print("CHECK 6: Layer norm weights (should be all 1.0)")
            print("=" * 60)
            for name, param in model.named_parameters():
                if "layer_norm_weight" in name and "layers.0" in name:
                    unique_vals = param.unique()
                    print(f"\n  {name}:")
                    print(f"    unique values: {unique_vals.tolist()}")
                    if len(unique_vals) == 1 and unique_vals[0].item() == 1.0:
                        print("    OK: All ones (correct)")
                    else:
                        print("    WARNING: Not all ones!")

            print("\n" + "=" * 60)
            print("DEBUG CHECKS COMPLETE - Entering pdb")
            print("You can now inspect the model interactively.")
            print("Type 'c' to continue to training, or 'q' to quit.")
            print("=" * 60 + "\n")

        # Run the checks
        debug_init_checks()

        # Drop into pdb for interactive inspection
        import pdb

        pdb.set_trace()
    # === END DEBUG INITIALIZATION BREAKPOINT ===

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
    micro_step = 0
    while step < args.num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa PLW2901

            micro_step += 1
            # Use no_sync to prevent gradient synchronization until the last microbatch
            with model.no_sync() if micro_step % args.grad_acc_steps != 0 else nullcontext():
                # Forward pass with mixed precision.
                # Note: FP8 is selectively applied inside the model (first/last layers stay in bf16)
                outputs = model(**batch, fp8_enabled=args.fp8_config.enabled, fp8_recipe=fp8_recipe)

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
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
