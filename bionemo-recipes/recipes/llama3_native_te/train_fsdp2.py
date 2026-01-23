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
    # Check if it's a DTensor by checking for to_local method
    if hasattr(tensor, "to_local"):
        return tensor.to_local()
    return tensor


@torch.no_grad()
def compute_tev(model: torch.nn.Module) -> tuple[float, float]:
    """Compute Token Embedding Variance (TEV) statistics.

    TEV measures the variance of embedding vectors across the vocabulary dimension.
    This metric is useful for monitoring embedding stability during training,
    especially when using Spike-No-More initialization (std=1.0).

    For FSDP2/DTensor, this gathers the FULL embedding across all ranks to compute
    accurate global TEV (matching Megatron's TEV calculation).

    Args:
        model: The model containing embeddings.

    Returns:
        Tuple of (tev_mean, tev_sd) - mean and standard deviation of per-dimension variances.
    """
    # Find embedding parameter
    embed = None
    for name, param in model.named_parameters():
        if "embed_tokens" in name and "weight" in name:
            try:
                # For FSDP2/DTensor, use full_tensor() to gather the complete embedding
                # This matches Megatron's TEV which is computed on the full embedding
                if hasattr(param.data, "full_tensor"):
                    embed = param.data.full_tensor().float()
                else:
                    embed = param.data.float()
            except Exception:
                # Fallback to local tensor if full_tensor fails
                try:
                    embed = _to_local_tensor(param.data).float()
                except Exception:
                    pass
            break

    if embed is None or embed.numel() == 0:
        return 0.0, 0.0

    try:
        # Calculate token embedding variance (TEV)
        # TEV = sqrt(mean((embed - mean(embed, dim=0))^2, dim=0))
        # This gives the std dev of each embedding dimension across all tokens
        tev = torch.sqrt(torch.mean(torch.pow(embed - embed.mean(dim=0), 2), dim=0))

        tev_mean = torch.mean(tev).item()
        # Compute std manually to avoid potential DTensor issues
        tev_var = torch.mean(torch.pow(tev - tev_mean, 2))
        tev_sd = torch.sqrt(tev_var).item()

        return tev_mean, tev_sd
    except RuntimeError:
        # Return zeros if computation fails on this rank
        return 0.0, 0.0


@torch.no_grad()
def log_init_stats(model: torch.nn.Module, dist_config: "DistributedConfig") -> dict[str, dict[str, float]]:
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
        # Log all key layers for debugging initialization:
        # - embed_tokens: token embeddings (spike-no-more uses std=1.0)
        # - lm_head: output projection
        # - o_proj/proj: attention output projection
        # - down_proj/fc2: MLP output projection
        # - q_proj/k_proj/v_proj: QKV projections
        # - gate_proj/up_proj/fc1: MLP input projections
        # - layernorm/rmsnorm/norm: normalization layers (should be ~1.0 for weights)
        keys_to_log = [
            "embed_tokens",
            "lm_head",
            # Attention
            "o_proj",
            ".proj.",
            "q_proj",
            "k_proj",
            "v_proj",
            # MLP
            "gate_proj",
            "up_proj",
            "down_proj",
            ".fc1.",
            ".fc2.",
            # Normalization (RMSNorm in Llama)
            "layernorm",
            "rmsnorm",
            "input_layernorm",
            "post_attention_layernorm",
            "norm",
        ]
        if not any(key in name.lower() for key in keys_to_log):
            continue

        # For FSDP2, convert DTensor to local tensor to avoid unsupported ops
        try:
            data = _to_local_tensor(param.data).float()
        except Exception:
            # Skip if we can't convert to local tensor
            continue

        # Skip empty tensors (e.g., _extra_state from Transformer Engine, or empty FSDP shards)
        if data.numel() == 0:
            continue

        # Compute stats manually to avoid DTensor op issues
        try:
            mean_val = data.mean().item()
            var_val = torch.mean(torch.pow(data - mean_val, 2)).item()
            std_val = var_val**0.5
            min_val = data.min().item()
            max_val = data.max().item()
        except RuntimeError:
            # Skip if stats computation fails (e.g., empty tensor edge cases)
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


def compute_megatron_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    is_thd: bool = False,
) -> tuple[torch.Tensor, int]:
    """Compute loss using Megatron-style per-token reduction.

    This matches the loss calculation in Megatron with --no-calculate-per-token-loss:
    1. Compute sum of per-token cross-entropy losses
    2. Return the sum and the count of valid tokens

    The caller is responsible for dividing by total tokens across all microbatches.

    Args:
        logits: Model logits, shape [batch, seq, vocab] or [total_tokens, vocab] for THD.
        labels: Target labels, shape [batch, seq] or [total_tokens] for THD.
        is_thd: Whether the input is in THD format (packed sequences).

    Returns:
        Tuple of (loss_sum, num_valid_tokens).
    """
    if is_thd:
        # THD format: logits are [total_tokens, vocab]
        # Labels are [batch, seq] - need to shift and flatten
        # The model handles THD internally, but labels are still batch format
        shift_labels = labels[..., 1:].contiguous()
        flat_logits = logits[: shift_labels.numel(), :]  # Match token count after shift
        flat_labels = shift_labels.view(-1)
    else:
        # BSHD format: logits are [batch, seq, vocab]
        # Need to shift for causal LM (predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        flat_logits = shift_logits.view(-1, logits.size(-1))
        flat_labels = shift_labels.view(-1)

    # Count valid tokens (not -100)
    valid_mask = flat_labels != -100
    num_valid_tokens = valid_mask.sum().item()

    # Compute sum of losses (not mean!)
    loss_sum = torch.nn.functional.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="sum")

    return loss_sum, num_valid_tokens


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


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    val_dataloader,
    num_batches: int,
    device: torch.device,
    dist_config: DistributedConfig,
    is_thd: bool = False,
) -> dict:
    """Run validation and compute loss metrics.

    Args:
        model: The model to evaluate.
        val_dataloader: DataLoader for validation data.
        num_batches: Number of batches to evaluate.
        device: Device to run on.
        dist_config: Distributed config for logging.
        is_thd: Whether using THD format.

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

        # Forward pass (no FP8 for validation to ensure consistent loss)
        outputs = model(**batch, fp8_enabled=False)

        # Compute loss with Megatron-style reduction for consistency
        loss_sum, num_tokens = compute_megatron_loss(
            logits=outputs.logits,
            labels=batch["labels"],
            is_thd=is_thd,
        )

        total_loss += loss_sum.item()
        total_tokens += num_tokens
        num_evaluated += 1

    # Aggregate across ranks
    loss_tensor = torch.tensor([total_loss, total_tokens], device=device)
    torch.distributed.all_reduce(loss_tensor)
    global_loss = loss_tensor[0].item()
    global_tokens = int(loss_tensor[1].item())

    avg_loss = global_loss / max(global_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    model.train()

    return {
        "val_loss": avg_loss,
        "val_ppl": perplexity,
        "val_tokens": global_tokens,
        "val_batches": num_evaluated * dist_config.world_size,
    }


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

    # Enable gradient checkpointing if requested (trades compute for memory)
    if getattr(args, "use_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    logger.info("Initialized Model:\n%s", model)

    # Shard the transformer layers with FSDP. For Llama3, the transformer stack is in model.model.layers.
    # Each decoder layer should be individually sharded before sharding the full model.
    for layer in model.model.layers:
        fully_shard(layer, mesh=device_mesh["dp"])
    fully_shard(model, mesh=device_mesh["dp"])

    if args.use_meta_device:
        model.to_empty(device=device)
        model.apply(model._init_weights)

    # Log initialization statistics if enabled (matches John's debugging approach)
    if getattr(args, "log_init_stats", False):
        log_init_stats(model, dist_config)
        # Also compute initial TEV
        tev_mean, tev_sd = compute_tev(model)
        if dist_config.rank == 0:
            logger.info(f"Initial TEV: mean={tev_mean:.4f}, sd={tev_sd:.4f}")

    # === DEBUG INITIALIZATION BREAKPOINT ===
    # Set debug_init=true in config or command line to enable
    if getattr(args, "debug_init", False) and dist_config.rank == 0:
        logger.info("=" * 60)
        logger.info("DEBUG INIT: Entering initialization debug mode")
        logger.info("=" * 60)

        # Helper function to run all init checks
        def debug_init_checks():
            import math

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
                    print(f"    expected 99% for Gaussian: ±{expected_99:.2f}")
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

    # Check if we should use Megatron-style per-token loss reduction
    use_megatron_loss = getattr(args, "use_megatron_loss_reduction", False)
    log_tev = getattr(args, "log_tev", False)
    is_thd = getattr(args.config_kwargs, "attn_input_format", "bshd") == "thd"

    if use_megatron_loss:
        logger.info("Using Megatron-style per-token loss reduction (sum/total_tokens)")
    else:
        logger.info("Using HuggingFace-style loss reduction (mean)")

    perf_logger = PerfLogger(dist_config, args, log_tev=log_tev)

    # Setup validation if enabled
    val_config = getattr(args, "validation", None)
    val_enabled = val_config is not None and getattr(val_config, "enabled", False)
    val_dataloader = None

    if val_enabled:
        val_data_path = getattr(val_config, "data_path", None)
        if val_data_path:
            logger.info(f"Setting up validation dataloader from {val_data_path}")

            # Create validation dataloader config - copy training config and override data path
            val_dataset_kwargs = OmegaConf.to_container(args.dataset, resolve=True)

            # Override load_dataset_kwargs for validation data
            # Keep same data format settings but use validation file with streaming for distributed loading
            # For single files (.jsonl.gz), use path="json" with data_files parameter
            val_dataset_kwargs["load_dataset_kwargs"] = {
                "path": "json",  # Tell HF to use JSON loader
                "data_files": val_data_path,  # The actual file path
                "split": "train",  # HF loads single files as "train" split
                "streaming": True,  # Stream for proper distributed sharding across ranks
            }

            # Don't use stateful dataloader for validation (not checkpointing val state)
            val_dataset_kwargs["use_stateful_dataloader"] = False

            # Optionally override validation batch size and sequence settings
            if hasattr(val_config, "micro_batch_size") and val_config.micro_batch_size is not None:
                val_dataset_kwargs["micro_batch_size"] = val_config.micro_batch_size
            if hasattr(val_config, "max_seq_length") and val_config.max_seq_length is not None:
                val_dataset_kwargs["max_seq_length"] = val_config.max_seq_length
            if hasattr(val_config, "stride") and val_config.stride is not None:
                val_dataset_kwargs["stride"] = val_config.stride

            # Use same data format as training (THD vs BSHD) for consistent loss computation
            if args.use_sequence_packing:
                if getattr(args.config_kwargs, "attn_input_format", "thd") == "bshd":
                    # BSHD with full packing (cross-boundary attention, no cu_seqlens)
                    val_dataloader, _ = create_bshd_packed_dataloader(dist_config, **val_dataset_kwargs)
                    logger.info("Validation using BSHD packed dataloader (matching training)")
                else:
                    # THD with packing (respects boundaries via cu_seqlens)
                    val_dataloader, _ = create_thd_dataloader(dist_config, **val_dataset_kwargs)
                    logger.info("Validation using THD dataloader (matching training)")
            else:
                # Standard BSHD with windowing (no packing)
                val_dataloader, _ = create_bshd_dataloader(dist_config, **val_dataset_kwargs)
                logger.info("Validation using BSHD dataloader (matching training)")

            logger.info(
                f"Validation enabled: every {val_config.eval_interval} steps, {val_config.num_batches} batches"
            )
        else:
            logger.warning("Validation enabled but no data_path specified, skipping validation")
            val_enabled = False

    gc.collect()
    torch.cuda.empty_cache()

    # Training loop
    logger.info(f"Starting training loop from step {start_step} to {args.num_train_steps}")
    step = start_step
    micro_step = 0

    # Accumulators for loss (across gradient accumulation steps)
    # We track sum and count to enable proper all-reduce across ranks (like Megatron)
    accumulated_loss_sum = 0.0
    accumulated_tokens = 0
    # For HF-style loss, we also track sum/count to enable all-reduce
    accumulated_hf_loss_sum = 0.0
    accumulated_hf_batch_count = 0

    while step < args.num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

            micro_step += 1

            # Forward pass with mixed precision.
            # Note: FP8 is selectively applied inside the model (first/last layers stay in bf16)
            outputs = model(**batch, fp8_enabled=args.fp8_config.enabled, fp8_recipe=fp8_recipe)

            if use_megatron_loss:
                # Megatron-style: compute sum of losses and accumulate tokens
                loss_sum, num_tokens = compute_megatron_loss(
                    logits=outputs.logits,
                    labels=batch["labels"],
                    is_thd=is_thd,
                )
                accumulated_loss_sum += loss_sum.item()
                accumulated_tokens += num_tokens

                # Scale loss for backward pass (will divide by total tokens at the end)
                # For now, just use sum/tokens for this microbatch, scaled by grad_acc_steps
                loss = loss_sum / max(num_tokens, 1) / args.grad_acc_steps
                loss.backward()

                # Log microbatch with Megatron-style metrics
                perf_logger.log_micro_step(
                    batch=batch,
                    outputs=outputs,
                    loss_sum=loss_sum.item(),
                    num_tokens=num_tokens,
                )
            else:
                # HuggingFace-style: use the mean loss directly
                loss = outputs.loss / args.grad_acc_steps
                loss.backward()

                # Track loss sum and count for all-reduce (to match Megatron's global averaging)
                accumulated_hf_loss_sum += outputs.loss.item()
                accumulated_hf_batch_count += 1

                # Log microbatch step data for accumulation metrics
                perf_logger.log_micro_step(batch=batch, outputs=outputs)

            # Gradient accumulation - only step optimizer after accumulating gradients
            if micro_step % args.grad_acc_steps == 0:
                micro_step = 0

                # Compute and clip gradient norms
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

                # Step optimizer.
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Compute TEV if logging is enabled (every 100 steps to reduce overhead)
                # full_tensor() does an all-gather, so we don't want to do it every step
                tev_mean, tev_sd = (0.0, 0.0)
                tev_log_interval = getattr(args, "tev_log_interval", 100)
                if log_tev and (step % tev_log_interval == 0 or step == 1):
                    tev_mean, tev_sd = compute_tev(model)

                # Log step with optional Megatron-style loss
                # Compute reduced_train_loss: globally aggregated across all DP ranks
                # This matches John's Megatron logging where loss is all-reduced before logging
                #
                # The math:
                #   Each rank has: accumulated_loss_sum (sum of CE losses across grad_acc_steps microbatches)
                #                  accumulated_tokens (count of valid tokens across those microbatches)
                #   Global loss = sum(all ranks' loss_sum) / sum(all ranks' tokens)
                #   This is the TRUE per-token loss across the entire global batch

                reduced_loss = None  # Will be set if we compute it

                if use_megatron_loss:
                    # Compute local (per-rank) megatron_loss for backward compatibility
                    local_megatron_loss = accumulated_loss_sum / max(accumulated_tokens, 1)

                    # Compute reduced_train_loss (all-reduced across DP ranks)
                    # This is always computed when use_megatron_loss=True, as it's cheap
                    loss_tensor = torch.tensor([accumulated_loss_sum, float(accumulated_tokens)], device=device)
                    torch.distributed.all_reduce(loss_tensor)
                    global_loss_sum = loss_tensor[0].item()
                    global_tokens = int(loss_tensor[1].item())
                    reduced_loss = global_loss_sum / max(global_tokens, 1)

                    perf_logger.log_step(
                        step=step,
                        grad_norm=total_norm,
                        lr=optimizer.param_groups[0]["lr"],
                        megatron_loss=local_megatron_loss,  # Per-rank loss
                        total_tokens=accumulated_tokens,  # Per-rank tokens
                        reduced_train_loss=reduced_loss,  # Global loss (matches John's)
                        tev_mean=tev_mean,
                        tev_sd=tev_sd,
                    )
                    # Reset accumulators
                    accumulated_loss_sum = 0.0
                    accumulated_tokens = 0
                else:
                    # HuggingFace-style loss
                    # Compute reduced_train_loss by all-reducing the HF loss sums
                    # HF loss is already mean-reduced per batch, so we average across ranks
                    loss_tensor = torch.tensor(
                        [accumulated_hf_loss_sum, float(accumulated_hf_batch_count)], device=device
                    )
                    torch.distributed.all_reduce(loss_tensor)
                    global_loss_sum = loss_tensor[0].item()
                    global_batch_count = int(loss_tensor[1].item())
                    reduced_loss = global_loss_sum / max(global_batch_count, 1)

                    perf_logger.log_step(
                        step=step,
                        grad_norm=total_norm,
                        lr=optimizer.param_groups[0]["lr"],
                        reduced_train_loss=reduced_loss,  # Global loss
                        tev_mean=tev_mean,
                        tev_sd=tev_sd,
                    )

                    # Reset HF accumulators
                    accumulated_hf_loss_sum = 0.0
                    accumulated_hf_batch_count = 0

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

                # Run validation at specified interval
                if val_enabled and val_dataloader is not None and step > 0 and step % val_config.eval_interval == 0:
                    val_metrics = run_validation(
                        model=model,
                        val_dataloader=val_dataloader,
                        num_batches=val_config.num_batches,
                        device=device,
                        dist_config=dist_config,
                        is_thd=is_thd,
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
