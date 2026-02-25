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

r"""Evaluate a Megatron/MBridge checkpoint directly on a training parquet shard (no dataloader).

PURPOSE
=======
Same goal as ``evaluate_train_shard_loss.py`` but for **Megatron-style** checkpoints
(those produced by MBridge / evo2_megatron training).  This lets you evaluate
John's Megatron model on the same raw parquet training shards and compare loss
numbers with the FSDP2/TE models.

DATA FORMAT
===========
Expects parquet files with a ``text`` column containing raw DNA sequences —
the same format as ``evaluate_train_shard_loss.py``.

The training parquet shards on Lepton are at: ``/data/opengenome2/parquet/``

METHODOLOGY
===========
Matches the **Megatron eden training data format** exactly:
  1. For each sequence in the parquet shard, create windows of ``eff_len``
     bases (= ``seq_length - 2 = 8190`` by default) with ``stride`` base
     overlap (default 7992, giving 198 bases overlap — matching eden training).
  2. Tokenize each window: ``[BOS(1), byte_tokens..., EOS(2), PAD(0)...]``
     padded to ``seq_length``.  The byte-level tokenizer maps each DNA
     character to its ASCII value (A=65, C=67, G=71, T=84).
  3. Create labels (shifted by 1) and loss_mask (masking special tokens
     BOS=1, EOS=2, SEP=3, PAD=0 — matching ``ShardedEdenDataset``).
  4. Forward pass through the Megatron model, compute per-token CE loss.

No shuffling.  No distributed sampler.  Just raw loss on raw data.

CHECKPOINT FORMAT
=================
Supports MBridge checkpoints: directories containing ``run_config.yaml``
and model weights (``iter_XXXXXXX/`` or flat directory structure).

ENVIRONMENT
===========
Requires ``megatron.bridge`` and ``megatron.core`` to be installed.
Run this from the evo2 container or an environment with these packages.

EXAMPLE USAGE
=============
::

    # ── Evaluate John's Megatron model on a training shard ──────────────
    torchrun --nproc_per_node=1 evaluate_train_shard_loss_megatron.py \
        --ckpt-dir /data/johns_checkpoints/eden_7b_megatron \
        --parquet-path /data/opengenome2/parquet/some_shard.parquet \
        --tokenizer-path ./tokenizers/nucleotide_fast_tokenizer \
        --max-sequences 500 \
        --output /data/savithas/eval_results/megatron_train_shard.json

    # ── With tensor parallelism (if checkpoint uses TP) ─────────────────
    torchrun --nproc_per_node=2 evaluate_train_shard_loss_megatron.py \
        --ckpt-dir /data/johns_checkpoints/eden_7b_megatron \
        --parquet-path /data/opengenome2/parquet/some_shard.parquet \
        --tensor-parallel-size 2 \
        --output /data/savithas/eval_results/megatron_tp2.json
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
from megatron.bridge.training.config import DistributedInitConfig, RNGConfig
from megatron.bridge.training.mixed_precision import MIXED_PRECISION_RECIPES, get_mixed_precision_config
from megatron.bridge.training.tokenizers.tokenizer import _HuggingFaceTokenizer
from megatron.bridge.training.utils.checkpoint_utils import (
    file_exists,
    get_checkpoint_run_config_filename,
    read_run_config,
)
from megatron.bridge.utils.common_utils import (
    get_local_rank_preinit,
    get_master_addr_safe,
    get_master_port_safe,
    get_rank_safe,
    get_world_size_safe,
)
from megatron.bridge.utils.instantiate_utils import instantiate
from megatron.core import parallel_state, tensor_parallel
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.core.tensor_parallel.mappings import _gather_along_last_dim
from megatron.core.transformer.module import Float16Module
from torch.nn.functional import cross_entropy


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Eden special token IDs (byte-level tokenizer with eden patches)
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3
PAD_ID = 0
SPECIAL_TOKEN_IDS = frozenset({PAD_ID, BOS_ID, EOS_ID, SEP_ID})

# Valid DNA token IDs: A(65), C(67), G(71), T(84) and lowercase a(97), c(99), g(103), t(116)
DNA_TOKENS = frozenset({65, 67, 71, 84, 97, 99, 103, 116})


# ---------------------------------------------------------------------------
# Parquet reading (same as HF eval script)
# ---------------------------------------------------------------------------


def read_sequences_from_parquet(
    parquet_path: str,
    text_column: str = "text",
    max_sequences: int | None = None,
) -> list[str]:
    """Read raw DNA sequences from one or more parquet files.

    Args:
        parquet_path: Path to a single parquet file, or a glob pattern.
        text_column: Name of the column containing DNA sequences.
        max_sequences: If set, read at most this many sequences.

    Returns:
        List of raw DNA sequence strings.
    """
    paths = sorted(glob.glob(parquet_path))
    if not paths:
        raise FileNotFoundError(f"No parquet files matched: {parquet_path}")

    logger.info("Found %d parquet file(s) matching '%s'", len(paths), parquet_path)

    sequences: list[str] = []
    for p in paths:
        df = pd.read_parquet(p, columns=[text_column])
        sequences.extend(df[text_column].tolist())
        if max_sequences is not None and len(sequences) >= max_sequences:
            sequences = sequences[:max_sequences]
            break

    logger.info("Loaded %d sequences from parquet", len(sequences))
    return sequences


# ---------------------------------------------------------------------------
# Tokenization + windowing (Megatron eden format)
# ---------------------------------------------------------------------------


def tokenize_with_windowing_megatron(
    sequences: list[str],
    seq_length: int = 8192,
    stride: int = 7992,
) -> list[dict[str, torch.Tensor]]:
    """Tokenize sequences with windowing matching Megatron eden training format.

    Each window is formatted as: [BOS(1), content_bytes..., EOS(2), PAD(0)...]
    padded to ``seq_length``.

    The byte-level tokenizer maps each character to its ASCII value.  We replicate
    this directly with ``ord()`` for simplicity and correctness.

    Args:
        sequences: Raw DNA strings.
        seq_length: Total token length per window (must match training).
        stride: Step between consecutive window starts in bases (must match training).

    Returns:
        List of dicts, each with ``tokens`` and ``position_ids`` tensors
        of shape ``(seq_length,)``.
    """
    # Effective content length per window (room for BOS + EOS)
    eff_len = seq_length - 2  # e.g. 8192 - 2 = 8190

    windows: list[dict[str, torch.Tensor]] = []

    for seq in sequences:
        seq_upper = seq.upper()
        seq_len = len(seq_upper)

        if seq_len == 0:
            continue

        # Compute window starts
        if seq_len <= eff_len:
            starts = [0]
        else:
            starts = list(range(0, seq_len - eff_len + 1, stride))
            # Ensure we cover the last part of the sequence
            if starts[-1] + eff_len < seq_len:
                starts.append(seq_len - eff_len)

        for start in starts:
            # Extract content bases for this window
            content = seq_upper[start : start + eff_len]

            # Byte-level tokenization: each character → ASCII value
            content_tokens = [ord(c) for c in content]

            # Build token sequence: [BOS, content..., EOS, PAD...]
            token_ids = [BOS_ID, *content_tokens, EOS_ID]

            # Pad to seq_length if needed
            pad_len = seq_length - len(token_ids)
            if pad_len > 0:
                token_ids = token_ids + [PAD_ID] * pad_len
            else:
                token_ids = token_ids[:seq_length]

            tokens = torch.tensor(token_ids, dtype=torch.long)
            position_ids = torch.arange(seq_length, dtype=torch.long)

            windows.append({"tokens": tokens, "position_ids": position_ids})

    logger.info(
        "Created %d windows from %d sequences (eff_len=%d, stride=%d)",
        len(windows),
        len(sequences),
        eff_len,
        stride,
    )
    return windows


# ---------------------------------------------------------------------------
# Checkpoint resolution (copied from predict.py to keep script self-contained)
# ---------------------------------------------------------------------------


def resolve_checkpoint_path(checkpoint_path: Path) -> Path:
    """Resolve a checkpoint path to the actual checkpoint directory.

    MBridge checkpoints can be organized in two ways:
    1. Direct checkpoint: A directory containing run_config.yaml directly
    2. Training output: A parent directory containing iter_XXXXXXX subdirectories

    Args:
        checkpoint_path: Path to either a direct checkpoint or a training output directory.

    Returns:
        Path to the checkpoint directory containing run_config.yaml.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path '{checkpoint_path}' does not exist.")
    if not checkpoint_path.is_dir():
        raise NotADirectoryError(f"Checkpoint path '{checkpoint_path}' must be a directory.")

    # Check if run_config.yaml exists directly in this path
    run_config_path = get_checkpoint_run_config_filename(str(checkpoint_path))
    if file_exists(run_config_path):
        return checkpoint_path

    # Look for iter_* subdirectories
    iter_dirs = [
        (child.name, child) for child in checkpoint_path.iterdir() if child.is_dir() and child.name.startswith("iter_")
    ]

    if not iter_dirs:
        raise FileNotFoundError(
            f"No valid checkpoint found at '{checkpoint_path}'. "
            "Expected either run_config.yaml in the directory or iter_* subdirectories."
        )

    # Find the latest iteration
    def _parse_iter_num(item: tuple[str, Path]) -> int:
        try:
            return int(item[0].replace("iter_", ""))
        except ValueError:
            return -1

    _, latest_iter_path = max(iter_dirs, key=_parse_iter_num)

    run_config_path = get_checkpoint_run_config_filename(str(latest_iter_path))
    if not file_exists(run_config_path):
        raise FileNotFoundError(f"Latest checkpoint directory '{latest_iter_path}' does not contain run_config.yaml.")

    logger.info("Resolved checkpoint path to: %s", latest_iter_path)
    return latest_iter_path


# ---------------------------------------------------------------------------
# Distributed initialization (copied from predict.py)
# ---------------------------------------------------------------------------


def initialize_inference_distributed(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    micro_batch_size: int = 1,
    global_batch_size: int = 1,
    rng_config: RNGConfig | None = None,
    dist_config: DistributedInitConfig | None = None,
) -> None:
    """Initialize distributed environment for inference (Megatron parallel state)."""
    if rng_config is None:
        rng_config = RNGConfig(seed=1234)
    if dist_config is None:
        dist_config = DistributedInitConfig()

    assert torch.cuda.is_available(), "Inference requires CUDA."

    device_count = torch.cuda.device_count()
    world_size = get_world_size_safe()
    model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    data_parallel_size = world_size // model_parallel_size

    init_num_microbatches_calculator(
        rank=get_rank_safe(),
        rampup_batch_size=None,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
        decrease_batch_size_if_needed=False,
    )

    if not dist.is_initialized():
        if get_rank_safe() == 0:
            print("> initializing torch distributed for inference ...", flush=True)

        if device_count > 0:
            torch.cuda.set_device(get_local_rank_preinit())

        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = get_master_addr_safe()
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(get_master_port_safe())

        dist.init_process_group(
            backend=dist_config.distributed_backend,
            world_size=world_size,
            rank=get_rank_safe(),
            timeout=datetime.timedelta(minutes=dist_config.distributed_timeout_minutes),
        )
        dist.barrier(device_ids=[get_local_rank_preinit()])

    if device_count > 0 and not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            distributed_timeout_minutes=dist_config.distributed_timeout_minutes,
        )

    # Set random seeds
    seed = rng_config.seed + (100 * parallel_state.get_pipeline_model_parallel_rank())
    if rng_config.data_parallel_random_init:
        seed = seed + (10 * parallel_state.get_data_parallel_rank())
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if device_count > 0:
        tensor_parallel.model_parallel_cuda_manual_seed(
            seed, rng_config.te_rng_tracker, rng_config.inference_rng_tracker
        )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_megatron_model(
    ckpt_dir: Path,
    tokenizer_path: str,
    tensor_parallel_size: int = 1,
    mixed_precision_recipe: str = "bf16_mixed",
) -> tuple[list, Path]:
    """Load a Megatron/MBridge model from checkpoint.

    This follows the same pattern as ``predict.py`` in the evo2_megatron recipe:
    1. Resolve checkpoint path
    2. Read run_config.yaml to instantiate the model provider
    3. Configure parallelism and precision
    4. Initialize distributed environment
    5. Create model and load weights

    Args:
        ckpt_dir: Path to the MBridge checkpoint directory.
        tokenizer_path: Path to the HuggingFace tokenizer.
        tensor_parallel_size: Tensor parallelism degree.
        mixed_precision_recipe: Mixed precision recipe name.

    Returns:
        Tuple of (model_list, resolved_checkpoint_path).
    """
    # Step 1: Resolve checkpoint
    resolved_ckpt_dir = resolve_checkpoint_path(ckpt_dir)
    logger.info("Loading configuration from checkpoint: %s", resolved_ckpt_dir)

    # Step 2: Read config and instantiate model provider
    run_config_filename = get_checkpoint_run_config_filename(str(resolved_ckpt_dir))
    run_config = read_run_config(run_config_filename)
    model_provider = instantiate(run_config["model"])
    logger.info("Instantiated model provider: %s", type(model_provider).__name__)

    # Step 3: Configure parallelism
    model_provider.tensor_model_parallel_size = tensor_parallel_size
    model_provider.pipeline_model_parallel_size = 1
    model_provider.context_parallel_size = 1
    model_provider.sequence_parallel = False

    # Step 4: Load tokenizer and set vocab size
    tokenizer_dir = resolved_ckpt_dir / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer = _HuggingFaceTokenizer(tokenizer_dir)
        logger.info("Loaded tokenizer from checkpoint: %s", tokenizer_dir)
    else:
        tokenizer = _HuggingFaceTokenizer(tokenizer_path)
        logger.info("Loaded tokenizer from: %s", tokenizer_path)

    model_provider.vocab_size = tokenizer.vocab_size
    model_provider.should_pad_vocab = True

    # Step 5: Configure mixed precision
    if mixed_precision_recipe in MIXED_PRECISION_RECIPES:
        mp_config = get_mixed_precision_config(mixed_precision_recipe)
    else:
        mp_config = get_mixed_precision_config("bf16_mixed")
    mp_config.finalize()
    mp_config.setup(model_provider)

    # Step 6: Initialize distributed
    rng_config = instantiate(run_config.get("rng")) if run_config.get("rng") else RNGConfig(seed=1234)
    dist_config = instantiate(run_config.get("dist")) if run_config.get("dist") else DistributedInitConfig()

    world_size = get_world_size_safe()
    data_parallel_size = world_size // tensor_parallel_size
    global_batch_size = 1 * data_parallel_size

    initialize_inference_distributed(
        tensor_model_parallel_size=tensor_parallel_size,
        micro_batch_size=1,
        global_batch_size=global_batch_size,
        rng_config=rng_config,
        dist_config=dist_config,
    )
    logger.info("Initialized distributed environment (TP=%d, world_size=%d)", tensor_parallel_size, world_size)

    # Step 7: Create model
    model_provider.finalize()
    model = model_provider.provide_distributed_model(
        ddp_config=None,
        wrap_with_ddp=False,
        data_parallel_random_init=False,
        bf16=mp_config.bf16,
        fp16=mp_config.fp16,
        mixed_precision_wrapper=Float16Module if (mp_config.bf16 or mp_config.fp16) else None,
    )
    for m in model:
        m.eval()

    # Log model info
    model_for_inspection = model[0]
    if hasattr(model_for_inspection, "module"):
        model_for_inspection = model_for_inspection.module
    if hasattr(model_for_inspection, "decoder") and hasattr(model_for_inspection.decoder, "layers"):
        num_layers = len(model_for_inspection.decoder.layers)
        logger.info("Model initialized with %d layers", num_layers)

    # Step 8: Load weights
    logger.info("Loading weights from: %s", resolved_ckpt_dir)
    _load_model_weights_from_checkpoint(
        checkpoint_path=str(resolved_ckpt_dir),
        model=model,
        dist_ckpt_strictness="ignore_all",
    )
    logger.info("Weights loaded successfully")

    return model, resolved_ckpt_dir


# ---------------------------------------------------------------------------
# Core: cross-entropy loss computation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_loss_on_windows(
    model: list,
    windows: list[dict[str, torch.Tensor]],
    micro_batch_size: int = 1,
    mask_degenerate_bases: bool = False,
) -> dict:
    """Compute cross-entropy loss over pre-tokenized windows using a Megatron model.

    Matches the loss computation from Megatron eden training:
      - Labels are shifted by 1 (causal LM: predict next token)
      - Special tokens (BOS=1, EOS=2, SEP=3, PAD=0) are masked in labels
      - Optionally, non-ACGT degenerate bases are masked too
      - Loss = per-token CE averaged over valid tokens

    Args:
        model: The loaded Megatron model list (model[0] is the actual model).
        windows: List of dicts with ``tokens`` and ``position_ids`` tensors.
        micro_batch_size: Number of windows per forward pass.
        mask_degenerate_bases: Whether to mask non-ACGT bases.  Default False
            to match Megatron eden training (which does NOT mask degenerate bases).

    Returns:
        Dictionary with loss statistics.
    """
    model_module = model[0]
    model_module.eval()
    device = torch.device(f"cuda:{get_local_rank_preinit()}")

    special_tokens_tensor = torch.tensor(sorted(SPECIAL_TOKEN_IDS), device=device)
    dna_tokens_tensor = torch.tensor(sorted(DNA_TOKENS), device=device)

    tp_size = parallel_state.get_tensor_model_parallel_world_size()

    total_hf_loss = 0.0
    total_token_loss = 0.0
    total_valid_tokens = 0
    total_windows = 0
    num_batches = 0
    per_window_losses: list[float] = []

    for start in range(0, len(windows), micro_batch_size):
        batch_windows = windows[start : start + micro_batch_size]
        batch_size = len(batch_windows)

        # Stack into batch tensors
        tokens = torch.stack([w["tokens"] for w in batch_windows]).to(device)
        position_ids = torch.stack([w["position_ids"] for w in batch_windows]).to(device)

        # Forward pass: model returns logits
        logits = model_module(input_ids=tokens, position_ids=position_ids, attention_mask=None)

        # Gather logits across tensor parallel ranks (vocab dim is sharded)
        if tp_size > 1:
            logits = _gather_along_last_dim(logits, group=parallel_state.get_tensor_model_parallel_group())

        # Handle sequence-first output format: [S, B, V] → [B, S, V]
        if logits.dim() == 3 and logits.shape[0] != batch_size and logits.shape[1] == batch_size:
            logits = logits.transpose(0, 1).contiguous()

        # Shift: logits[:, :-1] predicts tokens[:, 1:]
        pred_logits = logits[:, :-1, :].contiguous()  # [B, S-1, V]
        targets = tokens[:, 1:].contiguous()  # [B, S-1]

        # Create loss mask matching Megatron eden training:
        # Mask where target is a special token (BOS, EOS, SEP, PAD)
        loss_mask = torch.ones_like(targets, dtype=torch.float)
        loss_mask[torch.isin(targets, special_tokens_tensor)] = 0.0

        # Optionally mask degenerate (non-ACGT) bases
        if mask_degenerate_bases:
            not_dna = ~torch.isin(targets, dna_tokens_tensor)
            not_special = ~torch.isin(targets, special_tokens_tensor)
            loss_mask[not_dna & not_special] = 0.0

        # Per-token cross-entropy
        per_token_loss = cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)

        # Apply mask and compute batch loss
        masked_loss = per_token_loss * loss_mask
        valid_tokens = loss_mask.sum().item()
        batch_loss = masked_loss.sum().item() / max(valid_tokens, 1)

        total_hf_loss += batch_loss
        total_token_loss += masked_loss.sum().item()
        total_valid_tokens += int(valid_tokens)
        total_windows += batch_size
        num_batches += 1
        per_window_losses.append(batch_loss)

        if total_windows % 50 == 0 and total_windows > 0:
            running_avg = total_token_loss / max(total_valid_tokens, 1)
            logger.info(
                "  [%d/%d windows]  running_megatron_loss=%.4f  valid_tokens=%s",
                total_windows,
                len(windows),
                running_avg,
                f"{total_valid_tokens:,}",
            )

    # Compute final aggregates
    hf_avg_loss = total_hf_loss / max(num_batches, 1)
    megatron_avg_loss = total_token_loss / max(total_valid_tokens, 1)

    return {
        "avg_loss": hf_avg_loss,
        "perplexity": math.exp(min(hf_avg_loss, 100)),
        "megatron_loss": megatron_avg_loss,
        "megatron_perplexity": math.exp(min(megatron_avg_loss, 100)),
        "total_windows": total_windows,
        "total_valid_tokens": total_valid_tokens,
        "total_batches": num_batches,
        "per_window_losses": per_window_losses,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Evaluate a Megatron/MBridge checkpoint on a training parquet shard (no dataloader).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt-dir", type=Path, required=True, help="Path to MBridge checkpoint directory.")
    p.add_argument(
        "--parquet-path",
        type=str,
        required=True,
        help="Path to parquet file(s). Supports glob patterns.",
    )
    p.add_argument("--text-column", type=str, default="text", help="Column name for sequences in parquet files.")
    p.add_argument(
        "--tokenizer-path",
        type=str,
        default="./tokenizers/nucleotide_fast_tokenizer",
        help="Path to HF tokenizer (fallback if not in checkpoint).",
    )
    p.add_argument("--seq-length", type=int, default=8192, help="Total sequence length (must match training).")
    p.add_argument("--stride", type=int, default=7992, help="Window stride in bases (must match training).")
    p.add_argument("--micro-batch-size", type=int, default=1, help="Windows per forward pass.")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism degree.")
    p.add_argument(
        "--mixed-precision-recipe",
        type=str,
        default="bf16_mixed",
        choices=list(MIXED_PRECISION_RECIPES.keys()),
        help="Mixed precision recipe.",
    )
    p.add_argument("--max-sequences", type=int, default=None, help="Max sequences to read from parquet.")
    p.add_argument("--max-windows", type=int, default=None, help="Max windows to evaluate.")
    p.add_argument(
        "--mask-degenerate-bases",
        action="store_true",
        default=False,
        help="Mask non-ACGT bases in loss (default: False — matching Megatron eden training).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None, help="Path to write results JSON.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    args = parse_args()

    rank = get_rank_safe()

    if rank == 0:
        logger.info("=" * 70)
        logger.info("Training-Shard Loss Evaluation — Megatron/MBridge (no dataloader)")
        logger.info("=" * 70)
        logger.info("  ckpt_dir              : %s", args.ckpt_dir)
        logger.info("  parquet_path          : %s", args.parquet_path)
        logger.info("  tokenizer_path        : %s", args.tokenizer_path)
        logger.info("  seq_length            : %d", args.seq_length)
        logger.info("  stride                : %d", args.stride)
        logger.info("  micro_batch_size      : %d", args.micro_batch_size)
        logger.info("  tensor_parallel_size  : %d", args.tensor_parallel_size)
        logger.info("  mixed_precision       : %s", args.mixed_precision_recipe)
        logger.info("  mask_degenerate_bases : %s", args.mask_degenerate_bases)
        logger.info("  max_sequences         : %s", args.max_sequences)
        logger.info("  max_windows           : %s", args.max_windows)
        logger.info("  seed                  : %d", args.seed)
        logger.info("=" * 70)

    # ── read parquet data ────────────────────────────────────────────────
    if rank == 0:
        logger.info("Reading sequences from parquet …")
    sequences = read_sequences_from_parquet(
        args.parquet_path,
        text_column=args.text_column,
        max_sequences=args.max_sequences,
    )

    # ── tokenize with windowing (Megatron eden format) ───────────────────
    if rank == 0:
        logger.info(
            "Tokenizing sequences (seq_length=%d, stride=%d, eff_len=%d) …",
            args.seq_length,
            args.stride,
            args.seq_length - 2,
        )
    windows = tokenize_with_windowing_megatron(sequences, args.seq_length, args.stride)

    if args.max_windows is not None and len(windows) > args.max_windows:
        if rank == 0:
            logger.info("Limiting to %d windows (out of %d)", args.max_windows, len(windows))
        windows = windows[: args.max_windows]

    if rank == 0:
        logger.info("Total windows to evaluate: %d", len(windows))

    # ── load model ───────────────────────────────────────────────────────
    model, resolved_ckpt = load_megatron_model(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        tensor_parallel_size=args.tensor_parallel_size,
        mixed_precision_recipe=args.mixed_precision_recipe,
    )

    # ── evaluation ───────────────────────────────────────────────────────
    if rank == 0:
        logger.info("Computing loss on %d windows …", len(windows))

    results = compute_loss_on_windows(
        model=model,
        windows=windows,
        micro_batch_size=args.micro_batch_size,
        mask_degenerate_bases=args.mask_degenerate_bases,
    )

    # ── report ───────────────────────────────────────────────────────────
    if rank == 0:
        logger.info("=" * 70)
        logger.info(
            "RESULTS  (%d windows, %s valid tokens)", results["total_windows"], f"{results['total_valid_tokens']:,}"
        )
        logger.info("=" * 70)
        logger.info("  HF-style avg loss     : %.4f", results["avg_loss"])
        logger.info("  HF-style perplexity   : %.2f", results["perplexity"])
        logger.info("  Megatron-style loss   : %.4f", results["megatron_loss"])
        logger.info("  Megatron-style PPL    : %.2f", results["megatron_perplexity"])
        logger.info("  Total windows         : %d", results["total_windows"])
        logger.info("  Total valid tokens    : %s", f"{results['total_valid_tokens']:,}")
        logger.info("=" * 70)

        # ── save JSON ────────────────────────────────────────────────────
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "checkpoint": str(resolved_ckpt),
                "parquet_path": args.parquet_path,
                "text_column": args.text_column,
                "seq_length": args.seq_length,
                "stride": args.stride,
                "seed": args.seed,
                "mask_degenerate_bases": args.mask_degenerate_bases,
                "tensor_parallel_size": args.tensor_parallel_size,
                "mixed_precision_recipe": args.mixed_precision_recipe,
                "max_sequences": args.max_sequences,
                "max_windows": args.max_windows,
                "num_sequences_loaded": len(sequences),
                "num_windows_created": len(windows),
                "summary": {
                    "avg_loss": results["avg_loss"],
                    "perplexity": results["perplexity"],
                    "megatron_loss": results["megatron_loss"],
                    "megatron_perplexity": results["megatron_perplexity"],
                    "total_windows": results["total_windows"],
                    "total_valid_tokens": results["total_valid_tokens"],
                    "total_batches": results["total_batches"],
                },
            }
            with open(out, "w") as f:
                json.dump(payload, f, indent=2)
            logger.info("Results saved → %s", out)

    # ── cleanup ──────────────────────────────────────────────────────────
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
