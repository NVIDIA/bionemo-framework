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

r"""Evaluate a checkpoint directly on a training parquet shard (no dataloader).

PURPOSE
=======
This script is a **diagnostic tool** to verify that a trained model achieves low
cross-entropy loss on the data it was trained on.  It reads raw parquet files,
tokenizes sequences with windowing (matching the training pipeline), and computes
per-token cross-entropy loss — all without any dataloader or sampler machinery.

The idea: if the model genuinely learned from the training data, its loss on a
training shard should be low.  Running a *different* model (trained on different
data or untrained) on the same shard should produce higher loss.

DATA FORMAT
===========
Expects parquet files with a ``text`` column containing raw DNA sequences (the
same format used by the HuggingFace streaming dataset in ``train_fsdp2.py``).
The training parquet shards on Lepton are at: ``/data/opengenome2/parquet/``

METHODOLOGY
===========
For each sequence in the parquet shard:
  1. Tokenize with the nucleotide tokenizer (add_special_tokens=True).
  2. Apply windowing: split long sequences into overlapping windows of
     ``max_seq_length`` tokens with ``stride`` token overlap — matching
     training's ``return_overflowing_tokens`` windowing.
  3. For each window, run a forward pass and compute cross-entropy loss,
     masking non-ACGT (degenerate) bases to match training.
  4. Aggregate: per-token average cross-entropy across all windows.

No shuffling.  No distributed sampler.  Just raw loss on raw data.

CHECKPOINT FORMATS
==================
Supports the same formats as the other evaluation scripts:
  - FSDP2 DCP  (train_fsdp2/ subdirectory with step_N/ directories)
  - DDP        (checkpoint.pt files)
  - Safetensors (final_model/ directories)

EXAMPLE USAGE
=============
::

    cd bionemo-recipes/recipes/llama3_native_te

    # ── Your model on a training shard ──────────────────────────────────
    torchrun --nproc_per_node=1 evaluate_train_shard_loss.py \
        --checkpoint-dir /data/savithas/checkpoints/og2-7b-bf16-sanity \
        --checkpoint-step 20000 \
        --parquet-path /data/opengenome2/parquet/some_shard.parquet \
        --output /data/savithas/eval_results/eden_train_shard.json

    # ── Different model on the same shard (expect higher loss) ─────────
    torchrun --nproc_per_node=1 evaluate_train_shard_loss.py \
        --checkpoint-dir /data/savithas/checkpoints/og2-7b-bf16-fp32master-hf-ws \
        --checkpoint-step 20000 \
        --parquet-path /data/opengenome2/parquet/some_shard.parquet \
        --output /data/savithas/eval_results/hf_ws_train_shard.json

    # ── Compare ────────────────────────────────────────────────────────
    python -c "
    import json
    a = json.load(open('/data/savithas/eval_results/eden_train_shard.json'))
    b = json.load(open('/data/savithas/eval_results/hf_ws_train_shard.json'))
    print()
    print('=== Training-shard loss comparison ===')
    for tag, r in [('Eden-trained', a), ('HF-WS-trained', b)]:
        s = r['summary']
        print(f'  {tag:20s}:  loss={s[\"avg_loss\"]:.4f}  '
              f'ppl={s[\"perplexity\"]:.2f}  '
              f'windows={s[\"total_windows\"]}  '
              f'tokens={s[\"total_valid_tokens\"]:,}')
    "

If you want to evaluate on multiple shards, just use a glob::

    torchrun --nproc_per_node=1 evaluate_train_shard_loss.py \
        --checkpoint-dir /data/savithas/checkpoints/og2-7b-bf16-sanity \
        --parquet-path '/data/opengenome2/parquet/*.parquet' \
        --max-sequences 500 \
        --output /data/savithas/eval_results/eden_multi_shard.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import transformer_engine.pytorch
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoTokenizer

from checkpoint import AppState
from distributed_config import DistributedConfig
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM
from scheduler import get_cosine_annealing_schedule_with_warmup


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Valid DNA token IDs: A(65), C(67), G(71), T(84) and lowercase a(97), c(99), g(103), t(116)
DNA_TOKENS = frozenset({65, 67, 71, 84, 97, 99, 103, 116})


# ---------------------------------------------------------------------------
# Parquet reading + tokenization (NO dataloader)
# ---------------------------------------------------------------------------


def read_sequences_from_parquet(
    parquet_path: str,
    text_column: str = "text",
    max_sequences: int | None = None,
) -> list[str]:
    """Read raw DNA sequences from one or more parquet files.

    Args:
        parquet_path: Path to a single parquet file, or a glob pattern
            (e.g., ``/data/opengenome2/parquet/*.parquet``).
        text_column: Name of the column containing DNA sequences.
        max_sequences: If set, read at most this many sequences (for speed).

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


def tokenize_with_windowing(
    sequences: list[str],
    tokenizer,
    max_seq_length: int = 8192,
    stride: int = 200,
) -> list[dict[str, torch.Tensor]]:
    """Tokenize sequences with overlapping windowing — matching training exactly.

    This replicates the tokenization done by ``dataset.py::create_tokenized_dataset``
    which uses ``return_overflowing_tokens=True`` to split long sequences into
    overlapping windows.

    Args:
        sequences: Raw DNA strings.
        tokenizer: HuggingFace tokenizer.
        max_seq_length: Window size (must match training).
        stride: Overlap between windows (must match training).

    Returns:
        List of dicts, each with ``input_ids`` and ``attention_mask`` tensors
        of shape ``(max_seq_length,)``.
    """
    windows: list[dict[str, torch.Tensor]] = []

    for seq in sequences:
        encoded = tokenizer(
            seq,
            max_length=max_seq_length,
            stride=stride,
            truncation=True,
            return_overflowing_tokens=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )

        # encoded["input_ids"] has shape (num_windows, max_seq_length)
        num_windows = encoded["input_ids"].shape[0]
        windows.extend(
            {"input_ids": encoded["input_ids"][i], "attention_mask": encoded["attention_mask"][i]}
            for i in range(num_windows)
        )

    logger.info("Created %d windows from %d sequences", len(windows), len(sequences))
    return windows


# ---------------------------------------------------------------------------
# Checkpoint helpers (same as other eval scripts)
# ---------------------------------------------------------------------------


def find_checkpoint_path(checkpoint_dir: str, step: int | None = None) -> tuple[Path, str]:
    """Locate the checkpoint inside *checkpoint_dir* and return ``(path, type)``."""
    root = Path(checkpoint_dir)

    # 1. safetensors
    for candidate in [root, root / "final_model", root / "train_fsdp2" / "final_model"]:
        if (candidate / "model.safetensors").exists():
            return candidate, "safetensors"

    # 2. FSDP2 DCP step directories
    fsdp2_dir = root / "train_fsdp2" if (root / "train_fsdp2").exists() else root
    step_dirs = sorted(
        [d for d in fsdp2_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if step_dirs:
        if step is not None:
            target = fsdp2_dir / f"step_{step}"
            if not target.exists():
                raise FileNotFoundError(f"step_{step} not found. Available: {[d.name for d in step_dirs]}")
            chosen = target
        else:
            chosen = step_dirs[-1]
        if (chosen / ".metadata").exists() or any(chosen.glob("*.distcp")):
            return chosen, "dcp"
        if (chosen / "checkpoint.pt").exists():
            return chosen, "ddp"
        return chosen, "dcp"

    # 3. root itself
    if (root / "checkpoint.pt").exists():
        return root, "ddp"
    if (root / ".metadata").exists() or any(root.glob("*.distcp")):
        return root, "dcp"

    raise FileNotFoundError(f"No recognisable checkpoint in {checkpoint_dir}")


# ---------------------------------------------------------------------------
# Model building + loading
# ---------------------------------------------------------------------------


def _build_model_config(config_name_or_path: str, num_kv_heads: int = 8) -> NVLlamaConfig:
    """Build the 7B config."""
    return NVLlamaConfig.from_pretrained(
        config_name_or_path,
        dtype=torch.float32,
        vocab_size=256,
        num_hidden_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=8192,
        initializer_range=0.02,
        attn_input_format="bshd",
        self_attn_mask_type="causal",
        rope_theta=500000,
        rope_scaling={
            "type": "llama3",
            "factor": 1,
            "low_freq_factor": 1,
            "high_freq_factor": 4,
            "original_max_position_embeddings": 8192,
        },
    )


def load_model_from_checkpoint(
    ckpt_path: Path,
    ckpt_type: str,
    config: NVLlamaConfig,
    dist_config: DistributedConfig,
    device_mesh,
) -> NVLlamaForCausalLM:
    """Create, FSDP2-shard, and load weights from *ckpt_path*."""
    model = NVLlamaForCausalLM(config)
    if dist_config.rank == 0:
        logger.info("Model created (%s parameters)", f"{sum(p.numel() for p in model.parameters()):,}")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=False,
    )
    for layer in model.model.layers:
        fully_shard(layer, mesh=device_mesh["dp"], mp_policy=mp_policy)
    fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)

    if ckpt_type == "dcp":
        if dist_config.rank == 0:
            logger.info("Loading FSDP2 DCP checkpoint from %s …", ckpt_path)
        from torch.optim import AdamW

        optimizer = AdamW(model.parameters(), lr=1e-5)
        scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, num_warmup_steps=100, num_decay_steps=1000)
        app_state = AppState(model=model, optimizer=optimizer, scheduler=scheduler)
        dcp_load({"app": app_state}, checkpoint_id=ckpt_path, process_group=device_mesh.get_group("dp"))
        if dist_config.rank == 0:
            logger.info("DCP checkpoint loaded (step=%d, epoch=%d)", app_state.step, app_state.epoch)

    elif ckpt_type == "ddp":
        if dist_config.rank == 0:
            logger.info("Loading DDP checkpoint from %s …", ckpt_path)
        ckpt = torch.load(ckpt_path / "checkpoint.pt", map_location="cpu", weights_only=True)
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        set_model_state_dict(model, model_state_dict=ckpt["model"], options=StateDictOptions(strict=False))
        if dist_config.rank == 0:
            logger.info("DDP checkpoint loaded (step=%d)", ckpt.get("step", -1))

    elif ckpt_type == "safetensors":
        if dist_config.rank == 0:
            logger.info("Loading safetensors from %s …", ckpt_path)
        from safetensors.torch import load_file
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        weights = load_file(str(ckpt_path / "model.safetensors"))
        set_model_state_dict(model, model_state_dict=weights, options=StateDictOptions(strict=False))
        if dist_config.rank == 0:
            logger.info("Safetensors loaded")

    else:
        raise ValueError(f"Unknown checkpoint type: {ckpt_type}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Core: cross-entropy loss computation on raw windows
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_loss_on_windows(
    model: torch.nn.Module,
    windows: list[dict[str, torch.Tensor]],
    device: torch.device,
    micro_batch_size: int = 1,
    mask_degenerate_bases: bool = True,
) -> dict:
    """Compute cross-entropy loss over pre-tokenized windows.

    This is intentionally simple: no DataLoader, no sampler.  Just iterate
    over windows, forward pass, compute loss.

    The loss computation matches training:
      - Labels are shifted by 1 (causal LM: predict next token)
      - Non-ACGT bases are masked (labels set to -100)
      - Padding tokens are masked (labels set to -100)
      - Loss = CrossEntropyLoss with ignore_index=-100 (mean over valid tokens)

    We compute *two* aggregations:
      1. **HF-style** (avg of per-batch mean losses) — what ``model(**batch).loss`` returns
      2. **Megatron-style** (sum of per-token losses / total valid tokens) — true per-token avg

    Args:
        model: The loaded model in eval mode.
        windows: List of dicts with ``input_ids`` and ``attention_mask`` tensors.
        device: CUDA device.
        micro_batch_size: Number of windows per forward pass.
        mask_degenerate_bases: Whether to mask non-ACGT bases (should match training).

    Returns:
        Dictionary with loss statistics.
    """
    model.eval()

    dna_tokens_tensor = torch.tensor(sorted(DNA_TOKENS), device=device)

    total_hf_loss = 0.0  # Sum of per-batch mean losses
    total_token_loss = 0.0  # Sum of per-token losses (for Megatron-style)
    total_valid_tokens = 0
    total_windows = 0
    num_batches = 0
    per_window_losses: list[float] = []

    # Process in micro-batches
    for start in range(0, len(windows), micro_batch_size):
        batch_windows = windows[start : start + micro_batch_size]
        batch_size = len(batch_windows)

        # Stack into batch tensors
        input_ids = torch.stack([w["input_ids"] for w in batch_windows]).to(device)
        attention_mask = torch.stack([w["attention_mask"] for w in batch_windows]).to(device)

        # Create labels: shift input_ids left by 1 for causal LM
        # HF's DataCollatorForLanguageModeling sets labels = input_ids.clone()
        # and the model's loss_function handles the shift internally.
        # We replicate the same approach.
        labels = input_ids.clone()

        # Mask padding positions in labels
        labels[attention_mask == 0] = -100

        # Mask degenerate (non-ACGT) bases, matching GenomicDataCollator
        if mask_degenerate_bases:
            not_dna = ~torch.isin(labels, dna_tokens_tensor)
            labels[(not_dna) & (labels != -100)] = -100

        # Forward pass
        with transformer_engine.pytorch.fp8_autocast(enabled=False):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        if loss is not None:
            loss_val = loss.item()
            total_hf_loss += loss_val

            # Count valid tokens for Megatron-style averaging
            valid_tokens = (labels != -100).sum().item()
            total_valid_tokens += valid_tokens
            total_token_loss += loss_val * valid_tokens

            num_batches += 1
            total_windows += batch_size

            # Per-window loss (for detailed reporting)
            # When batch_size=1, loss_val is the per-window loss.
            # For larger batches, this is the batch-mean loss — still useful for tracking.
            per_window_losses.append(loss_val)

        if total_windows % 50 == 0 and total_windows > 0:
            running_avg = total_hf_loss / num_batches
            logger.info(
                "  [%d/%d windows]  running_avg_loss=%.4f  valid_tokens=%s",
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
        "perplexity": math.exp(min(hf_avg_loss, 100)),  # cap to avoid overflow
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
        description="Evaluate a checkpoint on a training parquet shard (no dataloader).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint-dir", type=str, required=True, help="Root checkpoint directory.")
    p.add_argument("--checkpoint-step", type=int, default=None, help="Specific step to load (latest if omitted).")
    p.add_argument(
        "--parquet-path",
        type=str,
        required=True,
        help="Path to parquet file(s). Supports glob patterns (e.g., '/data/opengenome2/parquet/*.parquet').",
    )
    p.add_argument("--text-column", type=str, default="text", help="Column name for sequences in the parquet files.")
    p.add_argument("--config-name-or-path", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--tokenizer", type=str, default="./tokenizers/nucleotide_fast_tokenizer")
    p.add_argument("--micro-batch-size", type=int, default=1, help="Windows per forward pass.")
    p.add_argument("--max-seq-length", type=int, default=8192, help="Window size (must match training).")
    p.add_argument("--stride", type=int, default=200, help="Window stride (must match training).")
    p.add_argument(
        "--num-kv-heads",
        type=int,
        default=8,
        help="Number of KV heads. 8 = GQA (default), 32 = MHA (older models).",
    )
    p.add_argument("--max-sequences", type=int, default=None, help="Max sequences to read from parquet (for speed).")
    p.add_argument("--max-windows", type=int, default=None, help="Max windows to evaluate (for speed).")
    p.add_argument(
        "--mask-degenerate-bases",
        action="store_true",
        default=True,
        help="Mask non-ACGT bases in loss (default: True — matches training).",
    )
    p.add_argument("--no-mask-degenerate-bases", action="store_false", dest="mask_degenerate_bases")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None, help="Path to write results JSON.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    args = parse_args()

    # ── distributed setup ─────────────────────────────────────────────────
    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)
    set_seed(args.seed)

    if dist_config.rank == 0:
        logger.info("=" * 70)
        logger.info("Training-Shard Loss Evaluation (no dataloader)")
        logger.info("=" * 70)
        logger.info("  checkpoint_dir  : %s", args.checkpoint_dir)
        logger.info("  checkpoint_step : %s", args.checkpoint_step)
        logger.info("  parquet_path    : %s", args.parquet_path)
        logger.info("  text_column     : %s", args.text_column)
        logger.info("  max_seq_length  : %d", args.max_seq_length)
        logger.info("  stride          : %d", args.stride)
        logger.info("  micro_batch_size: %d", args.micro_batch_size)
        logger.info("  mask_degenerate : %s", args.mask_degenerate_bases)
        logger.info("  max_sequences   : %s", args.max_sequences)
        logger.info("  max_windows     : %s", args.max_windows)
        logger.info("  num_kv_heads    : %d", args.num_kv_heads)
        logger.info("  seed            : %d", args.seed)
        logger.info("  world_size      : %d", dist_config.world_size)
        logger.info("=" * 70)

    # ── read parquet data ────────────────────────────────────────────────
    if dist_config.rank == 0:
        logger.info("Reading sequences from parquet …")
    sequences = read_sequences_from_parquet(
        args.parquet_path,
        text_column=args.text_column,
        max_sequences=args.max_sequences,
    )

    # ── tokenize with windowing ──────────────────────────────────────────
    if dist_config.rank == 0:
        logger.info("Tokenizing sequences with windowing (seq_len=%d, stride=%d) …", args.max_seq_length, args.stride)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    windows = tokenize_with_windowing(sequences, tokenizer, args.max_seq_length, args.stride)

    if args.max_windows is not None and len(windows) > args.max_windows:
        if dist_config.rank == 0:
            logger.info("Limiting to %d windows (out of %d)", args.max_windows, len(windows))
        windows = windows[: args.max_windows]

    if dist_config.rank == 0:
        logger.info("Total windows to evaluate: %d", len(windows))

    # ── checkpoint ────────────────────────────────────────────────────────
    ckpt_path, ckpt_type = find_checkpoint_path(args.checkpoint_dir, args.checkpoint_step)
    if dist_config.rank == 0:
        logger.info("Resolved checkpoint: %s  (type=%s)", ckpt_path, ckpt_type)

    # ── model ─────────────────────────────────────────────────────────────
    device_mesh = init_device_mesh("cuda", mesh_shape=(dist_config.world_size,), mesh_dim_names=("dp",))
    config = _build_model_config(args.config_name_or_path, num_kv_heads=args.num_kv_heads)
    model = load_model_from_checkpoint(ckpt_path, ckpt_type, config, dist_config, device_mesh)

    # ── evaluation ────────────────────────────────────────────────────────
    if dist_config.rank == 0:
        logger.info("Computing loss on %d windows …", len(windows))

    results = compute_loss_on_windows(
        model=model,
        windows=windows,
        device=device,
        micro_batch_size=args.micro_batch_size,
        mask_degenerate_bases=args.mask_degenerate_bases,
    )

    # ── report ────────────────────────────────────────────────────────────
    if dist_config.rank == 0:
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

        # ── save JSON ─────────────────────────────────────────────────────
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "checkpoint": str(ckpt_path),
                "checkpoint_type": ckpt_type,
                "checkpoint_step": args.checkpoint_step,
                "parquet_path": args.parquet_path,
                "text_column": args.text_column,
                "max_seq_length": args.max_seq_length,
                "stride": args.stride,
                "seed": args.seed,
                "mask_degenerate_bases": args.mask_degenerate_bases,
                "max_sequences": args.max_sequences,
                "max_windows": args.max_windows,
                "num_kv_heads": args.num_kv_heads,
                "world_size": dist_config.world_size,
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

    # ── cleanup ───────────────────────────────────────────────────────────
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
