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

r"""Evaluate a checkpoint on a globally shuffled test set for model comparison.

Computes per-token cross-entropy loss on held-out test data from the OG2 metagenomics
dataset.  Use this to compare models trained with different data strategies (e.g.,
HuggingFace window-shuffle vs Eden SQLite) at the same training step on an identically
shuffled held-out set.

Both models see the exact same globally-shuffled data in the same order (same seed,
same sharding, same file) so the losses are directly comparable regardless of how
their *training* data was ordered.

**Default mode (recommended): Eden / ShardedEdenDataset**

Uses the ShardedEdenDataset with ``DistributedSampler(shuffle=True)`` which produces
a *true* global permutation over all windows — every window has equal probability of
appearing at any position.  This is the gold-standard for "globally shuffled".

**Fallback mode: ``--use-hf-data``**

Loads a jsonl.gz test file via HuggingFace streaming with a large shuffle buffer.
This is only *pseudo*-global: items far apart in the file may never get mixed.
Use this only when the Eden SQLite test database is not available.

Supports checkpoint formats:
  - FSDP2 DCP checkpoints  (from train_fsdp2.py — the default)
  - DDP  checkpoints        (checkpoint.pt files from train_ddp.py)
  - Consolidated safetensors (final_model/ directories)

Run from the recipe directory with ``torchrun``::

    cd bionemo-recipes/recipes/llama3_native_te

    # ── Model 1: HF window-shuffle (evaluated on Eden test set) ────────
    torchrun --nproc_per_node=8 evaluate_test_loss.py \\
        --checkpoint-dir /data/savithas/checkpoints/og2-7b-bf16-fp32master-hf-ws \\
        --sequence-db-dir /data/bcr_eden/OG2_database_splits/ \\
        --test-window-db /data/bcr_eden/OG2_database_splits/og2__test__short.sqlite \\
        --val-window-db  /data/bcr_eden/OG2_database_splits/og2__validation__short.sqlite \\
        --num-eval-batches 500 \\
        --micro-batch-size 8 \\
        --seed 42 \\
        --output /data/savithas/eval_results/hf_ws_test_loss.json

    # ── Model 2: Eden dataset (evaluated on the same Eden test set) ────
    torchrun --nproc_per_node=8 evaluate_test_loss.py \\
        --checkpoint-dir /data/savithas/checkpoints/og2-7b-bf16-sanity \\
        --sequence-db-dir /data/bcr_eden/OG2_database_splits/ \\
        --test-window-db /data/bcr_eden/OG2_database_splits/og2__test__short.sqlite \\
        --val-window-db  /data/bcr_eden/OG2_database_splits/og2__validation__short.sqlite \\
        --num-eval-batches 500 \\
        --micro-batch-size 8 \\
        --seed 42 \\
        --output /data/savithas/eval_results/eden_test_loss.json

    # ── Compare results ────────────────────────────────────────────────
    python -c "
    import json
    hf   = json.load(open('/data/savithas/eval_results/hf_ws_test_loss.json'))
    eden = json.load(open('/data/savithas/eval_results/eden_test_loss.json'))
    print()
    print('=== Test-set loss comparison (globally shuffled) ===')
    for tag, r in [('HF Window-Shuffle', hf), ('Eden Dataset', eden)]:
        t = r['test_results']
        print(f'  {tag:20s}:  loss={t[\"avg_loss\"]:.4f}  ppl={t[\"perplexity\"]:.2f}  '
              f'(megatron: {t[\"megatron_loss\"]:.4f})  tokens={t[\"total_tokens\"]:,}')
    diff = abs(hf['test_results']['avg_loss'] - eden['test_results']['avg_loss'])
    print(f'  Loss difference:      {diff:.4f}')
    "
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.distributed as dist
import transformer_engine.pytorch
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from checkpoint import AppState
from distributed_config import DistributedConfig
from genomic_dataset import GenomicDataCollator
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM
from scheduler import get_cosine_annealing_schedule_with_warmup
from sharded_eden_dataset import ShardedEdenDataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a globally shuffled test / validation set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- checkpoint ---------------------------------------------------------
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help=(
            "Root checkpoint directory.  The script auto-detects the format: "
            "FSDP2 DCP (train_fsdp2/ subdirectory), DDP (checkpoint.pt), "
            "or safetensors (final_model/)."
        ),
    )
    p.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Specific training step to load.  If omitted the latest step is used.",
    )

    # -- Eden data (default — true global shuffling) ------------------------
    eden = p.add_argument_group(
        "Eden data (default)",
        "ShardedEdenDataset from SQLite — true global shuffling via DistributedSampler.",
    )
    eden.add_argument(
        "--sequence-db-dir",
        type=str,
        default="/data/bcr_eden/OG2_database_splits/",
        help="Directory containing per-sample SQLite sequence databases.",
    )
    eden.add_argument(
        "--test-window-db",
        type=str,
        default="/data/bcr_eden/OG2_database_splits/og2__test__short.sqlite",
        help="Path to the pre-computed test-split window database.",
    )
    eden.add_argument(
        "--val-window-db",
        type=str,
        default=None,
        help="Optional: path to a validation-split window database.",
    )
    eden.add_argument(
        "--eden-stride",
        type=int,
        default=7992,
        help="Stride used when the window database was pre-computed (default: 7992).",
    )

    # -- HF streaming fallback ----------------------------------------------
    hf = p.add_argument_group(
        "HF streaming fallback",
        "Use --use-hf-data to switch to HuggingFace jsonl.gz streaming (only pseudo-global shuffling via buffer).",
    )
    hf.add_argument(
        "--use-hf-data",
        action="store_true",
        default=False,
        help="Use HF streaming instead of Eden SQLite for the test set.",
    )
    hf.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to the HF test data file (jsonl.gz).  Required when --use-hf-data.",
    )
    hf.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Optional HF validation data file (jsonl.gz).",
    )
    hf.add_argument(
        "--hf-stride",
        type=int,
        default=200,
        help="Stride for HF tokenizer windowing (default: 200).",
    )
    hf.add_argument(
        "--buffer-size",
        type=int,
        default=100_000,
        help="HF shuffle-buffer size for pseudo-global shuffling.",
    )
    hf.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name for sequences in the jsonl data.",
    )

    # -- model --------------------------------------------------------------
    p.add_argument(
        "--config-name-or-path",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Base HF config name used during model creation (architecture template).",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="./tokenizers/nucleotide_fast_tokenizer",
        help="Path or name of the nucleotide tokenizer.",
    )

    # -- eval hyper-parameters ----------------------------------------------
    p.add_argument("--num-eval-batches", type=int, default=500, help="Number of micro-batches to evaluate.")
    p.add_argument("--micro-batch-size", type=int, default=8, help="Micro-batch size per GPU.")
    p.add_argument("--max-seq-length", type=int, default=8192, help="Max sequence / window length.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (identical across models = fair comparison).")

    # -- masking ------------------------------------------------------------
    p.add_argument(
        "--mask-degenerate-bases",
        action="store_true",
        default=True,
        help="Mask non-ACGT bases in labels (default: True — matches training).",
    )
    p.add_argument(
        "--no-mask-degenerate-bases",
        action="store_false",
        dest="mask_degenerate_bases",
    )

    # -- output -------------------------------------------------------------
    p.add_argument("--output", type=str, default=None, help="Path to write results as JSON.")

    args = p.parse_args()

    # Validation
    if args.use_hf_data and args.test_data is None:
        p.error("--test-data is required when using --use-hf-data")

    return args


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def find_checkpoint_path(checkpoint_dir: str, step: int | None = None) -> tuple[Path, str]:
    """Locate the actual checkpoint inside *checkpoint_dir*.

    Returns:
        ``(path, checkpoint_type)`` where *checkpoint_type* is one of
        ``"dcp"``, ``"ddp"``, or ``"safetensors"``.
    """
    root = Path(checkpoint_dir)

    # 1. final_model/ with safetensors
    for candidate in [root, root / "final_model", root / "train_fsdp2" / "final_model"]:
        if (candidate / "model.safetensors").exists():
            return candidate, "safetensors"

    # 2. train_fsdp2/ with DCP step directories
    fsdp2_dir = root / "train_fsdp2" if (root / "train_fsdp2").exists() else root

    step_dirs = sorted(
        [d for d in fsdp2_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    if step_dirs:
        if step is not None:
            target = fsdp2_dir / f"step_{step}"
            if not target.exists():
                avail = [d.name for d in step_dirs]
                raise FileNotFoundError(f"step_{step} not found.  Available: {avail}")
            chosen = target
        else:
            chosen = step_dirs[-1]  # latest

        if (chosen / ".metadata").exists() or any(chosen.glob("*.distcp")):
            return chosen, "dcp"
        if (chosen / "checkpoint.pt").exists():
            return chosen, "ddp"
        return chosen, "dcp"

    # 3. The root itself might be a step directory
    if (root / "checkpoint.pt").exists():
        return root, "ddp"
    if (root / ".metadata").exists() or any(root.glob("*.distcp")):
        return root, "dcp"

    raise FileNotFoundError(
        f"No recognisable checkpoint in {checkpoint_dir}.  "
        "Expected train_fsdp2/step_N/ (DCP), step_N/checkpoint.pt (DDP), "
        "or final_model/model.safetensors."
    )


# ---------------------------------------------------------------------------
# Dataloader creation
# ---------------------------------------------------------------------------


def create_eden_eval_dataloader(
    dist_config: DistributedConfig,
    sequence_db_dir: str,
    window_db_path: str,
    tokenizer_name_or_path: str,
    micro_batch_size: int = 8,
    seq_length: int = 8192,
    stride: int = 7992,
    seed: int = 42,
    mask_degenerate_bases: bool = True,
) -> DataLoader:
    """Create a *truly* globally-shuffled eval dataloader from a ShardedEdenDataset.

    Uses ``DistributedSampler(shuffle=True)`` which produces a full random
    permutation over all pre-computed windows — the gold-standard for
    "globally shuffled" evaluation.

    Returns:
        A ``DataLoader`` ready for the evaluation loop.
    """
    dataset = ShardedEdenDataset(
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_db_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        seq_length=seq_length,
        stride=stride,
        rc_aug=False,  # No augmentation during evaluation
        pad_in_getitem=True,  # BSHD: pad to seq_length
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist_config.world_size,
        rank=dist_config.rank,
        shuffle=True,  # TRUE global permutation
        seed=seed,
    )

    # Collator with genomic masking (same as training)
    base_collator = DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer, mlm=False)
    if mask_degenerate_bases:
        data_collator = GenomicDataCollator(
            base_collator=base_collator,
            uppercase_labels=False,
            mask_degenerate_bases=True,
        )
    else:
        data_collator = base_collator

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=micro_batch_size,
        collate_fn=data_collator,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    if dist_config.rank == 0:
        logger.info(
            "Eden eval dataloader: %d windows, mbs=%d, world=%d  (true global shuffle, seed=%d)",
            len(dataset),
            micro_batch_size,
            dist_config.world_size,
            seed,
        )

    return dataloader


def create_hf_eval_dataloader(
    dist_config: DistributedConfig,
    data_path: str,
    tokenizer_name_or_path: str,
    micro_batch_size: int = 8,
    max_seq_length: int = 8192,
    stride: int = 200,
    seed: int = 42,
    text_column: str = "text",
    mask_degenerate_bases: bool = True,
    buffer_size: int = 100_000,
) -> DataLoader:
    """Create a pseudo-globally-shuffled eval dataloader from a jsonl.gz file.

    Uses HuggingFace streaming with a large shuffle buffer.  This is *not*
    true global shuffling — use :func:`create_eden_eval_dataloader` when the
    Eden SQLite databases are available.

    Returns:
        A ``DataLoader`` ready for the evaluation loop.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading HF eval data from %s (pseudo-global shuffle, buffer=%d)", data_path, buffer_size)
    dataset = datasets.load_dataset("json", data_files=data_path, split="train", streaming=True)

    if dist_config.world_size > 1:
        dataset = dataset.shard(num_shards=dist_config.world_size, index=dist_config.rank)

    dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)

    def tokenize_with_windowing(examples):
        return tokenizer(
            examples[text_column],
            max_length=max_seq_length,
            stride=stride,
            truncation=True,
            return_overflowing_tokens=True,
            add_special_tokens=True,
        )

    tokenized = dataset.select_columns(text_column).map(
        tokenize_with_windowing,
        batched=True,
        batch_size=micro_batch_size * 4,
        remove_columns=[text_column],
    )

    base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if mask_degenerate_bases:
        data_collator = GenomicDataCollator(
            base_collator=base_collator,
            uppercase_labels=False,
            mask_degenerate_bases=True,
        )
    else:
        data_collator = base_collator

    dataloader = DataLoader(
        tokenized,
        batch_size=micro_batch_size,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
    )
    return dataloader


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_batches: int,
    device: torch.device,
    dist_config: DistributedConfig,
) -> dict:
    """Compute loss metrics over *num_batches* from *dataloader*.

    Returns a dictionary with both HF-style (mean-of-batch-means) and
    Megatron-style (true per-token average) loss and perplexity.
    """
    model.eval()

    total_loss = 0.0
    total_weighted_loss = 0.0
    total_tokens = 0
    num_evaluated = 0

    eval_iter = iter(dataloader)

    for batch_idx in range(num_batches):
        try:
            batch = next(eval_iter)
        except StopIteration:
            if dist_config.rank == 0:
                logger.info("Data exhausted after %d batches (requested %d)", batch_idx, num_batches)
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward in BF16 (no FP8 during eval for maximum comparability)
        with transformer_engine.pytorch.fp8_autocast(enabled=False):
            outputs = model(**batch)

        loss = outputs.loss
        if loss is not None:
            loss_val = loss.item()
            total_loss += loss_val

            labels = batch.get("labels")
            num_tokens = (labels != -100).sum().item() if labels is not None else batch["input_ids"].numel()
            total_tokens += num_tokens
            total_weighted_loss += loss_val * num_tokens
            num_evaluated += 1

            if batch_idx % 50 == 0 and dist_config.rank == 0:
                running_avg = total_loss / num_evaluated
                logger.info(
                    "  [%d/%d]  batch_loss=%.4f  running_avg=%.4f  tokens=%s",
                    batch_idx,
                    num_batches,
                    loss_val,
                    running_avg,
                    f"{total_tokens:,}",
                )

    # ── aggregate across ranks ────────────────────────────────────────────
    dist.barrier()
    stats = torch.tensor([total_loss, float(total_tokens), float(num_evaluated), total_weighted_loss], device=device)
    dist.all_reduce(stats)

    g_loss = stats[0].item()
    g_tokens = int(stats[1].item())
    g_batches = int(stats[2].item())
    g_weighted = stats[3].item()

    avg_loss = g_loss / max(g_batches, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    megatron_loss = g_weighted / max(g_tokens, 1)
    megatron_ppl = torch.exp(torch.tensor(megatron_loss)).item()

    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "megatron_loss": megatron_loss,
        "megatron_ppl": megatron_ppl,
        "total_tokens": g_tokens,
        "total_batches": g_batches,
    }


# ---------------------------------------------------------------------------
# Model + checkpoint loading
# ---------------------------------------------------------------------------


def _build_model_config(config_name_or_path: str) -> NVLlamaConfig:
    """Build the 7B GQA NVLlamaConfig that matches both training configs."""
    return NVLlamaConfig.from_pretrained(
        config_name_or_path,
        dtype=torch.float32,  # FP32 master weights
        vocab_size=256,
        num_hidden_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA
        max_position_embeddings=8192,
        initializer_range=0.02,
        attn_input_format="thd",  # Matches training; model auto-converts BSHD inputs
        self_attn_mask_type="padding_causal",
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
    """Create, shard, and load a model from the discovered checkpoint.

    Args:
        ckpt_path: Path returned by :func:`find_checkpoint_path`.
        ckpt_type: One of ``"dcp"``, ``"ddp"``, ``"safetensors"``.
        config: Model configuration.
        dist_config: Distributed training config.
        device_mesh: FSDP2 device mesh.

    Returns:
        The loaded model (already in eval mode).
    """
    # ── create model ──────────────────────────────────────────────────────
    model = NVLlamaForCausalLM(config)
    if dist_config.rank == 0:
        logger.info("Model created (%s parameters)", f"{sum(p.numel() for p in model.parameters()):,}")

    # ── FSDP2 sharding (needed for DCP; also fine for other formats) ──────
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=False,
    )
    for layer in model.model.layers:
        fully_shard(layer, mesh=device_mesh["dp"], mp_policy=mp_policy)
    fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)

    # ── load weights ──────────────────────────────────────────────────────
    if ckpt_type == "dcp":
        if dist_config.rank == 0:
            logger.info("Loading FSDP2 DCP checkpoint from %s …", ckpt_path)

        from torch.optim import AdamW

        optimizer = AdamW(model.parameters(), lr=1e-5)
        scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, num_warmup_steps=100, num_decay_steps=1000)
        app_state = AppState(model=model, optimizer=optimizer, scheduler=scheduler)
        state_dict = {"app": app_state}

        dcp_load(state_dict, checkpoint_id=ckpt_path, process_group=device_mesh.get_group("dp"))

        if dist_config.rank == 0:
            logger.info("DCP checkpoint loaded  (step=%d, epoch=%d)", app_state.step, app_state.epoch)

    elif ckpt_type == "ddp":
        if dist_config.rank == 0:
            logger.info("Loading DDP checkpoint from %s …", ckpt_path)

        ckpt = torch.load(ckpt_path / "checkpoint.pt", map_location="cpu", weights_only=True)

        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        set_model_state_dict(
            model,
            model_state_dict=ckpt["model"],
            options=StateDictOptions(strict=False),
        )

        if dist_config.rank == 0:
            logger.info("DDP checkpoint loaded  (step=%d)", ckpt.get("step", -1))

    elif ckpt_type == "safetensors":
        if dist_config.rank == 0:
            logger.info("Loading safetensors from %s …", ckpt_path)

        from safetensors.torch import load_file
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        weights = load_file(str(ckpt_path / "model.safetensors"))
        set_model_state_dict(
            model,
            model_state_dict=weights,
            options=StateDictOptions(strict=False),
        )

        if dist_config.rank == 0:
            logger.info("Safetensors loaded")

    else:
        raise ValueError(f"Unknown checkpoint type: {ckpt_type}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _log_results(tag: str, results: dict) -> None:
    """Pretty-print a results dictionary on rank 0."""
    logger.info("=" * 70)
    logger.info("%s RESULTS", tag)
    logger.info("=" * 70)
    logger.info("  HF-style loss      : %.4f", results["avg_loss"])
    logger.info("  HF-style PPL       : %.2f", results["perplexity"])
    logger.info("  Megatron-style loss : %.4f", results["megatron_loss"])
    logger.info("  Megatron-style PPL  : %.2f", results["megatron_ppl"])
    logger.info("  Total tokens       : %s", f"{results['total_tokens']:,}")
    logger.info("  Total batches      : %d", results["total_batches"])
    logger.info("=" * 70)


def main() -> None:
    """Entry point."""
    args = parse_args()

    # ── distributed setup ─────────────────────────────────────────────────
    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)
    set_seed(args.seed)

    data_mode = "HF streaming (pseudo-global shuffle)" if args.use_hf_data else "Eden SQLite (true global shuffle)"

    if dist_config.rank == 0:
        logger.info("=" * 70)
        logger.info("Evaluate Checkpoint on Globally Shuffled Test Data")
        logger.info("=" * 70)
        logger.info("  checkpoint_dir  : %s", args.checkpoint_dir)
        logger.info("  data_mode       : %s", data_mode)
        if args.use_hf_data:
            logger.info("  test_data       : %s", args.test_data)
        else:
            logger.info("  sequence_db_dir : %s", args.sequence_db_dir)
            logger.info("  test_window_db  : %s", args.test_window_db)
            if args.val_window_db:
                logger.info("  val_window_db   : %s", args.val_window_db)
        logger.info("  num_eval_batches: %d", args.num_eval_batches)
        logger.info("  micro_batch_size: %d", args.micro_batch_size)
        logger.info("  max_seq_length  : %d", args.max_seq_length)
        logger.info("  seed            : %d", args.seed)
        logger.info("  world_size      : %d", dist_config.world_size)
        logger.info("=" * 70)

    # ── find checkpoint ───────────────────────────────────────────────────
    ckpt_path, ckpt_type = find_checkpoint_path(args.checkpoint_dir, args.checkpoint_step)
    if dist_config.rank == 0:
        logger.info("Resolved checkpoint: %s  (type=%s)", ckpt_path, ckpt_type)

    # ── model ─────────────────────────────────────────────────────────────
    device_mesh = init_device_mesh("cuda", mesh_shape=(dist_config.world_size,), mesh_dim_names=("dp",))
    config = _build_model_config(args.config_name_or_path)
    model = load_model_from_checkpoint(ckpt_path, ckpt_type, config, dist_config, device_mesh)

    # ── test evaluation ───────────────────────────────────────────────────
    if dist_config.rank == 0:
        logger.info("Creating %s TEST dataloader …", "HF" if args.use_hf_data else "Eden")

    if args.use_hf_data:
        test_dl = create_hf_eval_dataloader(
            dist_config=dist_config,
            data_path=args.test_data,
            tokenizer_name_or_path=args.tokenizer,
            micro_batch_size=args.micro_batch_size,
            max_seq_length=args.max_seq_length,
            stride=args.hf_stride,
            seed=args.seed,
            text_column=args.text_column,
            mask_degenerate_bases=args.mask_degenerate_bases,
            buffer_size=args.buffer_size,
        )
    else:
        test_dl = create_eden_eval_dataloader(
            dist_config=dist_config,
            sequence_db_dir=args.sequence_db_dir,
            window_db_path=args.test_window_db,
            tokenizer_name_or_path=args.tokenizer,
            micro_batch_size=args.micro_batch_size,
            seq_length=args.max_seq_length,
            stride=args.eden_stride,
            seed=args.seed,
            mask_degenerate_bases=args.mask_degenerate_bases,
        )

    if dist_config.rank == 0:
        logger.info("Evaluating on TEST data (%d batches) …", args.num_eval_batches)
    test_results = evaluate_model(model, test_dl, args.num_eval_batches, device, dist_config)

    if dist_config.rank == 0:
        _log_results("TEST SET", test_results)

    # ── optional validation evaluation ────────────────────────────────────
    val_results = None
    has_val = (args.use_hf_data and args.val_data) or (not args.use_hf_data and args.val_window_db)

    if has_val:
        if dist_config.rank == 0:
            logger.info("Creating %s VALIDATION dataloader …", "HF" if args.use_hf_data else "Eden")

        if args.use_hf_data:
            val_dl = create_hf_eval_dataloader(
                dist_config=dist_config,
                data_path=args.val_data,
                tokenizer_name_or_path=args.tokenizer,
                micro_batch_size=args.micro_batch_size,
                max_seq_length=args.max_seq_length,
                stride=args.hf_stride,
                seed=args.seed,
                text_column=args.text_column,
                mask_degenerate_bases=args.mask_degenerate_bases,
                buffer_size=args.buffer_size,
            )
        else:
            val_dl = create_eden_eval_dataloader(
                dist_config=dist_config,
                sequence_db_dir=args.sequence_db_dir,
                window_db_path=args.val_window_db,
                tokenizer_name_or_path=args.tokenizer,
                micro_batch_size=args.micro_batch_size,
                seq_length=args.max_seq_length,
                stride=args.eden_stride,
                seed=args.seed,
                mask_degenerate_bases=args.mask_degenerate_bases,
            )

        if dist_config.rank == 0:
            logger.info("Evaluating on VALIDATION data (%d batches) …", args.num_eval_batches)
        val_results = evaluate_model(model, val_dl, args.num_eval_batches, device, dist_config)

        if dist_config.rank == 0:
            _log_results("VALIDATION SET", val_results)

    # ── save results ──────────────────────────────────────────────────────
    if args.output and dist_config.rank == 0:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)

        payload: dict = {
            "checkpoint": str(ckpt_path),
            "checkpoint_type": ckpt_type,
            "data_mode": data_mode,
            "seed": args.seed,
            "num_eval_batches": args.num_eval_batches,
            "micro_batch_size": args.micro_batch_size,
            "max_seq_length": args.max_seq_length,
            "world_size": dist_config.world_size,
            "mask_degenerate_bases": args.mask_degenerate_bases,
            "test_results": test_results,
        }
        if args.use_hf_data:
            payload["test_data"] = args.test_data
            payload["hf_stride"] = args.hf_stride
            payload["buffer_size"] = args.buffer_size
        else:
            payload["sequence_db_dir"] = args.sequence_db_dir
            payload["test_window_db"] = args.test_window_db
            payload["eden_stride"] = args.eden_stride

        if val_results is not None:
            payload["val_results"] = val_results
            if args.use_hf_data:
                payload["val_data"] = args.val_data
            else:
                payload["val_window_db"] = args.val_window_db

        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Results saved → %s", out)

    # ── cleanup ───────────────────────────────────────────────────────────
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
