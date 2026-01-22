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

"""Evaluate a checkpoint on validation data and compute LM loss.

This script loads an FSDP2 checkpoint and computes validation loss to compare
with John's Megatron runs. It matches John's validation setup:
- Uses the same validation data source
- Can limit to N batches (default 40, matching John's --limit-val-batches 40)

Usage:
    # For consolidated checkpoints (model.safetensors):
    python eval_checkpoint.py \
        --checkpoint-dir /path/to/consolidated/model \
        --config-name L2_og2_metagenome_7b \
        --val-data-path /path/to/validation.jsonl \
        --num-batches 40

    # For FSDP2 distributed checkpoints (.distcp files):
    torchrun --nproc_per_node=1 eval_checkpoint.py \
        --checkpoint-dir /path/to/step_5000 \
        --config-name L2_og2_metagenome_7b \
        --val-data-path /path/to/validation.jsonl \
        --num-batches 40
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer


# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomic_dataset import GenomicDataCollator
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_validation_data(data_path: str, tokenizer, max_seq_length: int, stride: int):
    """Load validation data from JSON/JSONL file."""
    import datasets

    logger.info(f"Loading validation data from {data_path}")

    # Determine if it's a compressed file
    if data_path.endswith(".gz"):
        ds = datasets.load_dataset("json", data_files=data_path, split="train")
    elif data_path.endswith(".jsonl") or data_path.endswith(".json"):
        ds = datasets.load_dataset("json", data_files=data_path, split="train")
    else:
        # Assume it's a HuggingFace dataset path
        ds = datasets.load_dataset(data_path, split="validation", streaming=False)

    logger.info(f"Loaded {len(ds)} samples")
    return ds


def create_eval_dataloader(
    dataset,
    tokenizer,
    micro_batch_size: int,
    max_seq_length: int,
    num_workers: int = 0,
    seed: int = 42,
):
    """Create a dataloader for evaluation."""
    from torch.utils.data import DataLoader

    # Tokenize function
    def tokenize_fn(examples):
        # Get the text field (could be "sequence" or "text")
        if "sequence" in examples:
            texts = examples["sequence"]
        elif "text" in examples:
            texts = examples["text"]
        else:
            raise ValueError(f"No text field found. Available: {list(examples.keys())}")

        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None,
        )

        # Create labels (same as input_ids for CLM)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Apply tokenization
    tokenized_ds = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Create collator with genomic masking
    from transformers import DataCollatorForLanguageModeling

    base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    collator = GenomicDataCollator(
        base_collator=base_collator,
        mask_degenerate_bases=True,
        uppercase_labels=False,
    )

    # Create dataloader
    dataloader = DataLoader(
        tokenized_ds,
        batch_size=micro_batch_size,
        shuffle=True,  # Random sampling like John's validation
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )

    return dataloader


@torch.no_grad()
def evaluate(model, dataloader, num_batches: int, device: torch.device, desc: str = "Evaluating"):
    """Run evaluation and return loss metrics.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for evaluation data.
        num_batches: Max number of batches to evaluate. Use -1 for all batches.
        device: Device to run on.
        desc: Description for progress bar.

    Returns:
        Dictionary with loss metrics.
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_evaluated = 0

    total_batches = len(dataloader) if num_batches == -1 else num_batches
    pbar = tqdm(total=total_batches, desc=desc)

    for batch in dataloader:
        if num_batches != -1 and num_evaluated >= num_batches:
            break

        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

        # Forward pass
        outputs = model(**batch)

        # Count valid tokens (where labels != -100)
        valid_tokens = (batch["labels"] != -100).sum().item()

        # Accumulate loss (weighted by token count)
        total_loss += outputs.loss.item() * valid_tokens
        total_tokens += valid_tokens
        num_evaluated += 1

        pbar.update(1)
        pbar.set_postfix({"loss": total_loss / total_tokens if total_tokens > 0 else 0})

    pbar.close()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "val_loss": avg_loss,
        "val_ppl": perplexity,
        "total_tokens": total_tokens,
        "num_batches": num_evaluated,
    }


class ModelOnlyAppState:
    """A minimal AppState that only loads model weights, skipping optimizer/scheduler.

    This matches the {"app": {...}} checkpoint structure but only restores model weights.
    """

    def __init__(self, model: torch.nn.Module):
        """Initialize with just a model."""
        self.model = model
        self.step = 0
        self.epoch = 0

    def state_dict(self):
        """Get state dict structure matching what was saved."""
        from torch.distributed.fsdp import FullyShardedDataParallel
        from torch.distributed.fsdp.api import StateDictType

        # Get model state dict from FSDP
        with FullyShardedDataParallel.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
            model_state_dict = self.model.state_dict()

        return {
            "model": model_state_dict,
            "optim": {},  # Placeholder
            "scheduler": {},  # Placeholder
            "step": self.step,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict: dict):
        """Load only the model weights from state dict."""
        from torch.distributed.fsdp import FullyShardedDataParallel
        from torch.distributed.fsdp.api import StateDictType

        # Load only model weights, skip optimizer/scheduler
        with FullyShardedDataParallel.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
            self.model.load_state_dict(state_dict["model"])

        self.step = state_dict.get("step", 0)
        self.epoch = state_dict.get("epoch", 0)


def load_fsdp2_checkpoint(model, checkpoint_path: Path, device: torch.device):
    """Load FSDP2 distributed checkpoint using proper distributed loading.

    This handles the {"app": AppState} checkpoint structure used by training,
    but only loads model weights (skips optimizer/scheduler).
    """
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

    logger.info("Detected FSDP2 distributed checkpoint format (.distcp files)")
    logger.info("Initializing distributed environment...")

    # Initialize distributed if not already done
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    logger.info(f"Rank {rank}/{world_size}, local_rank={local_rank}")

    # Create device mesh
    init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

    # Move model to GPU and wrap with FSDP (NO_SHARD for simple extraction)
    model = model.to(f"cuda:{local_rank}")
    model = FullyShardedDataParallel(model, sharding_strategy=ShardingStrategy.NO_SHARD, device_id=local_rank)

    # Build state dict matching checkpoint structure with model-only loader
    app_state = ModelOnlyAppState(model=model)
    state_dict = {"app": app_state}

    # Load distributed checkpoint
    logger.info(f"Loading distributed checkpoint from {checkpoint_path}...")
    dcp.load(
        state_dict=state_dict,
        checkpoint_id=str(checkpoint_path),
    )
    logger.info(f"Successfully loaded FSDP2 checkpoint from {checkpoint_path}")
    logger.info(f"Checkpoint was saved at step {app_state.step}, epoch {app_state.epoch}")

    # Return the unwrapped model
    return model.module if hasattr(model, "module") else model


def main():
    """Run evaluation on a checkpoint."""
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on validation data")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to checkpoint directory (step_XXXX)")
    parser.add_argument(
        "--config-name",
        type=str,
        default="L2_og2_metagenome_7b",
        help="Config name (used to get model architecture)",
    )
    parser.add_argument("--val-data-path", type=str, required=True, help="Path to validation JSON/JSONL file")
    parser.add_argument("--num-batches", type=int, default=40, help="Number of batches to evaluate (default: 40)")
    parser.add_argument("--micro-batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="Max sequence length (default: 8192)")
    parser.add_argument("--stride", type=int, default=7992, help="Stride for windowing (default: 7992)")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="./tokenizers/nucleotide_fast_tokenizer",
        help="Path to tokenizer",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--skip-full-eval", action="store_true", help="Skip full dataset evaluation")
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / "hydra_config" / f"{args.config_name}.yaml"
    if config_path.exists():
        config = OmegaConf.load(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = OmegaConf.create({})

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Create model config
    model_config_kwargs = OmegaConf.to_container(config.get("config_kwargs", {}), resolve=True)
    model_config_kwargs.setdefault("vocab_size", 256)
    model_config_kwargs.setdefault("attn_input_format", "bshd")

    logger.info(f"Creating model with config: {model_config_kwargs}")
    model_config = NVLlamaConfig.from_pretrained(
        config.get("config_name_or_path", "meta-llama/Llama-3.1-8B"),
        **model_config_kwargs,
    )

    # Create model
    model = NVLlamaForCausalLM(model_config)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint_dir)

    # Check for different checkpoint formats
    if (checkpoint_path / "model.safetensors").exists():
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path / "model.safetensors")
        model.load_state_dict(state_dict)
        logger.info(f"Loaded checkpoint from {checkpoint_path / 'model.safetensors'}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    elif (checkpoint_path / "pytorch_model.bin").exists():
        state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info(f"Loaded checkpoint from {checkpoint_path / 'pytorch_model.bin'}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    elif list(checkpoint_path.glob("*.distcp")):
        # FSDP2 distributed checkpoint format - requires torchrun
        model = load_fsdp2_checkpoint(model, checkpoint_path, torch.device("cuda"))
        device = next(model.parameters()).device

    else:
        # Try to find any .pt or .safetensors files
        pt_files = list(checkpoint_path.glob("*.pt")) + list(checkpoint_path.glob("*.safetensors"))
        if pt_files:
            logger.info(f"Found checkpoint files: {pt_files}")
            raise ValueError(f"Please specify exact checkpoint file. Found: {pt_files}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

    model.eval()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params:,} parameters ({num_params / 1e9:.2f}B)")
    logger.info(f"Using device: {device}")

    # Load validation data
    dataset = load_validation_data(args.val_data_path, tokenizer, args.max_seq_length, args.stride)

    # Create dataloader
    dataloader = create_eval_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        micro_batch_size=args.micro_batch_size,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )

    # Run evaluation on subsample (matching John's --limit-val-batches)
    logger.info(
        f"Running evaluation on {args.num_batches} batches ({args.num_batches * args.micro_batch_size} samples)"
    )
    subsample_metrics = evaluate(
        model, dataloader, args.num_batches, device, desc=f"Eval (subsample {args.num_batches} batches)"
    )

    # Run evaluation on FULL dataset (unless skipped)
    full_metrics = None
    if not args.skip_full_eval:
        logger.info("Running evaluation on FULL validation set...")
        # Recreate dataloader with same seed for reproducibility
        full_dataloader = create_eval_dataloader(
            dataset=dataset,
            tokenizer=tokenizer,
            micro_batch_size=args.micro_batch_size,
            max_seq_length=args.max_seq_length,
            seed=args.seed + 1,  # Different seed for full eval
        )
        full_metrics = evaluate(model, full_dataloader, -1, device, desc="Eval (full dataset)")

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Validation data: {args.val_data_path}")
    print(f"Total samples in dataset: {len(dataset):,}")
    print("=" * 70)

    print("\n--- SUBSAMPLE (for comparison with John's wandb curve) ---")
    print(f"Num batches: {subsample_metrics['num_batches']}")
    print(f"Num samples: {subsample_metrics['num_batches'] * args.micro_batch_size}")
    print(f"Total tokens: {subsample_metrics['total_tokens']:,}")
    print(f"Val Loss: {subsample_metrics['val_loss']:.6f}")
    print(f"Val PPL:  {subsample_metrics['val_ppl']:.2f}")

    if full_metrics:
        print("\n--- FULL DATASET ---")
        print(f"Num batches: {full_metrics['num_batches']}")
        print(f"Total tokens: {full_metrics['total_tokens']:,}")
        print(f"Val Loss: {full_metrics['val_loss']:.6f}")
        print(f"Val PPL:  {full_metrics['val_ppl']:.2f}")
    print("=" * 70)

    # Cleanup distributed if initialized
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
