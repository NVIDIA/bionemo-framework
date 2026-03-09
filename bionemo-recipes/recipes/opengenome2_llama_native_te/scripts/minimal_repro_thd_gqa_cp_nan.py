# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Minimal reproduction of THD + GQA + CP NaN bug in TransformerEngine.

This script demonstrates that a single TransformerLayer with:
- THD format (packed sequences)
- GQA (num_attention_heads != num_key_value_heads)
- Context Parallelism (cp_size=2)

produces NaN outputs, while MHA (num_attention_heads == num_key_value_heads) works fine.

Usage:
    torchrun --nproc_per_node=2 scripts/minimal_repro_thd_gqa_cp_nan.py

Environment:
    - Container: nvcr.io/nvidia/pytorch:25.11-py3
    - TransformerEngine: 2.9.0+70f53666
    - PyTorch: 2.10.0a0+b558c986e8.nv25.11

Expected behavior:
    MHA (6/6 heads): loss is finite
    GQA (6/2 heads): loss is NaN  <-- BUG
"""

import argparse

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from torch.distributed.device_mesh import init_device_mesh


def create_mock_thd_batch(
    num_sequences: int,
    seq_length: int,
    hidden_size: int,
    vocab_size: int,
    cp_size: int,
    device: torch.device,
) -> dict:
    """Create synthetic THD-format batch for CP training.

    IMPORTANT: For THD + CP, TE expects:
    - hidden_states: (total_tokens_per_rank, hidden_size) — already split across CP ranks
    - cu_seqlens: FULL sequence cu_seqlens (NOT divided by cp_size) — TE divides internally
    - max_seqlen: FULL max sequence length (NOT divided) — TE divides internally

    This matches our real training pipeline where the collator splits tokens per CP rank
    but passes the original cu_seqlens_padded unchanged.
    """
    # Per-rank token count: each rank gets seq_length // cp_size tokens per sequence
    tokens_per_rank = seq_length // cp_size
    total_tokens_per_rank = num_sequences * tokens_per_rank

    # Hidden states for this rank's share of tokens
    hidden_states = torch.randn(total_tokens_per_rank, hidden_size, device=device, dtype=torch.bfloat16)

    # cu_seqlens for FULL sequences (TE internally divides by cp_size)
    # e.g., 4 sequences of 256 each: [0, 256, 512, 768, 1024]
    cu_seqlens_full = torch.arange(0, num_sequences * seq_length + 1, seq_length, device=device, dtype=torch.int32)

    # Labels for loss computation (per-rank tokens)
    labels = torch.randint(0, vocab_size, (total_tokens_per_rank,), device=device)

    return {
        "hidden_states": hidden_states,
        "cu_seqlens": cu_seqlens_full,
        "max_seqlen": seq_length,  # Full sequence length, not per-rank
        "labels": labels,
    }


def run_test(num_kv_heads: int, num_attn_heads: int = 6, num_steps: int = 20):
    """Run forward/backward through a single TransformerLayer with CP."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Tiny config (similar to L0_sanity) for fast reproduction
    cp_size = 2
    hidden_size = 384
    ffn_hidden_size = 1536
    vocab_size = 256
    seq_length = 256  # Must be divisible by 2 * cp_size = 4
    num_sequences = 4  # Number of packed sequences per batch

    # Create device mesh for CP
    device_mesh = init_device_mesh("cuda", mesh_shape=(1, cp_size), mesh_dim_names=("dp", "cp"))
    cp_group = device_mesh["cp"].get_group()
    cp_ranks = dist.get_process_group_ranks(cp_group)

    # Create a single TransformerLayer
    torch.manual_seed(42)
    layer = te.TransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attn_heads,
        num_gqa_groups=num_kv_heads,
        bias=False,
        normalization="RMSNorm",
        activation="swiglu",
        attn_input_format="thd",
        self_attn_mask_type="padding_causal",
    ).to(device=device, dtype=torch.bfloat16)

    # Attach CP group
    layer.set_context_parallel_group(cp_group, cp_ranks, torch.cuda.Stream())

    # Simple linear head for loss
    lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=torch.bfloat16)

    # RoPE embeddings
    rope = te.attention.RotaryPositionEmbedding(dim=hidden_size // num_attn_heads)
    rope_emb = rope(max_seq_len=seq_length).to(device)

    optimizer = torch.optim.AdamW(list(layer.parameters()) + list(lm_head.parameters()), lr=1e-4)

    head_label = (
        f"GQA ({num_attn_heads}/{num_kv_heads})"
        if num_kv_heads != num_attn_heads
        else f"MHA ({num_attn_heads}/{num_kv_heads})"
    )
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Testing: {head_label} + THD + CP={cp_size}")
        print(f"  hidden_size={hidden_size}, seq_length={seq_length}, num_sequences={num_sequences}")
        print(
            f"  tokens_per_rank={seq_length // cp_size}, total_tokens_per_rank={num_sequences * seq_length // cp_size}"
        )
        print(f"{'=' * 60}")

    for step in range(num_steps):
        batch = create_mock_thd_batch(
            num_sequences=num_sequences,
            seq_length=seq_length,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            cp_size=cp_size,
            device=device,
        )

        optimizer.zero_grad()

        with te.autocast(enabled=False):
            output = layer(
                batch["hidden_states"],
                attention_mask=None,
                rotary_pos_emb=rope_emb,
                cu_seqlens_q=batch["cu_seqlens"],
                cu_seqlens_kv=batch["cu_seqlens"],
                cu_seqlens_q_padded=batch["cu_seqlens"],
                cu_seqlens_kv_padded=batch["cu_seqlens"],
                max_seqlen_q=batch["max_seqlen"],
                max_seqlen_kv=batch["max_seqlen"],
                pad_between_seqs=True,
            )

        # Compute loss
        logits = lm_head(output)
        loss = torch.nn.functional.cross_entropy(logits, batch["labels"])

        is_nan = torch.isnan(loss).item()
        if rank == 0:
            print(f"  Step {step:3d}: loss = {loss.item():.4f}{' *** NaN! ***' if is_nan else ''}")

        if is_nan:
            if rank == 0:
                print(f"  RESULT: {head_label} + THD + CP={cp_size} -> NaN at step {step}")
            return False

        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0)
        optimizer.step()

    if rank == 0:
        print(f"  RESULT: {head_label} + THD + CP={cp_size} -> OK ({num_steps} steps)")
    return True


def main():
    """Run THD+GQA+CP NaN reproduction tests."""
    parser = argparse.ArgumentParser(description="Minimal repro for THD+GQA+CP NaN in TransformerEngine")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps per test")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    if rank == 0:
        import transformer_engine

        print("=" * 60)
        print("Minimal repro: THD + GQA + CP NaN bug in TransformerEngine")
        print(f"TE version: {transformer_engine.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"World size: {dist.get_world_size()}")
        print("=" * 60)

    # Test 1: MHA (should work)
    mha_ok = run_test(num_kv_heads=6, num_attn_heads=6, num_steps=args.steps)

    dist.barrier()

    # Test 2: GQA (should NaN)
    gqa_ok = run_test(num_kv_heads=2, num_attn_heads=6, num_steps=args.steps)

    dist.barrier()

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("SUMMARY:")
        print(f"  MHA (6/6) + THD + CP=2: {'PASS' if mha_ok else 'FAIL (NaN)'}")
        print(f"  GQA (6/2) + THD + CP=2: {'PASS' if gqa_ok else 'FAIL (NaN)'}")
        print(f"{'=' * 60}")
        if mha_ok and not gqa_ok:
            print("\nCONCLUSION: Bug confirmed in TE's THD + GQA + CP path.")
            print("The NaN occurs only when num_kv_heads != num_attention_heads")
            print("with packed sequences (THD) and context parallelism enabled.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
