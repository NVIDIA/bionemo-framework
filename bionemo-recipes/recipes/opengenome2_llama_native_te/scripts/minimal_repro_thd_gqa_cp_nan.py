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
- GQA (num_attention_heads=32, num_key_value_heads=8)
- Context Parallelism (cp_size=2)

produces NaN outputs, while MHA (num_key_value_heads=32) works fine.

Usage:
    torchrun --nproc_per_node=2 scripts/minimal_repro_thd_gqa_cp_nan.py

Environment:
    - Container: nvcr.io/nvidia/pytorch:25.11-py3
    - TransformerEngine: 2.9.0+70f53666
    - PyTorch: 2.10.0a0+b558c986e8.nv25.11

Expected behavior:
    MHA (32/32 heads): loss is finite
    GQA (32/8 heads):  loss is NaN  <-- BUG
"""

import argparse

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from torch.distributed.device_mesh import init_device_mesh
from transformer_engine.common.recipe import DelayedScaling, Format


def create_mock_thd_batch(
    num_sequences: int,
    seq_length: int,
    hidden_size: int,
    vocab_size: int,
    cp_rank: int,
    cp_size: int,
    device: torch.device,
) -> dict:
    """Create synthetic THD-format batch for a single CP rank.

    Each CP rank gets seq_length // cp_size tokens per sequence (dual-chunk zigzag).
    cu_seqlens_padded tracks the padded boundaries.
    """
    tokens_per_rank = seq_length // cp_size

    # Total tokens across all sequences for this rank
    total_tokens = num_sequences * tokens_per_rank

    # Create random hidden states (what embedding layer would produce)
    hidden_states = torch.randn(total_tokens, hidden_size, device=device, dtype=torch.bfloat16)

    # Create cu_seqlens for packed sequences (padded to equal length)
    cu_seqlens = torch.arange(0, total_tokens + 1, tokens_per_rank, device=device, dtype=torch.int32)

    # Create random labels for loss computation
    labels = torch.randint(0, vocab_size, (total_tokens,), device=device)

    return {
        "hidden_states": hidden_states,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": tokens_per_rank,
        "labels": labels,
    }


def run_test(num_kv_heads: int, num_attn_heads: int = 32, num_steps: int = 20):
    """Run forward/backward through a single TransformerLayer with CP."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    cp_size = 2
    hidden_size = 4096
    ffn_hidden_size = 14336
    vocab_size = 256
    seq_length = 8192  # Total sequence length before CP split
    num_sequences = 4  # Number of packed sequences per batch

    # Create device mesh for CP
    device_mesh = init_device_mesh("cuda", mesh_shape=(1, cp_size), mesh_dim_names=("dp", "cp"))
    cp_group = device_mesh["cp"].get_group()
    cp_ranks = dist.get_process_group_ranks(cp_group)
    cp_rank = device_mesh["cp"].get_local_rank()

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

    # FP8 recipe (disabled, but needed for autocast)
    fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID)

    optimizer = torch.optim.AdamW(list(layer.parameters()) + list(lm_head.parameters()), lr=1e-4)

    head_label = (
        f"GQA ({num_attn_heads}/{num_kv_heads})"
        if num_kv_heads != num_attn_heads
        else f"MHA ({num_attn_heads}/{num_kv_heads})"
    )
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Testing: {head_label} + THD + CP={cp_size}")
        print(f"{'=' * 60}")

    for step in range(num_steps):
        batch = create_mock_thd_batch(
            num_sequences=num_sequences,
            seq_length=seq_length,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            cp_rank=cp_rank,
            cp_size=cp_size,
            device=device,
        )

        optimizer.zero_grad()

        with te.autocast(enabled=False, recipe=fp8_recipe):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    if rank == 0:
        print("=" * 60)
        print("Minimal repro: THD + GQA + CP NaN bug in TransformerEngine")
        import transformer_engine

        print(f"TE version: {transformer_engine.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"World size: {dist.get_world_size()}")
        print("=" * 60)

    # Test 1: MHA (should work)
    mha_ok = run_test(num_kv_heads=32, num_attn_heads=32, num_steps=args.steps)

    dist.barrier()

    # Test 2: GQA (should NaN)
    gqa_ok = run_test(num_kv_heads=8, num_attn_heads=32, num_steps=args.steps)

    dist.barrier()

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("SUMMARY:")
        print(f"  MHA (32/32) + THD + CP=2: {'PASS' if mha_ok else 'FAIL (NaN)'}")
        print(f"  GQA (32/8)  + THD + CP=2: {'PASS' if gqa_ok else 'FAIL (NaN)'}")
        print(f"{'=' * 60}")
        if mha_ok and not gqa_ok:
            print("\nCONCLUSION: Bug confirmed in TE's THD + GQA + CP path.")
            print("The NaN occurs only when num_kv_heads != num_attention_heads")
            print("with packed sequences (THD) and context parallelism enabled.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
