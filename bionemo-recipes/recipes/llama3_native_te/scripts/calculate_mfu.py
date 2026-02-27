#!/usr/bin/env python3

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

"""Calculate Model FLOPs Utilization (MFU) for a GQA transformer.

MFU = (FLOPs per step / step time) / (num_gpus * peak GPU FLOP/s)

FLOPs are computed per-layer accounting for GQA (fewer KV heads than query heads),
SwiGLU MLP, and the LM head.

Usage:
    # From wandb train/step_time metric:
    python scripts/calculate_mfu.py --step-time 10.5

    # Override model config:
    python scripts/calculate_mfu.py --step-time 10.5 --num-kv-heads 32  # MHA instead of GQA

    # Different cluster size:
    python scripts/calculate_mfu.py --step-time 10.5 --num-gpus 96 --num-nodes 12
"""

import argparse


def compute_flops_per_token(
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    seq_length: int,
    vocab_size: int,
) -> dict:
    """Compute forward-pass FLOPs per token for a GQA transformer.

    Per-layer forward FLOPs per token:
        Self-attention:
            Q proj:     2 * H * H
            K proj:     2 * H * d_kv       (d_kv = head_dim * num_kv_heads)
            V proj:     2 * H * d_kv
            QK^T:       2 * S * H           (all query heads, each attending over S keys)
            attn * V:   2 * S * H
            O proj:     2 * H * H
        MLP (SwiGLU):
            gate_proj:  2 * H * I
            up_proj:    2 * H * I
            down_proj:  2 * I * H

    LM head: 2 * H * V
    """
    head_dim = hidden_size // num_attention_heads
    d_kv = head_dim * num_kv_heads

    # Per layer, per token, forward pass
    attn_qo = 2 * 2 * hidden_size * hidden_size  # Q proj + O proj
    attn_kv = 2 * 2 * hidden_size * d_kv  # K proj + V proj
    attn_scores = 2 * 2 * seq_length * hidden_size  # QK^T + attn*V
    mlp = 3 * 2 * hidden_size * intermediate_size  # gate + up + down (SwiGLU)

    per_layer = attn_qo + attn_kv + attn_scores + mlp
    lm_head = 2 * hidden_size * vocab_size

    forward_per_token = num_layers * per_layer + lm_head
    # Forward + backward ≈ 3x forward (backward is ~2x forward for matmuls)
    total_per_token = 3 * forward_per_token

    return {
        "per_layer_forward": per_layer,
        "attn_qo": attn_qo,
        "attn_kv": attn_kv,
        "attn_scores": attn_scores,
        "mlp": mlp,
        "lm_head": lm_head,
        "forward_per_token": forward_per_token,
        "total_per_token": total_per_token,
        "d_kv": d_kv,
        "head_dim": head_dim,
    }


def main():
    """Calculate and print MFU for a GQA transformer given a measured step time."""
    parser = argparse.ArgumentParser(description="Calculate MFU for a GQA transformer")

    # Step time from wandb (train/step_time)
    parser.add_argument("--step-time", type=float, required=True, help="Seconds per step (from wandb train/step_time)")

    # Model architecture (defaults = OG2 7B GQA config)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8, help="KV heads for GQA (8=GQA, 32=MHA)")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seq-length", type=int, default=8192)

    # Training config
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-acc-steps", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=48, help="Total GPUs (num_nodes * gpus_per_node)")
    parser.add_argument("--num-nodes", type=int, default=6)
    parser.add_argument("--gpus-per-node", type=int, default=8)

    # Hardware (H100 SXM BF16 peak)
    parser.add_argument("--peak-tflops", type=float, default=989.4, help="Peak BF16 TFLOP/s per GPU (H100 SXM=989.4)")

    args = parser.parse_args()

    num_gpus = args.num_gpus if args.num_gpus != 48 else args.num_nodes * args.gpus_per_node

    # Compute FLOPs
    flops = compute_flops_per_token(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_attention_heads,
        num_kv_heads=args.num_kv_heads,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
    )

    # Global batch size
    gbs = args.micro_batch_size * num_gpus * args.grad_acc_steps
    tokens_per_step = gbs * args.seq_length
    flops_per_step = flops["total_per_token"] * tokens_per_step

    # Cluster peak
    peak_flops_per_gpu = args.peak_tflops * 1e12
    cluster_peak = num_gpus * peak_flops_per_gpu

    # MFU
    achieved_flops = flops_per_step / args.step_time
    mfu = achieved_flops / cluster_peak

    # Approximate parameter count (for reference)
    head_dim = args.hidden_size // args.num_attention_heads
    d_kv = head_dim * args.num_kv_heads
    params_per_layer = (
        args.hidden_size * args.hidden_size  # Q
        + args.hidden_size * d_kv  # K
        + args.hidden_size * d_kv  # V
        + args.hidden_size * args.hidden_size  # O
        + 3 * args.hidden_size * args.intermediate_size  # gate + up + down
        + 2 * args.hidden_size  # RMSNorm (input + post-attn)
    )
    total_params = (
        args.num_layers * params_per_layer
        + args.hidden_size * args.vocab_size  # embedding
        + args.hidden_size * args.vocab_size  # lm_head (untied)
        + args.hidden_size  # final RMSNorm
    )

    # Print results
    print("=" * 70)
    print("MFU CALCULATION")
    print("=" * 70)
    print("\nModel Architecture:")
    print(f"  Layers:              {args.num_layers}")
    print(f"  Hidden size:         {args.hidden_size}")
    print(f"  Intermediate size:   {args.intermediate_size}")
    print(f"  Attention heads:     {args.num_attention_heads}")
    print(f"  KV heads (GQA):      {args.num_kv_heads}")
    print(f"  Head dim:            {flops['head_dim']}")
    print(f"  d_kv (KV dim):       {flops['d_kv']}")
    print(f"  Vocab size:          {args.vocab_size}")
    print(f"  Seq length:          {args.seq_length}")
    print(f"  Approx params:       {total_params:,.0f} ({total_params / 1e9:.2f}B)")

    print("\nTraining Config:")
    print(f"  Micro batch size:    {args.micro_batch_size}")
    print(f"  Grad acc steps:      {args.grad_acc_steps}")
    print(f"  GPUs:                {num_gpus} ({args.num_nodes} nodes x {args.gpus_per_node} GPUs)")
    print(f"  Global batch size:   {gbs}")
    print(f"  Tokens per step:     {tokens_per_step:,}")

    print("\nFLOPs Breakdown (per token, forward only):")
    print(f"  Attn Q+O proj:       {flops['attn_qo']:,.0f}")
    print(f"  Attn K+V proj (GQA): {flops['attn_kv']:,.0f}")
    print(f"  Attn scores:         {flops['attn_scores']:,.0f}")
    print(f"  MLP (SwiGLU):        {flops['mlp']:,.0f}")
    print(f"  Per layer total:     {flops['per_layer_forward']:,.0f}")
    print(f"  LM head:             {flops['lm_head']:,.0f}")
    print(f"  Forward/token:       {flops['forward_per_token']:,.0f} ({flops['forward_per_token'] / 1e9:.2f} GFLOP)")
    print(f"  Fwd+Bwd/token (3x):  {flops['total_per_token']:,.0f} ({flops['total_per_token'] / 1e9:.2f} GFLOP)")

    print("\nMFU Result:")
    print(f"  Step time:           {args.step_time:.2f} s")
    print(f"  FLOPs per step:      {flops_per_step:.3e}")
    print(f"  Achieved FLOP/s:     {achieved_flops:.3e} ({achieved_flops / 1e12:.1f} TFLOP/s)")
    print(f"  Peak per GPU:        {peak_flops_per_gpu:.3e} ({args.peak_tflops} TFLOP/s)")
    print(f"  Cluster peak:        {cluster_peak:.3e} ({num_gpus * args.peak_tflops:.1f} TFLOP/s)")
    print("  ┌─────────────────────────────────┐")
    print(f"  │  MFU = {mfu * 100:.1f}%                     │")
    print("  └─────────────────────────────────┘")
    print("=" * 70)


if __name__ == "__main__":
    main()
