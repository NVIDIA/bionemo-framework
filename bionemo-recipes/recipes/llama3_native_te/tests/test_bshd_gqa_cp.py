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

"""Test GQA + CP with BSHD format -- reproduces TE bug and verifies workaround.

Two bugs in TransformerEngine's GQA + context parallelism (TE 2.9-2.10):

  Bug 1 (BSHD): GQA + CP NaNs when fuse_qkv_params is not set (default=False).
      Workaround: set fuse_qkv_params=True (which NVLlamaForCausalLM does).

  Bug 2 (THD):  GQA + CP NaNs regardless of fuse_qkv_params. No workaround.
      See test_thd_gqa_cp_nan.py for that test.

MHA (num_kv_heads == num_attn_heads) works fine in all configurations.

Verified on:
    - TE 2.9.0+70f53666  / PyTorch 2.10.0a0+b558c986e8.nv25.11 (H100)
    - TE 2.10.0+769ed778  / PyTorch 2.10.0a0+b4e4ee81d3.nv25.12 (RTX 5090)

Usage:
    # Bug repro — bare GQA + CP NaNs (no fuse_qkv_params):
    torchrun --nproc_per_node=2 tests/test_bshd_gqa_cp.py --num-kv-heads 2

    # Workaround — passes with production params (fuse_qkv_params=True):
    torchrun --nproc_per_node=2 tests/test_bshd_gqa_cp.py --num-kv-heads 2 --match-production

    # MHA control (always passes):
    torchrun --nproc_per_node=2 tests/test_bshd_gqa_cp.py --num-kv-heads 6

Exit codes:
    0: All training steps produced finite loss.
    1: NaN loss detected.
"""

import argparse
import sys

import torch
import torch.distributed as dist
import transformer_engine
import transformer_engine.pytorch as te
from torch.distributed.device_mesh import init_device_mesh


VOCAB_SIZE = 256
SEQ_LENGTH = 256
CP_SIZE = 2
NUM_STEPS = 5


def run_forward_backward(
    num_attn_heads: int,
    num_kv_heads: int,
    hidden_size: int,
    device: torch.device,
    cp_group,
    cp_ranks,
    match_production: bool = False,
) -> bool:
    """Run forward/backward through a single TransformerLayer with BSHD + CP.

    Args:
        match_production: If True, use the same TransformerLayer params as
            NVLlamaForCausalLM (fuse_qkv_params=True, qkv_weight_interleaved=True).
            This is the workaround for the BSHD + GQA + CP NaN bug.

    Returns True if all steps produce finite loss, False if NaN is encountered.
    """
    rank = dist.get_rank()
    ffn_hidden_size = hidden_size * 4
    head_label = (
        f"GQA ({num_attn_heads}/{num_kv_heads})"
        if num_kv_heads != num_attn_heads
        else f"MHA ({num_attn_heads}/{num_kv_heads})"
    )
    if match_production:
        head_label += " [fuse_qkv=True]"

    extra_kwargs = {}
    if match_production:
        extra_kwargs = {
            "fuse_qkv_params": True,
            "qkv_weight_interleaved": True,
            "hidden_dropout": 0,
            "attention_dropout": 0,
            "layer_number": 1,
        }

    torch.manual_seed(42)
    layer = te.TransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attn_heads,
        num_gqa_groups=num_kv_heads,
        bias=False,
        normalization="RMSNorm",
        activation="swiglu",
        attn_input_format="bshd",
        self_attn_mask_type="causal",
        **extra_kwargs,
    ).to(device=device, dtype=torch.bfloat16)

    layer.set_context_parallel_group(cp_group, cp_ranks, torch.cuda.Stream())

    lm_head = torch.nn.Linear(hidden_size, VOCAB_SIZE, bias=False, device=device, dtype=torch.bfloat16)
    rope = te.attention.RotaryPositionEmbedding(dim=hidden_size // num_attn_heads)
    rope_emb = rope(max_seq_len=SEQ_LENGTH).to(device)
    optimizer = torch.optim.AdamW(list(layer.parameters()) + list(lm_head.parameters()), lr=1e-4)

    seq_per_rank = SEQ_LENGTH // CP_SIZE

    for step in range(NUM_STEPS):
        hidden = torch.randn(1, seq_per_rank, hidden_size, device=device, dtype=torch.bfloat16)
        labels = torch.randint(0, VOCAB_SIZE, (seq_per_rank,), device=device)

        optimizer.zero_grad()

        with te.autocast(enabled=False):
            output = layer(hidden, attention_mask=None, rotary_pos_emb=rope_emb)

        logits = lm_head(output.squeeze(0))
        loss = torch.nn.functional.cross_entropy(logits, labels)

        if rank == 0:
            print(f"  {head_label} step {step}: loss={loss.item():.4f}")

        if torch.isnan(loss).item():
            if rank == 0:
                print(f"  FAIL: {head_label} + BSHD + CP={CP_SIZE} produced NaN at step {step}")
            return False

        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0)
        optimizer.step()

    if rank == 0:
        print(f"  PASS: {head_label} + BSHD + CP={CP_SIZE} completed {NUM_STEPS} steps")
    return True


def main():
    """Run BSHD + CP test with configurable head counts and hidden size."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-attn-heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--num-kv-heads", type=int, required=True, help="Number of KV heads")
    parser.add_argument(
        "--hidden-size", type=int, default=384, help="Hidden size (must be divisible by num-attn-heads)"
    )
    parser.add_argument(
        "--match-production",
        action="store_true",
        help="Use fuse_qkv_params=True (workaround for BSHD + GQA + CP NaN bug)",
    )
    args = parser.parse_args()

    assert args.hidden_size % args.num_attn_heads == 0, (
        f"hidden_size ({args.hidden_size}) must be divisible by num_attn_heads ({args.num_attn_heads})"
    )
    assert args.num_attn_heads % args.num_kv_heads == 0, (
        f"num_attn_heads ({args.num_attn_heads}) must be divisible by num_kv_heads ({args.num_kv_heads})"
    )

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    device_mesh = init_device_mesh("cuda", mesh_shape=(1, CP_SIZE), mesh_dim_names=("dp", "cp"))
    cp_group = device_mesh["cp"].get_group()
    cp_ranks = dist.get_process_group_ranks(cp_group)

    if rank == 0:
        print(f"TE version: {transformer_engine.__version__}, PyTorch: {torch.__version__}")
        print(
            f"Testing BSHD + num_attn_heads={args.num_attn_heads}, num_kv_heads={args.num_kv_heads},"
            f" hidden_size={args.hidden_size}, head_dim={args.hidden_size // args.num_attn_heads}, CP={CP_SIZE}"
        )
        if args.match_production:
            print("  Using production params: fuse_qkv_params=True, qkv_weight_interleaved=True")

    ok = run_forward_backward(
        num_attn_heads=args.num_attn_heads,
        num_kv_heads=args.num_kv_heads,
        hidden_size=args.hidden_size,
        device=device,
        cp_group=cp_group,
        cp_ranks=cp_ranks,
        match_production=args.match_production,
    )

    dist.barrier()
    dist.destroy_process_group()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
