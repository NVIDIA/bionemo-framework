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

"""Test that BSHD + GQA + CP works correctly (no NaN).

Unlike THD + GQA + CP (which produces NaN due to a TE bug), the BSHD format
with GQA and context parallelism works correctly. This test verifies that.

Usage:
    torchrun --nproc_per_node=2 tests/test_bshd_gqa_cp.py --num-kv-heads 2

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


# Tiny model config for fast reproduction
HIDDEN_SIZE = 384
FFN_HIDDEN_SIZE = 1536
VOCAB_SIZE = 256
SEQ_LENGTH = 256  # Must be divisible by 2 * cp_size = 4
NUM_ATTN_HEADS = 6
CP_SIZE = 2
NUM_STEPS = 5


def run_forward_backward(num_kv_heads: int, device: torch.device, cp_group, cp_ranks) -> bool:
    """Run forward/backward through a single TransformerLayer with BSHD + CP.

    Returns True if all steps produce finite loss, False if NaN is encountered.
    """
    rank = dist.get_rank()
    head_label = (
        f"GQA ({NUM_ATTN_HEADS}/{num_kv_heads})"
        if num_kv_heads != NUM_ATTN_HEADS
        else f"MHA ({NUM_ATTN_HEADS}/{num_kv_heads})"
    )

    torch.manual_seed(42)
    layer = te.TransformerLayer(
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        num_attention_heads=NUM_ATTN_HEADS,
        num_gqa_groups=num_kv_heads,
        bias=False,
        normalization="RMSNorm",
        activation="swiglu",
        attn_input_format="bshd",
        self_attn_mask_type="causal",
    ).to(device=device, dtype=torch.bfloat16)

    layer.set_context_parallel_group(cp_group, cp_ranks, torch.cuda.Stream())

    lm_head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False, device=device, dtype=torch.bfloat16)
    rope = te.attention.RotaryPositionEmbedding(dim=HIDDEN_SIZE // NUM_ATTN_HEADS)
    rope_emb = rope(max_seq_len=SEQ_LENGTH).to(device)
    optimizer = torch.optim.AdamW(list(layer.parameters()) + list(lm_head.parameters()), lr=1e-4)

    # BSHD format: each CP rank gets seq_length / cp_size tokens
    seq_per_rank = SEQ_LENGTH // CP_SIZE

    for step in range(NUM_STEPS):
        hidden = torch.randn(1, seq_per_rank, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
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
    """Run BSHD + CP test with configurable num_kv_heads.

    Exits 0 if all steps produce finite loss, 1 if NaN is detected.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-kv-heads", type=int, required=True, help="Number of KV heads (6=MHA, 2=GQA)")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    device_mesh = init_device_mesh("cuda", mesh_shape=(1, CP_SIZE), mesh_dim_names=("dp", "cp"))
    cp_group = device_mesh["cp"].get_group()
    cp_ranks = dist.get_process_group_ranks(cp_group)

    if rank == 0:
        print(f"TE version: {transformer_engine.__version__}, PyTorch: {torch.__version__}")
        print(f"Testing BSHD + num_kv_heads={args.num_kv_heads}, num_attn_heads={NUM_ATTN_HEADS}, CP={CP_SIZE}")

    ok = run_forward_backward(num_kv_heads=args.num_kv_heads, device=device, cp_group=cp_group, cp_ranks=cp_ranks)

    dist.barrier()
    dist.destroy_process_group()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
