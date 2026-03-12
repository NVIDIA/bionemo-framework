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

Bug 2 of two GQA + context parallelism bugs in TE 2.9-2.10:

  Bug 1 (BSHD): GQA + CP NaNs when fuse_qkv_params is not set (default=False).
      Workaround: set fuse_qkv_params=True (which NVLlamaForCausalLM does).
      See test_bshd_gqa_cp.py for that test.

  Bug 2 (THD):  GQA + CP NaNs regardless of fuse_qkv_params. No workaround.
      This is the test for Bug 2.

MHA (num_kv_heads == num_attn_heads) works fine in all configurations.

Verified on:
    - TE 2.9.0+70f53666  / PyTorch 2.10.0a0+b558c986e8.nv25.11 (H100)
    - TE 2.10.0+769ed778  / PyTorch 2.10.0a0+b4e4ee81d3.nv25.12 (RTX 5090)

Usage:
    # MHA control (should pass):
    torchrun --nproc_per_node=2 tests/test_thd_gqa_cp_nan.py --num-kv-heads 6

    # GQA bug repro (NaN -> exit 1):
    torchrun --nproc_per_node=2 tests/test_thd_gqa_cp_nan.py --num-kv-heads 2

    # GQA with production params (still NaN -> exit 1, no workaround for THD):
    torchrun --nproc_per_node=2 tests/test_thd_gqa_cp_nan.py --num-kv-heads 2 --match-production

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
NUM_SEQUENCES = 4
NUM_ATTN_HEADS = 6
CP_SIZE = 2
NUM_STEPS = 5


def create_mock_thd_batch(device: torch.device) -> dict:
    """Create synthetic THD-format batch for CP training.

    For THD + CP, TE expects:
    - hidden_states: (total_tokens_per_rank, hidden_size) -- already split across CP ranks
    - cu_seqlens: FULL sequence cu_seqlens (NOT divided by cp_size) -- TE divides internally
    - max_seqlen: FULL max sequence length (NOT divided) -- TE divides internally
    """
    tokens_per_rank = SEQ_LENGTH // CP_SIZE
    total_tokens_per_rank = NUM_SEQUENCES * tokens_per_rank

    hidden_states = torch.randn(total_tokens_per_rank, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    cu_seqlens = torch.arange(0, NUM_SEQUENCES * SEQ_LENGTH + 1, SEQ_LENGTH, device=device, dtype=torch.int32)
    labels = torch.randint(0, VOCAB_SIZE, (total_tokens_per_rank,), device=device)

    return {
        "hidden_states": hidden_states,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": SEQ_LENGTH,
        "labels": labels,
    }


def run_forward_backward(
    num_kv_heads: int, device: torch.device, cp_group, cp_ranks, match_production: bool = False
) -> bool:
    """Run forward/backward through a single TransformerLayer with CP.

    Returns True if all steps produce finite loss, False if NaN is encountered.
    """
    rank = dist.get_rank()
    head_label = (
        f"GQA ({NUM_ATTN_HEADS}/{num_kv_heads})"
        if num_kv_heads != NUM_ATTN_HEADS
        else f"MHA ({NUM_ATTN_HEADS}/{num_kv_heads})"
    )
    if match_production:
        head_label += " [production params]"

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
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        num_attention_heads=NUM_ATTN_HEADS,
        num_gqa_groups=num_kv_heads,
        bias=False,
        normalization="RMSNorm",
        activation="swiglu",
        attn_input_format="thd",
        self_attn_mask_type="padding_causal",
        **extra_kwargs,
    ).to(device=device, dtype=torch.bfloat16)

    layer.set_context_parallel_group(cp_group, cp_ranks, torch.cuda.Stream())

    lm_head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False, device=device, dtype=torch.bfloat16)
    rope = te.attention.RotaryPositionEmbedding(dim=HIDDEN_SIZE // NUM_ATTN_HEADS)
    rope_emb = rope(max_seq_len=SEQ_LENGTH).to(device)
    optimizer = torch.optim.AdamW(list(layer.parameters()) + list(lm_head.parameters()), lr=1e-4)

    for step in range(NUM_STEPS):
        batch = create_mock_thd_batch(device)
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

        logits = lm_head(output)
        loss = torch.nn.functional.cross_entropy(logits, batch["labels"])

        if rank == 0:
            print(f"  {head_label} step {step}: loss={loss.item():.4f}")

        if torch.isnan(loss).item():
            if rank == 0:
                print(f"  FAIL: {head_label} + THD + CP={CP_SIZE} produced NaN at step {step}")
            return False

        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0)
        optimizer.step()

    if rank == 0:
        print(f"  PASS: {head_label} + THD + CP={CP_SIZE} completed {NUM_STEPS} steps")
    return True


def main():
    """Run THD + CP test with configurable num_kv_heads.

    Exits 0 if all steps produce finite loss, 1 if NaN is detected.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-kv-heads", type=int, required=True, help="Number of KV heads (6=MHA, 2=GQA)")
    parser.add_argument(
        "--match-production",
        action="store_true",
        help="Use same TransformerLayer params as NVLlamaForCausalLM (fuse_qkv_params, qkv_weight_interleaved, etc.)",
    )
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
        print(f"Testing THD + num_kv_heads={args.num_kv_heads}, num_attn_heads={NUM_ATTN_HEADS}, CP={CP_SIZE}")
        if args.match_production:
            print("  Using production params: fuse_qkv_params=True, qkv_weight_interleaved=True")

    ok = run_forward_backward(
        num_kv_heads=args.num_kv_heads,
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
