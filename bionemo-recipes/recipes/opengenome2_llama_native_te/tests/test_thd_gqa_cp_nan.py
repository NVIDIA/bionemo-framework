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

This test demonstrates that a single TE TransformerLayer with:
- THD format (packed sequences)
- GQA (num_attention_heads != num_key_value_heads)
- Context Parallelism (cp_size=2)

produces NaN outputs, while MHA (num_attention_heads == num_key_value_heads) works fine.

Run directly:
    torchrun --nproc_per_node=2 tests/test_thd_gqa_cp_nan.py

Or via pytest (test_train_two_gpu.py wraps this script).

Environment:
    - Container: nvcr.io/nvidia/pytorch:25.11-py3
    - TransformerEngine: 2.9.0+70f53666
    - PyTorch: 2.10.0a0+b558c986e8.nv25.11

Expected behavior:
    MHA (6/6 heads): loss is finite (PASS)
    GQA (6/2 heads): loss is NaN  (BUG in TE)
"""

import sys

import torch
import torch.distributed as dist
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


def run_forward_backward(num_kv_heads: int, device: torch.device, cp_group, cp_ranks) -> bool:
    """Run forward/backward through a single TransformerLayer with CP.

    Returns True if all steps produce finite loss, False if NaN is encountered.
    """
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

        if torch.isnan(loss).item():
            return False

        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0)
        optimizer.step()

    return True


def main():
    """Run THD+GQA+CP NaN reproduction test.

    Exit codes:
        0: Bug reproduced (MHA works, GQA NaNs) -- expected result
        1: Unexpected result (both work or both fail)
    """
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    device_mesh = init_device_mesh("cuda", mesh_shape=(1, CP_SIZE), mesh_dim_names=("dp", "cp"))
    cp_group = device_mesh["cp"].get_group()
    cp_ranks = dist.get_process_group_ranks(cp_group)

    if rank == 0:
        print(f"TE version: {te.__version__}, PyTorch: {torch.__version__}")

    # Test 1: MHA (6/6) -- should work
    mha_ok = run_forward_backward(num_kv_heads=NUM_ATTN_HEADS, device=device, cp_group=cp_group, cp_ranks=cp_ranks)
    dist.barrier()

    # Test 2: GQA (6/2) -- should NaN due to TE bug
    gqa_ok = run_forward_backward(num_kv_heads=2, device=device, cp_group=cp_group, cp_ranks=cp_ranks)
    dist.barrier()

    if rank == 0:
        print(f"MHA (6/6) + THD + CP=2: {'PASS' if mha_ok else 'FAIL (NaN)'}")
        print(f"GQA (6/2) + THD + CP=2: {'PASS' if gqa_ok else 'FAIL (NaN)'}")

        if mha_ok and not gqa_ok:
            print("BUG CONFIRMED: THD + GQA + CP produces NaN in TransformerEngine")
        elif mha_ok and gqa_ok:
            print("BUG FIXED: GQA no longer produces NaN (TE may have been updated)")
        else:
            print(f"UNEXPECTED: MHA={'ok' if mha_ok else 'NaN'}, GQA={'ok' if gqa_ok else 'NaN'}")

    dist.destroy_process_group()

    # Exit 0 if bug is confirmed (expected), 1 otherwise
    if mha_ok and not gqa_ok:
        sys.exit(0)
    elif mha_ok and gqa_ok:
        # Bug is fixed -- also exit 0, this is the happy path
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
