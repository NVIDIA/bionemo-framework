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

"""Golden value tests for context parallelism correctness with FSDP2.

Validates that FSDP2 + CP produces equivalent results to non-CP execution.
Uses the same FSDP2 + CP setup as compare_mfu_multigpu.py and train_fsdp2_cp.py.

Strategy:
1. Init distributed, create FSDP2 model with CP
2. Gather full weights to rank 0 for non-CP baseline
3. Rank 0 runs non-CP baseline with identical weights
4. All ranks run CP forward+backward
5. Compare loss, logits (cosine sim), gradients (cosine sim)

Tests both TE (set_context_parallel_group) and HF (PyTorch native context_parallel).

Usage:
    cd bionemo-recipes/recipes/llama3_native_te
    torchrun --nproc_per_node=2 compare_mfu_validate.py
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import context_parallel_unshard
from torch.distributed.tensor.experimental._context_parallel._load_balancer import _HeadTailLoadBalancer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from collator import _split_batch_by_cp_rank
from compare_mfu_common import create_te_model_on_gpu
from modeling_llama_te import NVLlamaConfig


SEED = 42
LOSS_ATOL = 0.5
LOSS_RTOL = 0.25
LOGITS_COSINE_MIN = 0.99
GRAD_COSINE_MIN = 0.8


def seed_everything(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dummy_data(vocab_size, batch_size=2, seq_length=64):
    """Create deterministic dummy data for golden value tests."""
    seed_everything(SEED + 1000)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def reconstruct_logits_from_cp(logits_list, full_seq_len, cp_world_size):
    """Reconstruct full-sequence logits from TE CP-sharded chunks (zigzag pattern)."""
    batch_size, _, vocab_size = logits_list[0].shape
    total_chunks = 2 * cp_world_size
    chunk_size = full_seq_len // total_chunks
    reconstructed = torch.zeros(
        (batch_size, full_seq_len, vocab_size), dtype=logits_list[0].dtype, device=logits_list[0].device
    )
    for batch_idx in range(batch_size):
        for cp_idx, logits_shard in enumerate(logits_list):
            chunk_indices = [cp_idx, total_chunks - cp_idx - 1]
            for chunk_pos, chunk_idx in enumerate(chunk_indices):
                start_idx = chunk_idx * chunk_size
                end_idx = start_idx + chunk_size
                shard_start = chunk_pos * chunk_size
                shard_end = shard_start + chunk_size
                reconstructed[batch_idx, start_idx:end_idx, :] = logits_shard[batch_idx, shard_start:shard_end, :]
    return reconstructed


def capture_gradients(model, layer_accessor):
    """Capture gradients from sample layers for comparison."""
    gradients = {}
    for i, layer in enumerate(layer_accessor(model)):
        for name, param in layer.named_parameters():
            if param.grad is not None:
                gradients[f"layer_{i}.{name}"] = param.grad.detach().clone().cpu()
    return gradients


def compare_results(name, ref_loss, ref_logits, ref_grads, cp_loss, cp_logits, cp_grads, rank):
    """Compare CP results against non-distributed reference on rank 0."""
    if rank != 0:
        return True
    all_passed = True

    try:
        torch.testing.assert_close(cp_loss.cpu(), ref_loss.cpu(), atol=LOSS_ATOL, rtol=LOSS_RTOL)
        print(f"  [{name}] Loss: PASS (ref={ref_loss.item():.6f}, cp={cp_loss.item():.6f})")
    except AssertionError as e:
        print(f"  [{name}] Loss: FAIL - {e}")
        all_passed = False

    if ref_logits is not None and cp_logits is not None:
        assert cp_logits.shape == ref_logits.shape, f"Shape mismatch: {cp_logits.shape} vs {ref_logits.shape}"
        cosine_sim = torch.nn.functional.cosine_similarity(
            cp_logits.flatten().float().cuda(), ref_logits.flatten().float().cuda(), dim=0
        )
        passed = cosine_sim > LOGITS_COSINE_MIN
        print(
            f"  [{name}] Logits cosine sim: {'PASS' if passed else 'FAIL'} ({cosine_sim:.6f}, min={LOGITS_COSINE_MIN})"
        )
        if not passed:
            all_passed = False

    if ref_grads and cp_grads:
        for key in ref_grads:
            if key in cp_grads:
                cosine_sim = torch.nn.functional.cosine_similarity(
                    cp_grads[key].flatten().float(), ref_grads[key].flatten().float(), dim=0
                )
                passed = cosine_sim > GRAD_COSINE_MIN
                print(f"  [{name}] Grad {key}: {'PASS' if passed else 'FAIL'} (cosine={cosine_sim:.4f})")
                if not passed:
                    all_passed = False
    return all_passed


def validate_te(te_config, vocab_size, b, s, device, local_rank, device_mesh, cp_group, cp_rank, cp_size, rank):
    """Validate TE model: DDP + CP vs non-CP with identical weights."""
    if rank == 0:
        print(f"Test 1: TE model (DDP + CP={cp_size} vs non-CP baseline)")

    # DDP process group for gradient synchronization (matches test_cp_bshd.py)
    group_dp_cp = device_mesh[("dp", "cp")]._flatten("dp_cp").get_group()

    # Create CP model on all ranks with identical weights
    seed_everything(SEED)
    te_model = create_te_model_on_gpu(te_config)
    for param in te_model.parameters():
        dist.broadcast(param.data, src=0)

    # DDP + CP (matches test_cp_bshd.py pattern)
    te_model = torch.nn.parallel.DistributedDataParallel(
        te_model,
        device_ids=[local_rank],
        output_device=local_rank,
        process_group=group_dp_cp,
    )
    for layer in te_model.module.model.layers:
        layer.set_context_parallel_group(cp_group, dist.get_process_group_ranks(cp_group), torch.cuda.Stream())
    te_model.train()

    # --- Non-CP baseline on rank 0 (same weights) ---
    ref_loss = ref_logits = None
    ref_grads = {}
    if rank == 0:
        seed_everything(SEED)
        ref_model = create_te_model_on_gpu(te_config)
        # Copy weights from the CP model to ensure exact match
        ref_model.load_state_dict(
            {k: v for k, v in te_model.state_dict().items() if not k.endswith("_extra_state")}, strict=False
        )
        ref_model.train()
        batch = get_dummy_data(vocab_size, b, s)
        batch_cuda = {k: v.to(device) for k, v in batch.items()}
        ref_out = ref_model(**batch_cuda)
        ref_out.loss.backward()
        ref_loss = ref_out.loss.detach().clone().cpu()
        ref_logits = ref_out.logits.detach().clone().cpu()
        ref_grads = capture_gradients(
            ref_model,
            lambda m: [
                m.model.layers[0].self_attention.core_attention,
                m.model.layers[0].self_attention.layernorm_qkv,
            ],
        )
        print(f"  Baseline loss: {ref_loss.item():.6f}")
        del ref_model, ref_out, batch_cuda
        gc.collect()
        torch.cuda.empty_cache()
    dist.barrier()

    # --- CP forward+backward ---
    batch = get_dummy_data(vocab_size, b, s)
    batch_cuda = {k: v.detach().to(device) for k, v in batch.items()}
    batch_shard = dict(
        zip(
            ["input_ids", "labels"],
            _split_batch_by_cp_rank(
                None,
                batch_cuda["input_ids"],
                batch_cuda["labels"],
                qvk_format="bshd",
                cp_rank=cp_rank,
                cp_world_size=cp_size,
            ),
        )
    )
    batch_shard["max_length_q"] = batch_shard["max_length_k"] = s

    dist.barrier()
    te_out = te_model(**batch_shard)

    # All-gather losses
    losses = [torch.zeros_like(te_out.loss) for _ in range(cp_size)]
    dist.all_gather(losses, te_out.loss, group=cp_group)
    cp_loss = torch.mean(torch.stack(losses)).cpu() if rank == 0 else None

    # All-gather + reconstruct logits
    logits_list = [torch.zeros_like(te_out.logits.contiguous()) for _ in range(cp_size)]
    dist.all_gather(logits_list, te_out.logits.contiguous(), group=cp_group)
    cp_logits = reconstruct_logits_from_cp(logits_list, s, cp_size).cpu() if rank == 0 else None

    te_out.loss.backward()  # DDP all-reduces gradients automatically
    cp_grads = capture_gradients(
        te_model.module,
        lambda m: [m.model.layers[0].self_attention.core_attention, m.model.layers[0].self_attention.layernorm_qkv],
    )

    passed = compare_results("TE CP", ref_loss, ref_logits, ref_grads, cp_loss, cp_logits, cp_grads, rank)
    del te_model, te_out
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return passed


def validate_hf(hf_config, vocab_size, b, s, device, local_rank, device_mesh, cp_group, cp_rank, cp_size, rank):
    """Validate HF model: DDP + PyTorch native CP vs non-CP with identical weights."""
    if rank == 0:
        print(f"\nTest 2: HF model (DDP + PyTorch native CP={cp_size} vs non-CP baseline)")

    group_dp_cp = device_mesh[("dp", "cp")]._flatten("dp_cp").get_group()

    seed_everything(SEED + 100)
    hf_model = LlamaForCausalLM(hf_config).to(dtype=torch.bfloat16, device=device)
    for param in hf_model.parameters():
        dist.broadcast(param.data, src=0)

    # DDP for gradient synchronization
    hf_model = torch.nn.parallel.DistributedDataParallel(
        hf_model,
        device_ids=[local_rank],
        output_device=local_rank,
        process_group=group_dp_cp,
    )
    hf_model.train()

    # --- Non-CP baseline on rank 0 (same weights, already on GPU) ---
    ref_loss = ref_logits = None
    ref_grads = {}
    if rank == 0:
        ref_model = LlamaForCausalLM(hf_config).to(dtype=torch.bfloat16, device=device)
        ref_model.load_state_dict(hf_model.module.state_dict())
        ref_model.train()
        batch = get_dummy_data(vocab_size, b, s)
        batch_cuda = {k: v.to(device) for k, v in batch.items()}
        pos_ids = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
        ref_out = ref_model(position_ids=pos_ids, **batch_cuda)
        ref_out.loss.backward()
        ref_loss = ref_out.loss.detach().clone().cpu()
        ref_logits = ref_out.logits.detach().clone().cpu()
        ref_grads = capture_gradients(ref_model, lambda m: [m.model.layers[0].self_attn, m.model.layers[0].mlp])
        print(f"  Baseline loss: {ref_loss.item():.6f}")
        del ref_model, ref_out, batch_cuda
        gc.collect()
        torch.cuda.empty_cache()
    dist.barrier()

    # --- CP forward+backward ---
    batch = get_dummy_data(vocab_size, b, s)
    hf_full_ids = batch["input_ids"].to(device)
    hf_full_labels = batch["labels"].to(device)
    hf_full_pos = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)

    cp_mesh = device_mesh["cp"]
    with context_parallel(cp_mesh, buffers=(hf_full_ids, hf_full_labels, hf_full_pos), buffer_seq_dims=(1, 1, 1)):
        hf_out = hf_model(input_ids=hf_full_ids, labels=hf_full_labels, position_ids=hf_full_pos)
        cp_loss_local = hf_out.loss.detach().clone()
        cp_logits_local = hf_out.logits.detach().clone()
        hf_out.loss.backward()  # DDP all-reduces gradients automatically
        cp_grads = capture_gradients(hf_model.module, lambda m: [m.model.layers[0].self_attn, m.model.layers[0].mlp])

    # All-gather losses
    losses = [torch.zeros_like(cp_loss_local) for _ in range(cp_size)]
    dist.all_gather(losses, cp_loss_local, group=cp_group)
    cp_loss = torch.mean(torch.stack(losses)).cpu() if rank == 0 else None

    # Reconstruct logits with load balancer
    load_balancer = _HeadTailLoadBalancer(seq_length=s, world_size=cp_size, device=device)
    (cp_logits_full,) = context_parallel_unshard(cp_mesh, [cp_logits_local], [1], load_balancer=load_balancer)
    cp_logits = cp_logits_full.cpu() if rank == 0 else None

    passed = compare_results("HF CP", ref_loss, ref_logits, ref_grads, cp_loss, cp_logits, cp_grads, rank)
    del hf_model, hf_out
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return passed


def main():
    """Run golden value tests for CP correctness with FSDP2."""
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    parser = argparse.ArgumentParser(description="Golden value tests for CP with FSDP2")
    parser.add_argument("--config-path", default="./model_configs/lingua-1B", help="Model config directory")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    args = parser.parse_args()

    config_dict = json.loads(Path(args.config_path, "config.json").read_text())
    vocab_size = config_dict["vocab_size"]
    b, s = args.batch_size, args.seq_len

    cp_size = world_size
    dp_size = 1
    device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, cp_size), mesh_dim_names=("dp", "cp"))

    cp_group = device_mesh["cp"].get_group()
    cp_rank = device_mesh["cp"].get_local_rank()

    if s % (2 * cp_size) != 0:
        if rank == 0:
            print(f"ERROR: seq_len ({s}) must be divisible by {2 * cp_size}")
        dist.destroy_process_group()
        sys.exit(1)

    if rank == 0:
        print(f"Golden Value Tests (FSDP2 + CP): B={b}, S={s}, CP={cp_size}")
        print()

    te_config = NVLlamaConfig.from_pretrained(
        args.config_path,
        dtype=torch.bfloat16,
        attn_input_format="bshd",
        self_attn_mask_type="causal",
    )
    hf_config = LlamaConfig.from_pretrained(args.config_path)
    hf_config._attn_implementation = "sdpa"

    te_passed = validate_te(
        te_config, vocab_size, b, s, device, local_rank, device_mesh, cp_group, cp_rank, cp_size, rank
    )
    hf_passed = validate_hf(
        hf_config, vocab_size, b, s, device, local_rank, device_mesh, cp_group, cp_rank, cp_size, rank
    )

    if rank == 0:
        print()
        print(f"Summary: TE [{'PASS' if te_passed else 'FAIL'}], HF [{'PASS' if hf_passed else 'FAIL'}]")
        if not (te_passed and hf_passed):
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
