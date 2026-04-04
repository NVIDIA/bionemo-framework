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

"""Multi-GPU MFU comparison: TE CP vs HF CP head-to-head.

Compares MFU with context parallelism for both TE (via set_context_parallel_group)
and HF (via PyTorch native context_parallel with ring attention).

Usage:
    cd bionemo-recipes/recipes/llama3_native_te
    torchrun --nproc_per_node=2 compare_mfu_multigpu.py
    torchrun --nproc_per_node=2 compare_mfu_multigpu.py --seq-len 32768
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from compare_mfu_common import (
    cleanup_model,
    compute_flops_first_principles,
    compute_flops_readme,
    count_flops_with_model,
    create_te_model_on_gpu,
    detect_gpu_peak_tflops,
    estimate_cp_comm_bytes,
    format_bytes,
    format_flops,
    measure_bus_bandwidth,
    measure_step_time,
    print_breakdown,
    split_for_cp_bshd,
)
from modeling_llama_te import NVLlamaConfig


def main():
    """Run multi-GPU MFU comparison: TE CP vs HF CP."""
    # --- Distributed setup ---
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    # --- Parse arguments ---
    parser = argparse.ArgumentParser(description="Multi-GPU MFU comparison: TE CP vs HF CP")
    parser.add_argument("--config-path", default="./model_configs/lingua-1B", help="Model config directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro batch size per GPU")
    parser.add_argument("--seq-len", type=int, default=16384, help="Total sequence length (split across CP ranks)")
    parser.add_argument("--cp-size", type=int, default=None, help="CP size (default: world_size)")
    parser.add_argument("--peak-tflops", type=float, default=None, help="Override GPU peak bf16 TFLOPS")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup iterations before timing")
    parser.add_argument("--timed-steps", type=int, default=20, help="Timed iterations to average")
    args = parser.parse_args()

    cp_size = args.cp_size or world_size
    dp_size = world_size // cp_size
    if dp_size * cp_size != world_size:
        if rank == 0:
            print(f"ERROR: dp_size ({dp_size}) * cp_size ({cp_size}) != world_size ({world_size})")
        dist.destroy_process_group()
        sys.exit(1)

    # --- Device mesh ---
    device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, cp_size), mesh_dim_names=("dp", "cp"))
    cp_group = device_mesh["cp"].get_group()
    cp_ranks = dist.get_process_group_ranks(cp_group)
    cp_rank = device_mesh["cp"].get_local_rank()

    # --- Load model config ---
    config_path = Path(args.config_path) / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    b = args.batch_size
    s = args.seq_len
    h = config_dict["hidden_size"]
    num_layers = config_dict["num_hidden_layers"]
    vocab_size = config_dict["vocab_size"]
    n_kv_heads = config_dict["num_key_value_heads"]
    n_heads = config_dict["num_attention_heads"]
    head_dim = h // n_heads
    ffn_hidden_size = config_dict["intermediate_size"]
    s_local = s // cp_size

    if s % (2 * cp_size) != 0:
        if rank == 0:
            print(f"ERROR: seq_len ({s}) must be divisible by {2 * cp_size} (2 * cp_size)")
        dist.destroy_process_group()
        sys.exit(1)

    # --- GPU detection ---
    if args.peak_tflops:
        peak_tflops = args.peak_tflops
        device_name = torch.cuda.get_device_name(0)
    else:
        peak_tflops, device_name = detect_gpu_peak_tflops()
        if peak_tflops is None:
            if rank == 0:
                print(f"ERROR: Could not auto-detect GPU peak TFLOPS for: {device_name}")
            dist.destroy_process_group()
            sys.exit(1)

    peak_flops_per_sec = peak_tflops * 1e12

    if rank == 0:
        print(f"GPU: {world_size}x {device_name} (Peak: {peak_tflops:.1f} TFLOPS bf16 each)")
        print(f"Parallelism: dp={dp_size}, cp={cp_size} ({world_size} GPUs)")
        print(
            f"Config: H={h}, L={num_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads},"
            f" head_dim={head_dim}, I={ffn_hidden_size}, V={vocab_size}"
        )
        print(f"Batch: B={b}, S={s} (S_local={s_local} per GPU)")
        print()

    # =========================================================================
    # Table 1: FLOPs Counting
    # =========================================================================
    total_flops_readme = compute_flops_readme(b, s, h, num_layers, vocab_size)
    total_flops_fp, breakdown, lm_head_fwd = compute_flops_first_principles(
        b, s, h, num_layers, n_kv_heads, head_dim, ffn_hidden_size, vocab_size
    )

    if rank == 0:
        print("Counting FLOPs with HF model (meta device)...")
    hf_config_meta = LlamaConfig.from_pretrained(args.config_path)
    hf_config_meta._attn_implementation = "eager"
    hf_config_meta.max_position_embeddings = max(hf_config_meta.max_position_embeddings, s)
    with torch.device("meta"):
        hf_model_meta = LlamaForCausalLM(hf_config_meta)
    meta_input_ids = torch.randint(0, vocab_size, (b, s), device="meta")
    total_flops_hf_counter = count_flops_with_model(hf_model_meta, meta_input_ids)
    del hf_model_meta
    if rank == 0:
        print(f"  HF FlopCounter: {format_flops(total_flops_hf_counter)} (training, full batch)")

    per_gpu_flops = total_flops_fp // world_size

    # =========================================================================
    # Table 2: MFU — TE CP vs HF CP
    # =========================================================================

    # --- HF with PyTorch native CP (run first to avoid NCCL memory fragmentation) ---
    if rank == 0:
        print(f"\n[1/2] HF model with PyTorch native CP={cp_size} (S={s})...")
    hf_config_gpu = LlamaConfig.from_pretrained(args.config_path)
    hf_config_gpu._attn_implementation = "sdpa"  # Required for context_parallel
    hf_config_gpu.max_position_embeddings = max(hf_config_gpu.max_position_embeddings, s)
    hf_model = LlamaForCausalLM(hf_config_gpu).to(dtype=torch.bfloat16, device=device)
    hf_model.train()

    # Full-size inputs — context_parallel shards them each iteration
    hf_full_ids = torch.randint(0, vocab_size, (b, s), device=device)
    hf_full_labels = hf_full_ids.clone()
    cp_mesh = device_mesh["cp"]

    def make_hf_cp_ctx(_ids=hf_full_ids, _labels=hf_full_labels):
        return context_parallel(cp_mesh, buffers=(_ids, _labels), buffer_seq_dims=(1, 1))

    if rank == 0:
        print(f"Measuring HF CP step time ({args.warmup_steps} warmup + {args.timed_steps} timed)...")
    hf_cp_time = measure_step_time(
        hf_model,
        hf_full_ids,
        args.warmup_steps,
        args.timed_steps,
        distributed=True,
        cp_context_fn=make_hf_cp_ctx,
        labels=hf_full_labels,
    )
    if rank == 0:
        print(f"  HF CP step time: {hf_cp_time:.4f}s")
    cleanup_model(hf_model)
    del hf_full_ids, hf_full_labels

    # --- TE with CP via set_context_parallel_group ---
    if rank == 0:
        print(f"\n[2/2] TE model with CP={cp_size} (S={s})...")
    te_config = NVLlamaConfig.from_pretrained(
        args.config_path,
        dtype=torch.bfloat16,
        attn_input_format="bshd",
        self_attn_mask_type="causal",
    )
    te_config.max_position_embeddings = max(te_config.max_position_embeddings, s)
    te_model = create_te_model_on_gpu(te_config)
    for layer in te_model.model.layers:
        layer.set_context_parallel_group(cp_group, cp_ranks, torch.cuda.Stream())
    te_model.train()

    full_ids = torch.randint(0, vocab_size, (b, s), device=device)
    te_local_ids = split_for_cp_bshd(full_ids, cp_rank, cp_size)

    if rank == 0:
        print(f"Measuring TE CP step time ({args.warmup_steps} warmup + {args.timed_steps} timed)...")
    te_cp_time = measure_step_time(te_model, te_local_ids, args.warmup_steps, args.timed_steps, distributed=True)
    model_params = sum(p.numel() for p in te_model.parameters())
    if rank == 0:
        print(f"  TE CP step time: {te_cp_time:.4f}s")
    cleanup_model(te_model)

    # =========================================================================
    # Communication overhead
    # =========================================================================
    if rank == 0:
        print("\nMeasuring inter-GPU bandwidth...")
    bus_bw_gbps = measure_bus_bandwidth(device, world_size)
    if rank == 0:
        print(f"  Bus bandwidth: {bus_bw_gbps:.1f} GB/s")

    cp_comm_bytes = estimate_cp_comm_bytes(b, s, num_layers, n_kv_heads, head_dim, cp_size)
    cp_comm_time = cp_comm_bytes / (bus_bw_gbps * 1e9) if bus_bw_gbps > 0 else 0.0

    # =========================================================================
    # Print results (rank 0 only)
    # =========================================================================
    if rank == 0:
        print()
        print("=" * 75)
        print(f"MFU Comparison: Lingua-1B (B={b}, S={s}, bf16, CP={cp_size})")
        print(f"GPU: {world_size}x {device_name} (Peak: {peak_tflops:.1f} TFLOPS bf16 each)")
        print("=" * 75)

        # --- Table 1 ---
        print()
        print("--- Table 1: FLOPs Counting (per training step) ---")
        hdr1 = f"{'Method':<24} {'Total FLOPs':>14} {'Per-GPU FLOPs':>14}"
        print(hdr1)
        print("-" * len(hdr1))
        for name, total in [
            ("README Formula", total_flops_readme),
            ("First Principles", total_flops_fp),
            ("FlopCounter (HF)", total_flops_hf_counter),
        ]:
            print(f"{name:<24} {format_flops(total):>14} {format_flops(total // world_size):>14}")

        # --- Table 2 ---
        print()
        print(f"--- Table 2: MFU (per GPU, CP={cp_size}) ---")
        hdr2 = f"{'Model':<16} {'Per-GPU FLOPs':>14} {'Step (s)':>9} {'TFLOPS/s':>9} {'MFU':>7}"
        print(hdr2)
        print("-" * len(hdr2))

        for name, step_time in [("TE (CP)", te_cp_time), ("HF (CP)", hf_cp_time)]:
            tflops = per_gpu_flops / step_time / 1e12
            mfu = per_gpu_flops / step_time / peak_flops_per_sec * 100
            print(f"{name:<16} {format_flops(per_gpu_flops):>14} {step_time:>8.3f}s {tflops:>8.2f} {mfu:>6.1f}%")

        print()
        print(f"TE vs HF speedup: {hf_cp_time / te_cp_time:.2f}x")

        # --- Communication overhead ---
        print()
        print("--- Communication Overhead ---")
        print(f"Measured bus bandwidth:     {bus_bw_gbps:.1f} GB/s")
        print(f"CP ring attention (cp={cp_size}):  {format_bytes(cp_comm_bytes):>12}/step  (~{cp_comm_time:.4f}s)")
        if te_cp_time > 0:
            print(f"  As % of TE step:  {cp_comm_time / te_cp_time * 100:.1f}%")
        if hf_cp_time > 0:
            print(f"  As % of HF step:  {cp_comm_time / hf_cp_time * 100:.1f}%")

        # --- Breakdown ---
        print_breakdown(breakdown, lm_head_fwd, num_layers, total_flops_fp, model_params)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
