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

"""Shared utilities for MFU comparison scripts.

Provides FLOPs counting formulas, GPU detection, model creation helpers,
step time measurement, and formatting utilities used by both single-GPU
and multi-GPU MFU comparison scripts.
"""

import gc
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.flop_counter import FlopCounterMode

from modeling_llama_te import NVLlamaForCausalLM


# Peak bf16 TFLOPS for common NVIDIA GPUs (tensor core ops).
GPU_PEAK_TFLOPS_BF16 = {
    "H100": 989.0,
    "H200": 989.0,
    "A100": 312.0,
    "A6000": 155.0,
    "A5000": 111.0,
    "L40": 181.0,
    "RTX 4090": 330.0,
    "RTX 3090": 142.0,
    "GH200": 989.0,
    "B200": 2250.0,
    "GB200": 2250.0,
}


def detect_gpu_peak_tflops():
    """Auto-detect GPU peak bf16 TFLOPS from device name via substring match."""
    device_name = torch.cuda.get_device_name(0)
    for gpu_key, tflops in GPU_PEAK_TFLOPS_BF16.items():
        if gpu_key.lower() in device_name.lower():
            return tflops, device_name
    return None, device_name


def compute_flops_readme(b, s, h, num_layers, vocab_size):
    """README formula: assumes standard MHA + standard MLP (I=4H, 2 projections)."""
    return (24 * b * s * h * h + 4 * b * s * s * h) * (3 * num_layers) + (6 * b * s * h * vocab_size)


def compute_flops_first_principles(b, s, h, num_layers, n_kv_heads, head_dim, ffn_hidden_size, vocab_size):
    """First-principles FLOPs for GQA + SwiGLU architecture.

    Returns:
        total_training_flops: Total FLOPs for one training step (3x forward).
        breakdown: Per-component forward FLOPs for one layer.
        lm_head_fwd: Forward FLOPs for the LM head.
    """
    kv_dim = n_kv_heads * head_dim

    breakdown = {
        "Q projection": 2 * b * s * h * h,
        "K projection": 2 * b * s * h * kv_dim,
        "V projection": 2 * b * s * h * kv_dim,
        "O projection": 2 * b * s * h * h,
        "Attn logits": 2 * b * s * s * h,
        "Attn values": 2 * b * s * s * h,
        "Gate projection": 2 * b * s * h * ffn_hidden_size,
        "Up projection": 2 * b * s * h * ffn_hidden_size,
        "Down projection": 2 * b * s * ffn_hidden_size * h,
    }

    per_layer_fwd = sum(breakdown.values())
    lm_head_fwd = 2 * b * s * h * vocab_size
    total_fwd = num_layers * per_layer_fwd + lm_head_fwd
    total_training = 3 * total_fwd

    return total_training, breakdown, lm_head_fwd


def count_flops_with_model(model, input_ids):
    """Count forward FLOPs using PyTorch's FlopCounterMode, return 3x for training."""
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        model(input_ids=input_ids)
    return flop_counter.get_total_flops() * 3


def create_te_model_on_gpu(config):
    """Create a TE model on GPU using the meta device + init_empty_weights pattern."""
    with torch.device("meta"):
        model = NVLlamaForCausalLM(config)
    model.init_empty_weights()
    return model


def measure_step_time(
    model, input_ids, num_warmup=10, num_timed=20, distributed=False, cp_context_fn=None, labels=None
):
    """Measure average training step time (forward + backward).

    Args:
        model: The model to benchmark.
        input_ids: Input tensor. For CP with context_parallel, pass full-size tensors
            (the cp_context_fn will shard them).
        num_warmup: Number of warmup iterations (discarded).
        num_timed: Number of timed iterations to average.
        distributed: Whether to use dist.barrier() for synchronization.
        cp_context_fn: Optional callable returning a context manager (e.g., context_parallel).
            Called fresh each iteration since it shards/restores buffers.
        labels: Optional labels tensor. If None, uses input_ids as labels.
    """
    if labels is None:
        labels = input_ids

    for _ in range(num_warmup):
        ctx = cp_context_fn() if cp_context_fn else nullcontext()
        with ctx:
            output = model(input_ids=input_ids, labels=labels)
            output.loss.backward()
        model.zero_grad(set_to_none=True)

    if distributed:
        dist.barrier()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()

        ctx = cp_context_fn() if cp_context_fn else nullcontext()
        with ctx:
            output = model(input_ids=input_ids, labels=labels)
            output.loss.backward()
        model.zero_grad(set_to_none=True)

        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)

    return sum(times) / len(times)


def split_for_cp_bshd(tensor, cp_rank, cp_size):
    """Split a BSHD tensor for CP using the dual-chunk zigzag pattern.

    For cp_size=2: rank 0 gets chunks [0, 3], rank 1 gets chunks [1, 2].
    """
    if cp_size <= 1:
        return tensor
    total_chunks = 2 * cp_size
    seq_len = tensor.size(1)
    chunk_size = seq_len // total_chunks
    chunk_indices = [cp_rank, total_chunks - cp_rank - 1]
    slices = [tensor[:, idx * chunk_size : (idx + 1) * chunk_size] for idx in chunk_indices]
    return torch.cat(slices, dim=1)


def measure_bus_bandwidth(device, world_size, num_iters=20, num_elements=10_000_000):
    """Measure inter-GPU bus bandwidth using NCCL all-reduce."""
    if world_size <= 1:
        return 0.0

    tensor = torch.randn(num_elements, device=device, dtype=torch.bfloat16)
    for _ in range(5):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters

    data_bytes = tensor.nelement() * tensor.element_size()
    bus_bw = 2 * (world_size - 1) / world_size * data_bytes / elapsed
    return bus_bw / 1e9  # GB/s


def estimate_cp_comm_bytes(b, s, num_layers, n_kv_heads, head_dim, cp_size, dtype_bytes=2):
    """Estimate total bytes transferred for CP ring attention per training step."""
    if cp_size <= 1:
        return 0
    s_local = s // cp_size
    kv_dim = n_kv_heads * head_dim
    per_layer_fwd = (cp_size - 1) * b * s_local * 2 * kv_dim * dtype_bytes
    return 2 * num_layers * per_layer_fwd


def format_flops(flops):
    """Format FLOPs value with appropriate unit."""
    if flops >= 1e15:
        return f"{flops / 1e15:.2f} P"
    elif flops >= 1e12:
        return f"{flops / 1e12:.2f} T"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} G"
    else:
        return f"{flops:.2e}"


def format_bytes(num_bytes):
    """Format bytes with appropriate unit."""
    if num_bytes >= 1e9:
        return f"{num_bytes / 1e9:.2f} GB"
    elif num_bytes >= 1e6:
        return f"{num_bytes / 1e6:.2f} MB"
    elif num_bytes >= 1e3:
        return f"{num_bytes / 1e3:.2f} KB"
    else:
        return f"{num_bytes:.0f} B"


def cleanup_model(model):
    """Delete a model and free GPU memory."""
    del model
    gc.collect()
    torch.cuda.empty_cache()


def print_breakdown(breakdown, lm_head_fwd, num_layers, total_flops, model_params):
    """Print first-principles FLOPs breakdown."""
    print()
    print("--- First Principles Breakdown (forward pass, per layer) ---")
    per_layer_total = sum(breakdown.values())
    for component, flops_val in breakdown.items():
        pct = flops_val / per_layer_total * 100
        print(f"  {component:<20} {format_flops(flops_val):>12} ({pct:>5.1f}%)")
    total_fwd = num_layers * per_layer_total + lm_head_fwd
    print(f"  {'LM head':<20} {format_flops(lm_head_fwd):>12}")
    print(f"  {'Per-layer total':<20} {format_flops(per_layer_total):>12}")
    print(f"  {'All layers (x' + str(num_layers) + ')':<20} {format_flops(num_layers * per_layer_total):>12}")
    print(f"  {'Total forward':<20} {format_flops(total_fwd):>12}")
    print(f"  {'Total training (3x)':<20} {format_flops(total_flops):>12}")
    print(f"  {'Model params':<20} {model_params / 1e9:.2f}B")
