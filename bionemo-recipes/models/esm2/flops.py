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

"""Architecture-independent FLOPs counting, MFU calculation, and communication overhead estimation.

Supports transformer architectures (Llama, ESM2, CodonFM, etc.) and Hyena (Evo2).
Designed to be copied to any recipe via check_copied_files.py and hooked into
training scripts for live MFU tracking.

Usage as a library (in training scripts):
    from flops import MFUTracker, from_hf_config
    tracker = MFUTracker.from_config_dict(config_dict, batch_size=4, seq_len=4096, num_gpus=2)
    mfu_info = tracker.compute_mfu(step_time=0.5)

Usage as a CLI:
    python flops.py gpu-info
    python flops.py flops --config-path ./model_configs/lingua-1B
    python flops.py cp-comm --config-path ./model_configs/lingua-1B --cp-size 2
    torchrun --nproc_per_node=2 flops.py bandwidth
"""

import gc
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.flop_counter import FlopCounterMode


# =============================================================================
# GPU Peak TFLOPS
# =============================================================================

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


# =============================================================================
# Model FLOPs Config
# =============================================================================

# Model types that use gated MLP (SwiGLU/GeGLU) with 3 projections instead of 2.
GATED_MLP_MODEL_TYPES = frozenset({"llama", "mistral", "qwen2"})


@dataclass(frozen=True)
class ModelFLOPsConfig:
    """Architecture-independent parameters for FLOPs calculation.

    Can be constructed manually or via from_hf_config() for auto-detection.
    """

    hidden_size: int  # H
    num_hidden_layers: int  # L
    num_attention_heads: int  # n_heads
    num_kv_heads: int  # n_kv (== n_heads for MHA)
    head_dim: int  # H // n_heads
    intermediate_size: int  # I (FFN intermediate dimension)
    num_mlp_projections: int  # 2 (standard FFN) or 3 (SwiGLU/GLU)
    vocab_size: int  # V
    has_lm_head: bool  # True for LM models, False for ViT etc.


def from_hf_config(config_dict, **overrides):
    """Create ModelFLOPsConfig from an HF-compatible config dict.

    Auto-detects architecture:
    - GQA vs MHA: from num_key_value_heads (absent = MHA)
    - Gated MLP (3 proj) vs standard FFN (2 proj): from model_type
    - LM head: from vocab_size > 0

    Args:
        config_dict: Dict with standard HF config keys (hidden_size, num_hidden_layers, etc.).
            Works with config.json dicts, config.to_dict(), or MODEL_PRESETS dicts.
        **overrides: Explicit overrides for any field (e.g., num_mlp_projections=3).
    """
    h = config_dict["hidden_size"]
    n_heads = config_dict["num_attention_heads"]
    n_kv = config_dict.get("num_key_value_heads", n_heads)
    vocab = config_dict.get("vocab_size", 0)
    model_type = config_dict.get("model_type", "")

    # Detect gated MLP (3 projections) vs standard FFN (2 projections).
    # Llama/Mistral/Qwen use SwiGLU (gate + up + down = 3 projections).
    # ESM2/CodonFM/Geneformer/BERT use standard FFN (up + down = 2 projections).
    num_mlp_proj = 3 if model_type in GATED_MLP_MODEL_TYPES else 2

    kwargs = {
        "hidden_size": h,
        "num_hidden_layers": config_dict["num_hidden_layers"],
        "num_attention_heads": n_heads,
        "num_kv_heads": n_kv,
        "head_dim": h // n_heads,
        "intermediate_size": config_dict["intermediate_size"],
        "num_mlp_projections": num_mlp_proj,
        "vocab_size": vocab,
        "has_lm_head": vocab > 0,
    }
    kwargs.update(overrides)
    return ModelFLOPsConfig(**kwargs)


# =============================================================================
# FLOPs Formulas
# =============================================================================


def compute_flops_analytical(config, batch_size, seq_len):
    """First-principles FLOPs for any transformer (GQA/MHA, SwiGLU/GELU).

    Counts matmul FLOPs only (2 FLOPs per multiply-accumulate). Excludes softmax,
    layer norms, activations, and element-wise ops.

    Handles:
    - GQA vs MHA: K/V projection sizes based on config.num_kv_heads
    - SwiGLU vs standard FFN: 2 or 3 MLP projections
    - LM head presence

    Returns:
        (total_training_flops, per_layer_breakdown_dict, lm_head_forward_flops)
    """
    b, s, h = batch_size, seq_len, config.hidden_size
    kv_dim = config.num_kv_heads * config.head_dim

    breakdown = {
        "Q projection": 2 * b * s * h * h,
        "K projection": 2 * b * s * h * kv_dim,
        "V projection": 2 * b * s * h * kv_dim,
        "O projection": 2 * b * s * h * h,
        "Attn logits": 2 * b * s * s * h,
        "Attn values": 2 * b * s * s * h,
    }

    ffn = config.intermediate_size
    if config.num_mlp_projections == 3:
        # SwiGLU/GeGLU: gate + up + down = 3 matmuls
        breakdown["Gate projection"] = 2 * b * s * h * ffn
        breakdown["Up projection"] = 2 * b * s * h * ffn
        breakdown["Down projection"] = 2 * b * s * ffn * h
    else:
        # Standard FFN: up + down = 2 matmuls
        breakdown["Up projection"] = 2 * b * s * h * ffn
        breakdown["Down projection"] = 2 * b * s * ffn * h

    per_layer_fwd = sum(breakdown.values())
    lm_head_fwd = 2 * b * s * h * config.vocab_size if config.has_lm_head else 0
    total_fwd = config.num_hidden_layers * per_layer_fwd + lm_head_fwd
    total_training = 3 * total_fwd

    return total_training, breakdown, lm_head_fwd


def compute_flops_simplified(batch_size, seq_len, hidden_size, num_layers, vocab_size):
    """Simplified formula assuming standard MHA + standard FFN with I=4H.

    This is the formula from the Llama3 README:
    (24*B*S*H^2 + 4*B*S^2*H) * 3*L + 6*B*S*H*V

    The 24*H^2 coefficient assumes: 8*H^2 for 4 attention projections (MHA) +
    16*H^2 for 2 MLP projections with I=4H.
    """
    b, s, h = batch_size, seq_len, hidden_size
    return (24 * b * s * h * h + 4 * b * s * s * h) * (3 * num_layers) + (6 * b * s * h * vocab_size)


def compute_flops_hyena(config, batch_size, seq_len, hyena_layer_counts=None):
    """FLOPs for Hyena-based models (Evo2).

    Based on evo2_provider.py. Hyena replaces attention with O(N) FFT convolution.

    Args:
        config: ModelFLOPsConfig with model dimensions.
        batch_size: Batch size.
        seq_len: Sequence length.
        hyena_layer_counts: Optional dict {"S": n, "D": n, "H": n, "A": n} for
            short/medium/long conv and attention layer counts. If None, assumes
            all layers are long-conv Hyena (H=num_layers, no attention).
    """
    b, s, h = batch_size, seq_len, config.hidden_size
    ffn = config.intermediate_size

    if hyena_layer_counts is None:
        hyena_layer_counts = {"S": 0, "D": 0, "H": config.num_hidden_layers, "A": 0}

    # Common per-layer FLOPs
    pre_attn_qkv_proj = 2 * 3 * b * s * h * h
    post_attn_proj = 2 * b * s * h * h
    glu_ffn = 2 * 3 * b * s * ffn * h

    # Layer-type-specific FLOPs (defaults from evo2_provider.py)
    attn = 2 * 2 * b * h * s * s  # Standard S^2 attention
    hyena_proj = 2 * 3 * b * s * 3 * h  # short_conv_L=3 default
    hyena_short_conv = 2 * b * s * 7 * h  # short_conv_len=7
    hyena_medium_conv = 2 * b * s * 128 * h  # medium_conv_len=128
    hyena_long_fft = b * 10 * s * math.log2(max(s, 2)) * h

    n_s = hyena_layer_counts.get("S", 0)
    n_d = hyena_layer_counts.get("D", 0)
    n_h = hyena_layer_counts.get("H", 0)
    n_a = hyena_layer_counts.get("A", 0)

    logits = 2 * b * s * h * config.vocab_size if config.has_lm_head else 0

    total_fwd = (
        logits
        + config.num_hidden_layers * (pre_attn_qkv_proj + post_attn_proj + glu_ffn)
        + n_a * attn
        + (n_s + n_d + n_h) * hyena_proj
        + n_s * hyena_short_conv
        + n_d * hyena_medium_conv
        + int(n_h * hyena_long_fft)
    )

    return 3 * total_fwd


# Backward-compatible wrappers for existing compare_mfu*.py scripts.


def compute_flops_first_principles(b, s, h, num_layers, n_kv_heads, head_dim, ffn_hidden_size, vocab_size):
    """Backward-compatible wrapper. Assumes SwiGLU (3 MLP projections)."""
    config = ModelFLOPsConfig(
        hidden_size=h,
        num_hidden_layers=num_layers,
        num_attention_heads=h // head_dim,
        num_kv_heads=n_kv_heads,
        head_dim=head_dim,
        intermediate_size=ffn_hidden_size,
        num_mlp_projections=3,
        vocab_size=vocab_size,
        has_lm_head=True,
    )
    return compute_flops_analytical(config, b, s)


def compute_flops_readme(b, s, h, num_layers, vocab_size):
    """Backward-compatible wrapper for the simplified README formula."""
    return compute_flops_simplified(b, s, h, num_layers, vocab_size)


# =============================================================================
# MFU Tracker
# =============================================================================


class MFUTracker:
    """Tracks MFU during training. Initialize once, call compute_mfu() per step.

    Usage:
        tracker = MFUTracker.from_config_dict(config_dict, batch_size=4, seq_len=4096, num_gpus=2)
        # In training loop:
        mfu_info = tracker.compute_mfu(step_time=0.5)
        print(f"MFU: {mfu_info['mfu']:.1f}%")
    """

    def __init__(
        self,
        config,
        batch_size,
        seq_len,
        num_gpus=1,
        parallelism=None,
        peak_tflops=None,
        formula="analytical",
        hyena_layer_counts=None,
    ):
        """Initialize MFU tracker.

        Args:
            config: ModelFLOPsConfig instance.
            batch_size: Micro batch size per GPU.
            seq_len: Sequence length.
            num_gpus: Total number of GPUs.
            parallelism: Dict of parallelism dimensions, e.g. {"dp": 1, "cp": 2, "tp": 1}.
                Used for communication overhead estimation.
            peak_tflops: GPU peak bf16 TFLOPS. Auto-detected if None.
            formula: "analytical", "simplified", or "hyena".
            hyena_layer_counts: For Hyena formula, dict of layer type counts.
        """
        self.config = config
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_gpus = num_gpus
        self.parallelism = parallelism or {}
        self.formula = formula

        if formula == "analytical":
            self.total_flops, self.breakdown, self.lm_head_flops = compute_flops_analytical(
                config, batch_size, seq_len
            )
        elif formula == "simplified":
            self.total_flops = compute_flops_simplified(
                batch_size, seq_len, config.hidden_size, config.num_hidden_layers, config.vocab_size
            )
            self.breakdown = None
            self.lm_head_flops = 0
        elif formula == "hyena":
            self.total_flops = compute_flops_hyena(config, batch_size, seq_len, hyena_layer_counts)
            self.breakdown = None
            self.lm_head_flops = 0
        else:
            raise ValueError(f"Unknown formula: {formula!r}. Use 'analytical', 'simplified', or 'hyena'.")

        self.per_gpu_flops = self.total_flops // max(num_gpus, 1)

        if peak_tflops is not None:
            self.peak_tflops = peak_tflops
            self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
        else:
            detected, self.device_name = detect_gpu_peak_tflops()
            self.peak_tflops = detected

        self.comm_bytes = self._estimate_comm()

    @classmethod
    def from_config_dict(cls, config_dict, batch_size, seq_len, **kwargs):
        """Create from an HF config dict with auto-detection."""
        config = from_hf_config(config_dict)
        return cls(config, batch_size, seq_len, **kwargs)

    def compute_mfu(self, step_time):
        """Compute MFU from measured step time.

        Args:
            step_time: Wall-clock time for one training step (seconds).

        Returns:
            Dict with mfu (%), tflops_per_gpu, per_gpu_flops, total_flops, step_time.
        """
        tflops = self.per_gpu_flops / step_time / 1e12
        mfu = tflops / self.peak_tflops * 100 if self.peak_tflops else 0.0
        return {
            "mfu": mfu,
            "tflops_per_gpu": tflops,
            "per_gpu_flops": self.per_gpu_flops,
            "total_flops": self.total_flops,
            "step_time": step_time,
        }

    def estimate_comm_overhead(self, step_time, measured_bw_gbps=None):
        """Estimate communication overhead as a fraction of step time.

        Args:
            step_time: Measured step time in seconds.
            measured_bw_gbps: Measured P2P bandwidth in GB/s. If None, uses a default estimate.

        Returns:
            Dict with comm_bytes, estimated_comm_time, comm_pct.
        """
        bw = measured_bw_gbps or 6.0  # Default ~PCIe Gen3 x8
        comm_time = self.comm_bytes / (bw * 1e9) if bw > 0 else 0.0
        comm_pct = comm_time / step_time * 100 if step_time > 0 else 0.0
        return {"comm_bytes": self.comm_bytes, "estimated_comm_time": comm_time, "comm_pct": comm_pct}

    def _estimate_comm(self):
        """Estimate total communication bytes per step based on parallelism."""
        total = 0
        cp_size = self.parallelism.get("cp", 1)
        dp_size = self.parallelism.get("dp", 1)

        if cp_size > 1:
            total += estimate_cp_comm_bytes(
                self.batch_size,
                self.seq_len,
                self.config.num_hidden_layers,
                self.config.num_kv_heads,
                self.config.head_dim,
                cp_size,
            )

        if dp_size > 1:
            # FSDP reduce-scatter estimate: ~2 * model_params * dtype_bytes * (dp-1)/dp
            model_params = _estimate_model_params(self.config)
            total += 2 * model_params * 2 * (dp_size - 1) // dp_size

        return total

    def summary(self):
        """Return a human-readable summary string."""
        lines = [
            f"MFUTracker: {self.formula} formula, {self.num_gpus} GPU(s)",
            f"  Model: H={self.config.hidden_size}, L={self.config.num_hidden_layers},"
            f" heads={self.config.num_attention_heads}, kv_heads={self.config.num_kv_heads},"
            f" I={self.config.intermediate_size}, V={self.config.vocab_size}",
            f"  MLP projections: {self.config.num_mlp_projections}"
            f" ({'SwiGLU/GLU' if self.config.num_mlp_projections == 3 else 'standard FFN'})",
            f"  Batch: B={self.batch_size}, S={self.seq_len}",
            f"  Total FLOPs/step: {format_flops(self.total_flops)} ({format_flops_exact(self.total_flops)})",
            f"  Per-GPU FLOPs: {format_flops(self.per_gpu_flops)}",
            f"  GPU: {self.device_name} (Peak: {self.peak_tflops} TFLOPS)" if self.peak_tflops else "  GPU: unknown",
        ]
        if self.parallelism:
            lines.append(f"  Parallelism: {self.parallelism}")
        if self.comm_bytes > 0:
            lines.append(f"  Estimated comm: {format_bytes(self.comm_bytes)}/step")
        return "\n".join(lines)


# =============================================================================
# Communication Estimation
# =============================================================================


def estimate_cp_comm_bytes(b, s, num_layers, n_kv_heads, head_dim, cp_size, dtype_bytes=2):
    """Estimate total bytes transferred for CP ring attention per training step.

    Ring attention sends local KV chunks around the ring. Per layer forward:
    (cp-1) steps, each sending B * (S/cp) * 2 * kv_dim * dtype_bytes.
    Training = ~2x forward communication (forward sends KV, backward sends dKV).
    """
    if cp_size <= 1:
        return 0
    s_local = s // cp_size
    kv_dim = n_kv_heads * head_dim
    per_layer_fwd = (cp_size - 1) * b * s_local * 2 * kv_dim * dtype_bytes
    return 2 * num_layers * per_layer_fwd


def _estimate_model_params(config):
    """Rough parameter count estimate from config dimensions."""
    h = config.hidden_size
    kv_dim = config.num_kv_heads * config.head_dim
    attn_params = h * h + 2 * h * kv_dim + h * h  # Q + K + V + O
    mlp_params = config.num_mlp_projections * h * config.intermediate_size
    layer_params = attn_params + mlp_params
    total = config.num_hidden_layers * layer_params
    if config.has_lm_head:
        total += config.vocab_size * h * 2  # embed + lm_head
    return total


# =============================================================================
# Step Time Measurement
# =============================================================================


def measure_step_time(
    model,
    input_ids,
    num_warmup=10,
    num_timed=20,
    distributed=False,
    cp_context_fn=None,
    labels=None,
    position_ids=None,
    **extra_fwd_kwargs,
):
    """Measure average training step time (forward + backward).

    Args:
        model: The model to benchmark.
        input_ids: Input tensor.
        num_warmup: Warmup iterations (discarded).
        num_timed: Timed iterations to average.
        distributed: Whether to use dist.barrier() for synchronization.
        cp_context_fn: Optional callable returning a context manager for CP.
        labels: Optional labels tensor. If None, uses input_ids.
        position_ids: Optional position_ids for correct RoPE with CP.
        **extra_fwd_kwargs: Additional kwargs for model forward.
    """
    if labels is None:
        labels = input_ids

    fwd_kwargs = {"input_ids": input_ids, "labels": labels, **extra_fwd_kwargs}
    if position_ids is not None:
        fwd_kwargs["position_ids"] = position_ids

    for _ in range(num_warmup):
        ctx = cp_context_fn() if cp_context_fn else nullcontext()
        with ctx:
            output = model(**fwd_kwargs)
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
            output = model(**fwd_kwargs)
            output.loss.backward()
        model.zero_grad(set_to_none=True)

        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)

    return sum(times) / len(times)


# =============================================================================
# Utilities
# =============================================================================


def split_for_cp_bshd(tensor, cp_rank, cp_size):
    """Split a BSHD tensor for CP using the dual-chunk zigzag pattern."""
    if cp_size <= 1:
        return tensor
    total_chunks = 2 * cp_size
    seq_len = tensor.size(1)
    chunk_size = seq_len // total_chunks
    chunk_indices = [cp_rank, total_chunks - cp_rank - 1]
    slices = [tensor[:, idx * chunk_size : (idx + 1) * chunk_size] for idx in chunk_indices]
    return torch.cat(slices, dim=1)


def measure_bus_bandwidth(device, world_size, num_iters=20, num_elements=10_000_000):
    """Measure unidirectional P2P bandwidth via send/recv (matches CP ring pattern)."""
    if world_size <= 1:
        return 0.0

    rank = dist.get_rank()
    tensor = torch.randn(num_elements, device=device, dtype=torch.bfloat16)
    peer = 1 - rank

    for _ in range(5):
        if rank == 0:
            dist.send(tensor, dst=peer)
            dist.recv(tensor, src=peer)
        else:
            dist.recv(tensor, src=peer)
            dist.send(tensor, dst=peer)
    torch.cuda.synchronize()

    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        if rank == 0:
            dist.send(tensor, dst=peer)
        else:
            dist.recv(tensor, src=peer)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    data_bytes = tensor.nelement() * tensor.element_size()
    return num_iters * data_bytes / elapsed / 1e9


def count_flops_with_model(model, input_ids):
    """Count forward FLOPs using PyTorch's FlopCounterMode, return 3x for training."""
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        model(input_ids=input_ids)
    return flop_counter.get_total_flops() * 3


def load_model_config(config_path):
    """Load model config dict from a local path or HuggingFace model ID.

    Supports:
    - Local directory: ./model_configs/lingua-1B (reads config.json inside)
    - Local file: ./model_configs/lingua-1B/config.json
    - HF model ID: nvidia/esm2_t36_3B_UR50D (fetches from HuggingFace Hub)
    """
    import json
    from pathlib import Path

    path = Path(config_path)
    if path.is_dir():
        path = path / "config.json"
    if path.exists():
        return json.loads(path.read_text())

    # Fall back to HuggingFace Hub
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    return hf_config.to_dict()


def cleanup_model(model):
    """Delete a model and free GPU memory."""
    del model
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# Formatting
# =============================================================================


def format_flops(flops):
    """Format FLOPs with appropriate unit (G/T/P)."""
    if flops >= 1e15:
        return f"{flops / 1e15:.2f} P"
    elif flops >= 1e12:
        return f"{flops / 1e12:.2f} T"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} G"
    else:
        return f"{flops:.2e}"


def format_flops_exact(flops):
    """Format FLOPs as the full integer with commas."""
    return f"{int(flops):,}"


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


# =============================================================================
# CLI
# =============================================================================


def _cli_bandwidth():
    """Measure P2P bandwidth. Launch with: torchrun --nproc_per_node=2 flops.py bandwidth."""
    import os

    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Measuring P2P bandwidth between {world_size} GPUs...")
        for i in range(world_size):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    bw = measure_bus_bandwidth(device, world_size)
    if rank == 0:
        print(f"\nUnidirectional P2P bandwidth: {bw:.2f} GB/s")

    dist.destroy_process_group()


def _cli_gpu_info():
    """Print GPU info and peak TFLOPS."""
    peak, name = detect_gpu_peak_tflops()
    print(f"GPU: {name}")
    if peak:
        print(f"Peak bf16 TFLOPS: {peak:.1f}")
    else:
        print("Peak bf16 TFLOPS: unknown (use --peak-tflops to override)")
    print()
    print("Known GPUs:")
    for gpu, tflops in GPU_PEAK_TFLOPS_BF16.items():
        print(f"  {gpu:<16} {tflops:>8.1f} TFLOPS")


def _cli_flops():
    """Compute FLOPs for a model config."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--config-path", default="./model_configs/lingua-1B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--formula", default="analytical", choices=["analytical", "simplified"])
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for per-GPU FLOPs and comm estimates")
    parser.add_argument("--cp-size", type=int, default=1, help="Context parallelism size for comm overhead estimate")
    parser.add_argument("--p2p-bw", type=float, default=6.0, help="P2P bandwidth in GB/s for comm time estimate")
    args = parser.parse_args()

    cfg_dict = load_model_config(args.config_path)
    config = from_hf_config(cfg_dict)
    b, s = args.batch_size, args.seq_len

    print(
        f"Config: H={config.hidden_size}, L={config.num_hidden_layers},"
        f" n_heads={config.num_attention_heads}, n_kv={config.num_kv_heads},"
        f" I={config.intermediate_size}, V={config.vocab_size}"
    )
    print(
        f"MLP: {config.num_mlp_projections} projections"
        f" ({'SwiGLU/GLU' if config.num_mlp_projections == 3 else 'standard FFN'})"
    )
    print(f"Batch: B={b}, S={s}, GPUs={args.num_gpus}, CP={args.cp_size}")
    print()

    simplified = compute_flops_simplified(b, s, config.hidden_size, config.num_hidden_layers, config.vocab_size)
    analytical, breakdown, lm_head = compute_flops_analytical(config, b, s)

    print(f"{'Method':<24} {'FLOPs/step':>14} {'Per-GPU':>14} {'Exact':>30}")
    print("-" * 86)
    for name, flops in [("Simplified (README)", simplified), ("Analytical", analytical)]:
        per_gpu = flops // max(args.num_gpus, 1)
        print(f"{name:<24} {format_flops(flops):>14} {format_flops(per_gpu):>14} {format_flops_exact(flops):>30}")

    if simplified != analytical:
        diff = analytical - simplified
        print(f"\nDifference: {format_flops_exact(diff)} ({diff / simplified * 100:+.2f}%)")
    else:
        print("\nFormulas agree exactly for this config.")

    # Communication overhead estimate
    if args.cp_size > 1:
        dp_size = args.num_gpus // args.cp_size
        parallelism = {"dp": dp_size, "cp": args.cp_size}
        tracker = MFUTracker(config, b, s, num_gpus=args.num_gpus, parallelism=parallelism)
        print(f"\n--- Communication Overhead (CP={args.cp_size}, P2P BW={args.p2p_bw} GB/s) ---")
        print(
            f"  CP ring attention: {format_bytes(tracker.comm_bytes)}/step ({format_flops_exact(tracker.comm_bytes)} bytes)"
        )
        comm_time = tracker.comm_bytes / (args.p2p_bw * 1e9) if args.p2p_bw > 0 else 0
        print(f"  Estimated comm time: {comm_time:.4f}s")

    model_params = _estimate_model_params(config)
    print_breakdown(breakdown, lm_head, config.num_hidden_layers, analytical, model_params)


def _cli_cp_comm():
    """Estimate CP communication volume."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--config-path", default="./model_configs/lingua-1B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--cp-size", type=int, default=2)
    args = parser.parse_args()

    cfg_dict = load_model_config(args.config_path)
    config = from_hf_config(cfg_dict)
    b, s = args.batch_size, args.seq_len

    comm = estimate_cp_comm_bytes(b, s, config.num_hidden_layers, config.num_kv_heads, config.head_dim, args.cp_size)
    print(
        f"CP={args.cp_size}, B={b}, S={s}, L={config.num_hidden_layers},"
        f" n_kv_heads={config.num_kv_heads}, head_dim={config.head_dim}"
    )
    print(f"Estimated CP ring attention communication: {format_bytes(comm)}/step ({format_flops_exact(comm)} bytes)")


if __name__ == "__main__":
    import sys

    commands = {
        "bandwidth": ("Measure P2P bandwidth (requires torchrun --nproc_per_node=2)", _cli_bandwidth),
        "gpu-info": ("Print GPU info and peak TFLOPS", _cli_gpu_info),
        "flops": ("Compute FLOPs for a model config", _cli_flops),
        "cp-comm": ("Estimate CP communication volume", _cli_cp_comm),
    }

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or sys.argv[1] not in commands:
        print("Usage: python flops.py <command> [options]")
        print("       torchrun --nproc_per_node=2 flops.py bandwidth")
        print()
        print("Commands:")
        for cmd, (desc, _) in commands.items():
            print(f"  {cmd:<16} {desc}")
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 1)

    commands[sys.argv[1]][1]()
