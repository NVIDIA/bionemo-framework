#!/usr/bin/env python3
"""
GEMM Benchmarking Script with TFLOPS Measurement and Plotting

Benchmarks matrix multiplication performance across different precisions
(BF16, FP8 via Transformer Engine) and generates a comparison plot.

Usage:
    python gemm_benchmark.py [--output plot.png] [--num-iters 100]
"""

import argparse
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import math

# Optional TE import - gracefully handle if not available
try:
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.common.recipe import Format, MXFP8BlockScaling, NVFP4BlockScaling
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("Warning: Transformer Engine not available. FP8/FP4 benchmarks will be skipped.")

# Check for Blackwell (SM100+) for FP4 support
def is_blackwell_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10  # SM100 = Blackwell


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    tflops: float
    avg_time_ms: float
    shape: tuple[int, int, int]
    precision: str


def compute_gemm_flops(M: int, K: int, N: int) -> int:
    """
    Compute theoretical FLOP count for GEMM C = A @ B.
    
    A: (M, K), B: (K, N), C: (M, N)
    Each output element requires K multiply-adds = 2K FLOPs
    Total: 2 * M * N * K
    """
    return 2 * M * N * K


def benchmark_torch_matmul(
    M: int, 
    K: int, 
    N: int, 
    dtype: torch.dtype,
    num_warmup: int = 10,
    num_iters: int = 100
) -> BenchmarkResult:
    """Benchmark torch.matmul at specified precision."""
    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)
    
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    
    # Large matrix for leading kernel to saturate GPU
    A_large = torch.randn(4096, 4096, dtype=dtype, device=device)
    B_large = torch.randn(4096, 4096, dtype=dtype, device=device)
    
    # Warmup - critical for accurate timing
    for _ in range(num_warmup):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Timed iterations using CUDA events
    # Start with a long-running kernel to avoid measuring CPU dispatch overhead
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Leading kernel keeps GPU busy while CPU queues the timed kernels
    _ = torch.matmul(A_large, B_large)
    
    start_event.record()
    for _ in range(num_iters):
        _ = torch.matmul(A, B)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iters
    avg_time_s = avg_time_ms / 1000.0
    
    tflops = (flops / avg_time_s) / 1e12
    
    precision_name = {
        torch.bfloat16: "BF16",
        torch.float16: "FP16", 
        torch.float32: "FP32",
    }.get(dtype, str(dtype))
    
    return BenchmarkResult(
        tflops=tflops,
        avg_time_ms=avg_time_ms,
        shape=(M, K, N),
        precision=precision_name
    )


def benchmark_te_fp8(
    M: int,
    K: int, 
    N: int,
    num_warmup: int = 10,
    num_iters: int = 100
) -> Optional[BenchmarkResult]:
    """Benchmark FP8 GEMM via Transformer Engine Linear layer."""
    if not TE_AVAILABLE:
        return None
    
    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)
    
    # TE Linear: input (M, K) @ weight (K, N) -> output (M, N)
    linear = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).to(device)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    
    # Large tensors for leading kernel
    linear_large = te.Linear(4096, 4096, bias=False, params_dtype=torch.bfloat16).to(device)
    x_large = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
    
    fp8_recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
    
    # Keep autocast context open for warmup and timing
    with te.autocast(enabled=True, recipe=fp8_recipe):
        # Warmup
        for _ in range(num_warmup):
            _ = linear(x)
        torch.cuda.synchronize()
        
        # Timed iterations with leading kernel
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Leading kernel keeps GPU busy while CPU queues the timed kernels
        _ = linear_large(x_large)
        
        start_event.record()
        for _ in range(num_iters):
            _ = linear(x)
        end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iters
    avg_time_s = avg_time_ms / 1000.0
    
    tflops = (flops / avg_time_s) / 1e12
    
    return BenchmarkResult(
        tflops=tflops,
        avg_time_ms=avg_time_ms,
        shape=(M, K, N),
        precision="MXFP8"
    )


def benchmark_te_fp8_prequantized(
    M: int,
    K: int,
    N: int,
    workspace_size: int,
    num_warmup: int = 10,
    num_iters: int = 100
) -> Optional[BenchmarkResult]:
    """Benchmark FP8 GEMM with pre-quantized inputs (measures raw kernel throughput)."""
    if not TE_AVAILABLE:
        return None
    
    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)
    
    try:
        # Create quantizer (requires fp8_dtype argument)
        quantizer = te.MXFP8Quantizer(tex.DType.kFloat8E4M3)
        
        # Create BF16 tensors
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        B = torch.randn(N, K, dtype=torch.bfloat16, device=device)  # Transposed for GEMM
        
        # Pre-quantize to FP8
        A_fp8 = quantizer.quantize(A)
        B_fp8 = quantizer.quantize(B)
        
        # Large tensors for leading kernel
        A_large = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
        B_large = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
        A_large_fp8 = quantizer.quantize(A_large)
        B_large_fp8 = quantizer.quantize(B_large)
        
        # Output buffers
        D = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        D_large = torch.empty(4096, 4096, dtype=torch.bfloat16, device=device)
        
        # Allocate workspace (estimate size)
        workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
        
        # Warmup
        for _ in range(num_warmup):
            tex.generic_gemm(
                A_fp8, False,           # A, transA
                B_fp8, True,            # B, transB
                D,                      # output
                None,                   # quantizer (None = no output quantization)
                tex.DType.kBFloat16,    # output_dtype
                None,                   # bias
                tex.DType.kBFloat16,    # bias_type (unused when bias=None)
                False,                  # gelu
                None,                   # gelu_in
                False,                  # grad
                workspace,              # workspace
                workspace_size,         # workspace_size
                False,                  # accumulate
                False,                  # use_split_accumulator
            )
        torch.cuda.synchronize()
        
        # Timed iterations with leading kernel
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Leading kernel keeps GPU busy while CPU queues the timed kernels
        tex.generic_gemm(
            A_large_fp8, False, B_large_fp8, True, D_large, None,
            tex.DType.kBFloat16, None, tex.DType.kBFloat16,
            False, None, False, workspace, workspace_size, False, False,
        )
        
        start_event.record()
        for _ in range(num_iters):
            tex.generic_gemm(
                A_fp8, False,
                B_fp8, True,
                D,
                None,
                tex.DType.kBFloat16,
                None,
                tex.DType.kBFloat16,
                False,
                None,
                False,
                workspace,
                workspace_size,
                False,
                False,
            )
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        avg_time_ms = elapsed_ms / num_iters
        avg_time_s = avg_time_ms / 1000.0
        
        tflops = (flops / avg_time_s) / 1e12
        
        return BenchmarkResult(
            tflops=tflops,
            avg_time_ms=avg_time_ms,
            shape=(M, K, N),
            precision="MXFP8"
        )
    except Exception as e:
        print(
            f"Warning: FP8 prequantized benchmark failed for shape {M}x{K}x{N}: {e}\n"
            f"  Tip: try increasing --workspace-mb (current={workspace_size / (1024 * 1024):.0f}MB) "
            "or run without --pre-quantize to use te.Linear()."
        )
        return None


def benchmark_te_fp4(
    M: int,
    K: int,
    N: int,
    num_warmup: int = 10,
    num_iters: int = 100
) -> Optional[BenchmarkResult]:
    """Benchmark FP4 GEMM via Transformer Engine Linear layer (Blackwell only)."""
    if not TE_AVAILABLE:
        return None
    
    if not is_blackwell_available():
        return None
    
    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)
    
    # TE Linear: input (M, K) @ weight (K, N) -> output (M, N)
    linear = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).to(device)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    
    # Large tensors for leading kernel
    linear_large = te.Linear(4096, 4096, bias=False, params_dtype=torch.bfloat16).to(device)
    x_large = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
    
    fp4_recipe = NVFP4BlockScaling(fp4_format=Format.E2M1)
    
    # Keep autocast context open for warmup and timing
    with te.autocast(enabled=True, recipe=fp4_recipe):
        # Warmup
        for _ in range(num_warmup):
            _ = linear(x)
        torch.cuda.synchronize()
        
        # Timed iterations with leading kernel
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Leading kernel keeps GPU busy while CPU queues the timed kernels
        _ = linear_large(x_large)
        
        start_event.record()
        for _ in range(num_iters):
            _ = linear(x)
        end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iters
    avg_time_s = avg_time_ms / 1000.0
    
    tflops = (flops / avg_time_s) / 1e12
    
    return BenchmarkResult(
        tflops=tflops,
        avg_time_ms=avg_time_ms,
        shape=(M, K, N),
        precision="NVFP4"
    )


def benchmark_te_fp4_prequantized(
    M: int,
    K: int,
    N: int,
    workspace_size: int,
    num_warmup: int = 10,
    num_iters: int = 100
) -> Optional[BenchmarkResult]:
    """Benchmark FP4 GEMM with pre-quantized inputs (Blackwell only, measures raw kernel throughput)."""
    if not TE_AVAILABLE:
        return None
    
    if not is_blackwell_available():
        return None
    
    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)
    
    try:
        # Create quantizer (uses default kFloat4E2M1, but being explicit)
        quantizer = te.NVFP4Quantizer(tex.DType.kFloat4E2M1)
        
        # Create BF16 tensors
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        B = torch.randn(N, K, dtype=torch.bfloat16, device=device)  # Transposed for GEMM
        
        # Pre-quantize to FP4
        A_fp4 = quantizer.quantize(A)
        B_fp4 = quantizer.quantize(B)
        
        # Large tensors for leading kernel
        A_large = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
        B_large = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
        A_large_fp4 = quantizer.quantize(A_large)
        B_large_fp4 = quantizer.quantize(B_large)
        
        # Output buffers
        D = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        D_large = torch.empty(4096, 4096, dtype=torch.bfloat16, device=device)
        
        # Allocate workspace (estimate size)
        workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
        
        # Warmup
        for _ in range(num_warmup):
            tex.generic_gemm(
                A_fp4, False,           # A, transA
                B_fp4, True,            # B, transB
                D,                      # output
                None,                   # quantizer (None = no output quantization)
                tex.DType.kBFloat16,    # output_dtype
                None,                   # bias
                tex.DType.kBFloat16,    # bias_type (unused when bias=None)
                False,                  # gelu
                None,                   # gelu_in
                False,                  # grad
                workspace,              # workspace
                workspace_size,         # workspace_size
                False,                  # accumulate
                False,                  # use_split_accumulator
            )
        torch.cuda.synchronize()
        
        # Timed iterations with leading kernel
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Leading kernel keeps GPU busy while CPU queues the timed kernels
        tex.generic_gemm(
            A_large_fp4, False, B_large_fp4, True, D_large, None,
            tex.DType.kBFloat16, None, tex.DType.kBFloat16,
            False, None, False, workspace, workspace_size, False, False,
        )
        
        start_event.record()
        for _ in range(num_iters):
            tex.generic_gemm(
                A_fp4, False,
                B_fp4, True,
                D,
                None,
                tex.DType.kBFloat16,
                None,
                tex.DType.kBFloat16,
                False,
                None,
                False,
                workspace,
                workspace_size,
                False,
                False,
            )
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        avg_time_ms = elapsed_ms / num_iters
        avg_time_s = avg_time_ms / 1000.0
        
        tflops = (flops / avg_time_s) / 1e12
        
        return BenchmarkResult(
            tflops=tflops,
            avg_time_ms=avg_time_ms,
            shape=(M, K, N),
            precision="NVFP4"
        )
    except Exception as e:
        print(
            f"Warning: FP4 prequantized benchmark failed for shape {M}x{K}x{N}: {e}\n"
            f"  Tip: try increasing --workspace-mb (current={workspace_size / (1024 * 1024):.0f}MB)."
        )
        return None


def get_default_matrix_shapes() -> list[tuple[int, int, int]]:
    """Return default matrix shapes for benchmarking (square matrices)."""
    return [
        (256, 256, 256),
        (512, 512, 512),
        (768, 768, 768),
        (1024, 1024, 1024),
        (1536, 1536, 1536),
        (2048, 2048, 2048),
        (3072, 3072, 3072),
        (4096, 4096, 4096),
        (6144, 6144, 6144),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
    ]


def parse_shapes_arg(shapes_arg: str) -> list[tuple[int, int, int]]:
    """Parse a shapes argument into a list of (M, K, N) tuples.

    Supports either:
    - Square sizes: "1024,2048,4096" -> [(1024,1024,1024), ...]
    - Explicit triplets: "8192x5120x15360,8192x5120x5120"
    """
    items = [s.strip() for s in shapes_arg.split(",") if s.strip()]
    if not items:
        raise ValueError("Empty --shapes argument.")

    shapes: list[tuple[int, int, int]] = []
    for item in items:
        if "x" in item:
            parts = [p.strip() for p in item.lower().split("x")]
            if len(parts) != 3:
                raise ValueError(f"Invalid shape '{item}'. Expected format 'MxKxN'.")
            m, k, n = (int(parts[0]), int(parts[1]), int(parts[2]))
            shapes.append((m, k, n))
        else:
            size = int(item)
            shapes.append((size, size, size))

    return shapes


def warmup_gpu(duration_seconds: float = 5.0):
    """
    Warmup the GPU to stabilize clocks before benchmarking.
    
    Runs sustained matmuls to bring GPU out of idle state and 
    get clocks/thermals to steady state.
    """
    print(f"Warming up GPU for {duration_seconds:.1f} seconds...")
    
    device = torch.device("cuda")
    
    # Use a moderate size that keeps GPU busy without OOM
    M, K, N = 4096, 4096, 4096
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        # Run a batch of matmuls
        for _ in range(10):
            _ = torch.matmul(A, B)
        torch.cuda.synchronize()
    
    # Clear memory
    del A, B
    torch.cuda.empty_cache()
    
    print("GPU warmup complete.\n")


def run_benchmarks(
    shapes: list[tuple[int, int, int]],
    num_warmup: int = 10,
    num_iters: int = 100,
    include_fp8: bool = True,
    include_fp4: bool = True,
    gpu_warmup_seconds: float = 5.0,
    pre_quantize: bool = False,
    workspace_mb: int = 32,
) -> dict[str, list[float]]:
    """Run all benchmarks and return results organized by precision."""
    
    results = {"BF16": [], "MXFP8": [], "NVFP4": []}
    
    # Check hardware capabilities
    has_blackwell = is_blackwell_available()
    has_fp8 = is_fp8_available()
    run_fp8 = include_fp8 and TE_PYTORCH_AVAILABLE and has_fp8
    run_fp4 = include_fp4 and TE_PYTORCH_AVAILABLE and has_blackwell
    workspace_size = int(workspace_mb) * 1024 * 1024
    
    # Print header
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGEMM Benchmark on {gpu_name}")
    major, minor = torch.cuda.get_device_capability()
    print(f"CUDA capability: SM{major}{minor}")
    print(f"Warmup iterations: {num_warmup}, Timed iterations: {num_iters}")
    if pre_quantize:
        print("Mode: Pre-quantized inputs (raw kernel throughput)")
    else:
        print("Mode: Full quantize path (realistic training overhead)")
    if pre_quantize:
        print(f"Workspace: {workspace_mb}MB")
    if (include_fp8 or include_fp4) and not TE_PYTORCH_AVAILABLE:
        msg = "Note: FP8/FP4 requested but transformer_engine.pytorch import failed; skipping FP8/FP4."
        if TE_IMPORT_ERROR:
            msg += f" ImportError: {TE_IMPORT_ERROR}"
        print(msg)
    if pre_quantize and (include_fp8 or include_fp4) and not TE_TORCH_EXT_AVAILABLE:
        msg = (
            "Note: --pre-quantize requires transformer_engine_torch (tex.generic_gemm). "
            "transformer_engine_torch import failed; skipping FP8/FP4 pre-quantized benchmarks."
        )
        if TE_IMPORT_ERROR:
            msg += f" ImportError: {TE_IMPORT_ERROR}"
        print(msg)
    if include_fp8 and TE_AVAILABLE and not has_fp8:
        print("Note: FP8 requested but this GPU does not support FP8 Tensor Cores; skipping FP8.")
    if not has_blackwell and include_fp4:
        print("Note: NVFP4 requires Blackwell (SM100+), skipping FP4 benchmarks")
    
    # Warmup GPU to stabilize clocks
    if gpu_warmup_seconds > 0:
        warmup_gpu(gpu_warmup_seconds)
    
    print("=" * 80)
    
    # Build header dynamically
    header = f"{'Shape':<20} {'BF16 TFLOPS':>14}"
    if run_fp8:
        header += f" {'MXFP8 TFLOPS':>14}"
    if run_fp4:
        header += f" {'NVFP4 TFLOPS':>14}"
    header += f" {'Best Speedup':>12}"
    print(header)
    print("-" * 80)
    
    # Select benchmark functions based on pre_quantize flag
    if pre_quantize:
        # Pre-quantized path uses tex.generic_gemm, which requires the TE torch extension.
        if not TE_TORCH_EXT_AVAILABLE:
            return {"BF16": results["BF16"]}
        fp8_benchmark_fn = lambda m, k, n, nw, ni: benchmark_te_fp8_prequantized(  # noqa: E731
            m, k, n, workspace_size=workspace_size, num_warmup=nw, num_iters=ni
        )
        fp4_benchmark_fn = lambda m, k, n, nw, ni: benchmark_te_fp4_prequantized(  # noqa: E731
            m, k, n, workspace_size=workspace_size, num_warmup=nw, num_iters=ni
        )
    else:
        fp8_benchmark_fn = benchmark_te_fp8
        fp4_benchmark_fn = benchmark_te_fp4
    
    for M, K, N in shapes:
        shape_str = f"{M}x{K}x{N}"
        
        # BF16 benchmark
        bf16_result = benchmark_torch_matmul(M, K, N, torch.bfloat16, num_warmup, num_iters)
        results["BF16"].append(bf16_result.tflops)
        
        row = f"{shape_str:<20} {bf16_result.tflops:>14.1f}"
        best_tflops = bf16_result.tflops
        
        # FP8 benchmark
        if run_fp8:
            fp8_result = fp8_benchmark_fn(M, K, N, num_warmup, num_iters)
            if fp8_result:
                results["MXFP8"].append(fp8_result.tflops)
                row += f" {fp8_result.tflops:>14.1f}"
                best_tflops = max(best_tflops, fp8_result.tflops)
            else:
                results["MXFP8"].append(0)
                row += f" {'N/A':>14}"
        
        # FP4 benchmark (Blackwell only)
        if run_fp4:
            fp4_result = fp4_benchmark_fn(M, K, N, num_warmup, num_iters)
            if fp4_result:
                results["NVFP4"].append(fp4_result.tflops)
                row += f" {fp4_result.tflops:>14.1f}"
                best_tflops = max(best_tflops, fp4_result.tflops)
            else:
                results["NVFP4"].append(0)
                row += f" {'N/A':>14}"
        
        speedup = best_tflops / bf16_result.tflops
        row += f" {speedup:>11.2f}x"
        print(row)
    
    print("=" * 80)
    
    # Remove empty precision results
    results = {k: v for k, v in results.items() if v and any(x > 0 for x in v)}
    
    return results


def create_plot(
    shapes: list[tuple[int, int, int]],
    results: dict[str, list[float]],
    output_path: str = "gemm_benchmark.png",
    title: Optional[str] = None
):
    """Create a bar plot matching the style of the reference image."""
    
    gpu_name = torch.cuda.get_device_name(0)
    if title is None:
        title = f"Absolute Performance Comparison\nMeasured on {gpu_name}"
    
    # Create labels for x-axis
    labels = [f"{m}x{k}x{n}" for m, k, n in shapes]
    x = np.arange(len(labels))
    
    # Determine bar width based on number of kernels
    num_kernels = len(results)
    bar_width = 0.8 / num_kernels
    
    # Color scheme matching the reference plot
    colors = {
        "BF16": "#808080",   # Gray
        "MXFP8": "#4B0082",  # Indigo/Purple
        "NVFP4": "#B22222",  # Firebrick red (for future use)
    }
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot bars for each precision
    for i, (precision, tflops_list) in enumerate(results.items()):
        offset = (i - num_kernels / 2 + 0.5) * bar_width
        color = colors.get(precision, f"C{i}")
        bars = ax.bar(x + offset, tflops_list, bar_width, label=precision, color=color)
    
    # Customize the plot
    ax.set_xlabel("Matrix Shape (MxKxN)", fontsize=12)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    
    # Add legend
    ax.legend(title="Kernel", loc='upper left', fontsize=10)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_path_obj = Path(output_path)
    supported_formats = set(fig.canvas.get_supported_filetypes().keys())
    suffix = output_path_obj.suffix.lower().lstrip(".")
    if suffix not in supported_formats:
        output_path_obj = output_path_obj.with_suffix(".png")
        print(
            f"Warning: Output extension '.{suffix}' is not supported by matplotlib; "
            f"saving to '{output_path_obj}' instead."
        )
    plt.savefig(str(output_path_obj), dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path_obj}")
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="GEMM Benchmarking with TFLOPS measurement and plotting"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="gemm_benchmark.png",
        help="Output path for the plot (default: gemm_benchmark.png)"
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Number of timed iterations (default: 100)"
    )
    parser.add_argument(
        "--gpu-warmup",
        type=float,
        default=5.0,
        help="GPU warmup duration in seconds (default: 5.0, set to 0 to disable)"
    )
    parser.add_argument(
        "--no-fp8",
        action="store_true",
        help="Skip FP8 benchmarks"
    )
    parser.add_argument(
        "--no-fp4",
        action="store_true",
        help="Skip FP4 benchmarks (only available on Blackwell)"
    )
    parser.add_argument(
        "--pre-quantize",
        action="store_true",
        help="Use pre-quantized inputs to measure raw kernel throughput (excludes quantization overhead)"
    )
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=32,
        help="Workspace size in MB for pre-quantized generic_gemm() path (default: 32).",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default=None,
        help=(
            "Comma-separated list of GEMM shapes. Either square sizes like '1024,2048,4096' "
            "or explicit triplets like '8192x5120x15360,8192x5120x5120'."
        ),
    )
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires a GPU.")
        return 1
    
    # Parse custom shapes if provided
    if args.shapes:
        shapes = parse_shapes_arg(args.shapes)
    else:
        shapes = get_default_matrix_shapes()
    
    # Run benchmarks
    results = run_benchmarks(
        shapes=shapes,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
        include_fp8=not args.no_fp8,
        include_fp4=not args.no_fp4,
        gpu_warmup_seconds=args.gpu_warmup,
        pre_quantize=args.pre_quantize,
        workspace_mb=args.workspace_mb,
    )
    
    # Create plot
    create_plot(shapes, results, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())