#!/usr/bin/env python3
"""
GEMM Profiler with Power/Clock Monitoring

Detailed profiling of a specific GEMM size with GPU telemetry to understand
performance characteristics and potential throttling.

Usage:
    python profiler_gemm.py --size 1536 --precision bf16
    python profiler_gemm.py --size 1536 --precision fp8 --pre-quantize
    python profiler_gemm.py --size 1536 --precision fp4 --pre-quantize --with-leading-kernel
"""

import argparse
import time
import threading
import torch
from dataclasses import dataclass, field
from typing import Optional, List
import subprocess
import json

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. Install with: pip install pynvml")

# Optional TE import
try:
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.common.recipe import Format, MXFP8BlockScaling, NVFP4BlockScaling
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("Warning: Transformer Engine not available.")


@dataclass
class GPUTelemetry:
    """Container for GPU telemetry samples."""
    timestamps: List[float] = field(default_factory=list)
    power_watts: List[float] = field(default_factory=list)
    temperature_c: List[int] = field(default_factory=list)
    sm_clock_mhz: List[int] = field(default_factory=list)
    memory_clock_mhz: List[int] = field(default_factory=list)
    gpu_utilization: List[int] = field(default_factory=list)


class GPUMonitor:
    """Background thread for monitoring GPU telemetry."""
    
    def __init__(self, device_id: int = 0, sample_interval_ms: float = 10):
        self.device_id = device_id
        self.sample_interval = sample_interval_ms / 1000.0
        self.telemetry = GPUTelemetry()
        self._running = False
        self._thread = None
        self._handle = None
        
    def start(self):
        if not PYNVML_AVAILABLE:
            print("Warning: pynvml not available, skipping GPU monitoring")
            return
            
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
    def stop(self) -> GPUTelemetry:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if PYNVML_AVAILABLE:
            pynvml.nvmlShutdown()
        return self.telemetry
    
    def _monitor_loop(self):
        start_time = time.perf_counter()
        while self._running:
            try:
                now = time.perf_counter() - start_time
                
                # Power
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                power_w = power_mw / 1000.0
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Clocks
                sm_clock = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_MEM)
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                
                self.telemetry.timestamps.append(now)
                self.telemetry.power_watts.append(power_w)
                self.telemetry.temperature_c.append(temp)
                self.telemetry.sm_clock_mhz.append(sm_clock)
                self.telemetry.memory_clock_mhz.append(mem_clock)
                self.telemetry.gpu_utilization.append(util.gpu)
                
            except Exception as e:
                pass  # Ignore sampling errors
                
            time.sleep(self.sample_interval)


def get_gpu_info():
    """Get current GPU info using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,power.limit,clocks.max.sm,clocks.max.memory', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'name': parts[0],
                'power_limit_w': float(parts[1]),
                'max_sm_clock_mhz': int(parts[2]),
                'max_mem_clock_mhz': int(parts[3]),
            }
    except:
        pass
    return None


def compute_gemm_flops(M: int, K: int, N: int) -> int:
    return 2 * M * N * K


def profile_bf16_gemm(M: int, K: int, N: int, num_warmup: int, num_iters: int, 
                       with_leading_kernel: bool) -> tuple:
    """Profile BF16 GEMM with telemetry."""
    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    
    if with_leading_kernel:
        A_large = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
        B_large = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
    
    # Warmup
    for _ in range(num_warmup):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Start monitoring
    monitor = GPUMonitor(sample_interval_ms=5)
    monitor.start()
    
    # Give monitor a moment to start
    time.sleep(0.01)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    if with_leading_kernel:
        _ = torch.matmul(A_large, B_large)
    
    start_event.record()
    for _ in range(num_iters):
        _ = torch.matmul(A, B)
    end_event.record()
    torch.cuda.synchronize()
    
    # Stop monitoring
    telemetry = monitor.stop()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iters
    tflops = (flops / (avg_time_ms / 1000.0)) / 1e12
    
    return tflops, avg_time_ms, telemetry


def profile_fp8_gemm(M: int, K: int, N: int, num_warmup: int, num_iters: int,
                      with_leading_kernel: bool, pre_quantize: bool) -> tuple:
    """Profile FP8 GEMM with telemetry."""
    if not TE_AVAILABLE:
        return None, None, None
    
    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)
    
    if pre_quantize:
        quantizer = te.MXFP8Quantizer(tex.DType.kFloat8E4M3)
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        B = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        A_fp8 = quantizer.quantize(A)
        B_fp8 = quantizer.quantize(B)
        D = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        workspace_size = 32 * 1024 * 1024
        workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
        
        if with_leading_kernel:
            A_large = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            B_large = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            A_large_fp8 = quantizer.quantize(A_large)
            B_large_fp8 = quantizer.quantize(B_large)
            D_large = torch.empty(4096, 4096, dtype=torch.bfloat16, device=device)
        
        def run_gemm():
            tex.generic_gemm(
                A_fp8, False, B_fp8, True, D, None,
                tex.DType.kBFloat16, None, tex.DType.kBFloat16,
                False, None, False, workspace, workspace_size, False, False,
            )
        
        def run_large_gemm():
            tex.generic_gemm(
                A_large_fp8, False, B_large_fp8, True, D_large, None,
                tex.DType.kBFloat16, None, tex.DType.kBFloat16,
                False, None, False, workspace, workspace_size, False, False,
            )
    else:
        linear = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).to(device)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        
        if with_leading_kernel:
            linear_large = te.Linear(4096, 4096, bias=False, params_dtype=torch.bfloat16).to(device)
            x_large = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
        
        fp8_recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
    
    # Warmup
    if pre_quantize:
        for _ in range(num_warmup):
            run_gemm()
    else:
        with te.autocast(enabled=True, recipe=fp8_recipe):
            for _ in range(num_warmup):
                _ = linear(x)
    torch.cuda.synchronize()
    
    # Start monitoring
    monitor = GPUMonitor(sample_interval_ms=5)
    monitor.start()
    time.sleep(0.01)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    if pre_quantize:
        if with_leading_kernel:
            run_large_gemm()
        
        start_event.record()
        for _ in range(num_iters):
            run_gemm()
        end_event.record()
    else:
        with te.autocast(enabled=True, recipe=fp8_recipe):
            if with_leading_kernel:
                _ = linear_large(x_large)
            
            start_event.record()
            for _ in range(num_iters):
                _ = linear(x)
            end_event.record()
    
    torch.cuda.synchronize()
    telemetry = monitor.stop()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iters
    tflops = (flops / (avg_time_ms / 1000.0)) / 1e12
    
    return tflops, avg_time_ms, telemetry


def profile_fp4_gemm(M: int, K: int, N: int, num_warmup: int, num_iters: int,
                      with_leading_kernel: bool, pre_quantize: bool) -> tuple:
    """Profile FP4 GEMM with telemetry."""
    if not TE_AVAILABLE:
        return None, None, None
    
    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)
    
    if pre_quantize:
        quantizer = te.NVFP4Quantizer(tex.DType.kFloat4E2M1)
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        B = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        A_fp4 = quantizer.quantize(A)
        B_fp4 = quantizer.quantize(B)
        D = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        workspace_size = 32 * 1024 * 1024
        workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
        
        if with_leading_kernel:
            A_large = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            B_large = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            A_large_fp4 = quantizer.quantize(A_large)
            B_large_fp4 = quantizer.quantize(B_large)
            D_large = torch.empty(4096, 4096, dtype=torch.bfloat16, device=device)
        
        def run_gemm():
            tex.generic_gemm(
                A_fp4, False, B_fp4, True, D, None,
                tex.DType.kBFloat16, None, tex.DType.kBFloat16,
                False, None, False, workspace, workspace_size, False, False,
            )
        
        def run_large_gemm():
            tex.generic_gemm(
                A_large_fp4, False, B_large_fp4, True, D_large, None,
                tex.DType.kBFloat16, None, tex.DType.kBFloat16,
                False, None, False, workspace, workspace_size, False, False,
            )
    else:
        linear = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).to(device)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        
        if with_leading_kernel:
            linear_large = te.Linear(4096, 4096, bias=False, params_dtype=torch.bfloat16).to(device)
            x_large = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
        
        fp4_recipe = NVFP4BlockScaling(fp4_format=Format.E2M1)
    
    # Warmup
    if pre_quantize:
        for _ in range(num_warmup):
            run_gemm()
    else:
        with te.autocast(enabled=True, recipe=fp4_recipe):
            for _ in range(num_warmup):
                _ = linear(x)
    torch.cuda.synchronize()
    
    # Start monitoring
    monitor = GPUMonitor(sample_interval_ms=5)
    monitor.start()
    time.sleep(0.01)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    if pre_quantize:
        if with_leading_kernel:
            run_large_gemm()
        
        start_event.record()
        for _ in range(num_iters):
            run_gemm()
        end_event.record()
    else:
        with te.autocast(enabled=True, recipe=fp4_recipe):
            if with_leading_kernel:
                _ = linear_large(x_large)
            
            start_event.record()
            for _ in range(num_iters):
                _ = linear(x)
            end_event.record()
    
    torch.cuda.synchronize()
    telemetry = monitor.stop()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iters
    tflops = (flops / (avg_time_ms / 1000.0)) / 1e12
    
    return tflops, avg_time_ms, telemetry


def print_telemetry_summary(telemetry: GPUTelemetry, gpu_info: dict):
    """Print summary of GPU telemetry."""
    if not telemetry.timestamps:
        print("\nNo telemetry data collected (pynvml not available?)")
        return
    
    print("\n" + "=" * 60)
    print("GPU TELEMETRY SUMMARY")
    print("=" * 60)
    
    # Power
    avg_power = sum(telemetry.power_watts) / len(telemetry.power_watts)
    max_power = max(telemetry.power_watts)
    min_power = min(telemetry.power_watts)
    power_limit = gpu_info.get('power_limit_w', 0) if gpu_info else 0
    print(f"\nPower (W):")
    print(f"  Avg: {avg_power:.1f}  Min: {min_power:.1f}  Max: {max_power:.1f}  Limit: {power_limit:.0f}")
    if power_limit > 0:
        print(f"  Utilization: {100 * avg_power / power_limit:.1f}% of limit")
    
    # Temperature
    avg_temp = sum(telemetry.temperature_c) / len(telemetry.temperature_c)
    max_temp = max(telemetry.temperature_c)
    print(f"\nTemperature (°C):")
    print(f"  Avg: {avg_temp:.0f}  Max: {max_temp:.0f}")
    
    # SM Clock
    avg_sm = sum(telemetry.sm_clock_mhz) / len(telemetry.sm_clock_mhz)
    max_sm = max(telemetry.sm_clock_mhz)
    min_sm = min(telemetry.sm_clock_mhz)
    max_sm_possible = gpu_info.get('max_sm_clock_mhz', 0) if gpu_info else 0
    print(f"\nSM Clock (MHz):")
    print(f"  Avg: {avg_sm:.0f}  Min: {min_sm}  Max: {max_sm}  GPU Max: {max_sm_possible}")
    if max_sm_possible > 0:
        print(f"  Running at: {100 * avg_sm / max_sm_possible:.1f}% of max clock")
    
    # Check for throttling indicators
    print("\n" + "-" * 60)
    print("THROTTLING ANALYSIS:")
    if power_limit > 0 and max_power >= power_limit * 0.95:
        print("  ⚠️  Power usage near limit - possible power throttling")
    else:
        print("  ✓  Power usage below limit")
    
    if max_sm_possible > 0 and avg_sm < max_sm_possible * 0.9:
        print(f"  ⚠️  SM clocks below max ({avg_sm:.0f} vs {max_sm_possible} MHz)")
    else:
        print("  ✓  SM clocks near max")
    
    clock_variance = max_sm - min_sm
    if clock_variance > 100:
        print(f"  ⚠️  Clock variance: {clock_variance} MHz (unstable clocks)")
    else:
        print(f"  ✓  Clock variance: {clock_variance} MHz (stable)")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="GEMM Profiler with GPU Telemetry")
    parser.add_argument("--size", "-s", type=int, default=1536,
                        help="Matrix size (square MxKxN)")
    parser.add_argument("--precision", "-p", choices=['bf16', 'fp8', 'fp4'], default='bf16',
                        help="Precision to benchmark")
    parser.add_argument("--num-warmup", type=int, default=50,
                        help="Warmup iterations")
    parser.add_argument("--num-iters", type=int, default=500,
                        help="Timed iterations")
    parser.add_argument("--pre-quantize", action="store_true",
                        help="Use pre-quantized inputs (FP8/FP4 only)")
    parser.add_argument("--with-leading-kernel", action="store_true",
                        help="Run a large GEMM before the timed kernels")
    parser.add_argument("--compare", action="store_true",
                        help="Run both with and without leading kernel for comparison")
    parser.add_argument("--gpu-warmup", type=float, default=3.0,
                        help="Seconds to warm up GPU before profiling")
    
    args = parser.parse_args()
    
    M = K = N = args.size
    
    # Get GPU info
    gpu_info = get_gpu_info()
    print("\n" + "=" * 70)
    print("GEMM PROFILER")
    print("=" * 70)
    if gpu_info:
        print(f"GPU: {gpu_info['name']}")
        print(f"Power Limit: {gpu_info['power_limit_w']:.0f}W")
        print(f"Max SM Clock: {gpu_info['max_sm_clock_mhz']} MHz")
    
    print(f"\nConfiguration:")
    print(f"  Shape: {M}x{K}x{N}")
    print(f"  Precision: {args.precision.upper()}")
    print(f"  Iterations: {args.num_warmup} warmup + {args.num_iters} timed")
    print(f"  Pre-quantize: {args.pre_quantize}")
    
    # GPU warmup
    if args.gpu_warmup > 0:
        print(f"\nWarming up GPU for {args.gpu_warmup:.1f} seconds...")
        device = torch.device("cuda")
        warmup_a = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
        warmup_b = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
        start = time.time()
        while time.time() - start < args.gpu_warmup:
            _ = torch.matmul(warmup_a, warmup_b)
        torch.cuda.synchronize()
        del warmup_a, warmup_b
        torch.cuda.empty_cache()
        print("GPU warmup complete.")
    
    if args.compare:
        # Run both configurations
        configs = [
            ("Without leading kernel", False),
            ("With leading kernel", True),
        ]
    else:
        configs = [(
            "With leading kernel" if args.with_leading_kernel else "Without leading kernel",
            args.with_leading_kernel
        )]
    
    results = []
    
    for config_name, use_leading in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {config_name}")
        print('='*60)
        
        if args.precision == 'bf16':
            tflops, avg_ms, telemetry = profile_bf16_gemm(
                M, K, N, args.num_warmup, args.num_iters, use_leading
            )
        elif args.precision == 'fp8':
            tflops, avg_ms, telemetry = profile_fp8_gemm(
                M, K, N, args.num_warmup, args.num_iters, use_leading, args.pre_quantize
            )
        elif args.precision == 'fp4':
            tflops, avg_ms, telemetry = profile_fp4_gemm(
                M, K, N, args.num_warmup, args.num_iters, use_leading, args.pre_quantize
            )
        
        if tflops is not None:
            print(f"\nResults:")
            print(f"  TFLOPS: {tflops:.1f}")
            print(f"  Avg time: {avg_ms:.4f} ms")
            print_telemetry_summary(telemetry, gpu_info)
            results.append((config_name, tflops, avg_ms))
        else:
            print("Benchmark failed or not available")
    
    # Print comparison summary
    if len(results) == 2:
        print(f"\n{'=' * 60}")
        print("COMPARISON SUMMARY")
        print("=" * 60)
        name1, tflops1, ms1 = results[0]
        name2, tflops2, ms2 = results[1]
        
        print(f"\n  {name1}:")
        print(f"    {tflops1:.1f} TFLOPS, {ms1:.4f} ms")
        print(f"\n  {name2}:")
        print(f"    {tflops2:.1f} TFLOPS, {ms2:.4f} ms")
        
        diff_pct = 100 * (tflops2 - tflops1) / tflops1
        print(f"\n  Difference: {diff_pct:+.1f}%")
        
        if diff_pct < -5:
            print(f"\n  ⚠️  Leading kernel hurts performance")
            print(f"     Likely cause: power/thermal throttling from the leading kernel")
        elif diff_pct > 5:
            print(f"\n  ✓  Leading kernel helps performance")
            print(f"     Without it, CPU dispatch overhead was being measured")
        else:
            print(f"\n  ~  Minimal difference")
            print(f"     CPU dispatch overhead is not significant for this size")


if __name__ == "__main__":
    main()