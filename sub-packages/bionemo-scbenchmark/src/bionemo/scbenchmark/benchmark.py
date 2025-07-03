# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Benchmarking framework for any dataloader.

This module provides a comprehensive framework for benchmarking any dataloader
implementation. It supports both simple direct benchmarking and factory-based
benchmarking, with features like time-based limits, warmup phases, instantiation
measurement, and comprehensive performance metrics.
"""

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from tqdm import tqdm

from bionemo.scbenchmark.common import BenchmarkResult, get_batch_size, get_disk_size, measure_peak_memory_full


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking.

    This dataclass contains all the configuration parameters needed
    to run a benchmark. It supports both time-based and batch-based
    limits, as well as warmup phases.

    Attributes:
        name: Name of the benchmark
        num_epochs: Number of epochs to run
        max_batches: Maximum number of batches to process (None for all)
        max_time_seconds: Maximum time to run benchmark (None for no limit)
        warmup_batches: Number of warmup batches
        warmup_time_seconds: Time to warmup in seconds (overrides warmup_batches if set)
        print_progress: Whether to print progress during benchmarking
        data_path: Path to data files (for disk size measurement)
    """

    name: str
    num_epochs: int = 1
    max_batches: Optional[int] = None
    max_time_seconds: Optional[float] = None
    warmup_batches: int = 5
    warmup_time_seconds: Optional[float] = None
    print_progress: bool = True
    data_path: Optional[Union[str, Path]] = None


def run_benchmark(dataloader: Any, config: BenchmarkConfig) -> BenchmarkResult:
    """Run the actual benchmark and collect metrics.

    This method executes the main benchmark loop, collecting timing
    and memory metrics for each batch. It respects both time-based
    and batch-based limits, and provides progress reporting.

    Args:
        dataloader: The dataloader to benchmark
        config: Configuration for the benchmark run

    Returns:
        BenchmarkResult containing all collected data and calculated metrics
    """
    # Use measure_peak_memory_full to get memory info during benchmark
    gc.collect()

    def benchmark_iteration():
        gc.collect()

        total_samples = 0
        total_batches = 0
        pbar = tqdm(desc=f"{config.name} (for {config.max_time_seconds})")

        # Initialize warm-up timer - only if warmup_time_seconds is set
        if config.warmup_time_seconds is not None and config.warmup_time_seconds > 0:
            warm_up_start = time.perf_counter()
            warm_up_end = warm_up_start + config.warmup_time_seconds
            is_warming_up = True
            start_time = None  # Will be set after warmup
            end_time = None  # Will be set after warmup
        else:
            # No warmup - start timing immediately
            is_warming_up = False
            warm_up_start = None
            warm_up_end = None
            start_time = time.perf_counter()
            end_time = start_time + config.max_time_seconds

        for batch in dataloader:
            batch_size = get_batch_size(batch)

            current_time = time.perf_counter()

            if is_warming_up:
                # We're in warm-up period
                if current_time >= warm_up_end:
                    # Warm-up complete, start the actual timing
                    is_warming_up = False
                    total_samples = 0
                    start_time = time.perf_counter()
                    end_time = start_time + config.max_time_seconds
                    pbar.set_description(
                        f"{config.name} (warming up complete, testing for {config.max_time_seconds}s)"
                    )
                else:
                    pbar.set_description(
                        f"{config.name} (warming up: {current_time - warm_up_start:.1f}/{config.warmup_time_seconds}s)"
                    )
                    pbar.update(1)
                    continue

            # Now we're past the warm-up period
            total_samples += batch_size
            total_batches += 1
            elapsed = current_time - start_time
            pbar.set_postfix(samples=total_samples, elapsed=f"{elapsed:.2f}s")
            pbar.update(1)
            del batch
            gc.collect()

            if current_time >= end_time:
                break
        pbar.close()
        print("TIME", current_time - start_time)

        return total_samples, total_batches

    (total_samples, total_batches), baseline, peak, _, _, iteration_time = measure_peak_memory_full(
        benchmark_iteration
    )
    result = BenchmarkResult.from_raw_metrics(
        name=config.name,
        batch_times=[],  # Not collecting individual batch times
        memory_samples=[],  # Not collecting individual memory samples
        total_samples=total_samples,
        total_batches=total_batches,
        setup_time=0,  # Will be set by caller
        warmup_time=0.0,  # Will be set by caller
        iteration_time=iteration_time,
    )

    # Update with memory information from measure_peak_memory_full
    result.memory_before_instantiation_mb = baseline
    result.peak_memory_mb = peak

    return result


def benchmark_dataloader(
    name: str,
    dataloader_factory: Callable[[], Any],
    data_path: Optional[Union[str, Path]] = None,
    num_epochs: int = 1,
    max_batches: Optional[int] = None,
    max_time_seconds: Optional[float] = None,
    warmup_batches: int = 5,
    warmup_time_seconds: Optional[float] = None,
    print_progress: bool = True,
) -> BenchmarkResult:
    """Benchmark any dataloader using a factory function.

    This function measures both instantiation time/memory and iteration performance
    to give you a complete picture of dataloader performance. It uses a factory
    function to create the dataloader, allowing for flexible benchmarking without
    requiring inheritance or modifications to existing code.

    Args:
        name: Name for this dataloader benchmark
        dataloader_factory: Function that creates a dataloader object
        data_path: Path to data files (for disk size measurement)
        num_epochs: Number of epochs to run
        max_batches: Maximum number of batches to process (None for all)
        max_time_seconds: Maximum time to run benchmark (None for no limit)
        warmup_batches: Number of warmup batches
        warmup_time_seconds: Time to warmup in seconds (overrides warmup_batches if set)
        print_progress: Whether to print progress during benchmarking

    Returns:
        BenchmarkResult with performance metrics including instantiation

    Note:
        - If both max_batches and max_time_seconds are set, the first limit reached stops the benchmark
        - If both warmup_batches and warmup_time_seconds are set, warmup_time_seconds takes precedence
        - Instantiation is measured by default to give complete performance picture
        - The factory function is called once for instantiation measurement and once for setup
        - For existing dataloaders, simply wrap them in a factory function: lambda: your_dataloader
    """

    def log(message):
        if print_progress:
            print(message)

    log(f"\n{'=' * 60}")
    log(f"Benchmarking: {name}")
    log(f"{'=' * 60}")

    # Measure instantiation metrics if requested
    log("🔧 Measuring dataloader instantiation...")
    dataloader, baseline, peak, _, _, setup_time = measure_peak_memory_full(dataloader_factory)

    # Measure memory after

    instantiation_metrics = {
        "instantiation_time_seconds": setup_time,
        "peak_memory_during_instantiation_mb": peak,
        "memory_before_instantiation_mb": baseline,
    }

    log(f"✅ Setup completed in {setup_time:.4f}s")

    log(f"   Peak memory during: {instantiation_metrics['peak_memory_during_instantiation_mb']:.2f} MB")

    # Measure disk size
    disk_size_mb = 0.0
    if data_path:
        disk_size_mb = get_disk_size(data_path)
        log(f"📁 Disk size: {disk_size_mb:.2f} MB")

    # Create config and run benchmark
    config = BenchmarkConfig(
        name=name,
        num_epochs=num_epochs,
        max_batches=max_batches,
        max_time_seconds=max_time_seconds,
        warmup_batches=warmup_batches,
        warmup_time_seconds=warmup_time_seconds,
        print_progress=print_progress,
        data_path=data_path,
    )
    # Run the benchmark directly
    log("🏃 Running benchmark...")
    result = run_benchmark(dataloader, config)
    print("RESULT")
    print(result)
    del dataloader
    # Add instantiation metrics
    if instantiation_metrics:
        for key, value in instantiation_metrics.items():
            setattr(result, key, value)

    # Update timing and disk info
    result.setup_time_seconds += setup_time
    result.disk_size_mb = disk_size_mb
    # Print results
    _print_results(result)

    return result


def benchmark_multiple_dataloaders(
    dataloaders: List[Dict[str, Any]], output_dir: Optional[str] = None
) -> List[BenchmarkResult]:
    """Benchmark multiple dataloaders and compare results.

    This function benchmarks multiple dataloaders using their factory functions
    and provides a comparison of their performance. Results can be saved to
    individual JSON files for later analysis.

    Args:
        dataloaders: List of dicts with keys: name, dataloader_factory, data_path, etc.
                    The 'dataloader_factory' key should contain a Callable[[], Any]
                    Other keys are passed to benchmark_dataloader as kwargs
        output_dir: Directory to save individual results (optional)

    Returns:
        List of BenchmarkResult objects for each dataloader

    Note:
        - Each dataloader configuration should have at least 'name' and 'dataloader_factory' keys
        - If output_dir is provided, results are saved as JSON files
        - A comparison table is printed showing the best performers
    """
    results = []

    for dl_config in dataloaders:
        result = benchmark_dataloader(
            name=dl_config["name"],
            dataloader_factory=dl_config["dataloader_factory"],
            data_path=dl_config.get("data_path"),
            num_epochs=dl_config.get("num_epochs", 1),
            max_batches=dl_config.get("max_batches"),
            max_time_seconds=dl_config.get("max_time_seconds"),
            warmup_batches=dl_config.get("warmup_batches", 5),
            warmup_time_seconds=dl_config.get("warmup_time_seconds"),
            print_progress=dl_config.get("print_progress", True),
        )
        results.append(result)

        # Save individual result
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            result.save_to_file(os.path.join(output_dir, f"{result.name}_results.json"))

    # Print comparison
    if len(results) > 1:
        _print_comparison(results)

    return results


def _print_results(result: BenchmarkResult) -> None:
    """Print benchmark results in a formatted way.

    This function displays the benchmark results in a human-readable
    format, including performance metrics, memory usage, and any
    instantiation information if available.

    Args:
        result: BenchmarkResult to display
    """
    print(f"\n{'=' * 40}")
    print(f"RESULTS: {result.name}")
    print(f"{'=' * 40}")

    if result.errors:
        print(f"❌ ERRORS: {result.errors}")
        return

    print("📊 Performance:")
    print(f"   Setup time: {result.setup_time_seconds:.4f}s")
    print(f"   Warmup time: {result.warmup_time_seconds:.4f}s")
    print(f"   Total time: {result.total_iteration_time_seconds:.4f}s")
    print(f"   Avg batch time: {result.average_batch_time_seconds:.4f}s")

    # Print instantiation metrics if available
    if result.instantiation_time_seconds is not None:
        print("\n🔧 Instantiation:")
        print(f"   Instantiation time: {result.instantiation_time_seconds:.4f}s")
        print(f"   Peak memory during: {result.peak_memory_during_instantiation_mb:.2f} MB")

    print("\n🚀 Throughput:")
    print(f"   Total time: {result.total_iteration_time_seconds:.4f}s")
    print(f"   Total batches: {result.total_batches}")
    print(f"   Total samples: {result.total_samples}")
    print(f"   Samples/second: {result.samples_per_second:.2f}")
    print(f"   Batches/second: {result.batches_per_second:.2f}")

    print("\n💾 Memory:")
    print(f"   Peak memory: {(result.peak_memory_mb - result.memory_before_instantiation_mb):.2f} MB")
    print(f"   Average memory: {(result.peak_memory_mb - result.memory_before_instantiation_mb):.2f} MB")
    if result.gpu_memory_mb > 0:
        print(f"   GPU memory: {result.gpu_memory_mb:.2f} MB")

    if result.disk_size_mb > 0:
        print("\n💿 Storage:")
        print(f"   Disk size: {result.disk_size_mb:.2f} MB")


def _print_comparison(results: List[BenchmarkResult]) -> None:
    """Print comparison of multiple benchmark results.

    This function analyzes the results and prints a comparison table
    showing the best performers in different categories.

    Args:
        results: List of BenchmarkResult objects to compare
    """
    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPARISON")
    print(f"{'=' * 80}")

    # Find best performers
    valid_results = [r for r in results if not r.errors]
    if not valid_results:
        print("❌ No valid results to compare")
        return

    best_samples_per_sec = max(valid_results, key=lambda x: x.samples_per_second)
    best_batches_per_sec = max(valid_results, key=lambda x: x.batches_per_second)
    lowest_memory = min(valid_results, key=lambda x: x.peak_memory_mb)

    print(f"🏆 Best samples/second: {best_samples_per_sec.name} ({best_samples_per_sec.samples_per_second:.2f})")
    print(f"🏆 Best batches/second: {best_batches_per_sec.name} ({best_batches_per_sec.batches_per_second:.2f})")
    print(f"🏆 Lowest memory: {lowest_memory.name} ({lowest_memory.peak_memory_mb:.2f} MB)")

    # Check if we have instantiation metrics
    has_instantiation = any(r.instantiation_time_seconds is not None for r in valid_results)

    if has_instantiation:
        fastest_instantiation = min(
            [r for r in valid_results if r.instantiation_time_seconds is not None],
            key=lambda x: x.instantiation_time_seconds,
        )
        lowest_instantiation_memory = min(
            [r for r in valid_results if r.instantiation_time_seconds is not None],
            key=lambda x: x.peak_memory_during_instantiation_mb,
        )

        print(
            f"🏆 Fastest instantiation: {fastest_instantiation.name} ({fastest_instantiation.instantiation_time_seconds:.4f}s)"
        )
        print(
            f"🏆 Lowest instantiation memory: {lowest_instantiation_memory.name} ({lowest_instantiation_memory.peak_memory_during_instantiation_mb:.2f} MB)"
        )

    # Print detailed comparison table
    print(f"\n{'=' * 80}")
    print("DETAILED COMPARISON")
    print(f"{'=' * 80}")
    print(f"{'Name':<30} {'Samples/sec':<12} {'Batches/sec':<12} {'Peak Mem':<10} {'Inst Time':<10} {'Inst Mem':<10}")
    print("-" * 80)
    """
    for result in valid_results:
        inst_time = result.instantiation_time_seconds if result.instantiation_time_seconds is not None else 0.0
        inst_mem = result.memory_before_instantiation_mb if result.memory_before_instantiation_mb is not None else 0.0

        print(
            f"{result.name:<30} {result.samples_per_second:<12.2f} {result.batches_per_second:<12.2f} "
            f"{result.peak_memory_mb:<10.2f} {inst_time:<10.4f} {inst_mem:<10.2f}"
        )
    """
