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


"""Simple benchmarking for any dataloader.

This module provides functions to benchmark any dataloader implementation
by measuring disk space, iteration time, memory usage, and throughput.
The module uses factory functions to create dataloaders, allowing for
flexible benchmarking without requiring inheritance or modifications
to existing dataloader code.
"""

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .common import BenchmarkResult, measure_instantiation
from .core import BenchmarkConfig, run_benchmark_with_config
from .protocols import DataloaderProtocol


def benchmark_dataloader(
    name: str,
    dataloader_factory: Callable[[], DataloaderProtocol],
    data_path: Optional[Union[str, Path]] = None,
    num_epochs: int = 1,
    max_batches: Optional[int] = None,
    max_time_seconds: Optional[float] = None,
    warmup_batches: int = 5,
    warmup_time_seconds: Optional[float] = None,
    print_progress: bool = True,
    measure_instantiation_metrics: bool = True,
) -> BenchmarkResult:
    """Benchmark any dataloader using a factory function.

    This function measures both instantiation time/memory and iteration performance
    to give you a complete picture of dataloader performance. It uses a factory
    function to create the dataloader, allowing for flexible benchmarking without
    requiring inheritance or modifications to existing code.

    Args:
        name: Name for this dataloader benchmark
        dataloader_factory: Function that creates a DataloaderProtocol object
        data_path: Path to data files (for disk size measurement)
        num_epochs: Number of epochs to run
        max_batches: Maximum number of batches to process (None for all)
        max_time_seconds: Maximum time to run benchmark (None for no limit)
        warmup_batches: Number of warmup batches
        warmup_time_seconds: Time to warmup in seconds (overrides warmup_batches if set)
        print_progress: Whether to print progress during benchmarking
        measure_instantiation_metrics: Whether to measure instantiation (default: True)

    Returns:
        BenchmarkResult with performance metrics including instantiation

    Note:
        - If both max_batches and max_time_seconds are set, the first limit reached stops the benchmark
        - If both warmup_batches and warmup_time_seconds are set, warmup_time_seconds takes precedence
        - Instantiation is measured by default to give complete performance picture
        - The factory function is called once for instantiation measurement and once for setup
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {name}")
    print(f"{'=' * 60}")

    # Measure instantiation metrics if requested
    instantiation_metrics = None
    if measure_instantiation_metrics:
        print("🔧 Measuring dataloader instantiation...")
        instantiation_metrics = measure_instantiation(dataloader_factory, name)

        print(f"   Instantiation time: {instantiation_metrics.instantiation_time_seconds:.4f}s")
        print(f"   Memory delta: {instantiation_metrics.memory_delta_mb:.2f} MB")
        print(f"   Peak memory during: {instantiation_metrics.peak_memory_during_mb:.2f} MB")

    # Setup phase (includes factory execution)
    print("🔧 Setting up dataloader...")
    setup_start = time.perf_counter()
    try:
        dataloader = dataloader_factory()
        setup_time = time.perf_counter() - setup_start
        print(f"✅ Setup completed in {setup_time:.4f}s")
    except Exception as e:
        error_msg = f"Setup failed: {str(e)}"
        print(f"❌ {error_msg}")
        return BenchmarkResult(
            name=name,
            disk_size_mb=0.0,
            setup_time_seconds=0.0,
            total_iteration_time_seconds=0.0,
            average_batch_time_seconds=0.0,
            total_batches=0,
            total_samples=0,
            samples_per_second=0.0,
            batches_per_second=0.0,
            peak_memory_mb=0.0,
            average_memory_mb=0.0,
            errors=[error_msg],
            instantiation_metrics=instantiation_metrics,
        )

    # Create config for core benchmarking
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

    # Run the benchmark using core module
    result = run_benchmark_with_config(dataloader, config)

    # Add instantiation metrics
    result.instantiation_metrics = instantiation_metrics

    # Adjust setup time to include factory execution
    result.setup_time_seconds += setup_time

    return result


def benchmark_multiple_dataloaders(
    dataloaders: List[Dict[str, Any]], output_dir: Optional[str] = None
) -> List[BenchmarkResult]:
    """Benchmark multiple dataloaders and compare results.

    This function benchmarks multiple dataloaders using their factory functions
    and provides a comparison of their performance. Results can be saved to
    individual JSON files for later analysis.

    Args:
        dataloaders: List of dicts with keys: name, factory, data_path, etc.
                    The 'factory' key should contain a Callable[[], DataloaderProtocol]
                    Other keys are passed to benchmark_dataloader as kwargs
        output_dir: Directory to save individual results (optional)

    Returns:
        List of BenchmarkResult objects for each dataloader

    Note:
        - Each dataloader configuration should have at least 'name' and 'factory' keys
        - If output_dir is provided, results are saved as JSON files
        - A comparison table is printed showing the best performers
    """
    results = []

    for dl_config in dataloaders:
        result = benchmark_dataloader(
            name=dl_config["name"],
            dataloader_factory=dl_config["factory"],
            data_path=dl_config.get("data_path"),
            num_epochs=dl_config.get("num_epochs", 1),
            max_batches=dl_config.get("max_batches"),
            max_time_seconds=dl_config.get("max_time_seconds"),
            warmup_batches=dl_config.get("warmup_batches", 5),
            warmup_time_seconds=dl_config.get("warmup_time_seconds"),
            print_progress=dl_config.get("print_progress", True),
            measure_instantiation_metrics=dl_config.get("measure_instantiation_metrics", True),
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
    has_instantiation = any(r.instantiation_metrics for r in valid_results)

    if has_instantiation:
        fastest_instantiation = min(
            [r for r in valid_results if r.instantiation_metrics],
            key=lambda x: x.instantiation_metrics.instantiation_time_seconds,
        )
        lowest_instantiation_memory = min(
            [r for r in valid_results if r.instantiation_metrics],
            key=lambda x: x.instantiation_metrics.memory_delta_mb,
        )
        print(
            f"🏆 Fastest instantiation: {fastest_instantiation.name} ({fastest_instantiation.instantiation_metrics.instantiation_time_seconds:.4f}s)"
        )
        print(
            f"🏆 Lowest instantiation memory: {lowest_instantiation_memory.name} ({lowest_instantiation_memory.instantiation_metrics.memory_delta_mb:.2f} MB)"
        )

    # Comparison table
    if has_instantiation:
        print(
            f"\n{'Name':<20} {'Samples/s':<12} {'Batches/s':<12} {'Memory (MB)':<12} {'Inst. Time (s)':<15} {'Inst. Mem (MB)':<15}"
        )
        print("-" * 100)

        for result in results:
            if not result.errors:
                inst_time = (
                    result.instantiation_metrics.instantiation_time_seconds if result.instantiation_metrics else 0.0
                )
                inst_mem = result.instantiation_metrics.memory_delta_mb if result.instantiation_metrics else 0.0
                print(
                    f"{result.name:<20} "
                    f"{result.samples_per_second:<12.2f} "
                    f"{result.batches_per_second:<12.2f} "
                    f"{result.peak_memory_mb:<12.2f} "
                    f"{inst_time:<15.4f} "
                    f"{inst_mem:<15.2f}"
                )
            else:
                print(f"{result.name:<20} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<15} {'ERROR':<15}")
    else:
        print(f"\n{'Name':<20} {'Samples/s':<12} {'Batches/s':<12} {'Memory (MB)':<12} {'Disk (MB)':<12}")
        print("-" * 80)

        for result in results:
            if not result.errors:
                print(
                    f"{result.name:<20} "
                    f"{result.samples_per_second:<12.2f} "
                    f"{result.batches_per_second:<12.2f} "
                    f"{result.peak_memory_mb:<12.2f} "
                    f"{result.disk_size_mb:<12.2f}"
                )
            else:
                print(f"{result.name:<20} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")
