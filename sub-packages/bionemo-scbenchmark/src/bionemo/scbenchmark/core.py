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


"""Core benchmarking logic - modular and reusable.

This module contains the core benchmarking functionality that can be used
by both the simple and factory-based interfaces. It provides the underlying
infrastructure for measuring performance, handling time-based and batch-based
limits, and collecting comprehensive metrics.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import psutil
import torch

from .common import BenchmarkResult, get_batch_size, get_disk_size
from .protocols import DataloaderProtocol


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


@dataclass
class BenchmarkMetrics:
    """Raw metrics collected during benchmarking.

    This dataclass stores the raw data collected during a benchmark run.
    It contains timing information, memory samples, and counts that are
    used to calculate the final BenchmarkResult.

    Attributes:
        batch_times: List of individual batch processing times
        memory_samples: List of memory usage samples during benchmark
        total_samples: Total number of samples processed
        total_batches: Total number of batches processed
        setup_time: Time taken for setup phase
        warmup_time: Time taken for warmup phase
        iteration_time: Total time spent iterating through batches
    """

    batch_times: List[float]
    memory_samples: List[float]
    total_samples: int
    total_batches: int
    setup_time: float
    warmup_time: float
    iteration_time: float


class BenchmarkRunner:
    """Modular benchmark runner with time-based and batch-based limits.

    This class encapsulates the core benchmarking logic, providing
    methods for warmup phases and actual benchmark execution. It
    supports both time-based and batch-based limits, and handles
    progress reporting and error conditions gracefully.
    """

    def __init__(self, config: BenchmarkConfig):
        """Initialize the benchmark runner.

        Args:
            config: Configuration for the benchmark run
        """
        self.config = config
        self.process = psutil.Process()

    def run_warmup(self, dataloader: DataloaderProtocol) -> float:
        """Run warmup phase and return warmup time.

        This method runs a warmup phase to ensure the system is in a
        steady state before the actual benchmark. It supports both
        batch-based and time-based warmup limits.

        Args:
            dataloader: The dataloader to warmup

        Returns:
            Time taken for warmup phase in seconds
        """
        warmup_start = time.perf_counter()

        if self.config.warmup_batches > 0:
            if self.config.print_progress:
                print(f"🔥 Running {self.config.warmup_batches} warmup batches...")

            try:
                batch_count = 0
                for batch in dataloader:
                    batch_count += 1
                    if batch_count >= self.config.warmup_batches:
                        break
                if self.config.print_progress:
                    print("✅ Warmup completed")
            except Exception as e:
                if self.config.print_progress:
                    print(f"⚠️  Warmup failed: {str(e)}")

        elif self.config.warmup_time_seconds:
            if self.config.print_progress:
                print(f"🔥 Running warmup for {self.config.warmup_time_seconds:.2f}s...")

            try:
                warmup_end_time = warmup_start + self.config.warmup_time_seconds
                for batch in dataloader:
                    if time.perf_counter() >= warmup_end_time:
                        break
                if self.config.print_progress:
                    print("✅ Warmup completed")
            except Exception as e:
                if self.config.print_progress:
                    print(f"⚠️  Warmup failed: {str(e)}")

        return time.perf_counter() - warmup_start

    def run_benchmark(self, dataloader: DataloaderProtocol) -> BenchmarkMetrics:
        """Run the actual benchmark and collect metrics.

        This method executes the main benchmark loop, collecting timing
        and memory metrics for each batch. It respects both time-based
        and batch-based limits, and provides progress reporting.

        Args:
            dataloader: The dataloader to benchmark

        Returns:
            BenchmarkMetrics containing all collected raw data
        """
        batch_times = []
        memory_samples = []
        total_samples = 0
        total_batches = 0

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        iteration_start = time.perf_counter()

        try:
            for epoch in range(self.config.num_epochs):
                if self.config.print_progress:
                    print(f"  Epoch {epoch + 1}/{self.config.num_epochs}")

                epoch_batches = 0
                for batch in dataloader:
                    # Start timing
                    batch_start = time.perf_counter()

                    # Sample memory
                    memory_mb = self.process.memory_info().rss / (1024 * 1024)
                    memory_samples.append(memory_mb)

                    # Determine batch size
                    batch_size = get_batch_size(batch)
                    total_samples += batch_size

                    # End timing
                    batch_time = time.perf_counter() - batch_start
                    batch_times.append(batch_time)

                    total_batches += 1
                    epoch_batches += 1

                    # Check max batches limit
                    if self.config.max_batches and total_batches >= self.config.max_batches:
                        break

                    # Check max time limit
                    if self.config.max_time_seconds:
                        elapsed_time = time.perf_counter() - iteration_start
                        if elapsed_time >= self.config.max_time_seconds:
                            if self.config.print_progress:
                                print(f"    Time limit reached ({elapsed_time:.2f}s)")
                            break

                    # Print progress
                    if self.config.print_progress and total_batches % 10 == 0:
                        elapsed = time.perf_counter() - iteration_start
                        print(f"    Processed {total_batches} batches ({elapsed:.2f}s)")

                # Check limits after epoch
                if self.config.max_batches and total_batches >= self.config.max_batches:
                    break
                if self.config.max_time_seconds:
                    elapsed_time = time.perf_counter() - iteration_start
                    if elapsed_time >= self.config.max_time_seconds:
                        break

            iteration_time = time.perf_counter() - iteration_start

        except Exception as e:
            if self.config.print_progress:
                print(f"❌ Iteration failed: {str(e)}")
            iteration_time = time.perf_counter() - iteration_start

        return BenchmarkMetrics(
            batch_times=batch_times,
            memory_samples=memory_samples,
            total_samples=total_samples,
            total_batches=total_batches,
            setup_time=0.0,  # Will be set by caller
            warmup_time=0.0,  # Will be set by caller
            iteration_time=iteration_time,
        )

    def calculate_results(self, metrics: BenchmarkMetrics, disk_size_mb: float = 0.0) -> BenchmarkResult:
        """Calculate final benchmark results from raw metrics.

        This method processes the raw metrics collected during benchmarking
        to produce the final BenchmarkResult with calculated throughput
        and memory statistics.

        Args:
            metrics: Raw metrics collected during benchmark
            disk_size_mb: Size of data files on disk

        Returns:
            BenchmarkResult with calculated performance metrics
        """
        # Calculate timing metrics
        total_time = metrics.iteration_time
        avg_batch_time = total_time / len(metrics.batch_times) if metrics.batch_times else 0
        samples_per_sec = metrics.total_samples / total_time if total_time > 0 else 0
        batches_per_sec = len(metrics.batch_times) / total_time if total_time > 0 else 0

        # Calculate memory metrics
        peak_memory = max(metrics.memory_samples) if metrics.memory_samples else 0
        avg_memory = sum(metrics.memory_samples) / len(metrics.memory_samples) if metrics.memory_samples else 0

        return BenchmarkResult(
            name=self.config.name,
            disk_size_mb=disk_size_mb,
            setup_time_seconds=metrics.setup_time,
            total_iteration_time_seconds=total_time,
            average_batch_time_seconds=avg_batch_time,
            total_batches=metrics.total_batches,
            total_samples=metrics.total_samples,
            samples_per_second=samples_per_sec,
            batches_per_second=batches_per_sec,
            peak_memory_mb=peak_memory,
            average_memory_mb=avg_memory,
        )


def run_benchmark_with_config(dataloader: DataloaderProtocol, config: BenchmarkConfig) -> BenchmarkResult:
    """Run a benchmark with the given configuration.

    This is the core benchmarking function that can be used by both
    simple and factory-based interfaces. It orchestrates the entire
    benchmark process including setup, warmup, execution, and result
    calculation.

    Args:
        dataloader: The dataloader to benchmark
        config: Configuration for the benchmark run

    Returns:
        BenchmarkResult with comprehensive performance metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {config.name}")
    print(f"{'=' * 60}")

    # Measure disk size
    disk_size_mb = 0.0
    if config.data_path:
        disk_size_mb = get_disk_size(config.data_path)
        print(f"📁 Disk size: {disk_size_mb:.2f} MB")

    # Setup phase
    print("🔧 Starting benchmark...")
    setup_start = time.perf_counter()

    # Create benchmark runner
    runner = BenchmarkRunner(config)

    # Run warmup
    warmup_time = runner.run_warmup(dataloader)

    setup_time = time.perf_counter() - setup_start

    # Run actual benchmark
    if config.max_time_seconds:
        print(f"🏃 Running benchmark for max {config.max_time_seconds:.2f}s...")
    else:
        print(f"🏃 Running benchmark for {config.num_epochs} epoch(s)...")

    metrics = runner.run_benchmark(dataloader)

    # Set timing info
    metrics.setup_time = setup_time
    metrics.warmup_time = warmup_time

    # Calculate final results
    result = runner.calculate_results(metrics, disk_size_mb)

    # Print results
    _print_results(result)

    return result


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
    print(f"   Total time: {result.total_iteration_time_seconds:.4f}s")
    print(f"   Avg batch time: {result.average_batch_time_seconds:.4f}s")

    # Print instantiation metrics if available
    if result.instantiation_metrics:
        print("\n🔧 Instantiation:")
        print(f"   Instantiation time: {result.instantiation_metrics.instantiation_time_seconds:.4f}s")
        print(f"   Memory before: {result.instantiation_metrics.memory_before_mb:.2f} MB")
        print(f"   Memory after: {result.instantiation_metrics.memory_after_mb:.2f} MB")
        print(f"   Memory delta: {result.instantiation_metrics.memory_delta_mb:.2f} MB")
        print(f"   Peak memory during: {result.instantiation_metrics.peak_memory_during_mb:.2f} MB")

    print("\n🚀 Throughput:")
    print(f"   Total batches: {result.total_batches}")
    print(f"   Total samples: {result.total_samples}")
    print(f"   Samples/second: {result.samples_per_second:.2f}")
    print(f"   Batches/second: {result.batches_per_second:.2f}")

    print("\n💾 Memory:")
    print(f"   Peak memory: {result.peak_memory_mb:.2f} MB")
    print(f"   Average memory: {result.average_memory_mb:.2f} MB")
    if result.gpu_memory_mb > 0:
        print(f"   GPU memory: {result.gpu_memory_mb:.2f} MB")

    if result.disk_size_mb > 0:
        print("\n💿 Storage:")
        print(f"   Disk size: {result.disk_size_mb:.2f} MB")
