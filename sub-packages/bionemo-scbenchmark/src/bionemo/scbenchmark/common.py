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


"""Common components shared across the benchmarking framework.

This module contains shared dataclasses and utility functions to avoid
circular imports between modules. It provides the core data structures
and measurement utilities used throughout the benchmarking framework.
"""

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil


@dataclass
class BenchmarkResult:
    """Results from benchmarking a dataloader.

    This dataclass contains all the performance metrics collected during
    a benchmark run. It provides comprehensive information about timing,
    throughput, memory usage, and any errors that occurred.

    The class contains both raw data (for detailed analysis) and calculated
    metrics (for easy consumption). Raw data includes individual batch times
    and memory samples, while calculated metrics include averages and totals.

    Attributes:
        name: Name of the benchmarked dataloader
        disk_size_mb: Size of data files on disk
        setup_time_seconds: Time taken for setup phase
        warmup_time_seconds: Time taken for warmup phase
        total_iteration_time_seconds: Total time spent iterating
        average_batch_time_seconds: Average time per batch
        total_batches: Total number of batches processed
        total_samples: Total number of samples processed
        samples_per_second: Throughput in samples per second
        batches_per_second: Throughput in batches per second
        peak_memory_mb: Peak memory usage during benchmark
        average_memory_mb: Average memory usage during benchmark
        gpu_memory_mb: GPU memory usage (if available)
        errors: List of error messages encountered

        # Instantiation metrics
        instantiation_time_seconds: Time taken to instantiate the dataloader
        memory_before_instantiation_mb: Memory usage before instantiation
        memory_after_instantiation_mb: Memory usage after instantiation
        memory_delta_instantiation_mb: Change in memory usage during instantiation
        peak_memory_during_instantiation_mb: Peak memory usage during instantiation

        # Raw data for detailed analysis
        batch_times: List of individual batch processing times
        memory_samples: List of memory usage samples during benchmark
    """

    name: str
    disk_size_mb: float
    setup_time_seconds: float
    warmup_time_seconds: float
    total_iteration_time_seconds: float
    average_batch_time_seconds: float
    total_batches: int
    total_samples: int
    samples_per_second: float
    batches_per_second: float
    peak_memory_mb: float
    average_memory_mb: float
    gpu_memory_mb: float = 0.0
    errors: List[str] = None

    # Instantiation metrics (None if not measured)
    instantiation_time_seconds: Optional[float] = None
    memory_before_instantiation_mb: Optional[float] = None
    memory_after_instantiation_mb: Optional[float] = None
    memory_delta_instantiation_mb: Optional[float] = None
    peak_memory_during_instantiation_mb: Optional[float] = None

    # Raw data for detailed analysis
    batch_times: List[float] = None
    memory_samples: List[float] = None

    def __post_init__(self):
        """Initialize lists if not provided."""
        if self.errors is None:
            self.errors = []
        if self.batch_times is None:
            self.batch_times = []
        if self.memory_samples is None:
            self.memory_samples = []

    @classmethod
    def from_raw_metrics(
        cls,
        name: str,
        batch_times: List[float],
        memory_samples: List[float],
        total_samples: int,
        total_batches: int,
        setup_time: float,
        warmup_time: float,
        iteration_time: float,
        disk_size_mb: float = 0.0,
        gpu_memory_mb: float = 0.0,
        instantiation_metrics: Optional[Dict[str, float]] = None,
    ) -> "BenchmarkResult":
        """Create BenchmarkResult from raw metrics.

        This factory method calculates all the derived metrics from raw data
        and creates a complete BenchmarkResult object.

        Args:
            name: Name of the benchmark
            batch_times: List of individual batch processing times
            memory_samples: List of memory usage samples
            total_samples: Total number of samples processed
            total_batches: Total number of batches processed
            setup_time: Time taken for setup phase
            warmup_time: Time taken for warmup phase
            iteration_time: Total time spent iterating
            disk_size_mb: Size of data files on disk
            gpu_memory_mb: GPU memory usage
            instantiation_metrics: Instantiation metrics dictionary if available

        Returns:
            BenchmarkResult with all calculated metrics
        """
        # Calculate timing metrics
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        samples_per_sec = total_samples / iteration_time if iteration_time > 0 else 0
        batches_per_sec = len(batch_times) / iteration_time if iteration_time > 0 else 0

        # Calculate memory metrics
        peak_memory = max(memory_samples) if memory_samples else 0
        avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else 0

        # Extract instantiation metrics if provided
        instantiation_kwargs = {}
        if instantiation_metrics:
            instantiation_kwargs = instantiation_metrics

        return cls(
            name=name,
            disk_size_mb=disk_size_mb,
            setup_time_seconds=setup_time,
            warmup_time_seconds=warmup_time,
            total_iteration_time_seconds=iteration_time,
            average_batch_time_seconds=avg_batch_time,
            total_batches=total_batches,
            total_samples=total_samples,
            samples_per_second=samples_per_sec,
            batches_per_second=batches_per_sec,
            peak_memory_mb=peak_memory,
            average_memory_mb=avg_memory,
            gpu_memory_mb=gpu_memory_mb,
            batch_times=batch_times,
            memory_samples=memory_samples,
            **instantiation_kwargs,
        )

    def save_to_file(self, filepath: str) -> None:
        """Save results to JSON file.

        Args:
            filepath: Path to the output JSON file
        """
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "BenchmarkResult":
        """Load results from JSON file.

        Args:
            filepath: Path to the input JSON file

        Returns:
            BenchmarkResult object loaded from the file
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        return cls(**data)


def get_disk_size(path: Union[str, Path]) -> float:
    """Get disk size of a file or directory in MB.

    This function calculates the total size of a file or directory
    on disk. For directories, it recursively sums the sizes of all
    files within the directory.

    Args:
        path: Path to file or directory to measure

    Returns:
        Size in megabytes (0.0 if path doesn't exist or error occurs)
    """
    result = subprocess.run(["du", "-sb", path], stdout=subprocess.PIPE, text=True)
    size_in_bytes = int(result.stdout.split()[0])
    return size_in_bytes / (1024 * 1024)


def get_batch_size(batch: Any) -> int:
    """Determine the size of a batch.

    This function attempts to determine the batch size from various
    common batch formats including PyTorch tensors, lists, and
    dictionaries with common keys.

    Args:
        batch: The batch object to measure

    Returns:
        Number of samples in the batch (defaults to 1 if unknown)
    """
    # Try common batch size attributes
    if hasattr(batch, "__len__"):
        return len(batch)

    # For PyTorch tensors
    if hasattr(batch, "shape") and len(batch.shape) > 0:
        return batch.shape[0]

    # For dictionaries with common keys
    if isinstance(batch, dict):
        for key in ["input_ids", "labels", "data", "features", "x", "y"]:
            if key in batch and hasattr(batch[key], "__len__"):
                return len(batch[key])

    # Default to 1 if we can't determine
    return 1


def measure_instantiation(dataloader_factory: callable, name: str = "Unknown") -> Dict[str, float]:
    """Measure the time and memory usage of instantiating a dataloader.

    This function measures the overhead of creating a dataloader by
    calling the factory function and monitoring timing and memory
    usage before, during, and after instantiation.

    Args:
        dataloader_factory: Function that creates the dataloader
        name: Name for logging and error reporting

    Returns:
        Dictionary with instantiation metrics (time and memory information)

    Note:
        - Memory is measured using psutil for cross-platform compatibility
        - If instantiation fails, metrics are still returned with error information
        - Peak memory during instantiation is tracked
    """
    import time

    process = psutil.Process()

    # Measure memory before
    memory_before = process.memory_info().rss / (1024 * 1024)
    peak_memory_during = memory_before

    # Start timing
    start_time = time.perf_counter()

    try:
        # Create the dataloader
        dataloader_factory()

        # Measure peak memory during instantiation
        peak_memory = process.memory_info().rss / (1024 * 1024)
        peak_memory_during = max(peak_memory_during, peak_memory)

        instantiation_time = time.perf_counter() - start_time

        # Measure memory after
        memory_after = process.memory_info().rss / (1024 * 1024)
        memory_delta = memory_after - memory_before

        return {
            "instantiation_time_seconds": instantiation_time,
            "memory_before_instantiation_mb": memory_before,
            "memory_after_instantiation_mb": memory_after,
            "memory_delta_instantiation_mb": memory_delta,
            "peak_memory_during_instantiation_mb": peak_memory_during,
        }

    except Exception as e:
        instantiation_time = time.perf_counter() - start_time
        memory_after = process.memory_info().rss / (1024 * 1024)
        memory_delta = memory_after - memory_before

        # Note: This is a utility function that may be called outside of benchmark context
        # so we use a simple print here rather than requiring a logging function parameter
        print(f"⚠️  Instantiation failed for {name}: {str(e)}")

        return {
            "instantiation_time_seconds": instantiation_time,
            "memory_before_instantiation_mb": memory_before,
            "memory_after_instantiation_mb": memory_after,
            "memory_delta_instantiation_mb": memory_delta,
            "peak_memory_during_instantiation_mb": peak_memory_during,
        }
