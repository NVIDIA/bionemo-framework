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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil


@dataclass
class InstantiationMetrics:
    """Metrics for dataloader instantiation.

    This dataclass stores timing and memory information collected during
    the instantiation of a dataloader. It provides insights into the
    overhead of creating dataloaders.

    Attributes:
        instantiation_time_seconds: Time taken to instantiate the dataloader
        memory_before_mb: Memory usage before instantiation
        memory_after_mb: Memory usage after instantiation
        memory_delta_mb: Change in memory usage during instantiation
        peak_memory_during_mb: Peak memory usage during instantiation
    """

    instantiation_time_seconds: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    peak_memory_during_mb: float


@dataclass
class BenchmarkResult:
    """Results from benchmarking a dataloader.

    This dataclass contains all the performance metrics collected during
    a benchmark run. It provides comprehensive information about timing,
    throughput, memory usage, and any errors that occurred.

    Attributes:
        name: Name of the benchmarked dataloader
        disk_size_mb: Size of data files on disk
        setup_time_seconds: Time taken for setup phase
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
        instantiation_metrics: Metrics from dataloader instantiation
    """

    name: str
    disk_size_mb: float
    setup_time_seconds: float
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
    instantiation_metrics: Optional[InstantiationMetrics] = None

    def __post_init__(self):
        """Initialize errors list if not provided."""
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the benchmark result
        """
        result = {
            "name": self.name,
            "disk_size_mb": self.disk_size_mb,
            "setup_time_seconds": self.setup_time_seconds,
            "total_iteration_time_seconds": self.total_iteration_time_seconds,
            "average_batch_time_seconds": self.average_batch_time_seconds,
            "total_batches": self.total_batches,
            "total_samples": self.total_samples,
            "samples_per_second": self.samples_per_second,
            "batches_per_second": self.batches_per_second,
            "peak_memory_mb": self.peak_memory_mb,
            "average_memory_mb": self.average_memory_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "errors": self.errors,
        }

        if self.instantiation_metrics:
            result["instantiation_metrics"] = {
                "instantiation_time_seconds": self.instantiation_metrics.instantiation_time_seconds,
                "memory_before_mb": self.instantiation_metrics.memory_before_mb,
                "memory_after_mb": self.instantiation_metrics.memory_after_mb,
                "memory_delta_mb": self.instantiation_metrics.memory_delta_mb,
                "peak_memory_during_mb": self.instantiation_metrics.peak_memory_during_mb,
            }

        return result

    def save_to_file(self, filepath: str) -> None:
        """Save results to JSON file.

        Args:
            filepath: Path to the output JSON file
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

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

        # Handle instantiation_metrics if present
        if "instantiation_metrics" in data:
            metrics_data = data.pop("instantiation_metrics")
            data["instantiation_metrics"] = InstantiationMetrics(**metrics_data)

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
    try:
        if os.path.isfile(path):
            return os.path.getsize(path) / (1024 * 1024)
        elif os.path.isdir(path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.isfile(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)
        else:
            return 0.0
    except Exception:
        return 0.0


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


def measure_instantiation(dataloader_factory: callable, name: str = "Unknown") -> InstantiationMetrics:
    """Measure the time and memory usage of instantiating a dataloader.

    This function measures the overhead of creating a dataloader by
    calling the factory function and monitoring timing and memory
    usage before, during, and after instantiation.

    Args:
        dataloader_factory: Function that creates the dataloader
        name: Name for logging and error reporting

    Returns:
        InstantiationMetrics with timing and memory information

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

        return InstantiationMetrics(
            instantiation_time_seconds=instantiation_time,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_delta_mb=memory_delta,
            peak_memory_during_mb=peak_memory_during,
        )

    except Exception as e:
        instantiation_time = time.perf_counter() - start_time
        memory_after = process.memory_info().rss / (1024 * 1024)
        memory_delta = memory_after - memory_before

        print(f"⚠️  Instantiation failed for {name}: {str(e)}")

        return InstantiationMetrics(
            instantiation_time_seconds=instantiation_time,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_delta_mb=memory_delta,
            peak_memory_during_mb=peak_memory_during,
        )
