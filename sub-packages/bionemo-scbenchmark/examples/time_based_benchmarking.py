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


#!/usr/bin/env python3
"""Time-Based Benchmarking Examples.

This example shows how to use the new time-based benchmarking features
with warmup periods and maximum time limits. It demonstrates various
scenarios where time-based limits are useful for controlling benchmark
duration and ensuring fair comparisons.
"""

import time

import torch
from torch.utils.data import DataLoader, Dataset

# Import the benchmarking framework
from bionemo.scbenchmark import (
    BenchmarkConfig,
    benchmark_any_dataloader,
    benchmark_multiple_dataloaders_simple,
    run_benchmark_with_config,
)


class SlowDataset(Dataset):
    """A dataset that simulates slow loading for demonstration.

    This dataset artificially introduces delays during data loading
    to simulate real-world scenarios where data loading might be
    slow due to disk I/O, network latency, or complex preprocessing.

    Attributes:
        size: Number of samples in the dataset
        delay_ms: Delay in milliseconds per sample
        data: Random tensor data
    """

    def __init__(self, size=1000, delay_ms=10):
        """Initialize the slow dataset.

        Args:
            size: Number of samples in the dataset
            delay_ms: Delay in milliseconds per sample access
        """
        self.size = size
        self.delay_ms = delay_ms
        self.data = torch.randn(size, 100)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx):
        """Get a sample at the given index with artificial delay.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Sample data with artificial loading delay
        """
        # Simulate slow loading
        time.sleep(self.delay_ms / 1000.0)
        return self.data[idx]


class FastDataset(Dataset):
    """A fast dataset for comparison.

    This dataset provides fast access to data for comparison with
    slower datasets. It demonstrates the baseline performance
    achievable with minimal overhead.

    Attributes:
        size: Number of samples in the dataset
        data: Random tensor data
    """

    def __init__(self, size=1000):
        """Initialize the fast dataset.

        Args:
            size: Number of samples in the dataset
        """
        self.size = size
        self.data = torch.randn(size, 100)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx):
        """Get a sample at the given index.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Sample data
        """
        return self.data[idx]


def example_time_based_benchmarking():
    """Example: Time-based benchmarking with different limits.

    This example demonstrates various time-based benchmarking scenarios:
    1. Benchmark with a maximum time limit
    2. Benchmark with time-based warmup
    3. Compare multiple dataloaders with different time limits

    The examples show how time-based limits can be used to ensure
    fair comparisons and control benchmark duration.
    """
    print("=" * 60)
    print("EXAMPLE: Time-Based Benchmarking")
    print("=" * 60)

    # Create datasets
    slow_dataset = SlowDataset(size=500, delay_ms=5)  # 5ms delay per item
    fast_dataset = FastDataset(size=500)

    slow_dataloader = DataLoader(slow_dataset, batch_size=16, shuffle=True)
    fast_dataloader = DataLoader(fast_dataset, batch_size=16, shuffle=True)

    # Example 1: Benchmark with time limit (5 seconds)
    print("\n1. Benchmark with 5-second time limit:")
    benchmark_any_dataloader(
        dataloader=slow_dataloader,
        name="Slow Dataloader (5s limit)",
        max_time_seconds=5.0,
        warmup_batches=3,
        print_progress=True,
    )

    # Example 2: Benchmark with time-based warmup
    print("\n2. Benchmark with 2-second warmup:")
    benchmark_any_dataloader(
        dataloader=fast_dataloader,
        name="Fast Dataloader (2s warmup)",
        max_time_seconds=10.0,
        warmup_time_seconds=2.0,  # Warmup for 2 seconds
        print_progress=True,
    )

    # Example 3: Compare multiple dataloaders with time limits
    print("\n3. Compare multiple dataloaders with time limits:")
    dataloaders = [
        {
            "name": "Slow Dataloader (3s)",
            "dataloader": DataLoader(slow_dataset, batch_size=8),
            "max_time_seconds": 3.0,
            "warmup_time_seconds": 1.0,
            "print_progress": False,
        },
        {
            "name": "Fast Dataloader (3s)",
            "dataloader": DataLoader(fast_dataset, batch_size=8),
            "max_time_seconds": 3.0,
            "warmup_time_seconds": 1.0,
            "print_progress": False,
        },
        {
            "name": "Slow Dataloader (6s)",
            "dataloader": DataLoader(slow_dataset, batch_size=16),
            "max_time_seconds": 6.0,
            "warmup_time_seconds": 1.0,
            "print_progress": False,
        },
    ]

    benchmark_multiple_dataloaders_simple(dataloaders=dataloaders, output_dir="time_based_results")

    print("\n✅ Time-based benchmarks completed!")


def example_hybrid_limits():
    """Example: Using both batch and time limits together.

    This example demonstrates how to use both batch-based and time-based
    limits simultaneously. The benchmark will stop when either limit is
    reached, providing flexibility in controlling benchmark duration.

    This is useful when you want to ensure a benchmark doesn't run too
    long while also ensuring a minimum number of batches are processed.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE: Hybrid Limits (Batch + Time)")
    print("=" * 60)

    dataset = FastDataset(size=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Benchmark with both limits - stops when either is reached
    result = benchmark_any_dataloader(
        dataloader=dataloader,
        name="Hybrid Limits",
        max_batches=50,  # Stop after 50 batches
        max_time_seconds=3.0,  # OR stop after 3 seconds
        warmup_batches=5,
        print_progress=True,
    )

    print(
        f"✅ Hybrid benchmark completed: {result.total_batches} batches in {result.total_iteration_time_seconds:.2f}s"
    )


def example_custom_config():
    """Example: Using the core BenchmarkConfig for custom setups.

    This example shows how to use the BenchmarkConfig class directly
    for more advanced benchmarking scenarios. This provides access to
    all configuration options and can be useful for programmatic
    benchmark configuration.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE: Custom BenchmarkConfig")
    print("=" * 60)

    dataset = FastDataset(size=800)
    dataloader = DataLoader(dataset, batch_size=24, shuffle=True)

    # Create custom configuration
    config = BenchmarkConfig(
        name="Custom Config Example", num_epochs=2, max_time_seconds=4.0, warmup_time_seconds=1.5, print_progress=True
    )

    # Run benchmark with custom config
    run_benchmark_with_config(dataloader, config)

    print("✅ Custom config benchmark completed!")


def example_progress_monitoring():
    """Example: Monitoring progress with time-based benchmarks.

    This example demonstrates how progress monitoring works with
    time-based benchmarks. It shows how the framework provides
    real-time feedback during long-running benchmarks, even when
    using time-based limits.

    The example uses a dataset with variable processing times to
    show how progress reporting adapts to different scenarios.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE: Progress Monitoring")
    print("=" * 60)

    # Create a dataset that takes variable time
    class VariableSpeedDataset(Dataset):
        """Dataset with variable processing times for demonstration.

        This dataset simulates real-world scenarios where different
        samples take different amounts of time to process, such as
        when dealing with variable-length sequences or complex
        preprocessing steps.
        """

        def __init__(self, size=200):
            """Initialize the variable speed dataset.

            Args:
                size: Number of samples in the dataset
            """
            self.size = size
            self.data = torch.randn(size, 50)

        def __len__(self):
            """Return the number of samples in the dataset."""
            return self.size

        def __getitem__(self, idx):
            """Get a sample with variable processing time.

            Args:
                idx: Index of the sample to retrieve

            Returns:
                Sample data with variable delay
            """
            # Variable delay based on index
            delay = (idx % 10) * 0.001  # 0-9ms delay
            time.sleep(delay)
            return self.data[idx]

    dataset = VariableSpeedDataset(size=200)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Benchmark with progress monitoring
    benchmark_any_dataloader(
        dataloader=dataloader,
        name="Variable Speed Dataloader",
        max_time_seconds=8.0,
        warmup_time_seconds=2.0,
        print_progress=True,
    )

    print("✅ Progress monitoring example completed!")


if __name__ == "__main__":
    """Main execution function for time-based benchmarking examples.

    This function runs all the time-based benchmarking examples
    and provides a summary of the key features demonstrated.
    """
    print("BioNeMo SCDL Benchmarking Framework - Time-Based Examples")
    print("=" * 60)

    try:
        example_time_based_benchmarking()
        example_hybrid_limits()
        example_custom_config()
        example_progress_monitoring()

        print("\n" + "=" * 60)
        print("🎉 All time-based examples completed successfully!")
        print("=" * 60)
        print("\nKey features demonstrated:")
        print("- Time-based benchmarking with max_time_seconds")
        print("- Time-based warmup with warmup_time_seconds")
        print("- Hybrid limits (batch + time)")
        print("- Custom configurations with BenchmarkConfig")
        print("- Progress monitoring during time-based benchmarks")

    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        print("This might be due to missing dependencies.")
        print("The framework itself should work with any iterable dataloader.")
