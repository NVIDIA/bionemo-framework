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
"""
Simple test script to verify the benchmarking framework works.

This script provides basic tests to verify that the benchmarking
framework is functioning correctly. It tests both the simple
interface and time-based benchmarking features.
"""

import torch
from torch.utils.data import DataLoader, Dataset

# Import the benchmarking framework
from bionemo.scbenchmark import benchmark_any_dataloader


class SimpleDataset(Dataset):
    """A simple PyTorch dataset for testing purposes.

    This dataset creates random data and labels for use in
    testing the benchmarking framework. It provides a basic
    implementation that can be used to verify functionality.

    Attributes:
        data: Random tensor data with shape (size, 100)
        labels: Random integer labels with shape (size,)
    """

    def __init__(self, size=1000):
        """Initialize the simple dataset.

        Args:
            size: Number of samples in the dataset
        """
        self.data = torch.randn(size, 100)
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a sample at the given index.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (data, label) for the sample
        """
        return self.data[idx], self.labels[idx]


def test_simple_benchmark():
    """Test the simple benchmarking interface.

    This function tests the basic functionality of the simple
    benchmarking interface. It creates a standard PyTorch
    DataLoader and benchmarks it using the simple interface
    without requiring factory functions.

    Returns:
        BenchmarkResult from the simple benchmark test
    """
    print("=" * 60)
    print("TESTING: Simple Benchmarking Interface")
    print("=" * 60)

    # Create your dataloader normally
    dataset = SimpleDataset(size=500)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # Benchmark it directly - no factory function needed!
    result = benchmark_any_dataloader(
        dataloader=dataloader,  # Just pass the dataloader!
        name="Simple PyTorch DataLoader",
        num_epochs=1,
        max_batches=10,  # Limit to 10 batches for quick test
        warmup_batches=2,
    )

    print(f"✅ Test completed: {result.samples_per_second:.2f} samples/sec")
    return result


def test_time_based_benchmark():
    """Test time-based benchmarking.

    This function tests the time-based benchmarking features
    of the framework. It demonstrates how to use time limits
    and time-based warmup periods to control benchmark duration.

    Returns:
        BenchmarkResult from the time-based benchmark test
    """
    print("\n" + "=" * 60)
    print("TESTING: Time-Based Benchmarking")
    print("=" * 60)

    dataset = SimpleDataset(size=1000)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # Benchmark with time limit
    result = benchmark_any_dataloader(
        dataloader=dataloader,
        name="Time-Based Test",
        max_time_seconds=5.0,  # Run for max 5 seconds
        warmup_time_seconds=1.0,  # Warmup for 1 second
        print_progress=True,
    )

    print(
        f"✅ Time-based test completed: {result.total_batches} batches in {result.total_iteration_time_seconds:.2f}s"
    )
    return result


if __name__ == "__main__":
    """Main execution function for running framework tests.

    This function runs both the simple and time-based benchmark
    tests to verify that the framework is working correctly.
    It provides a summary of the test results and handles any
    errors that might occur during testing.
    """
    print("BioNeMo SCDL Benchmarking Framework - Test")
    print("=" * 60)

    try:
        # Test simple interface
        result1 = test_simple_benchmark()

        # Test time-based interface
        result2 = test_time_based_benchmark()

        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print("=" * 60)
        print(f"Simple test: {result1.samples_per_second:.2f} samples/sec")
        print(f"Time-based test: {result2.samples_per_second:.2f} samples/sec")

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
