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
"""Instantiation Benchmarking Examples.

This example shows how to benchmark the time and memory usage
of creating different types of dataloaders. It demonstrates how
to measure the overhead of dataloader instantiation, which is
important for understanding the complete performance picture.
"""

import time

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Import the benchmarking framework
from bionemo.scbenchmark import benchmark_dataloader, benchmark_multiple_dataloaders, measure_instantiation


class LargeDataset(Dataset):
    """A dataset that takes time to instantiate.

    This dataset simulates scenarios where dataset creation involves
    significant computation or I/O operations, such as loading large
    files, preprocessing data, or building complex data structures.

    Attributes:
        size: Number of samples in the dataset
        feature_dim: Number of features per sample
        data: Tensor containing the dataset features
        labels: Tensor containing the dataset labels
    """

    def __init__(self, size=10000, feature_dim=1000):
        """Initialize the large dataset with artificial delay.

        Args:
            size: Number of samples in the dataset
            feature_dim: Number of features per sample
        """
        print(f"    Creating dataset with {size} samples, {feature_dim} features...")
        self.size = size
        self.feature_dim = feature_dim

        # Simulate slow instantiation
        time.sleep(0.1)  # 100ms delay

        # Create large data
        self.data = torch.randn(size, feature_dim)
        self.labels = torch.randint(0, 10, (size,))

        print("    Dataset created successfully!")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx):
        """Get a sample at the given index.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (data, label) for the sample
        """
        return self.data[idx], self.labels[idx]


class SimpleDataset(Dataset):
    """A simple dataset for comparison.

    This dataset provides a baseline for comparison with more
    complex datasets. It has minimal instantiation overhead
    and represents the best-case scenario for dataloader creation.

    Attributes:
        size: Number of samples in the dataset
        data: Tensor containing the dataset features
        labels: Tensor containing the dataset labels
    """

    def __init__(self, size=1000):
        """Initialize the simple dataset.

        Args:
            size: Number of samples in the dataset
        """
        self.size = size
        self.data = torch.randn(size, 100)
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx):
        """Get a sample at the given index.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (data, label) for the sample
        """
        return self.data[idx], self.labels[idx]


def example_instantiation_metrics():
    """Example: Measure instantiation metrics for different dataloaders.

    This example demonstrates how to measure the time and memory
    overhead of creating different types of dataloaders. It shows
    three different scenarios:
    1. Simple dataloader with minimal overhead
    2. Large dataloader with weighted sampling
    3. Multi-worker dataloader with process creation overhead

    The example shows how instantiation metrics can vary significantly
    depending on the complexity of the dataloader configuration.
    """
    print("=" * 60)
    print("EXAMPLE: Instantiation Metrics")
    print("=" * 60)

    # Example 1: Simple dataloader
    def create_simple_dataloader():
        """Create a simple dataloader with minimal overhead."""
        dataset = SimpleDataset(size=1000)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    print("\n1. Simple Dataloader:")
    benchmark_dataloader(
        name="Simple Dataloader",
        dataloader_factory=create_simple_dataloader,
        num_epochs=1,
        max_batches=10,
        warmup_batches=2,
        measure_instantiation_metrics=True,
    )

    # Example 2: Large dataloader with weighted sampler
    def create_weighted_dataloader():
        """Create a large dataloader with weighted sampling."""
        dataset = LargeDataset(size=5000, feature_dim=500)

        # Create weights for weighted sampling
        weights = torch.ones(len(dataset))
        weights[dataset.labels == 1] = 2.0  # Make class 1 more likely

        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)

        return DataLoader(dataset, batch_size=16, sampler=sampler)

    print("\n2. Weighted Dataloader (Large Dataset):")
    benchmark_dataloader(
        name="Weighted Dataloader",
        dataloader_factory=create_weighted_dataloader,
        num_epochs=1,
        max_batches=10,
        warmup_batches=2,
        measure_instantiation_metrics=True,
    )

    # Example 3: Dataloader with multiple workers
    def create_multiworker_dataloader():
        """Create a dataloader with multiple worker processes."""
        dataset = SimpleDataset(size=2000)
        return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True)

    print("\n3. Multi-worker Dataloader:")
    benchmark_dataloader(
        name="Multi-worker Dataloader",
        dataloader_factory=create_multiworker_dataloader,
        num_epochs=1,
        max_batches=10,
        warmup_batches=2,
        measure_instantiation_metrics=True,
    )

    print("\n✅ Instantiation benchmarks completed!")


def example_standalone_instantiation():
    """Example: Measure instantiation separately from iteration.

    This example shows how to measure instantiation metrics independently
    of iteration performance. This is useful when you want to understand
    the overhead of dataloader creation without running the full benchmark.

    The example demonstrates the measure_instantiation function which
    provides detailed timing and memory information for the instantiation
    process alone.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE: Standalone Instantiation Measurement")
    print("=" * 60)

    def create_complex_dataloader():
        """Create a complex dataloader with multiple components."""
        # Simulate complex dataloader creation
        time.sleep(0.05)  # 50ms delay

        dataset = LargeDataset(size=3000, feature_dim=300)

        # Create complex sampler
        weights = torch.ones(len(dataset))
        for i in range(len(dataset)):
            weights[i] = 1.0 + (i % 5) * 0.5  # Variable weights

        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)

        return DataLoader(dataset, batch_size=24, sampler=sampler)

    print("Measuring instantiation only...")
    metrics = measure_instantiation(create_complex_dataloader, "Complex Dataloader")

    print("✅ Instantiation metrics:")
    print(f"   Time: {metrics.instantiation_time_seconds:.4f}s")
    print(f"   Memory before: {metrics.memory_before_mb:.2f} MB")
    print(f"   Memory after: {metrics.memory_after_mb:.2f} MB")
    print(f"   Memory delta: {metrics.memory_delta_mb:.2f} MB")
    print(f"   Peak memory: {metrics.peak_memory_during_mb:.2f} MB")


def example_multiple_comparison():
    """Example: Compare multiple dataloaders with instantiation metrics.

    This example demonstrates how to compare multiple dataloaders
    with instantiation metrics included. It shows how different
    dataloader configurations affect both instantiation time and
    iteration performance.

    The comparison includes:
    1. Basic dataloader with minimal overhead
    2. Large dataset with significant instantiation time
    3. Weighted sampling with additional complexity
    """
    print("\n" + "=" * 60)
    print("EXAMPLE: Multiple Dataloader Comparison")
    print("=" * 60)

    def create_basic_dataloader():
        """Create a basic dataloader."""
        dataset = SimpleDataset(size=1000)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def create_large_dataloader():
        """Create a dataloader with large dataset."""
        dataset = LargeDataset(size=2000, feature_dim=200)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def create_weighted_dataloader():
        """Create a dataloader with weighted sampling."""
        dataset = SimpleDataset(size=1000)
        weights = torch.ones(len(dataset))
        weights[dataset.labels == 1] = 3.0
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
        return DataLoader(dataset, batch_size=32, sampler=sampler)

    dataloaders = [
        {
            "name": "Basic Dataloader",
            "factory": create_basic_dataloader,
            "num_epochs": 1,
            "max_batches": 8,
            "warmup_batches": 2,
            "measure_instantiation_metrics": True,
        },
        {
            "name": "Large Dataloader",
            "factory": create_large_dataloader,
            "num_epochs": 1,
            "max_batches": 8,
            "warmup_batches": 2,
            "measure_instantiation_metrics": True,
        },
        {
            "name": "Weighted Dataloader",
            "factory": create_weighted_dataloader,
            "num_epochs": 1,
            "max_batches": 8,
            "warmup_batches": 2,
            "measure_instantiation_metrics": True,
        },
    ]

    benchmark_multiple_dataloaders(dataloaders=dataloaders, output_dir="instantiation_results")

    print("\n✅ Multiple dataloader comparison completed!")


if __name__ == "__main__":
    """Main execution function for instantiation benchmarking examples.

    This function runs all the instantiation benchmarking examples
    and demonstrates the key features of instantiation measurement.
    """
    print("BioNeMo SCDL Benchmarking Framework - Instantiation Examples")
    print("=" * 60)

    try:
        example_instantiation_metrics()
        example_standalone_instantiation()
        example_multiple_comparison()

        print("\n" + "=" * 60)
        print("🎉 All instantiation examples completed successfully!")
        print("=" * 60)
        print("\nKey features demonstrated:")
        print("- Measuring instantiation time and memory")
        print("- Comparing different dataloader configurations")
        print("- Standalone instantiation measurement")
        print("- Multiple dataloader comparison with instantiation metrics")

    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        print("This might be due to missing dependencies.")
        print("The framework itself should work with any iterable dataloader.")
