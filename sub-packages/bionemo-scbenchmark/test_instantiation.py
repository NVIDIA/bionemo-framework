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
Test script for instantiation metrics.
"""

import time

import torch
from torch.utils.data import DataLoader, Dataset

# Import the benchmarking framework
from bionemo.scbenchmark import benchmark_dataloader, measure_instantiation


class TestDataset(Dataset):
    """A test dataset that takes some time to instantiate."""

    def __init__(self, size=1000):
        print(f"    Creating dataset with {size} samples...")
        time.sleep(0.05)  # 50ms delay to simulate work
        self.data = torch.randn(size, 100)
        self.labels = torch.randint(0, 10, (size,))
        print("    Dataset created!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def test_instantiation_metrics():
    """Test the instantiation metrics functionality."""
    print("=" * 60)
    print("TESTING: Instantiation Metrics")
    print("=" * 60)

    def create_test_dataloader():
        dataset = TestDataset(size=500)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    # Test standalone instantiation measurement
    print("\n1. Standalone instantiation measurement:")
    metrics = measure_instantiation(create_test_dataloader, "Test Dataloader")

    print(f"   ✅ Instantiation time: {metrics.instantiation_time_seconds:.4f}s")
    print(f"   ✅ Memory delta: {metrics.memory_delta_mb:.2f} MB")
    print(f"   ✅ Peak memory: {metrics.peak_memory_during_mb:.2f} MB")

    # Test with full benchmark
    print("\n2. Full benchmark with instantiation metrics:")
    result = benchmark_dataloader(
        name="Test Dataloader with Instantiation",
        dataloader_factory=create_test_dataloader,
        num_epochs=1,
        max_batches=5,
        warmup_batches=1,
        measure_instantiation_metrics=True,
    )

    print(f"   ✅ Benchmark completed: {result.samples_per_second:.2f} samples/sec")
    if result.instantiation_metrics:
        print(f"   ✅ Instantiation time: {result.instantiation_metrics.instantiation_time_seconds:.4f}s")
        print(f"   ✅ Memory delta: {result.instantiation_metrics.memory_delta_mb:.2f} MB")

    return result


if __name__ == "__main__":
    print("BioNeMo SCDL Benchmarking Framework - Instantiation Test")
    print("=" * 60)

    try:
        result = test_instantiation_metrics()

        print("\n" + "=" * 60)
        print("🎉 Instantiation metrics test completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
