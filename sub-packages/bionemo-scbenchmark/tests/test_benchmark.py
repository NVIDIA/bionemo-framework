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


"""
Tests for the benchmarking framework.

This module contains comprehensive tests for the benchmarking framework,
covering both the factory-based and simple interfaces, as well as
protocol implementations and error handling scenarios.
"""

import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from bionemo.scbenchmark import (
    BenchmarkResult,
    benchmark_any_dataloader,
    benchmark_dataloader,
    benchmark_multiple_dataloaders,
    benchmark_multiple_dataloaders_simple,
)


class TestDataset(Dataset):
    """Simple test dataset for benchmarking tests.

    This dataset creates random data and labels for use in testing
    the benchmarking framework. It implements the standard PyTorch
    dataset interface.
    """

    def __init__(self, size=100):
        """Initialize the test dataset.

        Args:
            size: Number of samples in the dataset
        """
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 5, (size,))

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


class CustomDataloader:
    """Custom dataloader for testing purposes.

    This class demonstrates how to create a custom dataloader that
    provides an iterator over random tensors.
    """

    def __init__(self, size=20):
        """Initialize the custom dataloader.

        Args:
            size: Number of batches in the dataloader
        """
        self.data = [torch.randn(5) for _ in range(size)]
        self.current = 0

    def __iter__(self):
        """Return an iterator over the dataloader."""
        self.current = 0
        return self

    def __next__(self):
        """Get the next batch from the dataloader.

        Returns:
            Next batch of data

        Raises:
            StopIteration: When all batches have been yielded
        """
        if self.current >= len(self.data):
            raise StopIteration
        result = self.data[self.current]
        self.current += 1
        return result

    def __len__(self):
        """Return the number of batches in the dataloader."""
        return len(self.data)


def create_test_dataloader():
    """Factory function for test dataloader.

    This function creates a PyTorch DataLoader wrapped around a
    TestDataset for use in benchmarking tests.

    Returns:
        DataLoader instance with test data
    """
    dataset = TestDataset(size=50)
    return DataLoader(dataset, batch_size=8, shuffle=False)


def create_simple_iterator():
    """Factory function for simple iterator.

    This function creates a simple iterator that yields random tensors
    for testing purposes.

    Returns:
        Iterator over random tensors
    """
    return iter([torch.randn(5) for _ in range(20)])


def test_benchmark_dataloader():
    """Test basic dataloader benchmarking.

    This test verifies that the benchmark_dataloader function works
    correctly with a simple PyTorch DataLoader. It checks that all
    expected metrics are present and reasonable.
    """
    result = benchmark_dataloader(
        name="Test Dataloader",
        dataloader_factory=create_test_dataloader,
        num_epochs=1,
        max_batches=5,
        warmup_batches=1,
        print_progress=False,
    )

    # Check that we got a result
    assert isinstance(result, BenchmarkResult)
    assert result.name == "Test Dataloader"
    assert result.total_batches > 0
    assert result.total_samples > 0
    assert result.samples_per_second > 0
    assert result.batches_per_second > 0
    assert result.peak_memory_mb > 0
    assert len(result.errors) == 0


def test_benchmark_iterator():
    """Test benchmarking a simple iterator.

    This test verifies that the benchmark_dataloader function can
    handle simple iterators that don't implement the full DataLoader
    interface.
    """
    result = benchmark_dataloader(
        name="Test Iterator",
        dataloader_factory=create_simple_iterator,
        num_epochs=1,
        max_batches=5,
        warmup_batches=1,
        print_progress=False,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.name == "Test Iterator"
    assert result.total_batches > 0
    assert len(result.errors) == 0


def test_benchmark_with_disk_measurement():
    """Test benchmarking with disk measurement.

    This test verifies that disk size measurement works correctly
    when a data path is provided. It creates a temporary file and
    checks that the disk size is reported correctly.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test data" * 1000)  # Create some file size

        result = benchmark_dataloader(
            name="Test with Disk",
            dataloader_factory=create_test_dataloader,
            data_path=temp_dir,
            num_epochs=1,
            max_batches=3,
            warmup_batches=0,
            print_progress=False,
        )

        assert result.disk_size_mb > 0
        assert len(result.errors) == 0


def test_benchmark_multiple_dataloaders():
    """Test benchmarking multiple dataloaders.

    This test verifies that the benchmark_multiple_dataloaders function
    works correctly with multiple dataloader configurations. It checks
    that results are saved to files when an output directory is provided.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        dataloaders = [
            {
                "name": "Dataloader 1",
                "factory": create_test_dataloader,
                "num_epochs": 1,
                "max_batches": 3,
                "warmup_batches": 0,
                "print_progress": False,
            },
            {
                "name": "Dataloader 2",
                "factory": create_simple_iterator,
                "num_epochs": 1,
                "max_batches": 3,
                "warmup_batches": 0,
                "print_progress": False,
            },
        ]

        results = benchmark_multiple_dataloaders(dataloaders=dataloaders, output_dir=temp_dir)

        assert len(results) == 2
        assert all(isinstance(r, BenchmarkResult) for r in results)
        assert all(len(r.errors) == 0 for r in results)

        # Check that files were saved
        assert os.path.exists(os.path.join(temp_dir, "Dataloader 1_results.json"))
        assert os.path.exists(os.path.join(temp_dir, "Dataloader 2_results.json"))


def test_benchmark_result_serialization():
    """Test that BenchmarkResult can be saved and loaded.

    This test verifies that BenchmarkResult objects can be serialized
    to JSON files and deserialized back to objects correctly.
    """
    result = BenchmarkResult(
        name="Test",
        disk_size_mb=10.0,
        setup_time_seconds=1.0,
        total_iteration_time_seconds=5.0,
        average_batch_time_seconds=0.1,
        total_batches=50,
        total_samples=400,
        samples_per_second=80.0,
        batches_per_second=10.0,
        peak_memory_mb=100.0,
        average_memory_mb=50.0,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_file = f.name

    try:
        # Save
        result.save_to_file(temp_file)

        # Load
        loaded_result = BenchmarkResult.load_from_file(temp_file)

        # Check that all fields match
        assert loaded_result.name == result.name
        assert loaded_result.disk_size_mb == result.disk_size_mb
        assert loaded_result.setup_time_seconds == result.setup_time_seconds
        assert loaded_result.total_iteration_time_seconds == result.total_iteration_time_seconds
        assert loaded_result.average_batch_time_seconds == result.average_batch_time_seconds
        assert loaded_result.total_batches == result.total_batches
        assert loaded_result.total_samples == result.total_samples
        assert loaded_result.samples_per_second == result.samples_per_second
        assert loaded_result.batches_per_second == result.batches_per_second
        assert loaded_result.peak_memory_mb == result.peak_memory_mb
        assert loaded_result.average_memory_mb == result.average_memory_mb

    finally:
        os.unlink(temp_file)


def test_error_handling():
    """Test that errors are handled gracefully.

    This test verifies that the benchmarking framework properly
    handles errors during dataloader creation and iteration.
    It ensures that error information is captured and returned
    in the BenchmarkResult without crashing the benchmark.
    """

    def create_broken_dataloader():
        """Factory function that raises an error."""
        raise RuntimeError("This dataloader is broken")

    result = benchmark_dataloader(
        name="Broken Dataloader",
        dataloader_factory=create_broken_dataloader,
        num_epochs=1,
        max_batches=5,
        print_progress=False,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.name == "Broken Dataloader"
    assert len(result.errors) > 0
    assert "broken" in result.errors[0].lower()


# ============================================================================
# Tests for Simple Interface
# ============================================================================


def test_benchmark_any_dataloader():
    """Test the simple interface with any dataloader.

    This test verifies that the simple benchmarking interface
    works correctly with standard PyTorch DataLoaders. It
    demonstrates the ease of use of the simple interface
    compared to the factory-based approach.
    """
    # Create a dataloader normally
    dataset = TestDataset(size=50)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Benchmark it directly
    result = benchmark_any_dataloader(
        dataloader=dataloader,
        name="Simple Test Dataloader",
        num_epochs=1,
        max_batches=5,
        warmup_batches=1,
        print_progress=False,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.name == "Simple Test Dataloader"
    assert result.total_batches > 0
    assert result.total_samples > 0
    assert result.samples_per_second > 0
    assert result.batches_per_second > 0
    assert result.peak_memory_mb > 0
    assert len(result.errors) == 0


def test_benchmark_custom_dataloader():
    """Test benchmarking a custom dataloader.

    This test verifies that the simple interface works with custom
    dataloaders. It demonstrates the flexibility of the framework
    in handling different dataloader implementations.
    """
    custom_dl = CustomDataloader(size=15)

    result = benchmark_any_dataloader(
        dataloader=custom_dl,
        name="Custom Dataloader",
        num_epochs=1,
        max_batches=10,
        warmup_batches=2,
        print_progress=False,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.name == "Custom Dataloader"
    assert result.total_batches > 0
    assert len(result.errors) == 0


def test_benchmark_list_dataloader():
    """Test benchmarking a list-based dataloader.

    This test verifies that the framework can benchmark simple
    list-based dataloaders. It demonstrates the framework's
    ability to work with any iterable object, not just
    formal dataloader classes.
    """
    list_dataloader = [torch.randn(5) for _ in range(20)]

    result = benchmark_any_dataloader(
        dataloader=list_dataloader,
        name="List Dataloader",
        num_epochs=1,
        max_batches=10,
        warmup_batches=2,
        print_progress=False,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.name == "List Dataloader"
    assert result.total_batches > 0
    assert len(result.errors) == 0


def test_benchmark_generator_dataloader():
    """Test benchmarking a generator-based dataloader.

    This test verifies that the framework can benchmark generator
    functions that yield batches. It demonstrates the framework's
    flexibility in handling different types of data sources.
    """

    def generator_dataloader():
        """Generator function that yields random tensors."""
        for _ in range(20):
            yield torch.randn(5)

    result = benchmark_any_dataloader(
        dataloader=generator_dataloader(),
        name="Generator Dataloader",
        num_epochs=1,
        max_batches=10,
        warmup_batches=2,
        print_progress=False,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.name == "Generator Dataloader"
    assert result.total_batches > 0
    assert len(result.errors) == 0


def test_benchmark_multiple_dataloaders_simple():
    """Test the simple interface for multiple dataloaders.

    This test verifies that the simple interface works correctly
    when benchmarking multiple dataloaders simultaneously. It
    checks that results are saved to files and that all dataloaders
    are processed correctly.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create different types of dataloaders
        dataset = TestDataset(size=30)
        dataloader1 = DataLoader(dataset, batch_size=8, shuffle=False)
        dataloader2 = CustomDataloader(size=15)
        dataloader3 = [torch.randn(5) for _ in range(10)]

        dataloaders = [
            {
                "name": "PyTorch DataLoader",
                "dataloader": dataloader1,
                "num_epochs": 1,
                "max_batches": 3,
                "warmup_batches": 0,
                "print_progress": False,
            },
            {
                "name": "Custom Dataloader",
                "dataloader": dataloader2,
                "num_epochs": 1,
                "max_batches": 3,
                "warmup_batches": 0,
                "print_progress": False,
            },
            {
                "name": "List Dataloader",
                "dataloader": dataloader3,
                "num_epochs": 1,
                "max_batches": 3,
                "warmup_batches": 0,
                "print_progress": False,
            },
        ]

        results = benchmark_multiple_dataloaders_simple(dataloaders=dataloaders, output_dir=temp_dir)

        assert len(results) == 3
        assert all(isinstance(r, BenchmarkResult) for r in results)
        assert all(len(r.errors) == 0 for r in results)

        # Check that files were saved
        assert os.path.exists(os.path.join(temp_dir, "PyTorch DataLoader_results.json"))
        assert os.path.exists(os.path.join(temp_dir, "Custom Dataloader_results.json"))
        assert os.path.exists(os.path.join(temp_dir, "List Dataloader_results.json"))


if __name__ == "__main__":
    """Main execution function for running tests with pytest."""
    pytest.main([__file__])
