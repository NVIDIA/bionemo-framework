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


import gc
import math
import os
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bionemo.scbenchmark.benchmark import (
    BenchmarkConfig,
    benchmark_dataloaders_with_configs,
    benchmark_single_dataloader,
    run_benchmark,
)
from bionemo.scbenchmark.common import BenchmarkResult


class MockDataloader:
    """Mock dataloader for testing."""

    def __init__(self, data_list, num_workers=0):
        self.data_list = data_list
        self.num_workers = num_workers
        self._iter_count = 0

    def __iter__(self):
        self._iter_count = 0
        return self

    def __next__(self):
        if self._iter_count < len(self.data_list):
            result = self.data_list[self._iter_count]
            self._iter_count += 1
            return result
        raise StopIteration


class MockBatch:
    """Mock batch object with shape attribute."""

    def __init__(self, batch_size):
        self.shape = (batch_size, 10)  # (batch_size, features)


# run_benchmark tests
@mock.patch("bionemo.scbenchmark.benchmark.measure_peak_memory_full")
@mock.patch("builtins.print")
def test_run_benchmark_basic(mock_print, mock_measure_memory):
    """Test basic benchmark run."""
    # Create mock dataloader
    batches = [MockBatch(32) for _ in range(5)]
    dataloader = MockDataloader(batches, num_workers=0)

    # Mock memory measurement to return predictable results. This simulates measure_peak_memory_full
    mock_measure_memory.return_value = (
        (100, 5, 2.0, 10, 1, 0.5),  # (samples, batches, elapsed, warmup_samples, warmup_batches, warmup_time)
        100.0,  # baseline
        150.0,  # peak
        125.0,  # avg
        50.0,  # delta
        130.0,  # final
        2.5,  # duration
    )

    config = BenchmarkConfig(name="TestRun", num_epochs=1, warmup_time_seconds=0.5)

    result = run_benchmark(
        dataloader,
        config,
        dataset_instantiation_time_seconds=1.0,
        dataloader_instantiation_time_seconds=0.5,
        peak_memory_during_instantiation_mb=200.0,
        memory_before_instantiation_mb=100.0,
        memory_after_instantiation_mb=150.0,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.name == "TestRun"
    assert result.dataset_instantiation_time_seconds == 1.0
    assert result.dataloader_instantiation_time_seconds == 0.5
    assert len(result.epoch_results) == 1

    epoch_result = result.epoch_results[0]
    assert epoch_result["epoch"] == 1
    assert epoch_result["samples"] == 100
    assert epoch_result["batches"] == 5
    assert epoch_result["warmup_samples"] == 10
    assert epoch_result["warmup_batches"] == 1


@mock.patch("bionemo.scbenchmark.benchmark.measure_peak_memory_full")
@mock.patch("builtins.print")
def test_run_benchmark_multiple_epochs(mock_print, mock_measure_memory):
    """Test benchmark with multiple epochs."""
    batches = [MockBatch(16) for _ in range(3)]
    dataloader = MockDataloader(batches, num_workers=0)

    # Mock returns different results for each epoch
    mock_measure_memory.side_effect = [
        ((48, 3, 1.5, 5, 1, 0.2), 100.0, 140.0, 120.0, 40.0, 125.0, 1.7),  # Epoch 1
        ((48, 3, 1.4, 0, 0, 0.0), 100.0, 135.0, 118.0, 35.0, 122.0, 1.4),  # Epoch 2
    ]

    config = BenchmarkConfig(name="MultiEpoch", num_epochs=2, warmup_time_seconds=0.2)

    result = run_benchmark(dataloader, config)

    assert len(result.epoch_results) == 2

    # First epoch should have warmup data
    assert result.epoch_results[0]["warmup_samples"] == 5
    assert result.epoch_results[0]["warmup_batches"] == 1
    assert result.epoch_results[0]["warmup_time"] == 0.2

    # Second epoch should have no warmup
    assert result.epoch_results[1]["warmup_samples"] == 0
    assert result.epoch_results[1]["warmup_batches"] == 0
    assert result.epoch_results[1]["warmup_time"] == 0.0


@mock.patch("bionemo.scbenchmark.benchmark.measure_peak_memory_full")
@mock.patch("builtins.print")
def test_run_benchmark_no_warmup(mock_print, mock_measure_memory):
    """Test benchmark run without warmup."""
    batches = [MockBatch(32)]
    dataloader = MockDataloader(batches, num_workers=0)

    mock_measure_memory.return_value = ((32, 1, 1.0, 0, 0, 0.0), 100.0, 140.0, 120.0, 40.0, 125.0, 1.2)

    config = BenchmarkConfig(name="NoWarmup", warmup_time_seconds=None)

    result = run_benchmark(dataloader, config)

    assert result.epoch_results[0]["warmup_samples"] == 0
    assert result.epoch_results[0]["warmup_batches"] == 0
    assert result.epoch_results[0]["warmup_time"] == 0.0


# benchmark_single_dataloader tests
@mock.patch("bionemo.scbenchmark.benchmark.measure_peak_memory_full")
@mock.patch("bionemo.scbenchmark.benchmark.get_disk_size")
@mock.patch("bionemo.scbenchmark.benchmark.export_benchmark_results")
@mock.patch("bionemo.scbenchmark.benchmark._drop_caches")
@mock.patch("builtins.print")
def test_benchmark_single_dataloader_basic(
    mock_print, mock_drop_caches, mock_export, mock_get_disk_size, mock_measure_memory
):
    """Test basic single dataloader benchmarking."""
    # Mock disk size
    mock_get_disk_size.return_value = 100.0

    # Mock memory measurements
    mock_measure_memory.side_effect = [
        # Dataloader instantiation
        (MockDataloader([MockBatch(32)], 0), 100.0, 150.0, 125.0, 50.0, 130.0, 1.0),
        # Benchmark run
        ((32, 1, 1.0, 0, 0, 0.0), 100.0, 140.0, 120.0, 40.0, 125.0, 1.2),
    ]

    def dataloader_factory():
        return MockDataloader([MockBatch(32)], num_workers=0)

    result = benchmark_single_dataloader(
        dataloader_factory=dataloader_factory,
        data_path="/test/data",
        name="TestDataloader",
        dataset_factory=None,  # Dataloader creates everything internally
        num_epochs=1,
        max_batches=10,
        warmup_time_seconds=0.0,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.name == "TestDataloader"
    assert result.disk_size_mb == 100.0
    mock_export.assert_called_once()


@mock.patch("bionemo.scbenchmark.benchmark.measure_peak_memory_full")
@mock.patch("bionemo.scbenchmark.benchmark.get_disk_size")
@mock.patch("bionemo.scbenchmark.benchmark.export_benchmark_results")
@mock.patch("bionemo.scbenchmark.benchmark._drop_caches")
@mock.patch("builtins.print")
def test_benchmark_single_dataloader_multiple_runs(
    mock_print, mock_drop_caches, mock_export, mock_get_disk_size, mock_measure_memory
):
    """Test single dataloader with multiple runs."""
    mock_get_disk_size.return_value = 50.0

    # Mock for multiple runs
    mock_measure_memory.side_effect = [
        # Initial dataloader instantiation
        (MockDataloader([MockBatch(16)], 0), 100.0, 140.0, 120.0, 40.0, 125.0, 0.8),
        # Run 1
        ((16, 1, 0.8, 0, 0, 0.0), 100.0, 135.0, 118.0, 35.0, 122.0, 1.0),
        # Run 2
        ((16, 1, 0.7, 0, 0, 0.0), 100.0, 132.0, 116.0, 32.0, 120.0, 0.9),
    ]

    def dataloader_factory():
        return MockDataloader([MockBatch(16)], num_workers=0)

    results = benchmark_single_dataloader(
        dataloader_factory=dataloader_factory,
        data_path="/test/data",
        name="MultiRun",
        dataset_factory=None,  # Dataloader creates everything internally
        num_runs=2,
        warmup_time_seconds=0.0,
    )

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].name == "MultiRun_run_1"
    assert results[1].name == "MultiRun_run_2"
    assert mock_export.call_count == 2

    # Verify instantiation metrics are preserved across all runs
    for result in results:
        assert result.peak_memory_during_instantiation_mb == 40.0
        assert result.memory_before_instantiation_mb == 100.0
        assert result.memory_after_instantiation_mb == 125.0
        assert result.dataset_instantiation_time_seconds == 0.0
        assert result.dataloader_instantiation_time_seconds == 0.8
        assert result.disk_size_mb == 50.0


@mock.patch("bionemo.scbenchmark.benchmark.measure_peak_memory_full")
@mock.patch("bionemo.scbenchmark.benchmark.get_disk_size")
@mock.patch("bionemo.scbenchmark.benchmark.export_benchmark_results")
@mock.patch("bionemo.scbenchmark.benchmark._drop_caches")
@mock.patch("builtins.print")
def test_benchmark_with_dataset_factory(
    mock_print, mock_drop_caches, mock_export, mock_get_disk_size, mock_measure_memory
):
    """Test benchmarking with separate dataset factory."""
    mock_get_disk_size.return_value = 75.0

    mock_measure_memory.side_effect = [
        # Dataset instantiation
        ("mock_dataset", 50.0, 80.0, 65.0, 30.0, 70.0, 0.5),
        # Dataloader instantiation
        (MockDataloader([MockBatch(24)], 0), 70.0, 120.0, 95.0, 50.0, 110.0, 0.3),
        # Benchmark run
        ((24, 1, 1.2, 0, 0, 0.0), 70.0, 130.0, 100.0, 60.0, 115.0, 1.5),
    ]

    def dataset_factory():
        return "mock_dataset"

    def dataloader_factory(dataset):
        return MockDataloader([MockBatch(24)], num_workers=0)

    result = benchmark_single_dataloader(
        dataloader_factory=dataloader_factory,
        data_path="/test/dataset",
        name="WithDatasetFactory",
        dataset_factory=dataset_factory,
        warmup_time_seconds=0.0,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.name == "WithDatasetFactory"
    assert result.dataset_instantiation_time_seconds == 0.5
    assert result.dataloader_instantiation_time_seconds == 0.3

    # Verify all instantiation metrics when using dataset factory
    assert result.peak_memory_during_instantiation_mb == 70.0
    assert result.memory_before_instantiation_mb == 50.0  # dataset baseline
    assert result.memory_after_instantiation_mb == 110.0  # dataloader final memory
    assert result.disk_size_mb == 75.0  # From mock get_disk_size


@mock.patch("bionemo.scbenchmark.benchmark.get_disk_size")
@mock.patch("bionemo.scbenchmark.benchmark.measure_peak_memory_full")
@mock.patch("bionemo.scbenchmark.benchmark.export_benchmark_results")
@mock.patch("bionemo.scbenchmark.benchmark._drop_caches")
@mock.patch("builtins.print")
def test_benchmark_no_data_path(mock_print, mock_drop_caches, mock_export, mock_measure_memory, mock_get_disk_size):
    """Test benchmarking with combined dataloader factory."""
    mock_get_disk_size.return_value = 25.0

    mock_measure_memory.side_effect = [
        (MockDataloader([MockBatch(16)], 0), 100.0, 130.0, 115.0, 30.0, 120.0, 0.5),
        ((16, 1, 0.8, 0, 0, 0.0), 100.0, 125.0, 110.0, 25.0, 115.0, 1.0),
    ]

    def dataloader_factory():
        return MockDataloader([MockBatch(16)], num_workers=0)

    result = benchmark_single_dataloader(
        dataloader_factory=dataloader_factory,
        data_path="/test/combined",
        name="CombinedFactory",
        dataset_factory=None,  # Dataloader creates everything internally
        warmup_time_seconds=0.0,
    )

    assert result.disk_size_mb == 25.0  # From mock get_disk_size


# benchmark_dataloaders_with_configs tests
@mock.patch("bionemo.scbenchmark.benchmark.benchmark_single_dataloader")
@mock.patch("bionemo.scbenchmark.benchmark._drop_caches")
def test_benchmark_multiple_configs_basic(mock_drop_caches, mock_benchmark_single):
    """Test benchmarking multiple dataloader configurations."""
    # Mock results for each config
    mock_benchmark_single.side_effect = [
        BenchmarkResult(
            name="Config1",
            disk_size_mb=100,
            epoch_results=[{"samples": 100, "elapsed": 1.0, "peak_memory": 150, "avg_memory": 130}],
        ),
        BenchmarkResult(
            name="Config2",
            disk_size_mb=100,
            epoch_results=[{"samples": 200, "elapsed": 2.0, "peak_memory": 180, "avg_memory": 160}],
        ),
    ]

    def factory1():
        return MockDataloader([MockBatch(32)], 0)

    def factory2():
        return MockDataloader([MockBatch(64)], 2)

    configs = [
        {"name": "Config1", "dataloader_factory": factory1, "data_path": "/data1", "max_batches": 10},
        {"name": "Config2", "dataloader_factory": factory2, "data_path": "/data2", "num_workers": 2},
    ]

    results = benchmark_dataloaders_with_configs(
        configs,
        shared_dataset_factory=None,  # No shared dataset, each config creates its own
    )

    assert len(results) == 2
    assert results[0].name == "Config1"
    assert results[1].name == "Config2"
    assert mock_benchmark_single.call_count == 2


@mock.patch("bionemo.scbenchmark.benchmark.measure_peak_memory_full")
@mock.patch("bionemo.scbenchmark.benchmark.benchmark_single_dataloader")
@mock.patch("bionemo.scbenchmark.benchmark._drop_caches")
def test_benchmark_with_shared_dataset_factory(mock_drop_caches, mock_benchmark_single, mock_measure_memory):
    """Test benchmarking with shared dataset factory."""
    # Mock dataset creation
    mock_measure_memory.return_value = ("shared_dataset", 50.0, 80.0, 65.0, 30.0, 70.0, 1.0)

    # Mock benchmark results
    result1 = BenchmarkResult(
        name="Shared1", epoch_results=[{"samples": 50, "elapsed": 0.5, "peak_memory": 120, "avg_memory": 100}]
    )
    result2 = BenchmarkResult(
        name="Shared2", epoch_results=[{"samples": 75, "elapsed": 0.8, "peak_memory": 140, "avg_memory": 115}]
    )
    mock_benchmark_single.side_effect = [result1, result2]

    def dataset_factory():
        return "shared_dataset"

    def factory1(dataset):
        return MockDataloader([MockBatch(16)], 0)

    def factory2(dataset):
        return MockDataloader([MockBatch(24)], 1)

    configs = [
        {"name": "Shared1", "dataloader_factory": factory1},
        {"name": "Shared2", "dataloader_factory": factory2},
    ]

    results = benchmark_dataloaders_with_configs(dataloader_configs=configs, shared_dataset_factory=dataset_factory)

    assert len(results) == 2
    # Both results should have the shared dataset instantiation time
    assert results[0].dataset_instantiation_time_seconds == 1.0
    assert results[1].dataset_instantiation_time_seconds == 1.0


def test_benchmark_configs_missing_factory():
    """Test error handling for missing dataloader_factory."""
    configs = [
        {"name": "MissingFactory", "data_path": "/data"}  # No dataloader_factory
    ]

    with pytest.raises(ValueError, match="missing a 'dataloader_factory'"):
        benchmark_dataloaders_with_configs(configs, shared_dataset_factory=None)


@mock.patch("bionemo.scbenchmark.benchmark.benchmark_single_dataloader")
@mock.patch("bionemo.scbenchmark.benchmark._drop_caches")
def test_benchmark_configs_with_none_factory(mock_drop_caches, mock_benchmark_single):
    """Test error handling for None dataloader_factory."""
    configs = [{"name": "NoneFactory", "dataloader_factory": None}]

    with pytest.raises(ValueError, match="missing a 'dataloader_factory'"):
        benchmark_dataloaders_with_configs(configs, shared_dataset_factory=None)


# E2E Integration Tests with Real Synthetic Data


class SyntheticDataset(Dataset):
    """Synthetic dataset for e2e testing."""

    def __init__(self, num_samples=1000, num_features=100):
        """Create synthetic dataset with controllable size.

        Args:
            num_samples: Number of samples in the dataset
            num_features: Number of features per sample
        """
        self.num_samples = num_samples
        self.num_features = num_features
        # Create reproducible synthetic data
        torch.manual_seed(42)
        self.data = torch.randn(num_samples, num_features)
        self.labels = torch.randint(0, 10, (num_samples,))

        # Add some larger tensors to ensure measurable memory usage
        self.extra_data = torch.randn(num_samples, num_features * 2)  # Double size for memory impact

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "data": self.data[idx],
            "label": self.labels[idx],
            "extra": self.extra_data[idx],  # Additional data to increase memory footprint
        }


class MemoryIntensiveDataset(Dataset):
    """Alternative synthetic dataset that uses significant memory."""

    def __init__(self, num_samples=500, num_features=1000):
        """Create memory-intensive dataset.

        Args:
            num_samples: Number of samples in the dataset
            num_features: Number of features per sample (larger for more memory)
        """
        self.num_samples = num_samples
        self.num_features = num_features
        torch.manual_seed(42)

        # Create multiple large tensors to ensure measurable memory
        self.data1 = torch.randn(num_samples, num_features)
        self.data2 = torch.randn(num_samples, num_features)
        self.data3 = torch.randn(num_samples, num_features)
        self.labels = torch.randint(0, 10, (num_samples,))

        # Calculate approximate memory usage
        tensor_size_mb = (num_samples * num_features * 4 * 3) / (1024 * 1024)  # 3 tensors, 4 bytes per float32
        print(f"MemoryIntensiveDataset: ~{tensor_size_mb:.1f}MB")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "data1": self.data1[idx],
            "data2": self.data2[idx],
            "data3": self.data3[idx],
            "label": self.labels[idx],
        }


@pytest.fixture
def synthetic_dataset_factory():
    """Fixture providing a factory for creating synthetic datasets."""

    def factory(num_samples=1000):
        return SyntheticDataset(num_samples=num_samples, num_features=100)

    return factory


@pytest.fixture
def synthetic_dataloader_factory():
    """Fixture providing a factory for creating synthetic dataloaders."""

    def factory(num_samples=1000, batch_size=32, num_workers=0):
        def dataloader_factory():
            dataset = SyntheticDataset(num_samples=num_samples, num_features=100)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

        return dataloader_factory

    return factory


@pytest.fixture
def memory_intensive_dataloader_factory():
    """Fixture providing a factory for creating memory-intensive dataloaders that guarantee measurable memory usage."""

    def factory(num_samples=500, batch_size=32, num_workers=0):
        def dataloader_factory():
            dataset = MemoryIntensiveDataset(num_samples=num_samples, num_features=1000)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

        return dataloader_factory

    return factory


@pytest.fixture
def synthetic_dataloader_from_dataset_factory():
    """Fixture providing a factory that creates dataloaders from existing datasets."""

    def factory(batch_size=32, num_workers=0):
        def dataloader_factory(dataset):
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

        return dataloader_factory

    return factory


@pytest.fixture(autouse=True)
def memory_cleanup():
    """Fixture that cleans up memory before and after each test."""
    # Clean up before test
    manual_memory_cleanup()

    yield  # Run the test

    # Clean up after test
    manual_memory_cleanup()


def manual_memory_cleanup():
    """Public helper function for manual memory cleanup in tests.

    Call this function explicitly in tests when you need extra memory cleanup
    between operations within a single test.

    Example:
        def test_something():
            # ... do some work ...
            manual_memory_cleanup()  # Clean up before next step
            # ... continue test ...
    """
    # Multiple aggressive GC rounds
    for _ in range(5):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # Clear torch-specific memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Reset torch random state for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Clear numpy memory
    if hasattr(np, "clear_cache"):
        np.clear_cache()

    # Clear any module-level caches
    if "torch" in globals():
        # Clear torch tensor cache
        if hasattr(torch, "set_default_tensor_type"):
            torch.set_default_tensor_type(torch.FloatTensor)

    # Force release of any lingering references
    import sys

    if hasattr(sys, "intern"):
        # Clear string interning cache (if accessible)
        pass

    if hasattr(tqdm, "_instances"):
        tqdm._instances.clear()


@pytest.fixture
def e2e_memory_isolation():
    """Extra aggressive memory isolation for e2e tests with memory monitoring."""
    import psutil

    # Get baseline memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024

    # Aggressive cleanup before test
    manual_memory_cleanup()

    yield initial_memory

    # Cleanup and monitoring after test
    manual_memory_cleanup()

    final_memory = process.memory_info().rss / 1024 / 1024
    memory_diff = final_memory - initial_memory

    if memory_diff > 50:  # More than 50MB increase
        print(f"⚠️  Memory increased by {memory_diff:.1f}MB during test")


# E2E Integration Tests - Consolidated


def _create_synthetic_dataloader_factory(num_samples=1000, batch_size=32, num_workers=0):
    """Helper function to create synthetic dataloader factories without fixtures."""

    def dataloader_factory():
        dataset = SyntheticDataset(num_samples=num_samples, num_features=100)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

    return dataloader_factory


def _create_synthetic_dataloader_from_dataset_factory(batch_size=32, num_workers=0):
    """Helper function to create dataloader factories that work with existing datasets."""

    def dataloader_factory(dataset):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

    return dataloader_factory


def _validate_benchmark_result(result, test_name="", debug=True):
    """Helper function to validate benchmark results with consistent assertions."""
    if debug:
        print(f"DEBUG {test_name}: {result.name}")
        print(f"  - samples_per_second: {result.samples_per_second}")
        print(f"  - peak_memory_mb: {result.peak_memory_mb}")
        print(f"  - avg_memory_mb: {result.avg_memory_mb}")
        print(f"  - memory_before_instantiation_mb: {result.memory_before_instantiation_mb}")
        if result.epoch_results:
            print(f"  - epoch_results[0]: {result.epoch_results[0]}")

    # Basic structure validation
    assert isinstance(result, BenchmarkResult)
    assert result.samples_per_second > 0, f"{test_name}: Expected samples/s > 0, got {result.samples_per_second}"

    # Memory should not be NaN and should be non-negative (handle None values gracefully)
    peak_memory = result.peak_memory_mb if result.peak_memory_mb is not None else 0.0
    avg_memory = result.avg_memory_mb if result.avg_memory_mb is not None else 0.0

    assert not math.isnan(peak_memory), f"{test_name}: Peak memory is NaN"
    assert not math.isnan(avg_memory), f"{test_name}: Avg memory is NaN"
    assert peak_memory >= 0, f"{test_name}: Peak memory should be >= 0, got {peak_memory}"
    assert avg_memory >= 0, f"{test_name}: Avg memory should be >= 0, got {avg_memory}"

    # Timing should be non-negative (handle None values)
    dataset_time = (
        result.dataset_instantiation_time_seconds if result.dataset_instantiation_time_seconds is not None else 0.0
    )
    dataloader_time = (
        result.dataloader_instantiation_time_seconds
        if result.dataloader_instantiation_time_seconds is not None
        else 0.0
    )

    assert dataset_time >= 0, f"{test_name}: Dataset time should be >= 0, got {dataset_time}"
    assert dataloader_time >= 0, f"{test_name}: Dataloader time should be >= 0, got {dataloader_time}"

    # Should have epoch results with valid data
    assert result.epoch_results is not None, f"{test_name}: Epoch results should not be None"
    assert len(result.epoch_results) > 0, f"{test_name}: Should have at least one epoch result"

    epoch_result = result.epoch_results[0]
    assert epoch_result["samples"] > 0, f"{test_name}: Expected samples > 0, got {epoch_result['samples']}"
    assert epoch_result["batches"] > 0, f"{test_name}: Expected batches > 0, got {epoch_result['batches']}"
    assert epoch_result["elapsed"] > 0, f"{test_name}: Expected elapsed time > 0, got {epoch_result['elapsed']}"


@pytest.mark.parametrize(
    "test_scenario",
    [
        # (name, num_samples, batch_size, num_workers, description)
        ("small_dataset", 200, 16, 0, "Small dataset test"),
        ("medium_dataset", 500, 32, 0, "Medium dataset test"),
        ("large_batch", 300, 64, 0, "Large batch size test"),
    ],
)
@pytest.mark.slow
def test_e2e_comprehensive_benchmark_scenarios(tmpdir, e2e_memory_isolation, test_scenario):
    """Comprehensive e2e test covering various dataset sizes and batch configurations."""

    scenario_name, num_samples, batch_size, num_workers, description = test_scenario
    initial_memory_mb = e2e_memory_isolation
    print(f"\n--- {description} ---")
    print(f"Starting memory: {initial_memory_mb:.1f}MB")

    manual_memory_cleanup()

    # Test single dataloader
    result = benchmark_single_dataloader(
        dataloader_factory=_create_synthetic_dataloader_factory(
            num_samples=num_samples, batch_size=batch_size, num_workers=num_workers
        ),
        data_path=str(tmpdir),
        name=f"E2E_{scenario_name}",
        num_epochs=1,
        max_batches=5,
        max_time_seconds=5.0,
        warmup_time_seconds=0.0,
        num_runs=1,
    )

    _validate_benchmark_result(result, f"Single-{scenario_name}")

    # Test multiple configurations
    configs = [
        {
            "name": f"E2E_{scenario_name}_Config_A",
            "dataloader_factory": _create_synthetic_dataloader_factory(
                num_samples=num_samples // 2, batch_size=batch_size // 2, num_workers=0
            ),
            "data_path": str(tmpdir),
            "max_batches": 3,
            "max_time_seconds": 3.0,
            "warmup_time_seconds": 0.0,
            "num_runs": 1,
        },
        {
            "name": f"E2E_{scenario_name}_Config_B",
            "dataloader_factory": _create_synthetic_dataloader_factory(
                num_samples=num_samples // 2, batch_size=batch_size, num_workers=0
            ),
            "data_path": str(tmpdir),
            "max_batches": 3,
            "max_time_seconds": 3.0,
            "warmup_time_seconds": 0.0,
            "num_runs": 1,
        },
    ]

    manual_memory_cleanup()

    multi_results = benchmark_dataloaders_with_configs(
        dataloader_configs=configs, shared_dataset_factory=None, output_prefix=f"e2e_{scenario_name}_test"
    )

    assert len(multi_results) == 2, f"Expected 2 results, got {len(multi_results)}"
    for i, result in enumerate(multi_results):
        _validate_benchmark_result(result, f"Multi-{scenario_name}-{i}")


@pytest.mark.slow
def test_e2e_advanced_features(tmpdir, synthetic_dataset_factory, e2e_memory_isolation):
    """Test advanced benchmarking features: dataset reuse, multiple runs, error handling."""

    initial_memory_mb = e2e_memory_isolation
    print(f"\nStarting memory: {initial_memory_mb:.1f}MB")
    manual_memory_cleanup()

    # Test 1: Dataset reuse pattern
    print("\n--- Testing Dataset Reuse ---")
    configs = [
        {
            "name": "E2E_Reuse_A",
            "data_path": str(tmpdir),
            "dataloader_factory": _create_synthetic_dataloader_from_dataset_factory(batch_size=16, num_workers=0),
            "max_batches": 3,
            "max_time_seconds": 3.0,
            "warmup_time_seconds": 0.0,
        },
        {
            "name": "E2E_Reuse_B",
            "data_path": str(tmpdir),
            "dataloader_factory": _create_synthetic_dataloader_from_dataset_factory(batch_size=32, num_workers=0),
            "max_batches": 3,
            "max_time_seconds": 3.0,
            "warmup_time_seconds": 0.0,
        },
    ]

    def shared_dataset_factory_func():
        return synthetic_dataset_factory(num_samples=200)

    reuse_results = benchmark_dataloaders_with_configs(
        dataloader_configs=configs, shared_dataset_factory=shared_dataset_factory_func, output_prefix="e2e_reuse_test"
    )

    assert len(reuse_results) == 2
    # Both should have same dataset instantiation time (from shared dataset)
    dataset_time_1 = reuse_results[0].dataset_instantiation_time_seconds
    dataset_time_2 = reuse_results[1].dataset_instantiation_time_seconds
    assert dataset_time_1 == dataset_time_2, (
        f"Shared dataset times should be equal: {dataset_time_1} vs {dataset_time_2}"
    )

    for i, result in enumerate(reuse_results):
        _validate_benchmark_result(result, f"Reuse-{i}")

    manual_memory_cleanup()

    # Test 2: Multiple runs consistency
    print("\n--- Testing Multiple Runs ---")
    multi_run_results = benchmark_single_dataloader(
        dataloader_factory=_create_synthetic_dataloader_factory(num_samples=300, batch_size=20, num_workers=0),
        data_path=str(tmpdir),
        name="E2E_MultiRun",
        num_epochs=1,
        max_batches=4,
        max_time_seconds=4.0,
        warmup_time_seconds=0.0,
        num_runs=2,
    )

    assert isinstance(multi_run_results, list), f"Expected list for multiple runs, got {type(multi_run_results)}"
    assert len(multi_run_results) == 2, f"Expected 2 results, got {len(multi_run_results)}"
    assert multi_run_results[0].name == "E2E_MultiRun_run_1"
    assert multi_run_results[1].name == "E2E_MultiRun_run_2"

    for i, result in enumerate(multi_run_results):
        _validate_benchmark_result(result, f"MultiRun-{i + 1}")

        # Consistency checks between runs
        if i > 0:
            prev_result = multi_run_results[0]
            assert result.peak_memory_during_instantiation_mb == prev_result.peak_memory_during_instantiation_mb, (
                "Instantiation metrics should be consistent across runs"
            )

    # Test 3: Error handling - Factory signature mismatch
    print("\n--- Testing Error Handling ---")

    def bad_factory():
        def dataloader_factory():  # Missing dataset parameter for shared use
            dataset = SyntheticDataset(num_samples=50, num_features=50)
            return DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=False)

        return dataloader_factory

    def good_dataset_factory():
        return SyntheticDataset(num_samples=50, num_features=50)

    error_configs = [
        {
            "name": "Bad_Factory",
            "dataloader_factory": bad_factory(),
            "max_batches": 1,
            "max_time_seconds": 1.0,
        }
    ]

    with pytest.raises(TypeError, match="takes 0 positional arguments but 1 was given"):
        benchmark_dataloaders_with_configs(
            dataloader_configs=error_configs, shared_dataset_factory=good_dataset_factory, output_prefix="error_test"
        )

    print("✓ All advanced features working correctly")


def _create_memory_intensive_dataloader_factory(num_samples=500, batch_size=32, num_workers=0):
    """Helper function to create memory-intensive dataloader factories."""

    def dataloader_factory():
        dataset = MemoryIntensiveDataset(num_samples=num_samples, num_features=1000)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

    return dataloader_factory


@pytest.mark.slow
@pytest.mark.memory_intensive
def test_e2e_memory_intensive_benchmark(tmpdir, e2e_memory_isolation):
    """Optional memory-intensive test for environments where you want to verify actual memory measurement."""

    initial_memory_mb = e2e_memory_isolation
    print("\n--- Memory Intensive Test ---")
    print(f"Starting memory: {initial_memory_mb:.1f}MB")
    manual_memory_cleanup()

    result = benchmark_single_dataloader(
        dataloader_factory=_create_memory_intensive_dataloader_factory(num_samples=400, batch_size=32, num_workers=0),
        data_path=str(tmpdir),
        name="E2E_Memory_Test",
        num_epochs=1,
        max_batches=3,
        max_time_seconds=8.0,
        warmup_time_seconds=0.0,
        num_runs=1,
    )

    _validate_benchmark_result(result, "MemoryIntensive")

    # This test might actually show positive memory with the larger dataset
    print(f"Memory measurement successful - Peak: {result.peak_memory_mb:.1f}MB, Avg: {result.avg_memory_mb:.1f}MB")
