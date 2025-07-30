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


from unittest import mock

import pytest

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
        assert result.peak_memory_during_instantiation_mb == 140.0
        assert result.memory_before_instantiation_mb == 100.0
        assert result.memory_after_instantiation_mb == 125.0
        assert result.dataset_instantiation_time_seconds == 0.8  # Combined time when no dataset factory
        assert result.dataloader_instantiation_time_seconds == 0.0  # Always 0 when no separate dataset factory
        assert result.disk_size_mb == 50.0  # From mock get_disk_size


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
    assert result.peak_memory_during_instantiation_mb == 120.0  # max(dl_peak=120.0, dataset_peak=80.0)
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
            name="Config1", epoch_results=[{"samples": 100, "elapsed": 1.0, "peak_memory": 150, "avg_memory": 130}]
        ),
        BenchmarkResult(
            name="Config2", epoch_results=[{"samples": 200, "elapsed": 2.0, "peak_memory": 180, "avg_memory": 160}]
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
