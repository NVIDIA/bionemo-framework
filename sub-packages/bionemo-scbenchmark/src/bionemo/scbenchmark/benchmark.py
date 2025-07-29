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
import mmap
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from tqdm import tqdm

from bionemo.scbenchmark.common import (
    BenchmarkResult,
    export_benchmark_results,
    get_batch_size,
    get_disk_size,
    measure_peak_memory_full,
)


"""Benchmarking framework for any dataloader.

This module provides a comprehensive framework for benchmarking any dataloader
implementation. It supports both simple direct benchmarking and factory-based
benchmarking, with features like time-based limits, warmup phases, instantiation
measurement, and comprehensive performance metrics.
"""


def _drop_caches():
    """Helper function to drop system caches."""
    try:
        print("Dropping caches")
        subprocess.run(["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
    except subprocess.CalledProcessError:
        print("⚠️ Warning: failed to drop caches — are you running with sudo?")


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
        data_path: Path to data files (for disk size measurement)
    """

    name: str
    num_epochs: int = 1
    max_batches: Optional[int] = None
    max_time_seconds: Optional[float] = None
    warmup_batches: Optional[int] = None
    warmup_time_seconds: Optional[float] = None
    data_path: Optional[Union[str, Path]] = None
    madvise_interval: Optional[int] = None
    shuffle: bool = False
    num_workers: Optional[int] = None


def run_benchmark(
    dataloader: Any,
    config: BenchmarkConfig,
    run_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    **instantiation_kwargs,
) -> BenchmarkResult:
    """Run the actual benchmark and collect metrics.

    Args:
        dataloader: The dataloader to benchmark
        config: Configuration for the benchmark run
        run_name: Optional name for this run (used in iteration time logging)
        output_dir: Optional directory to save iteration time logs
        **instantiation_kwargs: Instantiation metrics (dataset_instantiation_time_seconds,
                               dataloader_instantiation_time_seconds, peak_memory_during_instantiation_mb,
                               memory_before_instantiation_mb, memory_after_instantiation_mb)

    Returns:
        BenchmarkResult containing all collected data and calculated metrics
    """
    # Use measure_peak_memory_full to get memory info during benchmark
    gc.collect()
    _drop_caches()

    def benchmark_iteration_single_epoch(epoch_num, do_warmup):
        """Run a single epoch of benchmarking, with optional warmup."""
        gc.collect()

        update_interval = 10
        epoch_samples = 0
        epoch_batches = 0
        warmup_samples = 0
        warmup_batches = 0
        warmup_time = 0.0
        pbar = tqdm(desc=f"{config.name} - Epoch {epoch_num + 1}/{config.num_epochs}")
        warm_up_start = time.perf_counter()
        if not do_warmup:
            config.warmup_time_seconds = 0
        warm_up_end = warm_up_start + config.warmup_time_seconds
        is_warming_up = True
        for num, batch in enumerate(dataloader):
            batch_size = get_batch_size(batch)

            if config.madvise_interval and num % config.madvise_interval == 0:
                dataloader.dataset.row_index._mmap.madvise(mmap.MADV_DONTNEED)
                dataloader.dataset.col_index._mmap.madvise(mmap.MADV_DONTNEED)
                dataloader.dataset.data._mmap.madvise(mmap.MADV_DONTNEED)
            current_time = time.perf_counter()

            if is_warming_up:
                # We're in warm-up period - count samples and batches
                warmup_samples += batch_size
                warmup_batches += 1

                if current_time >= warm_up_end:
                    # Warm-up complete and start the actual timing
                    warmup_time = current_time - warm_up_start

                    print(f"🔥 Warmup completed: {warmup_samples:,} samples, {warmup_batches:,} batches")

                    is_warming_up = False
                    start_time = time.perf_counter()
                    end_time = start_time + config.max_time_seconds if config.max_time_seconds is not None else None
                    pbar.set_description(f"{config.name} - Epoch {epoch_num + 1} (warmup complete)")
                else:
                    if warmup_batches % update_interval == 0:
                        elapsed_warmup = current_time - warm_up_start
                        current_warmup_speed = warmup_samples / elapsed_warmup if elapsed_warmup > 0 else 0
                        pbar.set_description(
                            f"{config.name} - Warmup: {elapsed_warmup:.1f}/{config.warmup_time_seconds}s, {current_warmup_speed:.1f} samples/sec"
                        )
                        pbar.update(update_interval)
                continue

            # Now we're past the warm-up period (or no warmup)
            epoch_samples += batch_size
            epoch_batches += 1
            elapsed = current_time - start_time if start_time else 0
            if epoch_batches % update_interval == 0:
                postfix_dict = {
                    "epoch": f"{epoch_num + 1}/{config.num_epochs}",
                    "samples": epoch_samples,
                    "elapsed": f"{elapsed:.2f}s",
                }

                pbar.set_postfix(**postfix_dict, refresh=False)
                pbar.update(update_interval)
            # Check max_batches limit
            if (config.max_batches and epoch_batches >= config.max_batches) or (end_time and current_time >= end_time):
                break

        pbar.close()

        return epoch_samples, epoch_batches, elapsed, warmup_samples, warmup_batches, warmup_time

    epoch_results = []
    for epoch in range(config.num_epochs):
        # Create a modified benchmark_iteration for this epoch

        result_tuple = measure_peak_memory_full(
            lambda: benchmark_iteration_single_epoch(epoch, epoch == 0), multi_worker=dataloader.num_workers > 0
        )
        (
            (epoch_samples, epoch_batches, elapsed, warmup_samples, warmup_batches, warmup_time),
            _,
            peak,
            avg,
            _,
            _,
            iteration_time,
        ) = result_tuple
        # Accumulate warmup data (only first epoch will have non-zero values)

        epoch_results.append(
            {
                "epoch": epoch + 1,
                "samples": epoch_samples,
                "batches": epoch_batches,
                "warmup_samples": warmup_samples,
                "warmup_batches": warmup_batches,
                "peak_memory": peak,
                "avg_memory": avg,
                "iteration_time": iteration_time,
                "elapsed": elapsed,
                "warmup_time": warmup_time,
            }
        )

        print(f"📊 Epoch {epoch + 1} completed: {epoch_samples:,} samples, {epoch_batches:,} batches")

    result = BenchmarkResult(
        name=config.name,
        madvise_interval=config.madvise_interval,
        data_path=str(config.data_path) if config.data_path else None,
        max_time_seconds=config.max_time_seconds,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        # Instantiation metrics passed as kwargs
        **instantiation_kwargs,
        epoch_results=epoch_results,
    )

    return result


def benchmark_dataloader(
    name: Optional[str] = None,
    dataloader_factory: Optional[Callable[[], Any]] = None,
    dataset_factory: Optional[Callable[[], Any]] = None,  # 🆕 NEW: For dataset reuse
    dataloaders: Optional[List[Dict[str, Any]]] = None,
    data_path: Optional[Union[str, Path]] = None,
    num_epochs: int = 1,
    max_batches: Optional[int] = None,
    max_time_seconds: Optional[float] = None,
    warmup_batches: int = 5,
    warmup_time_seconds: Optional[float] = None,
    madvise_interval: Optional[int] = None,
    shuffle: bool = False,
    num_runs: int = 1,
    output_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
) -> Union[BenchmarkResult, List[BenchmarkResult]]:
    """Benchmark one or multiple dataloaders using factory functions.

    This function can handle both single dataloader benchmarking and multiple
    dataloader comparison. It measures both instantiation time/memory and
    iteration performance to give you a complete picture of dataloader performance.

    Args:
        # Single dataloader mode parameters:
        name: Name for this dataloader benchmark
        dataloader_factory: Function that creates a dataloader object
        data_path: Path to data files (for disk size measurement)

        # Dataset reuse mode:
        dataset_factory: Function that creates a dataset once, then reused across multiple dataloaders
                        When provided, dataloader_factory functions receive the dataset as an argument

        # Multiple dataloaders mode parameter:
        dataloaders: List of dicts with keys: name, dataloader_factory, data_path, etc.
                    Each dict's keys are passed as kwargs to benchmark_dataloader

        # Common parameters:
        num_epochs: Number of epochs to run
        max_batches: Maximum number of batches to process (None for all)
        max_time_seconds: Maximum time to run benchmark (None for no limit)
        warmup_batches: Number of warmup batches
        warmup_time_seconds: Time to warmup in seconds (overrides warmup_batches if set)
        madvise_interval: Memory advice interval setting
        shuffle: Whether to shuffle the data
        num_runs: Number of times to run the benchmark (default: 1)
        output_dir: Directory to save individual results (for multiple dataloaders mode)
        num_workers: Number of worker processes for data loading

    Returns:
        Single BenchmarkResult if benchmarking one dataloader, or
        List[BenchmarkResult] if benchmarking multiple dataloaders.
        If num_runs > 1, returns individual results for each run.

    Examples:
        # Single dataloader
        result = benchmark_dataloader(
            name="My Dataloader",
            dataloader_factory=lambda: MyDataLoader(),
            max_time_seconds=30.0
        )

        # Dataset reuse mode
        result = benchmark_dataloader(
            dataset_factory=lambda: load_my_dataset(),  # Load once
            dataloaders=[
                {"name": "Config1", "dataloader_factory": lambda ds: DataLoader(ds, batch_size=32)},
                {"name": "Config2", "dataloader_factory": lambda ds: DataLoader(ds, batch_size=64)},
            ]
        )

    Note:
        - If both single and multiple parameters are provided, multiple mode takes precedence
        - For multiple dataloaders, each dict should have at least 'name' and 'dataloader_factory' keys
        - Add 'num_runs': N to run each configuration N times and get individual run results
        - If output_dir is provided in multiple mode, results are saved as JSON files
        - A comparison table is printed when benchmarking multiple dataloaders
        - When dataset_factory is provided, dataset is loaded once and reused across configs
        - Dataset vs dataloader instantiation times are tracked separately for dataset reuse
    """
    # Multiple dataloaders mode
    if dataloaders is not None:
        if dataset_factory is not None:
            return _benchmark_with_dataset_reuse(
                dataset_factory=dataset_factory,
                dataloaders=dataloaders,
                data_path=data_path,
                num_epochs=num_epochs,
                max_batches=max_batches,
                max_time_seconds=max_time_seconds,
                warmup_batches=warmup_batches,
                warmup_time_seconds=warmup_time_seconds,
                madvise_interval=madvise_interval,
                shuffle=shuffle,
                num_runs=num_runs,
                output_dir=output_dir,
                num_workers=num_workers,
            )
        else:
            # each config loads its own dataset
            results = []

            for dl_config in dataloaders:
                _drop_caches()

                # Extract parameters from config, with function parameters as defaults
                config_name = dl_config["name"]
                config_factory = dl_config["dataloader_factory"]

                result = _benchmark_single_dataloader(
                    name=config_name,
                    dataloader_factory=config_factory,
                    data_path=dl_config.get("data_path", data_path),
                    num_epochs=dl_config.get("num_epochs", num_epochs),
                    max_batches=dl_config.get("max_batches", max_batches),
                    max_time_seconds=dl_config.get("max_time_seconds", max_time_seconds),
                    warmup_batches=dl_config.get("warmup_batches", warmup_batches),
                    warmup_time_seconds=dl_config.get("warmup_time_seconds", warmup_time_seconds),
                    madvise_interval=dl_config.get("madvise_interval", madvise_interval),
                    shuffle=dl_config.get("shuffle", shuffle),
                    num_runs=dl_config.get("num_runs", 1),  # Default to 1 if not specified
                    output_dir=output_dir,
                    num_workers=dl_config.get("num_workers", num_workers),
                )
                # Handle both single result and list of results
                if isinstance(result, list):
                    results.extend(result)
                    # CSV already appended in _benchmark_single_dataloader for multiple runs
                    # Save individual results for multiple runs
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        for run_result in result:
                            run_result.save_to_file(os.path.join(output_dir, f"{run_result.name}_results.json"))
                else:
                    results.append(result)
                    # CSV already appended in _benchmark_single_dataloader for single runs
                    # Save single result
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        result.save_to_file(os.path.join(output_dir, f"{result.name}_results.json"))

            # Print comparison
            if len(results) > 1:
                _print_comparison(results)

            return results

    # Single dataloader mode
    elif name is not None and dataloader_factory is not None:
        return _benchmark_single_dataloader(
            name=name,
            dataloader_factory=dataloader_factory,
            data_path=data_path,
            num_epochs=num_epochs,
            max_batches=max_batches,
            max_time_seconds=max_time_seconds,
            warmup_batches=warmup_batches,
            warmup_time_seconds=warmup_time_seconds,
            madvise_interval=madvise_interval,
            shuffle=shuffle,
            num_runs=num_runs,
            output_dir=output_dir,
            num_workers=num_workers,
        )

    else:
        raise ValueError(
            "Must provide either:\n"
            "  1. (name + dataloader_factory) for single dataloader mode\n"
            "  2. (dataloaders) for multiple dataloaders mode\n"
            "  3. (dataset_factory + dataloaders) for dataset reuse mode"
        )


def _benchmark_single_dataloader(
    name: str,
    dataloader_factory: Callable[[], Any],
    data_path: Optional[Union[str, Path]] = None,
    num_epochs: int = 1,
    max_batches: Optional[int] = None,
    max_time_seconds: Optional[float] = None,
    warmup_batches: int = 5,
    warmup_time_seconds: Optional[float] = None,
    madvise_interval: Optional[int] = None,
    shuffle: bool = False,
    num_runs: int = 1,
    output_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
) -> Union[BenchmarkResult, List[BenchmarkResult]]:
    """Benchmark a single dataloader using a factory function.

    This function measures both instantiation time/memory and iteration performance
    to give you a complete picture of dataloader performance. It uses a factory
    function to create the dataloader, allowing for flexible benchmarking without
    requiring inheritance or modifications to existing code.

    Args:
        name: Name of the benchmark
        dataloader_factory: Factory function that creates a dataloader
        data_path: Path to the data file (optional)
        num_epochs: Number of epochs to run
        max_batches: Maximum number of batches per epoch (None for unlimited)
        max_time_seconds: Maximum time to run in seconds (None for unlimited)
        warmup_batches: Number of batches for warmup
        warmup_time_seconds: Time in seconds for warmup
        madvise_interval: Interval for memory advice calls
        shuffle: Whether to shuffle the data
        num_runs: Number of runs to perform
        output_dir: Directory to save results
        num_workers: Number of worker processes for data loading

    Returns:
        Single BenchmarkResult for num_runs=1, or List[BenchmarkResult] for multiple runs

    Note:
        - If both max_batches and max_time_seconds are set, the first limit reached stops the benchmark
        - If both warmup_batches and warmup_time_seconds are set, warmup_time_seconds takes precedence
        - Instantiation is measured by default to give complete performance picture
        - The factory function is called once for instantiation measurement and once for setup
        - For existing dataloaders, simply wrap them in a factory function: lambda: your_dataloader
    """
    print(f"🔧 Benchmarking: {name}")

    # Measure instantiation metrics if requested
    dataloader, baseline, peak, _, _, final_mib, setup_time = measure_peak_memory_full(dataloader_factory)

    # Measure memory after

    instantiation_metrics = {
        "peak_memory_during_instantiation_mb": peak,
        "memory_after_instantiation_mb": final_mib,
        "memory_before_instantiation_mb": baseline,
        "dataset_instantiation_time_seconds": setup_time,  # Full setup time (includes dataloader)
        "dataloader_instantiation_time_seconds": 0.0,  # Included in dataset time
    }

    # Measure disk size
    disk_size_mb = 0.0
    if data_path:
        disk_size_mb = get_disk_size(data_path)

    # Create config
    config = BenchmarkConfig(
        name=name,
        num_epochs=num_epochs,
        max_batches=max_batches,
        max_time_seconds=max_time_seconds,
        warmup_batches=warmup_batches,
        warmup_time_seconds=warmup_time_seconds,
        data_path=data_path,
        madvise_interval=madvise_interval,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    # Run benchmark(s)
    if num_runs == 1:
        # Single run
        result = run_benchmark(dataloader, config, name, output_dir, **instantiation_metrics)
        del dataloader

        gc.collect()
        # Update timing and disk info
        result.disk_size_mb = disk_size_mb

        # Print results
        _print_results(result)

        # 🆕 Export CSV after single run
        if output_dir:
            # Use output directory name as part of CSV prefix for consolidated results
            dir_name = os.path.basename(output_dir.rstrip("/"))
            csv_prefix = f"{dir_name}_consolidated_results"
        else:
            csv_prefix = "consolidated_benchmark_results"
        export_benchmark_results(result, output_prefix=csv_prefix, output_dir=output_dir)

        return result
    else:
        # Multiple runs
        print(f"🏃 Running {num_runs} times...")
        all_results = []

        for run_idx in range(num_runs):
            # Create fresh dataloader for each run
            if run_idx > 0:  # We already have one from instantiation measurement
                dataloader = dataloader_factory()

            # Update config name for this run
            run_config = BenchmarkConfig(
                name=f"{name}_run_{run_idx + 1}",
                num_epochs=num_epochs,
                max_batches=max_batches,
                max_time_seconds=max_time_seconds,
                warmup_batches=warmup_batches,
                warmup_time_seconds=warmup_time_seconds,
                data_path=data_path,
                madvise_interval=madvise_interval,
                shuffle=shuffle,
                num_workers=num_workers,
            )

            run_result = run_benchmark(
                dataloader, run_config, f"{name}_run_{run_idx + 1}", output_dir, **instantiation_metrics
            )
            all_results.append(run_result)

            # Print samples per second after every run
            print(f"Run {run_idx + 1}: {run_result.samples_per_second:.2f} samples/sec")

            if output_dir:
                # Use output directory name as part of CSV prefix for consolidated results
                dir_name = os.path.basename(output_dir.rstrip("/"))
                csv_prefix = f"{dir_name}_consolidated_results"
            else:
                csv_prefix = "consolidated_benchmark_results"
            export_benchmark_results(run_result, output_prefix=csv_prefix, output_dir=output_dir)

            del dataloader

        return all_results


def _benchmark_with_dataset_reuse(
    dataset_factory: Callable[[], Any],
    dataloaders: List[Dict[str, Any]],
    data_path: Optional[Union[str, Path]] = None,
    num_epochs: int = 1,
    max_batches: Optional[int] = None,
    max_time_seconds: Optional[float] = None,
    warmup_batches: int = 5,
    warmup_time_seconds: Optional[float] = None,
    madvise_interval: Optional[int] = None,
    shuffle: bool = False,
    num_runs: int = 1,
    output_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
) -> List[BenchmarkResult]:
    """Benchmark multiple dataloader configs using a shared dataset (dataset reuse mode).

    This function loads a dataset ONCE using dataset_factory, then tests multiple
    dataloader configurations on the same dataset. This is much more efficient
    for large datasets and provides fair comparisons.

    Args:
        dataset_factory: Function that creates the dataset (called once)
        dataloaders: List of dataloader configurations to test
        data_path: Path to data files for disk usage measurement
        num_epochs: Number of epochs to run for each configuration
        max_batches: Maximum number of batches per epoch
        max_time_seconds: Maximum time in seconds per epoch
        warmup_batches: Number of warmup batches
        warmup_time_seconds: Warmup time in seconds
        madvise_interval: Memory advice interval
        shuffle: Whether to shuffle data
        num_runs: Number of runs per configuration
        output_dir: Directory to save result files
        num_workers: Number of worker processes

    Returns:
        List[BenchmarkResult] with separate dataset vs dataloader instantiation times
    """
    print("🚀 Dataset Reuse Mode: Loading dataset once, testing multiple configs")

    # 🆕 Step 1: Load dataset ONCE and measure separate instantiation metrics
    _drop_caches()
    dataset, dataset_baseline, dataset_peak, _, _, dataset_final, dataset_time = measure_peak_memory_full(
        dataset_factory
    )

    print(f"✅ Dataset loaded in {dataset_time:.3f}s (peak memory: {dataset_peak:.1f} MB)")

    # 🆕 Step 2: Dataset instantiation metrics tracked separately in each run

    # Measure disk size once
    disk_size_mb = 0.0
    if data_path:
        disk_size_mb = get_disk_size(data_path)

    # 🆕 Step 3: Test each dataloader configuration
    all_results = []
    total_configs = len(dataloaders)

    # Set up CSV prefix once for all runs
    if output_dir:
        # Use output directory name as part of CSV prefix for consolidated results
        dir_name = os.path.basename(output_dir.rstrip("/"))
        csv_prefix = f"{dir_name}_consolidated_results"
    else:
        csv_prefix = "consolidated_benchmark_results"

    for config_idx, dl_config in enumerate(dataloaders, 1):
        print(f"📊 Testing config {config_idx}/{total_configs}: {dl_config['name']}")

        config_name = dl_config["name"]
        config_dataloader_factory = dl_config["dataloader_factory"]

        # 🚨 Check for None dataloader_factory
        if config_dataloader_factory is None:
            raise ValueError(
                f"Configuration '{config_name}' has None dataloader_factory. "
                f"Check that the factory function returned a valid callable."
            )

        config_num_runs = dl_config.get("num_runs", num_runs)

        # 🆕 Step 4: For each run of this config
        config_results = []

        for run_idx in range(config_num_runs):
            run_name = f"{config_name}_run_{run_idx + 1}" if config_num_runs > 1 else config_name

            # 🆕 Step 5: Measure dataloader instantiation time (separate from dataset)
            def create_dataloader_from_dataset():
                return config_dataloader_factory(dataset)  # Pass pre-loaded dataset

            dataloader, dl_baseline, dl_peak, _, _, dl_final, dl_time = measure_peak_memory_full(
                create_dataloader_from_dataset
            )

            # 🆕 Step 6: Calculate SEPARATE instantiation metrics
            dataset_time_per_run = dataset_time / len(dataloaders) / config_num_runs  # Amortize dataset cost
            instantiation_metrics = {
                "peak_memory_during_instantiation_mb": max(dl_peak, dataset_peak),
                "memory_after_instantiation_mb": dl_final,
                "memory_before_instantiation_mb": dataset_baseline,  # Memory BEFORE dataloader (after dataset loaded)
                "dataset_instantiation_time_seconds": dataset_time_per_run,  # Amortized dataset time
                "dataloader_instantiation_time_seconds": dl_time,  # Pure dataloader time
            }

            # 🆕 Step 7: Create benchmark config for this specific run
            config_params = BenchmarkConfig(
                name=run_name,
                num_epochs=dl_config.get("num_epochs", num_epochs),
                max_batches=dl_config.get("max_batches", max_batches),
                max_time_seconds=dl_config.get("max_time_seconds", max_time_seconds),
                warmup_batches=dl_config.get("warmup_batches", warmup_batches),
                warmup_time_seconds=dl_config.get("warmup_time_seconds", warmup_time_seconds),
                data_path=dl_config.get("data_path", data_path),
                madvise_interval=dl_config.get("madvise_interval", madvise_interval),
                shuffle=dl_config.get("shuffle", shuffle),
                num_workers=dl_config.get("num_workers", num_workers),
            )

            # 🆕 Step 8: Run the actual benchmark
            result = run_benchmark(dataloader, config_params, run_name, output_dir, **instantiation_metrics)
            result.disk_size_mb = disk_size_mb

            config_results.append(result)
            all_results.append(result)

            print(
                f"   Run {run_idx + 1}: {result.samples_per_second:.2f} samples/sec "
                f"(dataset: {dataset_time_per_run:.3f}s, dataloader: {dl_time:.3f}s)"
            )

            # 🆕 Append to CSV after EVERY individual run
            export_benchmark_results(result, output_prefix=csv_prefix, output_dir=output_dir)

            # Clean up dataloader for next run
            del dataloader

    # 🆕 Step 9: Save results and print comparison
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for result in all_results:
            result.save_to_file(os.path.join(output_dir, f"{result.name}_results.json"))

    if len(all_results) > 1:
        _print_comparison(all_results)

    print(f"✅ Dataset reuse completed! Dataset loaded once, tested {len(dataloaders)} configs")

    return all_results


def _print_results(result: BenchmarkResult) -> None:
    """Print benchmark results in a formatted way."""
    print(f"\n📊 {result.name}")
    print(f"   Samples/sec: {result.samples_per_second:.2f}")
    print(f"   Memory: {result.peak_memory_mb:.1f} MB")

    if hasattr(result, "dataset_instantiation_time_seconds") and result.dataset_instantiation_time_seconds is not None:
        dataset_time = result.dataset_instantiation_time_seconds
        dataloader_time = getattr(result, "dataloader_instantiation_time_seconds", 0.0) or 0.0

        print(
            f"   Setup: {dataset_time + dataloader_time:.3f}s (dataset: {dataset_time:.3f}s + dataloader: {dataloader_time:.3f}s)"
        )

    if hasattr(result, "disk_size_mb") and result.disk_size_mb:
        print(f"   Disk: {result.disk_size_mb:.1f} MB")


def _print_comparison(results: List[BenchmarkResult]) -> None:
    """Print comparison of multiple benchmark results."""
    if not results or len(results) < 2:
        return

    print(f"\n📊 COMPARISON ({len(results)} configurations)")

    # Show individual results
    for result in results:
        print(f"\n📋 {result.name}: {result.samples_per_second:.2f} samples/sec")
        print(f"   Memory: {result.peak_memory_mb:.1f} MB")

    # Find best performers
    best_samples_per_sec = max(results, key=lambda r: r.samples_per_second)
    lowest_memory = min(results, key=lambda r: r.peak_memory_mb)

    print("\n🏆 BEST PERFORMERS:")
    print(f"🏆 Best speed: {best_samples_per_sec.name} ({best_samples_per_sec.samples_per_second:.2f} samples/sec)")
    print(f"🏆 Lowest memory: {lowest_memory.name} ({lowest_memory.peak_memory_mb:.2f} MB)")
