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


"""Benchmarking framework for any dataloader.

This module provides a comprehensive framework for benchmarking any dataloader
implementation. It supports both simple direct benchmarking and factory-based
benchmarking, with features like time-based limits, warmup phases, instantiation
measurement, and comprehensive performance metrics.
"""

import gc
import mmap
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import psutil
from tqdm import tqdm

from bionemo.scbenchmark.common import (
    BenchmarkResult,
    aggregate_benchmark_results,
    export_benchmark_results,
    get_batch_size,
    get_disk_size,
    measure_peak_memory_full,
)


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
        print_progress: Whether to print progress during benchmarking
        data_path: Path to data files (for disk size measurement)
    """

    name: str
    num_epochs: int = 1
    max_batches: Optional[int] = None
    max_time_seconds: Optional[float] = None
    warmup_batches: Optional[int] = None
    warmup_time_seconds: Optional[float] = None
    print_progress: bool = True
    data_path: Optional[Union[str, Path]] = None
    madvise_interval: Optional[int] = None
    shuffle: bool = False
    track_iteration_times: bool = False
    log_iteration_times_to_file: Optional[int] = None  # None for no logging, or integer interval
    num_workers: Optional[int] = None


def run_benchmark(
    dataloader: Any,
    config: BenchmarkConfig,
    run_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    instantiation_metrics: Optional[Dict[str, float]] = None,
) -> BenchmarkResult:
    """Run the actual benchmark and collect metrics.

    Args:
        dataloader: The dataloader to benchmark
        config: Configuration for the benchmark run
        run_name: Optional name for this run (used in iteration time logging)
        output_dir: Optional directory to save iteration time logs
        instantiation_metrics: Optional dictionary containing instantiation metrics

    Returns:
        BenchmarkResult containing all collected data and calculated metrics
    """
    # Use measure_peak_memory_full to get memory info during benchmark
    gc.collect()

    def benchmark_iteration_single_epoch(epoch_num, do_warmup):
        """Run a single epoch of benchmarking, with optional warmup."""
        update_interval = 10
        epoch_samples = 0
        epoch_batches = 0
        warmup_samples = 0
        warmup_batches = 0
        warmup_time = 0.0
        print("Starting benchmark for ", config.name, config.print_progress)
        pbar = tqdm(
            desc=f"{config.name} - Epoch {epoch_num + 1}/{config.num_epochs}", disable=not config.print_progress
        )

        # Initialize warm-up timer - only if this is the first epoch and warmup is requested
        if do_warmup and config.warmup_time_seconds is not None and config.warmup_time_seconds > 0:
            warm_up_start = time.perf_counter()
            warm_up_end = warm_up_start + config.warmup_time_seconds
            is_warming_up = True
            start_time = None  # Will be set after warmup
            end_time = None  # Will be set after warmup
        else:
            # No warmup - start timing immediately
            is_warming_up = False
            warm_up_start = None
            warm_up_end = None
            start_time = time.perf_counter()
            # If max_time_seconds is None, end_time is None and we iterate over the whole dataloader
            end_time = start_time + config.max_time_seconds if config.max_time_seconds is not None else None

        prev_time = None
        iteration_times = []
        for num, batch in enumerate(dataloader):
            batch_size = get_batch_size(batch)
            current_time = time.perf_counter()

            if config.madvise_interval and num % config.madvise_interval == 0:
                dataloader.dataset.row_index._mmap.madvise(mmap.MADV_DONTNEED)
                dataloader.dataset.col_index._mmap.madvise(mmap.MADV_DONTNEED)
                dataloader.dataset.data._mmap.madvise(mmap.MADV_DONTNEED)

            if config.track_iteration_times or config.log_iteration_times_to_file is not None:
                if prev_time is not None:
                    iter_time = current_time - prev_time

                    # Log to file if interval is set and it's the right batch
                    interval = config.log_iteration_times_to_file
                    if interval is not None and interval > 0 and (num % interval == 0):
                        rss_bytes = psutil.Process(os.getpid()).memory_info().rss
                        rss_mb = rss_bytes / (1024 * 1024)
                        iteration_times.append(
                            {
                                "epoch": epoch_num + 1,
                                "batch": num,
                                "iteration_time_s": iter_time,
                                "rss_mb": rss_mb,
                                "run_name": config.name if config.name else "benchmark",
                            }
                        )
            prev_time = current_time

            if is_warming_up:
                # We're in warm-up period - count samples and batches
                warmup_samples += batch_size
                warmup_batches += 1

                if current_time >= warm_up_end:
                    # Warm-up complete and start the actual timing
                    warmup_time = current_time - warm_up_start

                    if config.print_progress:
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
            # print("Batch:", num)
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
                # Add iteration timing info if tracking is enabled and we have timing data
                # if config.track_iteration_times and prev_time is not None and 'iter_time' in locals():
                #    postfix_dict["iter_time"] = f"{iter_time:.4f}s"
                pbar.set_postfix(**postfix_dict, refresh=False)
                pbar.update(update_interval)

            # Check max_batches limit
            if config.max_batches and epoch_batches >= config.max_batches:
                break

            # Check time limit
            # Only break if end_time is set (i.e., max_time_seconds is not None)
            if end_time is not None and current_time >= end_time:
                break

        pbar.close()

        return epoch_samples, epoch_batches, warmup_samples, warmup_batches, warmup_time, iteration_times

    epoch_results = []
    total_warmup_samples = 0
    total_warmup_batches = 0
    total_warmup_time = 0.0
    all_iteration_times = []

    for epoch in range(config.num_epochs):
        # Create a modified benchmark_iteration for this epoch
        def epoch_benchmark_iteration():
            return benchmark_iteration_single_epoch(epoch, epoch == 0)  # Only warmup on first epoch

        result_tuple = measure_peak_memory_full(epoch_benchmark_iteration)
        (
            (epoch_samples, epoch_batches, warmup_samples, warmup_batches, warmup_time, iteration_times),
            baseline,
            peak,
            avg,
            _,
            _,
            iteration_time,
        ) = result_tuple

        # Accumulate warmup data (only first epoch will have non-zero values)
        total_warmup_samples += warmup_samples
        total_warmup_batches += warmup_batches
        total_warmup_time += warmup_time
        all_iteration_times.extend(iteration_times)

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
                "warmup_time": warmup_time,
            }
        )

        if config.print_progress:
            print(f"📊 Epoch {epoch + 1} completed: {epoch_samples:,} samples, {epoch_batches:,} batches")

    # Calculate totals across all epochs
    total_samples = sum(r["samples"] for r in epoch_results)
    total_batches = sum(r["batches"] for r in epoch_results)
    total_iteration_time = sum(r["iteration_time"] for r in epoch_results)  # Fixed: use total time, not average

    # Print epoch results summary - removed verbose output

    result = BenchmarkResult.from_raw_metrics(
        name=config.name,
        madvise_interval=config.madvise_interval,
        data_path=str(config.data_path) if config.data_path else None,
        max_time_seconds=config.max_time_seconds,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        total_samples=total_samples,
        total_batches=total_batches,
        setup_time=0,  # Will be set by caller
        warmup_time=total_warmup_time,
        iteration_time=total_iteration_time,  # Fixed: use total time
        warmup_samples=total_warmup_samples,
        warmup_batches=total_warmup_batches,
        **(instantiation_metrics if instantiation_metrics else {}),
    )

    # Add epoch results to the result object for further analysis
    result.epoch_results = epoch_results

    # Update with memory information from measure_peak_memory_full (use max across epochs)
    max_peak_memory = max(r["peak_memory"] for r in epoch_results)
    avg_memory = sum(r["avg_memory"] for r in epoch_results) / len(epoch_results)

    result.peak_memory_mb = max_peak_memory
    result.average_memory_mb = avg_memory

    # After all epochs, if logging is enabled, write iteration times to file
    if config.log_iteration_times_to_file is not None and all_iteration_times:
        log_dir = output_dir if output_dir else os.getcwd()
        log_file = os.path.join(log_dir, f"{run_name or config.name}_iteration_times.csv")
        pd.DataFrame(all_iteration_times).to_csv(log_file, index=False)
        print(f"📄 Per-iteration times logged to {os.path.abspath(log_file)} ({len(all_iteration_times)} entries)")

    return result


def benchmark_dataloader(
    name: Optional[str] = None,
    dataloader_factory: Optional[Callable[[], Any]] = None,
    dataloaders: Optional[List[Dict[str, Any]]] = None,
    data_path: Optional[Union[str, Path]] = None,
    num_epochs: int = 1,
    max_batches: Optional[int] = None,
    max_time_seconds: Optional[float] = None,
    warmup_batches: int = 5,
    warmup_time_seconds: Optional[float] = None,
    print_progress: bool = True,
    madvise_interval: Optional[int] = None,
    shuffle: bool = False,
    num_runs: int = 1,
    output_dir: Optional[str] = None,
    track_iteration_times: bool = False,
    log_iteration_times_to_file: Optional[int] = None,
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

        # Multiple dataloaders mode parameter:
        dataloaders: List of dicts with keys: name, dataloader_factory, data_path, etc.
                    Each dict's keys are passed as kwargs to benchmark_dataloader

        # Common parameters:
        num_epochs: Number of epochs to run
        max_batches: Maximum number of batches to process (None for all)
        max_time_seconds: Maximum time to run benchmark (None for no limit)
        warmup_batches: Number of warmup batches
        warmup_time_seconds: Time to warmup in seconds (overrides warmup_batches if set)
        print_progress: Whether to print progress during benchmarking
        madvise_interval: Memory advice interval setting
        shuffle: Whether to shuffle the data
        num_runs: Number of times to run the benchmark (default: 1)
        output_dir: Directory to save individual results (for multiple dataloaders mode)
        track_iteration_times: Whether to track and print the time taken for each iteration (default: False)
        log_iteration_times_to_file: Interval for logging iteration times to file (None to disable)
        num_workers: Number of worker processes for data loading

    Returns:
        Single BenchmarkResult if benchmarking one dataloader, or
        List[BenchmarkResult] if benchmarking multiple dataloaders.
        If num_runs > 1, returns aggregated statistics across all runs.

    Examples:
        # Single dataloader
        result = benchmark_dataloader(
            name="My Dataloader",
            dataloader_factory=lambda: MyDataLoader(),
            max_time_seconds=30.0
        )

        # Multiple dataloaders
        results = benchmark_dataloader(
            dataloaders=[
                {"name": "DL1", "dataloader_factory": lambda: DL1()},
                {"name": "DL2", "dataloader_factory": lambda: DL2()}
            ]
        )

    Note:
        - If both single and multiple parameters are provided, multiple mode takes precedence
        - For multiple dataloaders, each dict should have at least 'name' and 'dataloader_factory' keys
        - Add 'num_runs': N to run each configuration N times and get aggregated statistics
        - If output_dir is provided in multiple mode, results are saved as JSON files
        - A comparison table is printed when benchmarking multiple dataloaders
    """
    # Multiple dataloaders mode
    if dataloaders is not None:
        results = []
        for dl_config in dataloaders:
            try:
                print("Dropping caches")
                subprocess.run(["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
            except subprocess.CalledProcessError:
                print("⚠️ Warning: failed to drop caches — are you running with sudo?")

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
                print_progress=dl_config.get("print_progress", print_progress),
                madvise_interval=dl_config.get("madvise_interval", madvise_interval),
                shuffle=dl_config.get("shuffle", shuffle),
                num_runs=dl_config.get("num_runs", num_runs),
                output_dir=output_dir,
                track_iteration_times=dl_config.get("track_iteration_times", track_iteration_times),
                log_iteration_times_to_file=dl_config.get("log_iteration_times_to_file", log_iteration_times_to_file),
                num_workers=dl_config.get("num_workers", num_workers),
            )
            results.append(result)

            # Save individual result
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
            print_progress=print_progress,
            madvise_interval=madvise_interval,
            shuffle=shuffle,
            num_runs=num_runs,
            output_dir=output_dir,
            track_iteration_times=track_iteration_times,
            log_iteration_times_to_file=log_iteration_times_to_file,
            num_workers=num_workers,
        )

    else:
        raise ValueError(
            "Must provide either (name + dataloader_factory) for single dataloader mode "
            "or (dataloaders) for multiple dataloaders mode"
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
    print_progress: bool = True,
    madvise_interval: Optional[int] = None,
    shuffle: bool = False,
    num_runs: int = 1,
    output_dir: Optional[str] = None,
    track_iteration_times: bool = False,
    log_iteration_times_to_file: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> BenchmarkResult:
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
        print_progress: Whether to print progress information
        madvise_interval: Interval for memory advice calls
        shuffle: Whether to shuffle the data
        num_runs: Number of runs to perform
        output_dir: Directory to save iteration time logs
        track_iteration_times: Whether to track timing for each iteration
        log_iteration_times_to_file: Interval for logging iteration times to file (None to disable)
        num_workers: Number of worker processes for data loading

    Returns:
        BenchmarkResult object with aggregated or single run results

    Note:
        - If both max_batches and max_time_seconds are set, the first limit reached stops the benchmark
        - If both warmup_batches and warmup_time_seconds are set, warmup_time_seconds takes precedence
        - Instantiation is measured by default to give complete performance picture
        - The factory function is called once for instantiation measurement and once for setup
        - For existing dataloaders, simply wrap them in a factory function: lambda: your_dataloader
    """

    def log(message):
        if print_progress:
            print(message)

    log(f"🔧 Benchmarking: {name}")

    # Measure instantiation metrics if requested
    dataloader, baseline, peak, _, _, final_mib, setup_time = measure_peak_memory_full(dataloader_factory)

    # Measure memory after

    instantiation_metrics = {
        "instantiation_time_seconds": setup_time,
        "peak_memory_during_instantiation_mb": peak,
        "memory_before_instantiation_mb": baseline,
        "memory_after_instantiation_mb": final_mib,
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
        print_progress=print_progress,
        data_path=data_path,
        madvise_interval=madvise_interval,
        shuffle=shuffle,
        track_iteration_times=track_iteration_times,
        log_iteration_times_to_file=log_iteration_times_to_file,
        num_workers=num_workers,
    )

    # Run benchmark(s)
    if num_runs == 1:
        # Single run
        result = run_benchmark(dataloader, config, name, output_dir, instantiation_metrics)
        del dataloader

        # Update timing and disk info
        result.setup_time_seconds += setup_time
        result.disk_size_mb = disk_size_mb

        # Print results
        _print_results(result)
        return result
    else:
        # Multiple runs
        log(f"🏃 Running {num_runs} times...")
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
                print_progress=print_progress,
                data_path=data_path,
                madvise_interval=madvise_interval,
                shuffle=shuffle,
                track_iteration_times=track_iteration_times,
                log_iteration_times_to_file=log_iteration_times_to_file,
                num_workers=num_workers,
            )

            run_result = run_benchmark(
                dataloader, run_config, f"{name}_run_{run_idx + 1}", output_dir, instantiation_metrics
            )
            all_results.append(run_result)

            # Print samples per second after every run
            log(f"Run {run_idx + 1}: {run_result.samples_per_second:.2f} samples/sec")

            del dataloader

        # Calculate aggregated statistics
        result = aggregate_benchmark_results(all_results, name, instantiation_metrics, setup_time, disk_size_mb)

        # Print aggregated results
        export_benchmark_results([result])
        return result


def _print_results(result: BenchmarkResult) -> None:
    """Print benchmark results in a formatted way."""
    print(f"\n📊 {result.name}")
    print(f"   Samples/sec: {result.samples_per_second:.2f}")
    print(f"   Memory: {result.peak_memory_mb:.1f} MB")
    if result.instantiation_time_seconds:
        print(f"   Setup: {result.instantiation_time_seconds:.3f}s")


def _print_comparison(results: List[BenchmarkResult]) -> None:
    """Print comparison of multiple benchmark results."""
    if not results or len(results) < 2:
        return

    print(f"\n📊 COMPARISON ({len(results)} configurations)")

    # Show both aggregated and individual data
    for result in results:
        print(f"\n📋 {result.name}")
        if hasattr(result, "individual_results") and result.individual_results:
            # Multiple runs - show both individual and aggregated
            print("   Individual runs:")
            for i, ind_result in enumerate(result.individual_results, 1):
                print(f"     Run {i}: {ind_result.samples_per_second:.2f} samples/sec")
            print(f"   Aggregated: {result.samples_per_second:.2f} samples/sec (avg)")
        else:
            # Single run
            print(f"   Single run: {result.samples_per_second:.2f} samples/sec")
        print(f"   Memory: {result.peak_memory_mb:.1f} MB")

    # Find best performers (using aggregated data)
    best_samples_per_sec = max(results, key=lambda r: r.samples_per_second)
    lowest_memory = min(results, key=lambda r: r.peak_memory_mb)

    print("\n🏆 BEST PERFORMERS (Aggregated):")
    print(f"🏆 Best speed: {best_samples_per_sec.name} ({best_samples_per_sec.samples_per_second:.2f} samples/sec)")
    print(f"🏆 Lowest memory: {lowest_memory.name} ({lowest_memory.peak_memory_mb:.2f} MB)")
