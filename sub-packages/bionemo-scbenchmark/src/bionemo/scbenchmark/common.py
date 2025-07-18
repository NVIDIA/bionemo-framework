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

import gc
import json
import multiprocessing as mp
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil


@dataclass
class BenchmarkResult:
    """Results from benchmarking a dataloader.

    This class stores all metrics and metadata about a dataloader benchmark run,
    including timing, memory usage, throughput, configuration parameters, and
    the raw data for detailed analysis.

    Attributes:
        name: Name/description of the benchmark
        disk_size_mb: Size of the dataset files on disk in MB
        setup_time_seconds: Time spent setting up the dataloader
        warmup_time_seconds: Time spent in warmup phase
        total_iteration_time_seconds: Time spent iterating through data (excluding warmup)
        average_batch_time_seconds: Average time per batch
        total_batches: Number of batches processed (excluding warmup)
        total_samples: Number of samples processed (excluding warmup)
        samples_per_second: Throughput in samples per second (excluding warmup)
        batches_per_second: Throughput in batches per second (excluding warmup)
        peak_memory_mb: Peak memory usage during benchmark
        average_memory_mb: Average memory usage during benchmark
        gpu_memory_mb: GPU memory usage if applicable
        errors: List of error messages

        # Instantiation metrics
        instantiation_time_seconds: Time taken to instantiate the dataloader
        peak_memory_during_instantiation_mb: Peak memory usage during instantiation

        # Configuration metadata
        madvise_interval: Memory advice interval setting used
        data_path: Path to dataset used for benchmarking
        max_time_seconds: Maximum time limit set for the benchmark
        shuffle: Whether the dataloader was shuffled
    """

    name: str
    disk_size_mb: float
    setup_time_seconds: float
    warmup_time_seconds: float
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

    # Instantiation metrics (None if not measured)
    instantiation_time_seconds: Optional[float] = None
    peak_memory_during_instantiation_mb: Optional[float] = None

    # Configuration metadata
    madvise_interval: Optional[int] = None
    data_path: Optional[str] = None
    max_time_seconds: Optional[float] = None
    shuffle: Optional[bool] = None

    # Warmup metrics
    warmup_samples: int = 0
    warmup_batches: int = 0

    # Speed metrics
    total_speed_samples_per_second: float = 0.0  # Including warmup
    post_warmup_speed_samples_per_second: float = 0.0  # Excluding warmup

    def __post_init__(self):
        """Initialize lists if not provided."""
        if self.errors is None:
            self.errors = []

    @classmethod
    def from_raw_metrics(
        cls,
        name: str,
        madvise_interval: Optional[int] = None,
        data_path: Optional[str] = None,
        max_time_seconds: Optional[float] = None,
        shuffle: Optional[bool] = None,
        total_samples: int = 0,
        total_batches: int = 0,
        setup_time: float = 0.0,
        warmup_time: float = 0.0,
        iteration_time: float = 0.0,
        disk_size_mb: float = 0.0,
        gpu_memory_mb: float = 0.0,
        warmup_samples: int = 0,
        warmup_batches: int = 0,
        instantiation_metrics: Optional[Dict[str, float]] = None,
    ) -> "BenchmarkResult":
        """Create BenchmarkResult from raw metrics.

        This factory method calculates all the derived metrics from raw data
        and creates a complete BenchmarkResult object.

        Args:
            name: Name of the benchmark
            madvise_interval: Memory advice interval setting
            data_path: Path to dataset used for benchmarking
            max_time_seconds: Maximum time limit set for the benchmark
            shuffle: Whether data was shuffled during processing
            total_samples: Total number of samples processed (excluding warmup)
            total_batches: Total number of batches processed (excluding warmup)
            setup_time: Time taken for setup phase
            warmup_time: Time taken for warmup phase
            iteration_time: Total time spent iterating (excluding warmup)
            disk_size_mb: Size of data files on disk
            gpu_memory_mb: GPU memory usage
            warmup_samples: Number of samples processed during warmup
            warmup_batches: Number of batches processed during warmup
            instantiation_metrics: Instantiation metrics dictionary if available

        Returns:
            BenchmarkResult with all calculated metrics
        """
        # Calculate timing metrics
        avg_batch_time = iteration_time / total_batches if total_batches > 0 else 0
        samples_per_sec = total_samples / iteration_time if iteration_time > 0 else 0
        batches_per_sec = total_batches / iteration_time if iteration_time > 0 else 0

        # Calculate memory metrics (default to 0 if no data)
        peak_memory = 0
        avg_memory = 0

        # Calculate speed metrics
        total_samples_including_warmup = total_samples + warmup_samples
        total_time_including_warmup = warmup_time + iteration_time
        total_speed = (
            total_samples_including_warmup / total_time_including_warmup if total_time_including_warmup > 0 else 0
        )
        post_warmup_speed = total_samples / iteration_time if iteration_time > 0 else 0

        # Extract instantiation metrics if provided
        instantiation_kwargs = {}
        if instantiation_metrics:
            instantiation_kwargs = instantiation_metrics

        return cls(
            name=name,
            disk_size_mb=disk_size_mb,
            setup_time_seconds=setup_time,
            warmup_time_seconds=warmup_time,
            total_iteration_time_seconds=iteration_time,
            average_batch_time_seconds=avg_batch_time,
            total_batches=total_batches,
            total_samples=total_samples,
            samples_per_second=samples_per_sec,
            batches_per_second=batches_per_sec,
            peak_memory_mb=peak_memory,
            average_memory_mb=avg_memory,
            gpu_memory_mb=gpu_memory_mb,
            warmup_samples=warmup_samples,
            warmup_batches=warmup_batches,
            total_speed_samples_per_second=total_speed,
            post_warmup_speed_samples_per_second=post_warmup_speed,
            madvise_interval=madvise_interval,
            data_path=data_path,
            max_time_seconds=max_time_seconds,
            shuffle=shuffle,
            **instantiation_kwargs,
        )

    def save_to_file(self, filepath: str) -> None:
        """Save results to JSON file.

        Args:
            filepath: Path to the output JSON file
        """
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)


def calculate_derived_metrics(result: BenchmarkResult) -> Dict[str, float]:
    """Calculate derived metrics from a BenchmarkResult.

    This function extracts and calculates commonly used derived metrics
    to avoid duplication across the codebase.

    Args:
        result: BenchmarkResult to process

    Returns:
        Dictionary with derived metrics including:
        - warmup_samples_per_sec
        - num_epochs
        - dataset_samples_per_epoch
        - dataset_batches_per_epoch
        - avg_batch_size
        - dataset_size_k_samples
        - inst_memory
        - inst_time
    """
    # Calculate warmup speed
    warmup_samples_per_sec = (
        (result.warmup_samples / result.warmup_time_seconds) if result.warmup_time_seconds > 0 else 0.0
    )

    # Get number of epochs (try multiple sources)
    num_epochs = getattr(result, "num_epochs", 1) or 1
    if hasattr(result, "epoch_results") and result.epoch_results:
        num_epochs = len(result.epoch_results)

    # Calculate per-epoch values (actual dataset size), not cumulative across all epochs
    dataset_samples_per_epoch = result.total_samples // num_epochs if num_epochs > 0 else result.total_samples
    dataset_batches_per_epoch = result.total_batches // num_epochs if num_epochs > 0 else result.total_batches
    avg_batch_size = dataset_samples_per_epoch / dataset_batches_per_epoch if dataset_batches_per_epoch > 0 else 0
    dataset_size_k_samples = dataset_samples_per_epoch / 1000.0  # Convert to thousands

    # Get instantiation metrics (fallback to 0 if not available)
    inst_memory = getattr(result, "peak_memory_during_instantiation_mb", 0.0) or 0.0
    inst_time = getattr(result, "instantiation_time_seconds", 0.0) or 0.0

    return {
        "warmup_samples_per_sec": warmup_samples_per_sec,
        "num_epochs": num_epochs,
        "dataset_samples_per_epoch": dataset_samples_per_epoch,
        "dataset_batches_per_epoch": dataset_batches_per_epoch,
        "avg_batch_size": avg_batch_size,
        "dataset_size_k_samples": dataset_size_k_samples,
        "inst_memory": inst_memory,
        "inst_time": inst_time,
    }


def aggregate_benchmark_results(
    results: List[BenchmarkResult],
    base_name: str = "Aggregated",
    instantiation_metrics: Optional[Dict[str, float]] = None,
    setup_time: float = 0.0,
    disk_size_mb: float = 0.0,
) -> BenchmarkResult:
    """Aggregate multiple BenchmarkResult objects into a single BenchmarkResult with statistics."""
    import statistics

    if not results:
        raise ValueError("No results to aggregate.")
    metrics = {
        "total_samples": [r.total_samples for r in results],
        "total_batches": [r.total_batches for r in results],
        "samples_per_second": [r.samples_per_second for r in results],
        "batches_per_second": [r.batches_per_second for r in results],
        "total_iteration_time_seconds": [r.total_iteration_time_seconds for r in results],
        "average_batch_time_seconds": [r.average_batch_time_seconds for r in results],
        "peak_memory_mb": [r.peak_memory_mb for r in results],
        "average_memory_mb": [r.average_memory_mb for r in results],
        "warmup_samples": [r.warmup_samples for r in results],
        "warmup_batches": [r.warmup_batches for r in results],
        "post_warmup_speed_samples_per_second": [r.post_warmup_speed_samples_per_second for r in results],
        "total_speed_samples_per_second": [r.total_speed_samples_per_second for r in results],
    }

    def calc_stats(values):
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
        }

    stats = {key: calc_stats(values) for key, values in metrics.items()}
    mean_total_samples = stats["total_samples"]["mean"]
    mean_iteration_time = stats["total_iteration_time_seconds"]["mean"]
    mean_total_batches = stats["total_batches"]["mean"]
    aggregated_samples_per_sec = mean_total_samples / mean_iteration_time if mean_iteration_time > 0 else 0
    aggregated_batches_per_sec = mean_total_batches / mean_iteration_time if mean_iteration_time > 0 else 0
    mean_warmup_samples = stats["warmup_samples"]["mean"]
    mean_warmup_time = (
        mean_warmup_samples / stats["post_warmup_speed_samples_per_second"]["mean"]
        if stats["post_warmup_speed_samples_per_second"]["mean"] > 0
        else 0
    )
    total_samples_with_warmup = mean_total_samples + mean_warmup_samples
    total_time_with_warmup = mean_iteration_time + mean_warmup_time
    aggregated_total_speed = total_samples_with_warmup / total_time_with_warmup if total_time_with_warmup > 0 else 0
    first = results[0]
    result = BenchmarkResult(
        name=f"{base_name}_aggregated_{len(results)}runs",
        disk_size_mb=disk_size_mb,
        setup_time_seconds=setup_time,
        warmup_time_seconds=mean_warmup_time,
        total_iteration_time_seconds=mean_iteration_time,
        average_batch_time_seconds=stats["average_batch_time_seconds"]["mean"],
        total_batches=int(mean_total_batches),
        total_samples=int(mean_total_samples),
        samples_per_second=aggregated_samples_per_sec,
        batches_per_second=aggregated_batches_per_sec,
        peak_memory_mb=stats["peak_memory_mb"]["mean"],
        average_memory_mb=stats["average_memory_mb"]["mean"],
        warmup_samples=int(mean_warmup_samples),
        warmup_batches=int(stats["warmup_batches"]["mean"]),
        total_speed_samples_per_second=aggregated_total_speed,
        post_warmup_speed_samples_per_second=aggregated_samples_per_sec,
        madvise_interval=first.madvise_interval,
        data_path=first.data_path,
        max_time_seconds=first.max_time_seconds,
        shuffle=first.shuffle,
    )
    # Add instantiation metrics
    if instantiation_metrics:
        for key, value in instantiation_metrics.items():
            setattr(result, key, value)
    result.individual_results = results
    result.statistics = stats
    return result


def export_benchmark_results(results: List[BenchmarkResult], output_prefix: str = "benchmark_data") -> None:
    """Export benchmark results to summary, aggregated, statistics, and detailed breakdown CSVs."""
    from datetime import datetime

    import pandas as pd

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{output_prefix}_{timestamp}"
    summary_rows = []
    aggregated_rows = []
    statistics_rows = []
    detailed_rows = []
    for result in results:
        # If this result has individual runs, add them to summary and aggregate
        if hasattr(result, "individual_results") and result.individual_results:
            for i, ind in enumerate(result.individual_results):
                m = calculate_derived_metrics(ind)
                summary_rows.append(
                    {
                        "Configuration": result.name,
                        "Run_Number": i + 1,
                        "Run_Name": ind.name,
                        "Warmup_Time_s": ind.warmup_time_seconds,
                        "Warmup_Samples_per_sec": m["warmup_samples_per_sec"],
                        "Total_Time_s": ind.total_iteration_time_seconds,
                        "Total_Samples_per_sec": ind.samples_per_second,
                        "Instantiation_Time_s": m["inst_time"],
                        "Instantiation_Memory_MB": m["inst_memory"],
                        "Peak_Memory_MB": ind.peak_memory_mb,
                        "Average_Memory_MB": ind.average_memory_mb,
                        "Batches_per_Epoch": m["dataset_batches_per_epoch"],
                        "Average_Batch_Size": m["avg_batch_size"],
                        "Disk_Size_MB": ind.disk_size_mb,
                        "Dataset_Size_K_samples": m["dataset_size_k_samples"],
                        "Dataset_Path": ind.data_path,
                        "Madvise_Interval": ind.madvise_interval,
                        "Max_Time_Seconds": ind.max_time_seconds,
                        "Shuffle": ind.shuffle,
                        "Warmup_Samples": ind.warmup_samples,
                        "Warmup_Batches": ind.warmup_batches,
                        "Total_Samples_All_Epochs": ind.total_samples,
                        "Total_Batches_All_Epochs": ind.total_batches,
                        "Post_Warmup_Speed_Samples_per_sec": ind.post_warmup_speed_samples_per_second,
                        "Total_Speed_With_Warmup_Samples_per_sec": ind.total_speed_samples_per_second,
                        "Number_of_Epochs": m["num_epochs"],
                    }
                )
            # Aggregated row
            agg = aggregate_benchmark_results(result.individual_results, base_name=result.name)
            m = calculate_derived_metrics(agg)
            aggregated_rows.append(
                {
                    "Configuration": result.name,
                    "Number_of_Runs": len(result.individual_results),
                    "Warmup_Time_s": agg.warmup_time_seconds,
                    "Warmup_Samples_per_sec": m["warmup_samples_per_sec"],
                    "Total_Time_s": agg.total_iteration_time_seconds,
                    "Total_Samples_per_sec": agg.samples_per_second,
                    "Instantiation_Time_s": m["inst_time"],
                    "Instantiation_Memory_MB": m["inst_memory"],
                    "Peak_Memory_MB": agg.peak_memory_mb,
                    "Average_Memory_MB": agg.average_memory_mb,
                    "Batches_per_Epoch": m["dataset_batches_per_epoch"],
                    "Average_Batch_Size": m["avg_batch_size"],
                    "Disk_Size_MB": agg.disk_size_mb,
                    "Dataset_Size_K_samples": m["dataset_size_k_samples"],
                    "Dataset_Path": agg.data_path,
                    "Madvise_Interval": agg.madvise_interval,
                    "Max_Time_Seconds": agg.max_time_seconds,
                    "Shuffle": agg.shuffle,
                    "Warmup_Samples": agg.warmup_samples,
                    "Warmup_Batches": agg.warmup_batches,
                    "Total_Samples_All_Epochs": agg.total_samples,
                    "Total_Batches_All_Epochs": agg.total_batches,
                    "Post_Warmup_Speed_Samples_per_sec": agg.post_warmup_speed_samples_per_second,
                    "Total_Speed_With_Warmup_Samples_per_sec": agg.total_speed_samples_per_second,
                    "Number_of_Epochs": m["num_epochs"],
                }
            )
            # Statistics
            if hasattr(agg, "statistics") and agg.statistics:
                for metric_name, stats in agg.statistics.items():
                    statistics_rows.append(
                        {
                            "Configuration": result.name,
                            "Metric": metric_name,
                            "Mean": stats["mean"],
                            "Std": stats["std"],
                            "Min": stats["min"],
                            "Max": stats["max"],
                            "Median": stats["median"],
                            "Num_Runs": len(result.individual_results),
                        }
                    )
        else:
            m = calculate_derived_metrics(result)
            summary_rows.append(
                {
                    "Configuration": result.name,
                    "Run_Number": 1,
                    "Run_Name": result.name,
                    "Warmup_Time_s": result.warmup_time_seconds,
                    "Warmup_Samples_per_sec": m["warmup_samples_per_sec"],
                    "Total_Time_s": result.total_iteration_time_seconds,
                    "Total_Samples_per_sec": result.samples_per_second,
                    "Instantiation_Time_s": m["inst_time"],
                    "Instantiation_Memory_MB": m["inst_memory"],
                    "Peak_Memory_MB": result.peak_memory_mb,
                    "Average_Memory_MB": result.average_memory_mb,
                    "Batches_per_Epoch": m["dataset_batches_per_epoch"],
                    "Average_Batch_Size": m["avg_batch_size"],
                    "Disk_Size_MB": result.disk_size_mb,
                    "Dataset_Size_K_samples": m["dataset_size_k_samples"],
                    "Dataset_Path": result.data_path,
                    "Madvise_Interval": result.madvise_interval,
                    "Max_Time_Seconds": result.max_time_seconds,
                    "Shuffle": result.shuffle,
                    "Warmup_Samples": result.warmup_samples,
                    "Warmup_Batches": result.warmup_batches,
                    "Total_Samples_All_Epochs": result.total_samples,
                    "Total_Batches_All_Epochs": result.total_batches,
                    "Post_Warmup_Speed_Samples_per_sec": result.post_warmup_speed_samples_per_second,
                    "Total_Speed_With_Warmup_Samples_per_sec": result.total_speed_samples_per_second,
                    "Number_of_Epochs": m["num_epochs"],
                }
            )
        # Detailed breakdown: all runs and epochs
        results_to_process = []
        run_numbers = []
        if hasattr(result, "individual_results") and result.individual_results:
            results_to_process.extend(result.individual_results)
            run_numbers = list(range(1, len(result.individual_results) + 1))
        else:
            results_to_process.append(result)
            run_numbers = [1]
        for proc_result, run_num in zip(results_to_process, run_numbers):
            base_config = (
                proc_result.name.replace("_run_" + str(run_num), "")
                if "_run_" in proc_result.name
                else proc_result.name
            )
            # Overall run summary
            detailed_rows.append(
                {
                    "Configuration": base_config,
                    "Run_Number": run_num,
                    "Epoch": "OVERALL",
                    "Samples": proc_result.total_samples,
                    "Batches": proc_result.total_batches,
                    "Samples_per_sec": proc_result.samples_per_second,
                    "Peak_Memory_MB": proc_result.peak_memory_mb,
                    "Average_Memory_MB": proc_result.average_memory_mb,
                    "Total_Time_s": proc_result.total_iteration_time_seconds,
                    "Setup_Time_s": proc_result.setup_time_seconds,
                    "Warmup_Time_s": proc_result.warmup_time_seconds,
                    "Warmup_Samples": proc_result.warmup_samples,
                    "Warmup_Batches": proc_result.warmup_batches,
                    "Post_Warmup_Speed_Samples_per_sec": proc_result.post_warmup_speed_samples_per_second,
                    "Total_Speed_With_Warmup_Samples_per_sec": proc_result.total_speed_samples_per_second,
                    "Dataset_Path": proc_result.data_path,
                    "Madvise_Interval": proc_result.madvise_interval,
                    "Max_Time_Seconds": proc_result.max_time_seconds,
                    "Shuffle": getattr(proc_result, "shuffle", None),
                    "Instantiation_Time_s": getattr(proc_result, "instantiation_time_seconds", None),
                    "Instantiation_Memory_MB": getattr(proc_result, "peak_memory_during_instantiation_mb", None),
                }
            )
            # Per-epoch breakdown
            if hasattr(proc_result, "epoch_results") and proc_result.epoch_results:
                for epoch_info in proc_result.epoch_results:
                    avg_batch_size = epoch_info["samples"] / epoch_info["batches"] if epoch_info["batches"] > 0 else 0
                    detailed_rows.append(
                        {
                            "Configuration": base_config,
                            "Run_Number": run_num,
                            "Epoch": epoch_info["epoch"],
                            "Samples": epoch_info["samples"],
                            "Batches": epoch_info["batches"],
                            "Samples_per_sec": epoch_info["samples"] / epoch_info["iteration_time"]
                            if epoch_info["iteration_time"] > 0
                            else 0,
                            "Peak_Memory_MB": epoch_info["peak_memory"],
                            "Average_Memory_MB": epoch_info["avg_memory"],
                            "Total_Time_s": epoch_info["iteration_time"],
                            "Setup_Time_s": 0,
                            "Warmup_Time_s": proc_result.warmup_time_seconds if epoch_info["epoch"] == 1 else 0,
                            "Warmup_Samples": epoch_info["warmup_samples"],
                            "Warmup_Batches": epoch_info["warmup_batches"],
                            "Post_Warmup_Speed_Samples_per_sec": epoch_info["samples"] / epoch_info["iteration_time"]
                            if epoch_info["iteration_time"] > 0
                            else 0,
                            "Total_Speed_With_Warmup_Samples_per_sec": (
                                epoch_info["samples"] + epoch_info["warmup_samples"]
                            )
                            / epoch_info["iteration_time"]
                            if epoch_info["iteration_time"] > 0
                            else 0,
                            "Dataset_Path": proc_result.data_path,
                            "Madvise_Interval": proc_result.madvise_interval,
                            "Max_Time_Seconds": proc_result.max_time_seconds,
                            "Shuffle": getattr(proc_result, "shuffle", None),
                            "Instantiation_Time_s": None,
                            "Instantiation_Memory_MB": None,
                            "Average_Batch_Size": avg_batch_size,
                            "Batches_per_sec": epoch_info["batches"] / epoch_info["iteration_time"]
                            if epoch_info["iteration_time"] > 0
                            else 0,
                        }
                    )
    # Write CSVs
    pd.DataFrame(summary_rows).to_csv(f"{base_filename}_summary.csv", index=False)
    if aggregated_rows:
        pd.DataFrame(aggregated_rows).to_csv(f"{base_filename}_aggregated.csv", index=False)
    if statistics_rows:
        pd.DataFrame(statistics_rows).to_csv(f"{base_filename}_statistics.csv", index=False)
    if detailed_rows:
        pd.DataFrame(detailed_rows).to_csv(f"{base_filename}_detailed_breakdown.csv", index=False)
    print("Export complete.")


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
    import platform
    import subprocess
    
    try:
        # Use appropriate du command based on platform
        if platform.system() == "Darwin":  # macOS
            # macOS du doesn't have -b flag, use default (512-byte blocks) and convert
            result = subprocess.run(["du", "-s", str(path)], stdout=subprocess.PIPE, text=True, check=True)
            size_in_blocks = int(result.stdout.split()[0])
            size_in_bytes = size_in_blocks * 512  # macOS du uses 512-byte blocks by default
        else:  # Linux and others
            result = subprocess.run(["du", "-sb", str(path)], stdout=subprocess.PIPE, text=True, check=True)
            size_in_bytes = int(result.stdout.split()[0])
        
        return size_in_bytes / (1024 * 1024)
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        print(f"Warning: Could not determine disk size for {path}: {e}")
        return 0.0


def get_batch_size(batch: Any) -> int:
    """Determine the size of a batch.

    This function attempts to determine the batch size from various
    common batch formats including PyTorch tensors, lists, and
    dictionaries with common keys.

    Args:
        batch: The batch object to measure

    Returns:
        Number of samples in the batch
    """
    if hasattr(batch, "X"):
        # AnnCollection batch
        batch_size = batch.X.shape[0]
    else:
        batch_size = batch.shape[0] if hasattr(batch, "shape") else len(batch)
    return batch_size


def monitor_memory_dynamic_pss(parent_pid, stop_event, result_queue):
    """Monitor memory usage for a process and its children.

    Args:
        parent_pid: Process ID to monitor
        stop_event: Event to signal when to stop monitoring
        result_queue: Queue to store memory usage results
    """
    peak = 0
    samples = []
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        result_queue.put((0, 0.0))
        return

    while not stop_event.is_set():
        try:
            children = parent.children(recursive=True)
            all_pids = [parent_pid] + [c.pid for c in children if c.is_running()]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            all_pids = [parent_pid]

        # Try PSS first (Linux), fall back to RSS (macOS/Windows)
        total_mem = 0
        for pid in all_pids:
            if psutil.pid_exists(pid):
                try:
                    proc = psutil.Process(pid)
                    mem_info = proc.memory_full_info()
                    # PSS is Linux-specific, fall back to RSS on other platforms
                    if hasattr(mem_info, 'pss'):
                        total_mem += mem_info.pss
                    else:
                        total_mem += mem_info.rss
                except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                    continue
        mem = total_mem

        samples.append(mem)
        peak = max(peak, mem)
        time.sleep(0.05)

    avg = sum(samples) / len(samples) if samples else 0
    result_queue.put((peak, avg))


def measure_peak_memory_full(func, *args, **kwargs):
    """Measure peak memory usage while executing a function.

    Args:
        func: Function to execute while monitoring memory
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Tuple of (function_result, peak_memory_mb, memory_samples)
    """
    parent_pid = os.getpid()
    stop_event = mp.Event()
    result_queue = mp.Queue()
    monitor = mp.Process(target=monitor_memory_dynamic_pss, args=(parent_pid, stop_event, result_queue))

    gc.collect()
    # Use PSS on Linux, RSS on other platforms
    proc = psutil.Process(parent_pid)
    mem_info = proc.memory_full_info()
    baseline = mem_info.pss if hasattr(mem_info, 'pss') else mem_info.rss

    monitor.start()
    start = time.perf_counter()

    try:
        result = func(*args, **kwargs)
    finally:
        stop_event.set()
        monitor.join()

    duration = time.perf_counter() - start

    try:
        peak, avg = result_queue.get(timeout=2)
    except Exception:
        proc = psutil.Process(parent_pid)
        mem_info = proc.memory_full_info()
        peak = mem_info.pss if hasattr(mem_info, 'pss') else mem_info.rss
        avg = peak

    gc.collect()
    proc = psutil.Process(parent_pid)
    mem_info = proc.memory_full_info()
    final = mem_info.pss if hasattr(mem_info, 'pss') else mem_info.rss

    baseline_mib = baseline / 1024 / 1024
    peak_mib = peak / 1024 / 1024
    avg_mib = avg / 1024 / 1024
    delta_mib = peak_mib - baseline_mib
    final_mib = final / 1024 / 1024

    return result, baseline_mib, peak_mib, avg_mib, delta_mib, final_mib, duration
