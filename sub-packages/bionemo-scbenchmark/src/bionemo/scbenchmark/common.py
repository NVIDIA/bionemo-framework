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
import platform
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
        errors: List of error messages

        # Instantiation metrics
        instantiation_time_seconds: TOTAL time (dataset + dataloader)
        dataset_instantiation_time_seconds: Time to load/create dataset only
        dataloader_instantiation_time_seconds: Time to wrap dataset in dataloader only
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
    errors: List[str] = None
    # Instantiation metrics (None if not measured)
    instantiation_time_seconds: Optional[float] = None  # TOTAL (dataset + dataloader)
    dataset_instantiation_time_seconds: Optional[float] = None  # Dataset loading only
    dataloader_instantiation_time_seconds: Optional[float] = None  # DataLoader wrapping only
    peak_memory_during_instantiation_mb: Optional[float] = None
    memory_after_instantiation_mb: Optional[float] = None
    memory_before_instantiation_mb: Optional[float] = None

    # Configuration metadata
    madvise_interval: Optional[int] = None
    data_path: Optional[str] = None
    max_time_seconds: Optional[float] = None
    shuffle: Optional[bool] = None
    num_workers: Optional[int] = None

    # Warmup metrics
    warmup_samples: int = 0
    warmup_batches: int = 0

    # Speed metrics
    total_speed_samples_per_second: float = 0.0  # Including warmup

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
        elapsed_time: float = 0.0,
        disk_size_mb: float = 0.0,
        warmup_samples: int = 0,
        warmup_batches: int = 0,
        instantiation_metrics: Optional[Dict[str, float]] = None,
        num_workers: Optional[int] = None,
        epoch_results: Optional[List[Dict[str, Any]]] = None,
    ) -> "BenchmarkResult":
        """Create BenchmarkResult from raw metrics."""
        # Calculate timing metrics
        avg_batch_time = elapsed_time / total_batches if total_batches > 0 else 0
        samples_per_sec = total_samples / elapsed_time if elapsed_time > 0 else 0
        batches_per_sec = total_batches / elapsed_time if elapsed_time > 0 else 0

        # Calculate speed metrics
        iteration_time = elapsed_time + warmup_time
        total_samples_including_warmup = total_samples + warmup_samples
        total_speed = total_samples_including_warmup / (iteration_time) if iteration_time > 0 else 0

        # Calculate memory metrics from epoch results
        peak_memory_mb = 0.0
        average_memory_mb = 0.0
        if epoch_results:
            max_peak_memory = max(r["peak_memory"] for r in epoch_results)
            avg_memory = sum(r["avg_memory"] for r in epoch_results) / len(epoch_results)

            # Subtract baseline memory if available from instantiation metrics
            memory_before_instantiation = instantiation_metrics["memory_before_instantiation_mb"]

            peak_memory_mb = max_peak_memory - memory_before_instantiation
            average_memory_mb = avg_memory - memory_before_instantiation

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
            peak_memory_mb=peak_memory_mb,
            average_memory_mb=average_memory_mb,
            warmup_samples=warmup_samples,
            warmup_batches=warmup_batches,
            total_speed_samples_per_second=total_speed,
            madvise_interval=madvise_interval,
            data_path=data_path,
            max_time_seconds=max_time_seconds,
            shuffle=shuffle,
            num_workers=num_workers,
            **instantiation_metrics,
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


def export_benchmark_results(results: List[BenchmarkResult], output_prefix: str = "benchmark_data") -> None:
    """Export benchmark results to summary and detailed breakdown CSVs."""
    from datetime import datetime

    import pandas as pd

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{output_prefix}_{timestamp}"
    summary_rows = []
    detailed_rows = []
    for i, result in enumerate(results, 1):
        # Extract configuration name (remove run number suffix if present)
        config_name = result.name.replace(f"_run_{i}", "") if f"_run_{i}" in result.name else result.name

        m = calculate_derived_metrics(result)
        summary_rows.append(
            {
                "Configuration": config_name,
                "Run_Number": i,
                "Run_Name": result.name,
                "Warmup_Time_s": result.warmup_time_seconds,
                "Warmup_Samples_per_sec": m["warmup_samples_per_sec"],
                "Total_Time_s": result.total_iteration_time_seconds,
                "Total_Samples_per_sec": result.samples_per_second,
                "Instantiation_Time_s": m["inst_time"],  # TOTAL (dataset + dataloader)
                "Dataset_Instantiation_Time_s": getattr(result, "dataset_instantiation_time_seconds", None),  # 🆕 NEW
                "Dataloader_Instantiation_Time_s": getattr(
                    result, "dataloader_instantiation_time_seconds", None
                ),  # 🆕 NEW
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
                "Num_Workers": result.num_workers,
                "Warmup_Samples": result.warmup_samples,
                "Warmup_Batches": result.warmup_batches,
                "Total_Samples_All_Epochs": result.total_samples,
                "Total_Batches_All_Epochs": result.total_batches,
                "Total_Speed_With_Warmup_Samples_per_sec": result.total_speed_samples_per_second,
                "Number_of_Epochs": m["num_epochs"],
            }
        )

    # Detailed breakdown: all runs and epochs
    for i, result in enumerate(results, 1):
        base_config = result.name.replace("_run_" + str(i), "") if "_run_" in result.name else result.name

        # Per-epoch breakdown
        if hasattr(result, "epoch_results") and result.epoch_results:
            for epoch_info in result.epoch_results:
                avg_batch_size = epoch_info["samples"] / epoch_info["batches"] if epoch_info["batches"] > 0 else 0
                detailed_rows.append(
                    {
                        "Configuration": base_config,
                        "Run_Number": i,
                        "Epoch": epoch_info["epoch"],
                        "Samples": epoch_info["samples"],
                        "Batches": epoch_info["batches"],
                        "Samples_per_sec": epoch_info["samples"] / epoch_info["elapsed"]
                        if epoch_info["iteration_time"] > 0
                        else 0,
                        "Peak_Memory_MB": epoch_info["peak_memory"],
                        "Average_Memory_MB": epoch_info["avg_memory"],
                        "Total_Time_s": epoch_info["iteration_time"],
                        "Setup_Time_s": 0,
                        "Warmup_Time_s": result.warmup_time_seconds if epoch_info["epoch"] == 1 else 0,
                        "Warmup_Samples": epoch_info["warmup_samples"],
                        "Warmup_Batches": epoch_info["warmup_batches"],
                        "Total_Speed_With_Warmup_Samples_per_sec": (
                            epoch_info["samples"] + epoch_info["warmup_samples"]
                        )
                        / epoch_info["iteration_time"]
                        if epoch_info["iteration_time"] > 0
                        else 0,
                        "Dataset_Path": result.data_path,
                        "Madvise_Interval": result.madvise_interval,
                        "Max_Time_Seconds": result.max_time_seconds,
                        "Shuffle": getattr(result, "shuffle", None),
                        "Instantiation_Time_s": getattr(result, "instantiation_time_seconds", None),  # TOTAL
                        "Dataset_Instantiation_Time_s": getattr(
                            result, "dataset_instantiation_time_seconds", None
                        ),  # 🆕 NEW
                        "Dataloader_Instantiation_Time_s": getattr(
                            result, "dataloader_instantiation_time_seconds", None
                        ),  # 🆕 NEW
                        "Instantiation_Memory_MB": getattr(result, "peak_memory_during_instantiation_mb", None),
                        "Average_Batch_Size": avg_batch_size,
                        "Batches_per_sec": epoch_info["batches"] / epoch_info["elapsed"]
                        if epoch_info["iteration_time"] > 0
                        else 0,
                    }
                )
    # Write CSVs
    summary_csv = f"{base_filename}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"📄 Summary CSV: {os.path.abspath(summary_csv)} ({len(summary_rows)} rows)")

    if detailed_rows:
        detailed_csv = f"{base_filename}_detailed_breakdown.csv"
        pd.DataFrame(detailed_rows).to_csv(detailed_csv, index=False)
        print(f"📄 Detailed breakdown CSV: {os.path.abspath(detailed_csv)} ({len(detailed_rows)} rows)")

    print("Export complete.")


def append_benchmark_result(
    result: BenchmarkResult,
    output_prefix: str = "benchmark_data",
    create_headers: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    """Append a single benchmark result to detailed breakdown CSV only.

    Args:
        result: The benchmark result to append
        output_prefix: Prefix for the CSV filename
        create_headers: If True, create new files with headers. If False, append to existing files.
                       If None, automatically detect if files exist.
        output_dir: Directory where CSV files should be created. If None, uses current directory.
    """
    import pandas as pd

    # Handle output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        detailed_csv = os.path.join(output_dir, f"{output_prefix}_detailed_breakdown.csv")
    else:
        detailed_csv = f"{output_prefix}_detailed_breakdown.csv"

    # Auto-detect if we should create headers (if files don't exist yet)
    files_exist = os.path.exists(detailed_csv)
    should_create_headers = create_headers and not files_exist

    # Determine run number from result name
    run_number = 1
    if "_run_" in result.name:
        try:
            run_number = int(result.name.split("_run_")[-1])
        except ValueError:
            pass

    # Extract configuration name (remove run number suffix if present)
    config_name = result.name.replace(f"_run_{run_number}", "") if f"_run_{run_number}" in result.name else result.name

    # Create detailed rows for each epoch only (no summary)
    detailed_rows = []
    for epoch_info in result.epoch_results:
        avg_batch_size = epoch_info["samples"] / epoch_info["batches"] if epoch_info["batches"] > 0 else 0
        detailed_rows.append(
            {
                "Configuration": config_name,
                "Run_Number": run_number,
                "Run_Name": result.name,
                "Epoch": epoch_info["epoch"],
                "Batches": epoch_info["batches"],
                "Samples": epoch_info["samples"],
                "Samples_per_sec": epoch_info["samples"] / epoch_info["elapsed"] if epoch_info["elapsed"] > 0 else 0,
                "Peak_Memory_MB": epoch_info["peak_memory"],
                "Average_Memory_MB": epoch_info["avg_memory"],
                "Total_Time_s": epoch_info["iteration_time"],
                "Setup_Time_s": 0,
                "Warmup_Time_s": result.warmup_time_seconds if epoch_info["epoch"] == 1 else 0,
                "Warmup_Samples": epoch_info["warmup_samples"],
                "Warmup_Batches": epoch_info["warmup_batches"],
                "Total_Speed_With_Warmup_Samples_per_sec": (epoch_info["samples"] + epoch_info["warmup_samples"])
                / epoch_info["iteration_time"]
                if epoch_info["iteration_time"] > 0
                else 0,
                "Dataset_Path": result.data_path,
                "Madvise_Interval": result.madvise_interval,
                "Max_Time_Seconds": result.max_time_seconds,
                "Shuffle": getattr(result, "shuffle", None),
                "Instantiation_Time_s": getattr(result, "instantiation_time_seconds", None),  # TOTAL
                "Dataset_Instantiation_Time_s": getattr(result, "dataset_instantiation_time_seconds", None),  # 🆕 NEW
                "Dataloader_Instantiation_Time_s": getattr(
                    result, "dataloader_instantiation_time_seconds", None
                ),  # 🆕 NEW
                "Instantiation_Memory_MB": getattr(result, "peak_memory_during_instantiation_mb", None),
                "Average_Batch_Size": avg_batch_size,
                "Batches_per_sec": epoch_info["batches"] / epoch_info["elapsed"]
                if epoch_info["iteration_time"] > 0
                else 0,
            }
        )

    # Write or append detailed CSV only
    mode = "w" if should_create_headers else "a"
    header = should_create_headers

    if detailed_rows:
        pd.DataFrame(detailed_rows).to_csv(detailed_csv, mode=mode, header=header, index=False)
        if should_create_headers:
            print(f"📄 Created Detailed breakdown CSV: {os.path.abspath(detailed_csv)}")
        else:
            print(f"📄 Appended to Detailed breakdown CSV: {os.path.abspath(detailed_csv)}")


def get_disk_size(path: Union[str, Path]) -> float:
    """Get disk size of a file or directory in MB."""
    try:
        # Use appropriate du command based on platform
        if platform.system() == "Darwin":  # macOS
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


def _fast_pss_sampler(root_pid, stop_evt, result_queue, sample_interval):
    """In a separate process: every sample_interval seconds, sum up PSS of the root pid + all its live children via psutil.

    Args:
        root_pid: Process ID to monitor
        stop_evt: Event to signal when to stop monitoring
        result_queue: Queue to send results back
        sample_interval: How often to sample in seconds
    """
    parent = psutil.Process(root_pid)
    peak = 0
    total = 0
    count = 0

    while not stop_evt.is_set():
        # gather all alive children + parent
        try:
            procs = [parent] + parent.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            procs = [parent]

        # sum up PSS
        sample_pss = 0
        for p in procs:
            try:
                sample_pss += p.memory_full_info().pss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        peak = max(peak, sample_pss)
        total += sample_pss
        count += 1

        time.sleep(sample_interval)

    # return peak & avg
    avg = (total / count) if count else 0
    result_queue.put((peak, avg))


def _single_rss_sampler(root_pid, stop_evt, result_queue, sample_interval):
    """Light-weight sampler for a single process: polls rss via psutil."""
    proc = psutil.Process(root_pid)
    peak = 0
    total = 0
    count = 0

    while not stop_evt.is_set():
        try:
            rss = proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            rss = 0
        peak = max(peak, rss)
        total += rss
        count += 1
        time.sleep(sample_interval)

    avg = total / count if count else 0
    result_queue.put((peak, avg))


def measure_peak_memory_full(
    func,
    *args,
    sample_interval: float = 0.2,
    child_refresh_interval: float = 5.0,
    multi_worker: bool = False,
    **kwargs,
):
    """Measure peak & average memory while running `func`.

    If multi_worker=True, uses PSS across the process tree (slower but includes children).
    Otherwise uses RSS of just the main process (lightweight).

    Returns:
      (result,
       baseline_mib,
       peak_mib,
       avg_mib,
       delta_mib,
       final_mib,
       duration_s)
    """
    parent_pid = os.getpid()
    stop_event = mp.Event()
    result_queue = mp.Queue()

    # pick sampler
    if multi_worker:
        sampler_proc = mp.Process(
            target=_fast_pss_sampler,
            args=(
                parent_pid,
                stop_event,
                result_queue,
                sample_interval,
                child_refresh_interval,
            ),
        )
        # baseline via PSS
        with open(f"/proc/{parent_pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    baseline_kb = int(line.split()[1])
                    break
        baseline = baseline_kb * 1024
    else:
        sampler_proc = mp.Process(
            target=_single_rss_sampler,
            args=(parent_pid, stop_event, result_queue, sample_interval),
        )
        # baseline via RSS
        baseline = psutil.Process(parent_pid).memory_info().rss

    # start sampler
    sampler_proc.start()
    gc.collect()
    start = time.perf_counter()

    try:
        result = func(*args, **kwargs)
    finally:
        stop_event.set()
        sampler_proc.join()

    duration = time.perf_counter() - start

    # fetch stats
    try:
        peak, avg = result_queue.get_nowait()
    except mp.queues.Empty:
        peak = avg = baseline

    # final memory
    if multi_worker:
        with open(f"/proc/{parent_pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    final_kb = int(line.split()[1])
                    break
        final = final_kb * 1024
    else:
        final = psutil.Process(parent_pid).memory_info().rss

    # convert to MiB
    def to_mib(x):
        return x / 1024**2

    baseline_mib = to_mib(baseline)
    peak_mib = to_mib(peak)
    avg_mib = to_mib(avg)
    final_mib = to_mib(final)
    delta_mib = peak_mib - baseline_mib

    return (result, baseline_mib, peak_mib, avg_mib, delta_mib, final_mib, duration)
