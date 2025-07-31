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
import multiprocessing as mp
import os
import platform
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import psutil


@dataclass
class BenchmarkResult:
    """Results from benchmarking a dataloader.

    This class stores essential metrics and metadata about a dataloader benchmark run
    for CSV export and analysis.

    Attributes:
        name: Name/description of the benchmark
        warmup_time_seconds: Time spent in warmup phase

        # Instantiation metrics
        dataset_instantiation_time_seconds: Time to load/create dataset only
        dataloader_instantiation_time_seconds: Time to wrap dataset in dataloader only

        # Configuration metadata
        madvise_interval: Memory advice interval setting used
        data_path: Path to dataset used for benchmarking
        max_time_seconds: Maximum time limit set for the benchmark
        shuffle: Whether the dataloader was shuffled
        num_workers: Number of worker processes used for data loading

                # Input data
        epoch_results: List of per-epoch benchmark results
    """

    name: str
    warmup_time_seconds: float = 0.0

    # Instantiation metrics (always passed explicitly)
    dataset_instantiation_time_seconds: float = 0.0
    dataloader_instantiation_time_seconds: float = 0.0
    peak_memory_during_instantiation_mb: float = 0.0
    memory_before_instantiation_mb: float = 0.0
    memory_after_instantiation_mb: float = 0.0

    # Configuration metadata
    madvise_interval: Optional[int] = None
    data_path: Optional[str] = None
    max_time_seconds: Optional[float] = None
    shuffle: Optional[bool] = None
    num_workers: Optional[int] = None

    # Input data (always passed explicitly)
    epoch_results: Optional[List[Dict[str, Any]]] = None

    # Derived metrics
    samples_per_second: float = 0.0
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    disk_size_mb: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics from epoch results."""
        self.total_samples = sum(r["samples"] for r in self.epoch_results)
        self.total_time_seconds = sum(r["elapsed"] for r in self.epoch_results)
        self.samples_per_second = self.total_samples / self.total_time_seconds if self.total_time_seconds > 0 else 0.0
        self.peak_memory_mb = max(r["peak_memory"] for r in self.epoch_results) - self.memory_before_instantiation_mb
        self.avg_memory_mb = (
            sum(r["avg_memory"] for r in self.epoch_results) / len(self.epoch_results)
            - self.memory_before_instantiation_mb
        )
        self.instantiation_time_seconds = (
            self.dataset_instantiation_time_seconds + self.dataloader_instantiation_time_seconds
        )
        self.peak_memory_during_instantiation_mb = (
            self.peak_memory_during_instantiation_mb - self.memory_before_instantiation_mb
        )


def export_benchmark_results(
    results: Union[BenchmarkResult, List[BenchmarkResult]],
    output_prefix: str = "benchmark_data",
) -> None:
    """Append benchmark results to detailed breakdown CSV, never overwriting existing data.

    This function appends benchmark results to an existing CSV file or creates
    a new one if it doesn't exist. It never overwrites existing files.

    Args:
        results: Single BenchmarkResult or list of BenchmarkResults to append
        output_prefix: Prefix for the CSV filename
    """
    # Normalize results to always be a list
    if isinstance(results, BenchmarkResult):
        results = [results]

    # Use simple filename in current directory
    detailed_csv = f"{output_prefix}_detailed_breakdown.csv"

    # Always append, only write header if file does not exist
    file_exists = os.path.exists(detailed_csv)
    header = not file_exists

    # Build detailed rows
    detailed_rows = []

    for i, result in enumerate(results, 1):
        # Handle run numbering for single result vs multiple results
        if len(results) == 1:
            # Single result - extract run number from name
            run_number = 1
            if "_run_" in result.name:
                try:
                    run_number = int(result.name.split("_run_")[-1])
                except ValueError:
                    pass
        else:
            # Multiple results - use enumeration
            run_number = i

        # Create detailed rows for each epoch
        for epoch_info in result.epoch_results:
            detailed_rows.append(
                {
                    "Run_Name": result.name,
                    "Run_Number": run_number,
                    "Epoch": epoch_info["epoch"],
                    "Batches": epoch_info["batches"],
                    "Samples": epoch_info["samples"],
                    "Samples_per_sec": epoch_info["samples"] / epoch_info["elapsed"]
                    if epoch_info["elapsed"] > 0
                    else 0,
                    "Peak_Memory_MB": epoch_info["peak_memory"] - result.memory_before_instantiation_mb,
                    "Average_Memory_MB": epoch_info["avg_memory"] - result.memory_before_instantiation_mb,
                    "Total_Time_s": epoch_info["iteration_time"],
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
                    "Shuffle": result.shuffle,
                    "Dataset_Instantiation_Time_s": result.dataset_instantiation_time_seconds,
                    "Dataloader_Instantiation_Time_s": result.dataloader_instantiation_time_seconds,
                    "Peak_Instantiation_Memory_MB": result.peak_memory_during_instantiation_mb,
                    "Batches_per_sec": epoch_info["batches"] / epoch_info["elapsed"]
                    if epoch_info["elapsed"] > 0
                    else 0,
                }
            )

    # Write detailed CSV (always append, never overwrite)
    pd.DataFrame(detailed_rows).to_csv(detailed_csv, mode="a", header=header, index=False)
    if not file_exists:
        print(f"Created Detailed breakdown CSV: {os.path.abspath(detailed_csv)}")
    else:
        print(f"Appended to Detailed breakdown CSV: {os.path.abspath(detailed_csv)}")


def get_disk_size(path: Union[str, Path]) -> float:
    """Get disk size of a file or directory in MB.

    Tested on MacOS and Linux.
    """
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
        raise RuntimeError(f"Could not determine disk size for {path}: {e}")


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
    sample_interval: float = 0.05,
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
    peak_mib = to_mib(max(peak, final))
    avg_mib = to_mib(avg)
    final_mib = to_mib(final)
    delta_mib = peak_mib - baseline_mib
    print("baseline_mib", baseline_mib, "peak_mib", peak_mib)
    return (result, baseline_mib, peak_mib, avg_mib, delta_mib, final_mib, duration)


def download_example_dataset(cache_dir: str = ".", filename: str = "example_data.h5ad") -> str:
    """Download a small example AnnData dataset for testing.

    Args:
        cache_dir: Directory to save the downloaded file (default: current directory)
        filename: Name of the file to save (default: "example_data.h5ad")

    Returns:
        str: Path to the downloaded file

    Raises:
        RuntimeError: If download fails
    """
    cache_path = Path(cache_dir) / filename

    # If file already exists, return it
    if cache_path.exists():
        print(f"Using cached dataset: {cache_path}")
        return str(cache_path)

    # URL for a small example dataset (using 10x genomics pbmc3k dataset)
    url = "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.h5"

    try:
        print(f"Downloading example dataset to {cache_path}...")
        os.makedirs(cache_dir, exist_ok=True)

        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\rDownload progress: {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, cache_path, reporthook=progress_hook)
        print("\nDownload completed!")

        return str(cache_path)

    except Exception as e:
        # Clean up partial download
        if cache_path.exists():
            cache_path.unlink()
        raise RuntimeError(f"Failed to download example dataset: {e}")
