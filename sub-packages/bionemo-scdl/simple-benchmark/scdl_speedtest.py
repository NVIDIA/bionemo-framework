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
"""Standalone SCDL Benchmark Script.

This script benchmarks SingleCellMemMapDataset performance with different sampling strategies
and reports performance metrics including instantiation time, samples per second,
baseline memory, and peak memory.
"""

# Show banner only when run as script (not when imported)
if __name__ == "__main__":
    print("BioNeMo::SCDL SpeedTest Benchmarking")
    print("Loading dependencies...")

import argparse
import gc
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from anndata.experimental import AnnCollection, AnnLoader


def _dep_message():
    """Print message with instructions for installing all required dependencies."""
    print("To install all dependencies, run: pip install torch pandas psutil tqdm bionemo-scdl")


# Import with helpful error messages
try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    print("Error: PyTorch not found. Please install with:")
    print("   pip install torch")
    _dep_message()
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("Error: psutil not found. Please install with:")
    print("   pip install psutil")
    _dep_message()
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: pandas not found. Please install with:")
    print("   pip install pandas")
    _dep_message()
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm not found. Please install with:")
    print("   pip install tqdm")
    _dep_message()
    sys.exit(1)

try:
    from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
    from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch
except ImportError:
    print("Error: BioNeMo SCDL not found. Please install with:")
    print("   pip install bionemo-scdl")
    print("\nAlternatively, if you have the source code:")
    print("   cd /path/to/bionemo-framework")
    print("   pip install -e sub-packages/bionemo-scdl/")
    _dep_message()
    sys.exit(1)

# Optional AnnData import for baseline comparison
# These packages are only required when using --generate-baseline
try:
    import anndata
    import numpy as np
    import scipy.sparse as sp

    ANNDATA_AVAILABLE = True
except ImportError:
    # AnnData and/or scipy not available - baseline comparison will be disabled
    ANNDATA_AVAILABLE = False

# Show completion message after all imports are done
if __name__ == "__main__":
    print("Done.")
    print()


class AnnDataDataset:
    """Simple AnnData-backed dataset for baseline comparison."""

    def __init__(self, h5ad_path: str):
        """Initialize AnnDataDataset by loading an h5ad file.

        Args:
            h5ad_path (str): Path to the h5ad file to load.
        """
        print(f"Loading AnnData file: {Path(h5ad_path).name}")
        load_start = time.perf_counter()
        self.adata = anndata.read_h5ad(h5ad_path)
        load_end = time.perf_counter()
        self.load_time = load_end - load_start
        print(f"AnnData loading completed in {self.load_time:.2f} seconds")

        # Convert to CSR if not already
        if not sp.issparse(self.adata.X):
            print("Converting dense matrix to sparse...")
            self.adata.X = sp.csr_matrix(self.adata.X)
        elif not isinstance(self.adata.X, sp.csr_matrix):
            print("Converting sparse matrix to CSR format...")
            self.adata.X = self.adata.X.tocsr()

    def __len__(self):
        """Return the number of observations (rows) in the AnnData object."""
        return self.adata.n_obs

    def __getitem__(self, idx):
        """Get a single observation from the dataset.

        Args:
            idx: Index of the observation to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing (values, indices) where:
                - values: Non-zero values as float32 array
                - indices: Non-zero indices as int32 array
        """
        # Return sparse row data similar to SCDL format
        row_data = self.adata.X[idx]
        if sp.issparse(row_data):
            row_data = row_data.toarray().flatten()

        # Find non-zero indices and values
        nonzero_indices = np.nonzero(row_data)[0]
        nonzero_values = row_data[nonzero_indices]

        return (nonzero_values.astype(np.float32), nonzero_indices.astype(np.int32))


def collate_anndata_batch(batch):
    """Collate function for AnnData dataset to match SCDL output format."""
    if len(batch) == 1:
        values, indices = batch[0]
        # Create CSR tensor for single item
        n_features = indices.max() + 1 if len(indices) > 0 else 1
        return torch.sparse_csr_tensor(
            torch.tensor([0, len(values)], dtype=torch.int32),  # crow_indices
            torch.from_numpy(indices),  # col_indices
            torch.from_numpy(values),  # values
            size=(1, n_features),
        )

    # For batch > 1, create batch CSR tensor
    batch_values = []
    batch_col_indices = []
    batch_row_pointers = [0]
    max_features = 0

    for values, indices in batch:
        batch_values.extend(values)
        batch_col_indices.extend(indices)
        batch_row_pointers.append(batch_row_pointers[-1] + len(values))
        if len(indices) > 0:
            max_features = max(max_features, indices.max() + 1)

    if not batch_values:  # Empty batch
        max_features = 1

    return torch.sparse_csr_tensor(
        torch.tensor(batch_row_pointers, dtype=torch.int32),
        torch.tensor(batch_col_indices, dtype=torch.int32),
        torch.tensor(batch_values, dtype=torch.float32),
        size=(len(batch), max_features),
    )


def dense_to_csr(dense: torch.Tensor):
    """Converts a dense PyTorch tensor to a sparse CSR tensor.

    Args:
        dense (torch.Tensor): A 2D dense tensor.

    Returns:
        torch.Tensor: A sparse CSR tensor with the same shape as the input.
    """
    nz = dense != 0
    row, col = nz.nonzero(as_tuple=True)
    vals = dense[nz]
    counts = torch.bincount(row, minlength=dense.size(0))
    crow = torch.cat([torch.tensor([0], dtype=torch.long), counts.cumsum(0)])
    return torch.sparse_csr_tensor(crow, col.to(torch.long), vals, dense.shape)


def collate_annloader_batch(batch):
    """Collate function for AnnData dataset to match SCDL output format."""
    return dense_to_csr(batch.X)


@dataclass
class BenchmarkResult:
    """Results from benchmarking a dataloader."""

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

    # Instantiation metrics
    instantiation_time_seconds: Optional[float] = None
    peak_memory_during_instantiation_mb: Optional[float] = None
    memory_before_instantiation_mb: Optional[float] = None
    memory_after_instantiation_mb: Optional[float] = None

    # Configuration metadata
    madvise_interval: Optional[int] = None
    data_path: Optional[str] = None
    max_time_seconds: Optional[float] = None
    shuffle: Optional[bool] = None

    # Conversion metrics
    conversion_time_seconds: Optional[float] = None
    conversion_performed: bool = False

    # Load metrics (for AnnData baseline)
    load_time_seconds: Optional[float] = None
    load_performed: bool = False

    # Warmup metrics
    warmup_samples: int = 0
    warmup_batches: int = 0

    # Speed metrics
    total_speed_samples_per_second: float = 0.0
    post_warmup_speed_samples_per_second: float = 0.0

    def __post_init__(self):
        """Post-initialization hook for BenchmarkResult.

        Ensures that the 'errors' attribute is always a list after initialization,
        even if not provided. This prevents issues with mutable default arguments.
        """
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
        """Create BenchmarkResult from raw metrics."""
        # Calculate timing metrics
        total_time_excluding_warmup = iteration_time - warmup_time
        avg_batch_time = total_time_excluding_warmup / total_batches if total_batches > 0 else 0
        samples_per_sec = total_samples / total_time_excluding_warmup if total_time_excluding_warmup > 0 else 0
        batches_per_sec = total_batches / total_time_excluding_warmup if total_time_excluding_warmup > 0 else 0

        # Calculate speed metrics
        total_samples_including_warmup = total_samples + warmup_samples
        total_speed = total_samples_including_warmup / iteration_time if iteration_time > 0 else 0
        post_warmup_speed = total_samples / total_time_excluding_warmup if total_time_excluding_warmup > 0 else 0

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
            peak_memory_mb=0,  # Will be set by caller
            average_memory_mb=0,  # Will be set by caller
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


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

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
    log_iteration_times_to_file: Optional[int] = None


def get_batch_size(batch) -> int:
    """Get the batch size from a batch of data."""
    if hasattr(batch, "shape"):
        return batch.shape[0]
    elif isinstance(batch, (list, tuple)):
        if len(batch) > 0 and hasattr(batch[0], "shape"):
            return batch[0].shape[0]
        return len(batch)
    elif hasattr(batch, "__len__"):
        return len(batch)
    return 1


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


def monitor_memory_dynamic_pss(parent_pid, stop_event, result_queue):
    """Monitor memory usage for a process and its children."""
    peak = 0
    samples = []
    sample_count = 0

    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        result_queue.put((0, 0.0))
        return

    # Take initial measurement
    try:
        mem_info = parent.memory_full_info()
        initial_mem = mem_info.pss if hasattr(mem_info, "pss") else mem_info.rss
        peak = initial_mem
        samples.append(initial_mem)
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
        initial_mem = 0

    # Monitor with faster sampling for better accuracy
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
                    if hasattr(mem_info, "pss"):
                        total_mem += mem_info.pss
                    else:
                        total_mem += mem_info.rss
                except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                    continue

        if total_mem > 0:  # Only record valid measurements
            samples.append(total_mem)
            peak = max(peak, total_mem)
            sample_count += 1

        # Faster sampling for better accuracy (20ms instead of 50ms)
        time.sleep(0.02)

    avg = sum(samples) / len(samples) if samples else 0
    # Debug info - uncomment for troubleshooting
    # if sample_count == 0:
    #     print(f"Warning: Memory monitoring collected 0 samples!")
    # print(f"Debug: Memory monitoring collected {sample_count} samples, peak={peak/1024/1024:.1f}MB, avg={avg/1024/1024:.1f}MB")
    result_queue.put((peak, avg))


def measure_memory_simple(func, *args, **kwargs):
    """Simple memory measurement without multiprocessing."""
    gc.collect()
    proc = psutil.Process()
    mem_info = proc.memory_full_info()
    baseline = mem_info.pss if hasattr(mem_info, "pss") else mem_info.rss

    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration = time.perf_counter() - start

    gc.collect()
    mem_info = proc.memory_full_info()
    final = mem_info.pss if hasattr(mem_info, "pss") else mem_info.rss

    # For simple measurement, use final as peak (may underestimate but won't be 0)
    peak = max(baseline, final)
    avg = (baseline + final) / 2

    baseline_mib = baseline / 1024 / 1024
    peak_mib = peak / 1024 / 1024
    avg_mib = avg / 1024 / 1024
    delta_mib = peak_mib - baseline_mib
    final_mib = final / 1024 / 1024

    return result, baseline_mib, peak_mib, avg_mib, delta_mib, final_mib, duration


def measure_peak_memory_full(func, *args, **kwargs):
    """Measure peak memory usage while executing a function."""
    # Try multiprocessing approach first, fall back to simple if it fails
    try:
        parent_pid = os.getpid()
        stop_event = mp.Event()
        result_queue = mp.Queue()
        monitor = mp.Process(target=monitor_memory_dynamic_pss, args=(parent_pid, stop_event, result_queue))

        gc.collect()
        # Use PSS on Linux, RSS on other platforms
        proc = psutil.Process(parent_pid)
        mem_info = proc.memory_full_info()
        baseline = mem_info.pss if hasattr(mem_info, "pss") else mem_info.rss

        monitor.start()
        start = time.perf_counter()

        try:
            result = func(*args, **kwargs)
        finally:
            stop_event.set()
            monitor.join()

        duration = time.perf_counter() - start

        # Get memory measurements from monitoring process
        peak = baseline  # Default fallback
        avg = baseline
        monitoring_failed = False

        try:
            peak, avg = result_queue.get(timeout=5)  # Increase timeout
            if peak == 0 or peak < baseline:  # If monitoring returned invalid data
                monitoring_failed = True
                # print(f"Debug: Memory monitoring returned invalid data: peak={peak/1024/1024:.1f}MB, baseline={baseline/1024/1024:.1f}MB")
        except Exception:
            monitoring_failed = True
            # print(f"Debug: Memory monitoring failed: {e}")

        if monitoring_failed:
            # Fallback: measure current memory as both peak and average
            proc = psutil.Process(parent_pid)
            mem_info = proc.memory_full_info()
            current_mem = mem_info.pss if hasattr(mem_info, "pss") else mem_info.rss
            # Use the higher of baseline or current as peak
            peak = max(baseline, current_mem)
            avg = peak
            # print(f"Debug: Using fallback memory measurement: baseline={baseline/1024/1024:.1f}MB, fallback_peak={peak/1024/1024:.1f}MB")

        gc.collect()
        proc = psutil.Process(parent_pid)
        mem_info = proc.memory_full_info()
        final = mem_info.pss if hasattr(mem_info, "pss") else mem_info.rss

        baseline_mib = baseline / 1024 / 1024
        peak_mib = peak / 1024 / 1024
        avg_mib = avg / 1024 / 1024
        delta_mib = peak_mib - baseline_mib
        final_mib = final / 1024 / 1024

        # Debug output - can be removed in production
        # print(f"Debug: multiprocessing baseline={baseline_mib:.1f}MB, peak={peak_mib:.1f}MB, final={final_mib:.1f}MB, duration={duration:.2f}s")

        return result, baseline_mib, peak_mib, avg_mib, delta_mib, final_mib, duration

    except Exception:
        # If multiprocessing completely fails, use simple measurement
        # print(f"Debug: Falling back to simple memory measurement due to: {e}")
        return measure_memory_simple(func, *args, **kwargs)


def run_benchmark(dataloader: Any, config: BenchmarkConfig) -> BenchmarkResult:
    """Run the actual benchmark and collect metrics."""
    gc.collect()

    def benchmark_iteration_single_epoch(epoch_num, do_warmup):
        """Run a single epoch of benchmarking, with optional warmup.

        Args:
            epoch_num (int): The current epoch number (0-indexed).
            do_warmup (bool): Whether to perform warmup for this epoch.

        Returns:
            Tuple: Contains (epoch_samples, epoch_batches, warmup_samples,
                          warmup_batches, warmup_time, iteration_times)
        """
        update_interval = 10
        epoch_samples = 0
        epoch_batches = 0
        warmup_samples = 0
        warmup_batches = 0
        warmup_time = 0.0

        pbar = tqdm(
            desc=f"{config.name} - Epoch {epoch_num + 1}/{config.num_epochs}", disable=not config.print_progress
        )

        # Initialize warmup timer
        if do_warmup and config.warmup_time_seconds is not None and config.warmup_time_seconds > 0:
            warm_up_start = time.perf_counter()
            warm_up_end = warm_up_start + config.warmup_time_seconds
            is_warming_up = True
            start_time = None
            end_time = None
        else:
            is_warming_up = False
            start_time = time.perf_counter()
            end_time = start_time + config.max_time_seconds if config.max_time_seconds is not None else None
        for num, batch in enumerate(dataloader):
            batch_size = get_batch_size(batch)
            current_time = time.perf_counter()
            if is_warming_up:
                warmup_samples += batch_size
                warmup_batches += 1

                if current_time >= warm_up_end:
                    warmup_time = current_time - warm_up_start
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
                del batch
                continue

            # Main benchmark phase
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

            # Check limits
            if config.max_batches and epoch_batches >= config.max_batches:
                break
            if end_time is not None and current_time >= end_time:
                break

        pbar.close()
        # This addresses the case when it is all warm up time:
        if is_warming_up:
            warmup_time = current_time - warm_up_start
        return epoch_samples, epoch_batches, warmup_samples, warmup_batches, warmup_time, []

    epoch_results = []
    total_warmup_samples = 0
    total_warmup_batches = 0
    total_warmup_time = 0.0

    for epoch in range(config.num_epochs):

        def epoch_benchmark_iteration():
            return benchmark_iteration_single_epoch(epoch, epoch == 0)

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

        total_warmup_samples += warmup_samples
        total_warmup_batches += warmup_batches
        total_warmup_time += warmup_time
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
            print(f"Epoch {epoch + 1} completed: {epoch_samples:,} samples, {epoch_batches:,} batches")

    # Calculate totals
    total_samples = sum(r["samples"] for r in epoch_results)
    total_batches = sum(r["batches"] for r in epoch_results)
    total_iteration_time = sum(r["iteration_time"] for r in epoch_results)

    result = BenchmarkResult.from_raw_metrics(
        name=config.name,
        madvise_interval=config.madvise_interval,
        data_path=str(config.data_path) if config.data_path else None,
        max_time_seconds=config.max_time_seconds,
        shuffle=config.shuffle,
        total_samples=total_samples,
        total_batches=total_batches,
        setup_time=0,
        warmup_time=total_warmup_time,
        iteration_time=total_iteration_time,
        warmup_samples=total_warmup_samples,
        warmup_batches=total_warmup_batches,
    )

    # Add epoch results and memory info
    result.epoch_results = epoch_results
    max_peak_memory = max(r["peak_memory"] for r in epoch_results)
    avg_memory = sum(r["avg_memory"] for r in epoch_results) / len(epoch_results)
    result.peak_memory_mb = max_peak_memory
    result.average_memory_mb = avg_memory

    return result


def benchmark_dataloader(
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
) -> BenchmarkResult:
    """Benchmark a single dataloader using a factory function."""

    def log(message):
        if print_progress:
            print(message)

    log(f"Benchmarking: {name}")

    # Measure instantiation metrics
    dataloader, baseline, peak, _, _, final_mib, setup_time = measure_peak_memory_full(dataloader_factory)

    # Get conversion and load metrics if available
    conversion_time = 0.0
    conversion_performed = False
    load_time = 0.0
    load_performed = False

    if hasattr(dataloader_factory, "conversion_metrics"):
        conversion_time = dataloader_factory.conversion_metrics["time"]
        conversion_performed = dataloader_factory.conversion_metrics["performed"]

    if hasattr(dataloader_factory, "load_metrics"):
        load_time = dataloader_factory.load_metrics["time"]
        load_performed = dataloader_factory.load_metrics["performed"]

    instantiation_metrics = {
        "instantiation_time_seconds": setup_time,
        "peak_memory_during_instantiation_mb": peak,
        "memory_before_instantiation_mb": baseline,
        "memory_after_instantiation_mb": final_mib,
        "conversion_time_seconds": conversion_time,
        "conversion_performed": conversion_performed,
        "load_time_seconds": load_time,
        "load_performed": load_performed,
    }

    # Measure disk size using the actual data path used by the dataset
    disk_size_mb = 0.0
    actual_path = data_path
    if hasattr(dataloader_factory, "actual_data_path"):
        actual_path = dataloader_factory.actual_data_path["path"]

    if actual_path:
        disk_size_mb = get_disk_size(actual_path)

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
    )

    # Run benchmark
    result = run_benchmark(dataloader, config)
    del dataloader

    # Add instantiation metrics
    for key, value in instantiation_metrics.items():
        setattr(result, key, value)

    result.setup_time_seconds += setup_time
    result.disk_size_mb = disk_size_mb

    # Print results summary
    if print_progress:
        print(f"\n{result.name}")
        print(f"   Samples/sec: {result.samples_per_second:.2f}")
        print(f"   Memory: {result.peak_memory_mb:.1f} MB")
        print(f"   Setup: {result.instantiation_time_seconds:.3f}s")

    return result


def calculate_derived_metrics(result: BenchmarkResult) -> Dict[str, float]:
    """Calculate derived metrics from a BenchmarkResult."""
    warmup_samples_per_sec = (
        (result.warmup_samples / result.warmup_time_seconds) if result.warmup_time_seconds > 0 else 0.0
    )

    num_epochs = len(getattr(result, "epoch_results", [])) or 1
    dataset_samples_per_epoch = result.total_samples // num_epochs if num_epochs > 0 else result.total_samples
    dataset_batches_per_epoch = result.total_batches // num_epochs if num_epochs > 0 else result.total_batches
    avg_batch_size = dataset_samples_per_epoch / dataset_batches_per_epoch if dataset_batches_per_epoch > 0 else 0
    dataset_size_k_samples = dataset_samples_per_epoch / 1000.0

    inst_memory = getattr(result, "peak_memory_during_instantiation_mb", 0.0) or 0.0
    inst_time = getattr(result, "instantiation_time_seconds", 0.0) or 0.0
    conversion_time = getattr(result, "conversion_time_seconds", 0.0) or 0.0
    conversion_performed = getattr(result, "conversion_performed", False)
    load_time = getattr(result, "load_time_seconds", 0.0) or 0.0
    load_performed = getattr(result, "load_performed", False)

    return {
        "warmup_samples_per_sec": warmup_samples_per_sec,
        "num_epochs": num_epochs,
        "dataset_samples_per_epoch": dataset_samples_per_epoch,
        "dataset_batches_per_epoch": dataset_batches_per_epoch,
        "avg_batch_size": avg_batch_size,
        "dataset_size_k_samples": dataset_size_k_samples,
        "inst_memory": inst_memory,
        "inst_time": inst_time,
        "conversion_time": conversion_time,
        "conversion_performed": conversion_performed,
        "load_time": load_time,
        "load_performed": load_performed,
    }


def export_benchmark_results(results: List[BenchmarkResult], output_prefix: str = "benchmark_data") -> None:
    """Export benchmark results to CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{output_prefix}_{timestamp}"
    summary_rows = []
    detailed_rows = []

    for result in results:
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
                "Conversion_Time_s": getattr(result, "conversion_time_seconds", None),
                "Conversion_Performed": getattr(result, "conversion_performed", False),
                "Load_Time_s": getattr(result, "load_time_seconds", None),
                "Load_Performed": getattr(result, "load_performed", False),
            }
        )

        # Overall run summary
        detailed_rows.append(
            {
                "Configuration": result.name,
                "Run_Number": 1,
                "Epoch": "OVERALL",
                "Samples": result.total_samples,
                "Batches": result.total_batches,
                "Samples_per_sec": result.samples_per_second,
                "Peak_Memory_MB": result.peak_memory_mb,
                "Average_Memory_MB": result.average_memory_mb,
                "Total_Time_s": result.total_iteration_time_seconds,
                "Setup_Time_s": result.setup_time_seconds,
                "Warmup_Time_s": result.warmup_time_seconds,
                "Warmup_Samples": result.warmup_samples,
                "Warmup_Batches": result.warmup_batches,
                "Post_Warmup_Speed_Samples_per_sec": result.post_warmup_speed_samples_per_second,
                "Total_Speed_With_Warmup_Samples_per_sec": result.total_speed_samples_per_second,
                "Dataset_Path": result.data_path,
                "Madvise_Interval": result.madvise_interval,
                "Max_Time_Seconds": result.max_time_seconds,
                "Shuffle": getattr(result, "shuffle", None),
                "Instantiation_Time_s": getattr(result, "instantiation_time_seconds", None),
                "Instantiation_Memory_MB": getattr(result, "peak_memory_during_instantiation_mb", None),
                "Conversion_Time_s": getattr(result, "conversion_time_seconds", None),
                "Conversion_Performed": getattr(result, "conversion_performed", False),
                "Load_Time_s": getattr(result, "load_time_seconds", None),
                "Load_Performed": getattr(result, "load_performed", False),
            }
        )

        # Per-epoch breakdown
        if hasattr(result, "epoch_results") and result.epoch_results:
            for epoch_info in result.epoch_results:
                avg_batch_size = epoch_info["samples"] / epoch_info["batches"] if epoch_info["batches"] > 0 else 0
                detailed_rows.append(
                    {
                        "Configuration": result.name,
                        "Run_Number": 1,
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
                        "Warmup_Time_s": result.warmup_time_seconds if epoch_info["epoch"] == 1 else 0,
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
                        "Dataset_Path": result.data_path,
                        "Madvise_Interval": result.madvise_interval,
                        "Max_Time_Seconds": result.max_time_seconds,
                        "Shuffle": getattr(result, "shuffle", None),
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
    pd.DataFrame(detailed_rows).to_csv(f"{base_filename}_detailed_breakdown.csv", index=False)
    print("Export complete.")


def create_dataloader_factory_old(
    input_path: str, sampling_scheme: str, batch_size: int = 32, use_anndata: bool = False
):
    """Create a factory function for the dataloader."""
    # Track conversion metrics globally to be accessible later
    conversion_metrics = {"time": 0.0, "performed": False}
    load_metrics = {"time": 0.0, "performed": False}
    actual_data_path = {"path": input_path}  # Track the actual data path used

    def factory():
        if use_anndata:
            # Create AnnData-based dataset
            if not input_path.endswith(".h5ad"):
                raise ValueError("AnnData baseline requires .h5ad input file")
            dataset = AnnDataDataset(input_path)
            load_metrics["time"] = dataset.load_time
            load_metrics["performed"] = True
            actual_data_path["path"] = input_path  # AnnData uses the original h5ad file
            collate_fn = collate_anndata_batch
        else:
            # Create SCDL dataset
            if input_path.endswith(".h5ad"):
                # For h5ad files, check if SCMMAP cache already exists
                data_dir = str(Path(input_path).parent / Path(input_path).stem)

                # Check if the memory-mapped dataset already exists
                if Path(data_dir).exists():
                    # Load from existing SCMMAP cache
                    dataset = SingleCellMemMapDataset(data_path=data_dir)
                    conversion_metrics["performed"] = False
                else:
                    # Create new SCMMAP from h5ad file - measure conversion time
                    print(f"Converting h5ad to SCDL format: {Path(input_path).name}")
                    conversion_start = time.perf_counter()
                    dataset = SingleCellMemMapDataset(data_path=data_dir, h5ad_path=input_path)
                    conversion_end = time.perf_counter()
                    conversion_time = conversion_end - conversion_start
                    conversion_metrics["time"] = conversion_time
                    conversion_metrics["performed"] = True
                    print(f"Conversion completed in {conversion_time:.2f} seconds")
                actual_data_path["path"] = data_dir  # SCDL uses the converted directory
            else:
                # For scdl format, input_path is the data directory
                dataset = SingleCellMemMapDataset(data_path=input_path)
                conversion_metrics["performed"] = False
                actual_data_path["path"] = input_path  # SCDL directory is the input path

            collate_fn = collate_sparse_matrix_batch

        # Configure sampling based on sampling scheme
        if sampling_scheme == "sequential":
            # Access samples in order: 0, 1, 2, 3, ...
            shuffle = False
            sampler = None
        elif sampling_scheme == "shuffle":
            # Use PyTorch's built-in shuffle (reproducible based on generator state)
            shuffle = True
            sampler = None
        elif sampling_scheme == "random":
            # Random permutation with random seed (different each run)
            shuffle = False  # Don't use DataLoader's shuffle when using custom sampler
            # Use current time as seed for true randomness
            torch.manual_seed(int(time.time() * 1000000) % 2**32)
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            raise ValueError(f"Unknown sampling scheme: {sampling_scheme}")

        # Create and return the dataloader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    # Attach metrics to the factory function
    factory.conversion_metrics = conversion_metrics
    factory.load_metrics = load_metrics
    factory.actual_data_path = actual_data_path
    return factory


def get_sampler(sampling_scheme: str, dataset: torch.utils.data.Dataset):
    """Returns the shuffle flag and sampler object for a given sampling scheme.

    Args:
        sampling_scheme (str): The sampling strategy to use. One of "sequential", "shuffle", or "random".
        dataset (torch.utils.data.Dataset): The dataset to sample from.

    Returns:
        Tuple[bool, torch.utils.data.Sampler]:
            - shuffle (bool): Whether to use DataLoader's shuffle.
            - sampler (torch.utils.data.Sampler or None): The sampler object to use, or None if not applicable.

    Raises:
        ValueError: If the sampling_scheme is not recognized.
    """
    if sampling_scheme == "sequential":
        # Access samples in order: 0, 1, 2, 3, ...
        shuffle = False
        sampler = None
    elif sampling_scheme == "shuffle":
        # Use PyTorch's built-in shuffle (reproducible based on generator state)
        shuffle = True
        sampler = None
    elif sampling_scheme == "random":
        # Random permutation with random seed (different each run)
        shuffle = False  # Don't use DataLoader's shuffle when using custom sampler
        # Use current time as seed for true randomness
        torch.manual_seed(int(time.time() * 1000000) % 2**32)
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        raise ValueError(f"Unknown sampling scheme: {sampling_scheme}")
    return shuffle, sampler


def create_dataloader_factory(input_path: str, sampling_scheme: str, batch_size: int = 32, use_anndata: bool = False):
    """Create a factory function for the dataloader."""
    # Track conversion metrics globally to be accessible later
    conversion_metrics = {"time": 0.0, "performed": False}
    load_metrics = {"time": 0.0, "performed": False}
    actual_data_path = {"path": input_path}  # Track the actual data path used

    def factory():
        # Configure sampling based on sampling scheme

        if use_anndata:
            # Create AnnData-based dataset
            load_start = time.perf_counter()

            if input_path.endswith(".h5ad"):
                dataset = anndata.read_h5ad(input_path, backed="r")
            elif os.path.isdir(input_path) and any(f.endswith(".h5ad") for f in os.listdir(input_path)):
                h5ad_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".h5ad")]
                dataset = AnnCollection([anndata.read_h5ad(f, backed="r") for f in h5ad_files])
            else:
                raise ValueError("AnnData baseline requires a .h5ad input file or a directory containing .h5ad files")
            load_end = time.perf_counter()
            load_metrics["time"] = load_end - load_start
            load_metrics["performed"] = True
            actual_data_path["path"] = input_path  # AnnData uses the original h5ad file
            shuffle, sampler = get_sampler(sampling_scheme, dataset)
            if sampler:
                return AnnLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    sampler=sampler,
                    num_workers=0,
                    collate_fn=collate_annloader_batch,
                )
            else:
                return AnnLoader(
                    dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_annloader_batch
                )
        else:
            # Create SCDL dataset
            if input_path.endswith(".h5ad"):
                # For h5ad files, check if SCMMAP cache already exists
                data_dir = str(Path(input_path).parent / Path(input_path).stem)

                # Check if the memory-mapped dataset already exists
                if Path(data_dir).exists():
                    # Load from existing SCMMAP cache
                    dataset = SingleCellMemMapDataset(data_path=data_dir)
                    conversion_metrics["performed"] = False
                else:
                    # Create new SCMMAP from h5ad file - measure conversion time
                    print(f"Converting h5ad to SCDL format: {Path(input_path).name}")
                    conversion_start = time.perf_counter()
                    dataset = SingleCellMemMapDataset(data_path=data_dir, h5ad_path=input_path)
                    conversion_end = time.perf_counter()
                    conversion_time = conversion_end - conversion_start
                    conversion_metrics["time"] = conversion_time
                    conversion_metrics["performed"] = True
                    print(f"Conversion completed in {conversion_time:.2f} seconds")
                actual_data_path["path"] = data_dir  # SCDL uses the converted directory
            else:
                # For scdl format, input_path is the data directory
                dataset = SingleCellMemMapDataset(data_path=input_path)
                conversion_metrics["performed"] = False
                actual_data_path["path"] = input_path  # SCDL directory is the input path

            collate_fn = collate_sparse_matrix_batch
            shuffle, sampler = get_sampler(sampling_scheme, dataset)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                drop_last=False,
                collate_fn=collate_fn,
                num_workers=0,
            )

    # Attach metrics to the factory function
    factory.conversion_metrics = conversion_metrics
    factory.load_metrics = load_metrics
    factory.actual_data_path = actual_data_path
    return factory


def download_example_dataset(download_dir: Path = Path("example_data")) -> Path:
    """Download an example dataset for benchmarking."""
    # Example dataset from CellxGene (~25,000 cells)
    url = "https://datasets.cellxgene.cziscience.com/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad"
    filename = "cellxgene_example_25k.h5ad"

    # Create download directory
    download_dir.mkdir(exist_ok=True)
    filepath = download_dir / filename

    if filepath.exists():
        print(f"Example dataset already exists at: {filepath}")
        return filepath

    print("Downloading example dataset (~25,000 cells) from CellxGene...")
    print("This may take a few minutes depending on your internet connection.")
    print(f"Destination: {filepath}")

    try:
        # Download with progress indication
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            downloaded_mb = downloaded / (1024 * 1024)
            if total_size > 0:
                total_mb = total_size / (1024 * 1024)
                percent = min(100, (downloaded * 100) // total_size)
                print(f"\rDownload progress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
            else:
                print(f"\rDownloaded: {downloaded_mb:.1f} MB", end="", flush=True)

        urllib.request.urlretrieve(url, filepath, reporthook=show_progress)
        print("\nDownload completed successfully!")
        print(f"Dataset saved to: {filepath}")
        return filepath

    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        # Clean up partial download
        if filepath.exists():
            filepath.unlink()
        raise


def format_report(result, input_path: str, sampling_scheme: str, method: str = "SCDL") -> str:
    """Format the benchmark results into a readable report."""
    # Calculate epochs and per-epoch metrics
    num_epochs = len(getattr(result, "epoch_results", [])) or 1
    samples_per_epoch = result.total_samples // num_epochs if num_epochs > 0 else result.total_samples
    batches_per_epoch = result.total_batches // num_epochs if num_epochs > 0 else result.total_batches

    report = []
    report.append("=" * 60)
    report.append(f"{method.upper()} BENCHMARK REPORT")
    report.append("=" * 60)
    report.append("")
    report.append(f"Dataset: {Path(input_path).name}")
    report.append(f"Method: {method}")
    report.append(f"Sampling: {sampling_scheme}")
    report.append(f"Epochs: {num_epochs}")
    report.append("")
    report.append("PERFORMANCE METRICS:")
    report.append(f"  Throughput:        {result.samples_per_second:,.0f} samples/sec")
    report.append(f"  Instantiation:     {result.instantiation_time_seconds:.3f} seconds")
    if getattr(result, "conversion_performed", False):
        report.append(f"  H5AD -> SCDL:      {result.conversion_time_seconds:.3f} seconds")
    if getattr(result, "load_performed", False):
        report.append(f"  AnnData Load:      {result.load_time_seconds:.3f} seconds")
    report.append(f"  Avg Batch Time:    {result.average_batch_time_seconds:.4f} seconds")
    report.append("")
    report.append("MEMORY USAGE:")
    report.append(f"  Baseline:          {result.memory_before_instantiation_mb:.1f} MB")
    report.append(f"  Peak (Benchmark):  {result.peak_memory_mb:.1f} MB")
    report.append(f"  Dataset on Disk:   {result.disk_size_mb / 1024:.2f} GB")
    report.append("")
    report.append("DATA PROCESSED:")
    report.append(f"  Total Samples:     {result.total_samples:,} ({samples_per_epoch:,}/epoch)")
    report.append(f"  Total Batches:     {result.total_batches:,} ({batches_per_epoch:,}/epoch)")
    report.append("=" * 60)

    return "\n".join(report)


def format_comparison_report(scdl_result, anndata_result, input_path: str, sampling_scheme: str) -> str:
    """Format a comparison report between SCDL and AnnData results."""
    # Calculate speedup
    scdl_throughput = scdl_result.samples_per_second
    anndata_throughput = anndata_result.samples_per_second
    speedup = scdl_throughput / anndata_throughput if anndata_throughput > 0 else 0

    # Calculate memory efficiency
    scdl_peak = scdl_result.peak_memory_mb
    anndata_peak = anndata_result.peak_memory_mb
    memory_ratio = anndata_peak / scdl_peak if scdl_peak > 0 else 1

    # Calculate disk usage (convert MB to GB)
    scdl_disk_gb = scdl_result.disk_size_mb / 1024
    anndata_disk_gb = anndata_result.disk_size_mb / 1024
    disk_ratio = anndata_disk_gb / scdl_disk_gb if scdl_disk_gb > 0 else 1

    report = []
    report.append("=" * 80)
    report.append("SCDL vs ANNDATA COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")
    report.append(f"Dataset: {Path(input_path).name}")
    report.append(f"Sampling: {sampling_scheme}")
    report.append("")

    report.append("THROUGHPUT COMPARISON:")
    report.append(f"  SCDL:              {scdl_throughput:,.0f} samples/sec")
    report.append(f"  AnnData:           {anndata_throughput:,.0f} samples/sec")

    # Report performance ratio consistently from SCDL's perspective
    if speedup > 1:
        report.append(f"  Performance:       {speedup:.2f}x speedup with SCDL")
    elif speedup < 1 and speedup > 0:
        report.append(f"  Performance:       {1 / speedup:.2f}x slowdown with SCDL")
    else:
        report.append("  Performance:       Unable to calculate (division by zero)")
    report.append("")

    report.append("MEMORY COMPARISON:")
    report.append(f"  SCDL Peak:         {scdl_peak:.1f} MB")
    report.append(f"  AnnData Peak:      {anndata_peak:.1f} MB")

    # Report memory usage consistently from SCDL's perspective
    if memory_ratio > 1:
        report.append(f"  Memory Efficiency: SCDL uses {memory_ratio:.2f}x less memory")
    elif memory_ratio < 1 and memory_ratio > 0:
        report.append(f"  Memory Efficiency: SCDL uses {1 / memory_ratio:.2f}x more memory")
    else:
        report.append("  Memory Efficiency: Unable to calculate (division by zero)")
    report.append("")

    report.append("DISK USAGE COMPARISON:")
    report.append(f"  SCDL Size:         {scdl_disk_gb:.2f} GB")
    report.append(f"  AnnData Size:      {anndata_disk_gb:.2f} GB")

    # Report disk usage consistently from SCDL's perspective
    if scdl_disk_gb == 0 and anndata_disk_gb == 0:
        report.append("  Storage Efficiency: Both datasets have zero disk usage")
    elif scdl_disk_gb == 0:
        report.append("  Storage Efficiency: Unable to calculate (SCDL size is zero)")
    elif disk_ratio > 1:
        report.append(f"  Storage Efficiency: SCDL uses {disk_ratio:.2f}x less disk space")
    elif disk_ratio < 1:
        report.append(f"  Storage Efficiency: SCDL uses {1 / disk_ratio:.2f}x more disk space")
    else:  # disk_ratio == 1
        report.append("  Storage Efficiency: Both datasets use equal disk space")
    report.append("")

    report.append("LOADING TIME COMPARISON:")
    scdl_conversion = getattr(scdl_result, "conversion_time_seconds", 0) or 0
    anndata_load = getattr(anndata_result, "load_time_seconds", 0) or 0
    if scdl_conversion > 0 and anndata_load > 0:
        load_ratio = anndata_load / scdl_conversion
        report.append(f"  SCDL Conversion:   {scdl_conversion:.2f} seconds")
        report.append(f"  AnnData Load:      {anndata_load:.2f} seconds")
        report.append(f"  Load Time Ratio:   {load_ratio:.2f}x")
    elif anndata_load > 0:
        report.append("  SCDL Conversion:   0.00 seconds (cached)")
        report.append(f"  AnnData Load:      {anndata_load:.2f} seconds")
    report.append("")

    report.append("SUMMARY:")
    if speedup > 1:
        report.append(f"  SCDL provides {speedup:.1f}x throughput improvement")
    elif speedup < 1 and speedup > 0:
        report.append(f"  SCDL has {1 / speedup:.1f}x throughput slowdown")
    else:
        report.append("  Unable to determine throughput performance (invalid speedup)")

    if memory_ratio > 1:
        report.append(f"  SCDL uses {memory_ratio:.1f}x less memory")
    elif memory_ratio < 1 and memory_ratio > 0:
        report.append(f"  SCDL uses {1 / memory_ratio:.1f}x more memory")
    else:
        report.append("  Unable to determine memory efficiency (invalid ratio)")

    report.append(f"  SCDL disk usage: {scdl_disk_gb:.2f} GB")
    report.append(f"  AnnData disk usage: {anndata_disk_gb:.2f} GB")
    if scdl_disk_gb == 0 and anndata_disk_gb == 0:
        report.append("  Both datasets have zero disk usage")
    elif scdl_disk_gb == 0:
        report.append("  Unable to calculate storage efficiency (SCDL size is zero)")
    elif disk_ratio > 1:
        report.append(f"  SCDL uses {disk_ratio:.1f}x less disk space")
    elif disk_ratio < 1:
        report.append(f"  SCDL uses {1 / disk_ratio:.1f}x more disk space")
    else:  # disk_ratio == 1
        report.append("  Both datasets use equal disk space")

    report.append("=" * 80)

    return "\n".join(report)


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark SingleCellMemMapDataset performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scdl_speedtest.py                            # Quick benchmark (auto-downloads example data)
  python scdl_speedtest.py -i my_data.h5ad            # Benchmark your dataset
  python scdl_speedtest.py -s sequential              # Use sequential sampling
  python scdl_speedtest.py -o report.txt              # Save report to file
  python scdl_speedtest.py --csv                      # Export detailed CSV files
  python scdl_speedtest.py --generate-baseline        # Compare SCDL vs AnnData performance
        """,
    )

    parser.add_argument("-i", "--input", help="Dataset path (.h5ad file or scdl directory)")
    parser.add_argument("-o", "--output", help="Save report to file (default: print to screen)")
    parser.add_argument(
        "-s",
        "--sampling-scheme",
        choices=["shuffle", "sequential", "random"],
        default="shuffle",
        help="Sampling method (default: shuffle)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--max-time", type=float, default=120.0, help="Max runtime seconds (default: 30)")
    parser.add_argument("--warmup-time", type=float, default=30.0, help="Warmup seconds (default: 2)")
    parser.add_argument("--csv", action="store_true", help="Export detailed CSV files")
    parser.add_argument(
        "--generate-baseline", action="store_true", help="Generate AnnData baseline comparison (requires .h5ad input)"
    )
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs (default: 1)")
    parser.add_argument("--adata_path", type=str, default=None, help="Path to AnnData dataset")

    args = parser.parse_args()

    # Check if baseline generation is requested
    if args.generate_baseline:
        if not ANNDATA_AVAILABLE:
            print("Error: AnnData baseline comparison requires additional packages that are not installed.")
            print("")
            print("To use the --generate-baseline feature, please install:")
            print("   pip install anndata scipy")
            print("")
            print("These packages are needed to:")
            print("  - Load .h5ad files with AnnData")
            print("  - Handle sparse matrices with scipy")
            print("  - Run baseline comparisons between SCDL and AnnData datasets")
            print("")
            print("Alternatively, run without --generate-baseline to benchmark SCDL only.")
            sys.exit(1)

        if args.input and not args.input.endswith(".h5ad"):
            print("Error: --generate-baseline requires .h5ad input file")
            sys.exit(1)

    # Handle input path - download example if none provided or doesn't exist
    if args.input is None:
        print("No dataset specified. Downloading example dataset...")
        try:
            input_path = download_example_dataset()
        except Exception as e:
            print(f"Failed to download example dataset: {e}")
            sys.exit(1)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Dataset not found: '{args.input}'")
            response = input("Download example dataset instead? (y/N): ").strip().lower()
            if response in ["y", "yes"]:
                try:
                    input_path = download_example_dataset()
                except Exception as e:
                    print(f"Download failed: {e}")
                    sys.exit(1)
            else:
                print("Exiting...")
                sys.exit(1)

    try:
        if args.generate_baseline:
            # Run comparison benchmark
            print(f"\nRunning SCDL vs AnnData comparison: {Path(input_path).name}")
            print(f"Sampling: {args.sampling_scheme}")
            print("This will benchmark both SCDL and AnnData approaches...\n")

            # Run SCDL benchmark
            print("=== Running SCDL Benchmark ===")
            scdl_factory = create_dataloader_factory(
                str(input_path), args.sampling_scheme, args.batch_size, use_anndata=False
            )

            scdl_result = benchmark_dataloader(
                name=f"SCDL-{args.sampling_scheme}",
                dataloader_factory=scdl_factory,
                data_path=str(input_path),
                num_epochs=args.num_epochs,
                max_time_seconds=args.max_time,
                warmup_time_seconds=args.warmup_time,
                print_progress=True,
            )

            # Run AnnData benchmark
            adata_path = input_path
            if args.adata_path is not None:
                adata_path = args.data_path
            print("\n=== Running AnnData Benchmark ===")
            anndata_factory = create_dataloader_factory(
                str(adata_path), args.sampling_scheme, args.batch_size, use_anndata=True
            )

            anndata_result = benchmark_dataloader(
                name=f"AnnData-{args.sampling_scheme}",
                dataloader_factory=anndata_factory,
                data_path=str(adata_path),
                num_epochs=args.num_epochs,
                max_time_seconds=args.max_time,
                warmup_time_seconds=args.warmup_time,
                print_progress=True,
            )

            # Format and output comparison report
            comparison_report = format_comparison_report(
                scdl_result, anndata_result, str(input_path), args.sampling_scheme
            )

            if args.output:
                # Save individual reports and comparison
                scdl_report = format_report(scdl_result, str(input_path), args.sampling_scheme, "SCDL")
                anndata_report = format_report(anndata_result, str(input_path), args.sampling_scheme, "AnnData")

                full_report = f"{scdl_report}\n\n{anndata_report}\n\n{comparison_report}"
                with open(args.output, "w") as f:
                    f.write(full_report)
                print(f"\nComparison report saved to: {args.output}")
            else:
                print(f"\n{comparison_report}")

            # Export CSV files if requested
            if args.csv:
                csv_prefix = f"comparison_{args.sampling_scheme}"
                export_benchmark_results([scdl_result, anndata_result], output_prefix=csv_prefix)
                print(f"CSV files exported with prefix: {csv_prefix}_<timestamp>")

        else:
            # Regular SCDL-only benchmark
            factory = create_dataloader_factory(str(input_path), args.sampling_scheme, args.batch_size)

            print(f"\nBenchmarking: {Path(input_path).name}")
            print(f"Sampling: {args.sampling_scheme}")
            print("Running benchmark...\n")

            result = benchmark_dataloader(
                name=f"SCDL-{args.sampling_scheme}",
                dataloader_factory=factory,
                data_path=str(input_path),
                num_epochs=args.num_epochs,
                max_time_seconds=args.max_time,
                warmup_time_seconds=args.warmup_time,
                print_progress=True,
            )

            # Format and output report
            report = format_report(result, str(input_path), args.sampling_scheme)
            if args.output:
                with open(args.output, "w") as f:
                    f.write(report)
                print(f"\nReport saved to: {args.output}")
            else:
                print(f"\n{report}")

            # Export CSV files if requested
            if args.csv:
                csv_prefix = f"scdl_benchmark_{args.sampling_scheme}"
                export_benchmark_results([result], output_prefix=csv_prefix)
                print(f"CSV files exported with prefix: {csv_prefix}_<timestamp>")

    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        print("Try running with different parameters or check your dataset.")
        raise
        sys.exit(1)


if __name__ == "__main__":
    main()
