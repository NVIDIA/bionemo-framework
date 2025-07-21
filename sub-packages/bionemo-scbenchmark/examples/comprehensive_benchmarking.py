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


# Import AnnData support
# from arrayloaders.io.dask_loader import DaskDataset

import anndata as ad
from anndata.experimental import AnnCollection, AnnLoader
from torch.utils.data import DataLoader

# Import the benchmarking framework
from bionemo.scbenchmark import benchmark_dataloader
from bionemo.scbenchmark.common import export_benchmark_results
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch


def create_annloader_factory(batch_size=32, backed="r", shuffle=True, data_path=None, num_workers=0):
    """Create a factory function for AnnData dataloaders with different configurations.

    Args:
        batch_size: Number of samples per batch
        backed: AnnData backed mode ('r' for read-only, 'r+' for read-write)
        shuffle: Whether to shuffle the data
        data_path: Path to the data files
        num_workers: Number of worker processes for data loading

    Returns:
        Factory function that creates an AnnData dataloader
    """

    def factory():
        # Read all .h5ad files in data_path and create an AnnCollection from them
        print(f"Reading {data_path}...")
        # h5ad_files = [join(data_path, f) for f in os.listdir(data_path) if f.endswith('.h5ad')]
        h5ad_files = [data_path]
        print("H5AD FILES: ", h5ad_files)
        datasets = AnnCollection([ad.read_h5ad(data_path, backed="r")])
        print("Created AnnCollection ")
        return AnnLoader(datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return factory


def create_scdl_factory(batch_size=32, shuffle=True, adata_path=None, data_path=None, num_workers=0):
    """Create a factory function for AnnData dataloaders with different configurations.

    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        adata_path: Path to the AnnData file
        data_path: Path to the data files (for disk size measurement)
        num_workers: Number of worker processes for data loading

    Returns:
        Factory function that creates an SCDL dataloader
    """

    def factory():
        dataset = SingleCellMemMapDataset(data_path)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=collate_sparse_matrix_batch,
            num_workers=num_workers,
        )

    return factory


def comprehensive_benchmarking_example(num_epochs: int = 3, num_runs: int = 1):
    """Run comprehensive benchmarking with all features.

    This function demonstrates every capability of the benchmarking framework
    using AnnData dataloaders with different configurations, including
    multi-epoch benchmarking and multiple runs for statistical analysis.

    Args:
        num_epochs: Number of epochs to run for each benchmark (default: 3)
        num_runs: Number of times to run each configuration for statistics (default: 1)
    """
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARKING EXAMPLE")
    print("=" * 80)
    print()

    # Try to use real AnnData if available
    # adata_path = "/home/pbinder/bionemo-framework/sub-packages/bionemo-scdl/small_samples/sample_50000_19836_0.85.h5ad"
    # memmap_path = "/home/pbinder/bionemo-framework/sub-packages/bionemo-scdl/small_samples/s_memmap_zmdy090y"
    adata_path = (
        "/home/pbinder/bionemo-framework/tahoe_data/plate11_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
    )
    memmap_path = "/home/pbinder/bionemo-framework/tahoe_memmap/"
    # Create different configurations of the same dataloader

    # 7. MULTIPLE CONFIGURATIONS WITH STATISTICAL ANALYSIS
    print(f"🚀 Benchmarking {num_runs} run(s) each")
    print()
    """        {
            "name": "SCDL madvise_1",
            "dataloader_factory": create_scdl_factory(
                batch_size=64,
                shuffle=True,
                adata_path=adata_path,
                data_path=memmap_path,
                num_workers=0,
            ),
            "num_epochs": num_epochs,
            "max_time_seconds": 1000.0,
            "warmup_seconds": 10.0,
            "data_path": memmap_path,
            "madvise_interval": 1,
            "num_runs": num_runs
        },
    """
    configurations = [
        # Example: Enable per-iteration time and memory (RSS) logging every 5 batches
        {
            "name": "AnnLoader Regular",
            "dataloader_factory": create_annloader_factory(
                batch_size=64,
                shuffle=True,
                data_path=adata_path,
                num_workers=0,
            ),
            "num_epochs": num_epochs,
            "max_time_seconds": 10.0,
            "warmup_time_seconds": 2.0,
            "data_path": adata_path,
            "madvise_interval": None,
            "num_runs": num_runs,
            "log_iteration_times_to_file": None,
        },
        {
            "name": "SCDL Regular",
            "dataloader_factory": create_scdl_factory(
                batch_size=64,
                shuffle=True,
                adata_path=adata_path,
                data_path=memmap_path,
                num_workers=0,
            ),
            "num_epochs": num_epochs,
            "max_time_seconds": 10.0,
            "warmup_time_seconds": 2.0,
            "data_path": memmap_path,
            "madvise_interval": None,
            "num_runs": num_runs,
            "log_iteration_times_to_file": None,
        },
    ]
    """
            {
            "name": "SCDL Regular",
            "dataloader_factory": create_scdl_factory(
                batch_size=64,
                shuffle=True,
                adata_path=adata_path,
                data_path=memmap_path,
                num_workers=0,
            ),
            "num_epochs": num_epochs,
            "max_time_seconds": 120.0,
            "warmup_time_seconds": 10.0,
            "data_path": memmap_path,
            "madvise_interval": None,
            "num_runs": num_runs,
            "log_iteration_times_to_file": None,
        }
    """
    # To use: set 'log_iteration_times_to_file' to None (no logging) or an integer interval (e.g., 1 for every batch).
    # The log file will contain columns: epoch, batch, iteration_time_s, rss_mb, run_name

    results = benchmark_dataloader(dataloaders=configurations, output_dir="comprehensive_anndata_results")

    print("✅ Benchmarking completed!")
    print()

    # 9. SUMMARY AND ANALYSIS
    if results:
        # Find best performers
        best_samples_per_sec = max(results, key=lambda r: r.samples_per_second)
        lowest_memory = min(results, key=lambda r: r.peak_memory_mb)

        print("🏆 BEST PERFORMERS:")
        print(
            f"   Best speed: {best_samples_per_sec.name} ({best_samples_per_sec.samples_per_second:.2f} samples/sec)"
        )
        print(f"   Lowest memory: {lowest_memory.name} ({lowest_memory.peak_memory_mb:.2f} MB)")
        print()

        # Use shared utility for comprehensive export
        export_benchmark_results(results=results, output_prefix="comprehensive_benchmark_data")

    print("🎉 COMPREHENSIVE BENCHMARKING COMPLETED!")


if __name__ == "__main__":
    print("BioNeMo Benchmarking Framework - Comprehensive Example")
    print("=" * 80)

    # Run with default settings (3 epochs, 1 run each)
    # comprehensive_benchmarking_example()

    # Example: Run with statistical analysis (3 runs each for better confidence)
    print("\n" + "=" * 80)
    print("📊 MULTIPLE RUNS EXAMPLE")
    print("=" * 80)
    comprehensive_benchmarking_example(num_epochs=1, num_runs=3)
