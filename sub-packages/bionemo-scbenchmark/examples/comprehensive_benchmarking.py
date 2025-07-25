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

import os
from typing import Sequence, Union

import anndata
import numpy as np
import torch
from anndata.experimental import AnnCollection, AnnLoader
from torch.utils.data import DataLoader

# Import the benchmarking framework
from bionemo.scbenchmark import benchmark_dataloader
from bionemo.scbenchmark.common import export_benchmark_results
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch


# Optional import for scDataset
try:
    from scdataset import scDataset
except ImportError:
    scDataset = None


def create_annloader_factory(batch_size=64, backed="r", shuffle=True, data_path=None, num_workers=0):
    """Create a factory function for AnnData dataloaders with different configurations.

    Args:
        batch_size: Number of samples per batch
        backed: AnnData backed mode ('r' for read-only, 'r+' for read-write)
        shuffle: Whether to shuffle the data
        data_path: Path to a single h5ad file or multi h5ad files
        num_workers: Number of worker processes for data loading

    Returns:
        Factory function that creates an AnnData dataloader
    """

    def factory():
        if data_path.endswith(".h5ad"):
            dataset = anndata.read_h5ad(data_path, backed="r")
        elif os.path.isdir(data_path) and any(f.endswith(".h5ad") for f in os.listdir(data_path)):
            h5ad_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".h5ad")]
            dataset = AnnCollection([anndata.read_h5ad(f, backed="r") for f in h5ad_files])
        else:
            raise ValueError("AnnData baseline requires a .h5ad input file or a directory containing .h5ad files")
        return AnnLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return factory


def create_scdl_factory(batch_size=64, shuffle=True, adata_path=None, data_path=None, num_workers=0):
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


def fetch_callback_bionemo(self, idx: Union[int, slice, Sequence[int], np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Fetch callback for bionemo dataset when used with scDataset."""
    if isinstance(idx, int):
        # Single index
        return collate_sparse_matrix_batch([self.__getitem__(idx)]).to_dense()
    elif isinstance(idx, slice):
        # Slice: convert to a list of indices
        indices = list(range(*idx.indices(len(self))))
        batch_tensors = [self.__getitem__(i) for i in indices]
        return collate_sparse_matrix_batch(batch_tensors).to_dense()
    elif isinstance(idx, (list, np.ndarray, torch.Tensor)):
        # Batch indexing
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        batch_tensors = [self.__getitem__(int(i)) for i in idx]
        return collate_sparse_matrix_batch(batch_tensors).to_dense()
    else:
        raise TypeError(f"Unsupported index type: {type(idx)}")


def create_scdl_scdataset_factory(
    batch_size=64, block_size=1, shuffle=True, adata_path=None, data_path=None, num_workers=0, fetch_factor=1
):
    """Create a factory function for SCDL with scDataset wrapper.

    Args:
        batch_size: Number of samples per batch
        block_size: Block size for scDataset
        shuffle: Whether to shuffle the data
        adata_path: Path to the AnnData file (unused but kept for compatibility)
        data_path: Path to the data files
        num_workers: Number of worker processes for data loading
        fetch_factor: Fetch factor for scDataset

    Returns:
        Factory function that creates an SCDL dataloader with scDataset wrapper
    """

    def factory():
        if scDataset is None:
            raise ImportError("scDataset is not available. Please install scdataset package.")

        dataset = SingleCellMemMapDataset(data_path)

        wrapped_dataset = scDataset(
            data_collection=dataset,  # Use the created dataset as data_collection
            batch_size=batch_size,
            block_size=block_size,
            fetch_factor=fetch_factor,
            fetch_transform=None,
            **{"fetch_callback": fetch_callback_bionemo},
        )
        prefetch_factor = fetch_factor + 1 if num_workers > 0 else None
        return DataLoader(
            wrapped_dataset,
            batch_size=None,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
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
    #adata_path = "/home/pbinder/bionemo-framework/sub-packages/bionemo-scdl/small_samples/sample_50000_19836_0.85.h5ad"
    # memmap_path = "/home/pbinder/bionemo-framework/sub-packages/bionemo-scdl/small_samples/s_memmap_zmdy090y"
    adata_path = (
        "/home/pbinder/bionemo-framework/tahoe_data/plate11_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
    )
    memmap_path = "/home/pbinder/bionemo-framework/tahoe_memmap/"
    # Create different configurations of the same dataloader

    # 7. MULTIPLE CONFIGURATIONS WITH STATISTICAL ANALYSIS
    print(f"🚀 Benchmarking {num_runs} run(s) each")
    print()
    configurations = [
        # Example: Enable per-iteration time and memory (RSS) logging every 5 batches
        {
            "name": "SCDL Regular",
            "dataloader_factory": create_scdl_scdataset_factory(
                batch_size=64,
                shuffle=True,
                adata_path=adata_path,
                data_path=memmap_path,
                num_workers=0,
            ),
            "num_epochs": num_epochs,
            "max_time_seconds": 1.0,
            "warmup_time_seconds": 1.0,
            "data_path": memmap_path,
            "madvise_interval": None,
            "num_runs": num_runs,
        },
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
        }
    """


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
    comprehensive_benchmarking_example(num_epochs=2, num_runs=2)
