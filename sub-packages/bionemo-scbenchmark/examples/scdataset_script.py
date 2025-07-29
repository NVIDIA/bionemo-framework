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

import argparse
from datetime import datetime
from typing import Sequence, Union

import numpy as np
import torch
from comprehensive_benchmarking import (
    create_anndata_dataset_factory,
    create_annloader_factory,
    create_scdl_dataset_and_loader_factory,
)

# Optional import for scDataset
from scdataset import scDataset
from torch.utils.data import DataLoader

# Import the benchmarking framework
from bionemo.scbenchmark import benchmark_dataloader
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch


def fetch_transform_adata(batch):
    """Transform batch to AnnData format."""
    return batch.to_adata()


def create_scdataset_annloader_factory(batch_size=64, shuffle=True, block_size=1, fetch_factor=2, num_workers=0):
    """Create a dataloader factory that wraps a pre-loaded AnnData dataset.

    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        block_size: Block size for scDataset
        fetch_factor: Fetch factor for scDataset
        num_workers: Number of worker processes

    Returns:
        Factory function that creates AnnLoader from pre-loaded dataset
    """

    def factory(dataset):
        dataset = scDataset(
            data_collection=dataset,
            batch_size=batch_size,
            block_size=block_size,
            fetch_factor=fetch_factor,
            fetch_transform=fetch_transform_adata,
        )
        prefetch_factor = fetch_factor + 1 if num_workers > 0 else None
        dataset.set_mode("train")
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        return loader

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
        dataset = SingleCellMemMapDataset(data_path)
        wrapped_dataset = scDataset(
            data_collection=dataset,  # Use the created dataset as data_collection
            batch_size=batch_size,
            block_size=block_size,
            fetch_factor=fetch_factor,
            fetch_transform=None,
            **{"fetch_callback": fetch_callback_bionemo},
        )
        wrapped_dataset.set_mode("train")

        prefetch_factor = fetch_factor + 1 if num_workers > 0 else None
        return DataLoader(
            wrapped_dataset,
            batch_size=None,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    return factory


def comprehensive_benchmarking_example(
    num_epochs=1, num_runs=1, adata_path=None, memmap_path=None, fetch_factors=None, block_sizes=None
):
    """Comprehensive benchmarking example demonstrating various dataloader configurations.

    Args:
        num_epochs: Number of epochs to run for each configuration
        num_runs: Number of runs to perform for each configuration
        adata_path: Path to the AnnData file (.h5ad)
        memmap_path: Path to the SCDL memmap directory
        fetch_factors: List of fetch factors to test (default: [1])
        block_sizes: List of block sizes to test (default: [1, 2, 4, 8, 16, 32, 64])
    """
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARKING EXAMPLE")
    print("=" * 80)
    print()

    # Use provided paths or defaults
    if adata_path is None:
        adata_path = "/home/pbinder/bionemo-framework/tahoe_data"
    if memmap_path is None:
        memmap_path = "/home/pbinder/bionemo-framework/all_tahoe_memmap/"
    if fetch_factors is None:
        fetch_factors = [1]
    if block_sizes is None:
        block_sizes = [1, 2, 4, 8, 16, 32, 64]

    print(f"📊 Using AnnData path: {adata_path}")
    print(f"📊 Using SCDL memmap path: {memmap_path}")
    print(f"📊 Fetch factors: {fetch_factors}")
    print(f"📊 Block sizes: {block_sizes}")
    print()

    # Create single timestamped output directory for all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"scdataset_benchmark_results_{timestamp}"
    csv_prefix = f"{output_dir}_consolidated_results"
    print(f"📁 All results will be saved to: {output_dir}/")
    print(f"📊 Consolidated detailed CSV: {csv_prefix}_detailed_breakdown.csv")
    print()

    # Parameters
    warmup_time_seconds = 0
    max_time_seconds = 2

    print(f"🚀 Benchmarking {num_runs} run(s) each")
    print()
    # =============================================================================
    # Part1 2: SCDL Dataset with Multiple DataLoader Configurations
    # =============================================================================

    # First run SCDL Regular as baseline
    print("Running SCDL Regular baseline...")
    scdl_configurations = [
        {
            "name": "Baseline SCDL",
            "dataloader_factory": create_scdl_dataset_and_loader_factory(
                batch_size=64, shuffle=True, data_path=memmap_path, num_workers=0
            ),
            "num_epochs": num_epochs,
            "max_time_seconds": max_time_seconds,
            "warmup_time_seconds": warmup_time_seconds,
            "data_path": memmap_path,
            "num_runs": num_runs,
        },
    ]
    for fetch_factor in fetch_factors:
        for block_size in block_sizes:
            scdl_configurations.append(
                {
                    "name": f"ScDataset_{block_size}_{fetch_factor}",
                    "dataloader_factory": create_scdl_scdataset_factory(
                        batch_size=64,
                        shuffle=True,
                        data_path=memmap_path,
                        num_workers=0,
                        block_size=block_size,
                        fetch_factor=fetch_factor,
                    ),
                    "num_epochs": num_epochs,
                    "max_time_seconds": max_time_seconds,
                    "warmup_time_seconds": warmup_time_seconds,
                    "data_path": memmap_path,
                    "madvise_interval": None,
                    "num_runs": 1,
                }
            )
    benchmark_dataloader(
        dataloaders=scdl_configurations,  # 🔄 Each config loads dataset separately!
        output_dir=output_dir,
    )
    # =============================================================================
    # Part2: AnnData Dataset with ScDataset Configurations
    # =============================================================================
    anndata_configurations = [
        {
            "name": "AnnLoader_Baseline",
            "dataloader_factory": create_annloader_factory(batch_size=64, shuffle=True, num_workers=0),
            "num_epochs": num_epochs,
            "max_time_seconds": max_time_seconds,
            "warmup_time_seconds": warmup_time_seconds,
            "data_path": adata_path,
            "num_runs": num_runs,
        }
    ]
    for fetch_factor in fetch_factors:
        for block_size in block_sizes:
            anndata_configurations.append(
                {
                    "name": f"ScDataset_AnnData_{block_size}_{fetch_factor}",
                    "dataloader_factory": create_scdataset_annloader_factory(
                        batch_size=64, shuffle=True, block_size=block_size, fetch_factor=fetch_factor, num_workers=0
                    ),
                    "num_epochs": num_epochs,
                    "max_time_seconds": max_time_seconds,
                    "warmup_time_seconds": warmup_time_seconds,
                    "data_path": adata_path,
                    "num_runs": num_runs,
                }
            )

    # Call benchmark_dataloader AFTER building the complete configuration list
    benchmark_dataloader(
        dataset_factory=create_anndata_dataset_factory(adata_path),  # 🆕 Load once!
        dataloaders=anndata_configurations,  # All configs use same dataset
        output_dir=output_dir,
    )

    print("✅ Benchmarking completed!")
    print(f"📁 All results saved to: {output_dir}/")
    print("📊 Consolidated detailed CSV:")
    print(f"   - {output_dir}/{csv_prefix}_detailed_breakdown.csv")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioNeMo Benchmarking Framework - ScDataset Test")
    parser.add_argument(
        "--adata-path",
        type=str,
        default="/home/pbinder/bionemo-framework/tahoe_data",
        help="Path to the AnnData file (.h5ad). Default: %(default)s",
    )
    parser.add_argument(
        "--memmap-path",
        type=str,
        default="/home/pbinder/bionemo-framework/all_tahoe_memmap/",
        help="Path to the SCDL memmap directory. Default: %(default)s",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="Number of epochs to run for each configuration. Default: %(default)s",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs to perform for each configuration. Default: %(default)s",
    )
    parser.add_argument(
        "--fetch-factors", nargs="+", type=int, default=[1], help="List of fetch factors to test. Default: %(default)s"
    )
    parser.add_argument(
        "--block-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64],
        help="List of block sizes to test. Default: %(default)s",
    )

    args = parser.parse_args()

    print("BioNeMo Benchmarking Framework - ScDataset Test")
    print("=" * 80)
    comprehensive_benchmarking_example(
        num_epochs=args.num_epochs,
        num_runs=args.num_runs,
        adata_path=args.adata_path,
        memmap_path=args.memmap_path,
        fetch_factors=args.fetch_factors,
        block_sizes=args.block_sizes,
    )
