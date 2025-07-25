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

"""
Comprehensive Benchmarking Example: Dataset Reuse vs Traditional

This example demonstrates BOTH approaches:
1. Dataset Reuse (AnnData): Dataset loaded ONCE, multiple configs tested on SAME dataset
2. Traditional (SCDL): Each config loads its own dataset independently

This mixed approach shows:
- When to use dataset reuse (expensive loading, config comparison)
- When to use traditional (full isolation, different datasets)
- Flexibility to choose per use case

Key Benefits of Dataset Reuse:
- ⚡ Faster benchmarking (dataset loaded once, not N times)
- 💾 Memory efficient 
- 🔄 Fair comparison (all configs use identical data)
- 📊 Separate tracking of dataset vs dataloader instantiation times

Key Benefits of Traditional:
- 🔒 Full isolation between configs
- 🆕 Fresh dataset state per config
- 📈 Individual dataset loading performance measurement
"""

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


# =============================================================================
# DATASET FACTORY FUNCTIONS (Load data once)
# =============================================================================

def create_anndata_dataset_factory(data_path, backed = "r"):
    """Create a dataset factory that loads AnnData once.
    
    Args:
        data_path: Path to h5ad file or directory containing h5ad files
        
    Returns:
        Factory function that loads the dataset once
    """
    def factory():
        print(f"📂 Loading AnnData dataset from: {data_path}")
        if data_path.endswith(".h5ad"):
            dataset = anndata.read_h5ad(data_path, backed=backed)
        elif os.path.isdir(data_path) and any(f.endswith(".h5ad") for f in os.listdir(data_path)):
            h5ad_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".h5ad")]
            dataset = AnnCollection([anndata.read_h5ad(f, backed=backed) for f in h5ad_files])
        else:
            raise ValueError("AnnData requires a .h5ad file or directory with .h5ad files")
        print(f"✅ Dataset loaded: {dataset.shape[0]:,} cells x {dataset.shape[1]:,} genes")
        return dataset
    return factory




# =============================================================================
# DATALOADER FACTORY FUNCTIONS (Receive pre-loaded dataset)
# =============================================================================

def anndata_collate(batch):
    """Custom collate function for AnnData objects."""
    # Convert AnnCollectionView to tensors
    return torch.stack([item.X for item in batch])

def create_annloader_config(batch_size=64, shuffle=True, num_workers=0, collate_fn = anndata_collate):
    """Create a dataloader factory that wraps a pre-loaded AnnData dataset.
    
    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        Factory function that creates AnnLoader from pre-loaded dataset
    """
    def factory(dataset):
        return AnnLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return factory


def create_scdl_traditional_factory(batch_size=64, shuffle=True, data_path=None, num_workers=0):
    """Create a traditional SCDL dataloader factory that loads dataset each time.
    
    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        data_path: Path to the memmap data files
        num_workers: Number of worker processes
        
    Returns:
        Factory function that creates SCDL dataset and DataLoader (traditional approach)
    """
    def factory():
        print(f"📂 Loading SCDL dataset from: {data_path}")
        dataset = SingleCellMemMapDataset(data_path)
        print(f"✅ Dataset loaded: {len(dataset):,} samples")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=collate_sparse_matrix_batch,
            num_workers=num_workers,
        )
    return factory



# =============================================================================
# BENCHMARKING EXAMPLES
# =============================================================================

def dataset_reuse_benchmarking_example(num_epochs=1, num_runs=1):
    """Demonstrate dataset reuse functionality.
    
    This example shows how to:
    1. Load a dataset ONCE
    2. Test multiple dataloader configurations on the SAME dataset
    3. Get separate instantiation times for dataset vs dataloader creation
    
    Args:
        num_epochs: Number of epochs to run per configuration
        num_runs: Number of runs per configuration for statistical analysis
    """
    # Configure paths (adjust these for your environment)
    adata_path = "/home/pbinder/bionemo-framework/tahoe_data/plate11_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
    memmap_path = "/home/pbinder/bionemo-framework/tahoe_memmap/"
    
    print("🚀 Mixed Benchmarking Example: Dataset Reuse vs Traditional")
    print("=" * 60)
    print(f"📊 Testing {num_runs} run(s) each across different configs")
    print(f"⚡ AnnData: Dataset loaded ONCE (reuse)")
    print(f"🔄 SCDL: Dataset loaded PER CONFIG (traditional)")
    print()

    # =============================================================================
    # EXAMPLE 1: AnnData Dataset with Multiple DataLoader Configurations
    # =============================================================================
    
    print("🧬 EXAMPLE 1: AnnData with Multiple DataLoader Configs")
    print("-" * 60)
    
    anndata_configurations = [
        {
            "name": "AnnLoader_Multi_Worker",
            "dataloader_factory": create_annloader_config(batch_size=64, shuffle=True, num_workers=2),
            "num_epochs": num_epochs,
            "max_time_seconds": 5.0,
            "warmup_time_seconds": 1.0,
            "data_path": adata_path,
            "num_runs": num_runs,
        },
                {
            "name": "AnnLoader_Single_Worker",
            "dataloader_factory": create_annloader_config(batch_size=64, shuffle=True, num_workers=0),
            "num_epochs": num_epochs,
            "max_time_seconds": 5.0,
            "warmup_time_seconds": 1.0,
            "data_path": adata_path,
            "num_runs": num_runs,
        }

    ]

    """
        {
            "name": "AnnLoader_Small_Batch",
            "dataloader_factory": create_annloader_config(batch_size=32, shuffle=True, num_workers=0),
            "num_epochs": num_epochs,
            "max_time_seconds": 5.0,
            "warmup_time_seconds": 1.0,
            "data_path": adata_path,
            "num_runs": num_runs,
        },
        {
            "name": "AnnLoader_Large_Batch", 
            "dataloader_factory": create_annloader_config(batch_size=128, shuffle=True, num_workers=0),
            "num_epochs": num_epochs,
            "max_time_seconds": 5.0,
            "warmup_time_seconds": 1.0,
            "data_path": adata_path,
            "num_runs": num_runs,
        }
    ]
    
        {
            "name": "AnnLoader_No_Shuffle",
            "dataloader_factory": create_annloader_config(batch_size=64, shuffle=False, num_workers=0),
            "num_epochs": num_epochs,
            "max_time_seconds": 5.0,
            "warmup_time_seconds": 1.0,
            "data_path": adata_path,
            "num_runs": num_runs,
        },
        """

    # NEW: Use dataset_factory to load dataset once, then test multiple configs
    anndata_results = benchmark_dataloader(
        dataset_factory=create_anndata_dataset_factory(adata_path),  # 🆕 Load once!
        dataloaders=anndata_configurations,  # All configs use same dataset
        output_dir="dataset_reuse_anndata_results"
    )

    print()
    print("=" * 60)

    # =============================================================================
    # EXAMPLE 2: SCDL Dataset with Multiple DataLoader Configurations  
    # =============================================================================
    
    print("🔬 EXAMPLE 2: SCDL with Multiple DataLoader Configs (Traditional - Reload Each Time)")
    print("-" * 60)
    
    scdl_configurations = [
        {
            "name": "SCDL_Batch32_Shuffle",
            "dataloader_factory": create_scdl_traditional_factory(batch_size=32, shuffle=True, data_path=memmap_path, num_workers=0),
            "num_epochs": num_epochs,
            "max_time_seconds": 3.0,
            "warmup_time_seconds": 0.5,
            "data_path": memmap_path,
            "num_runs": num_runs,
        },
        {
            "name": "SCDL_Batch128_Shuffle",
            "dataloader_factory": create_scdl_traditional_factory(batch_size=128, shuffle=True, data_path=memmap_path, num_workers=0),
            "num_epochs": num_epochs,
            "max_time_seconds": 3.0,
            "warmup_time_seconds": 0.5,
            "data_path": memmap_path,
            "num_runs": num_runs,
        },
        {
            "name": "SCDL_Batch64_NoShuffle",
            "dataloader_factory": create_scdl_traditional_factory(batch_size=64, shuffle=False, data_path=memmap_path, num_workers=0),
            "num_epochs": num_epochs,
            "max_time_seconds": 3.0,
            "warmup_time_seconds": 0.5,
            "data_path": memmap_path,
            "num_runs": num_runs,
        },
    ]

    # Traditional approach: Each config loads its own dataset
    scdl_results = benchmark_dataloader(
        dataloaders=scdl_configurations,  # 🔄 Each config loads dataset separately!
        output_dir="dataset_reuse_scdl_results"
    )

    print()
    print("=" * 60)

    # =============================================================================
    # ANALYSIS AND SUMMARY
    # =============================================================================
    
    print("📊 ANALYSIS & COMPARISON")
    print("-" * 60)
    
    all_results = []
    if anndata_results:
        all_results.extend(anndata_results)
    if scdl_results:
        all_results.extend(scdl_results)
    
    if all_results:
        # Find best performers across all configurations
        best_samples_per_sec = max(all_results, key=lambda r: r.samples_per_second)
        lowest_memory = min(all_results, key=lambda r: r.peak_memory_mb)
        fastest_instantiation = min(all_results, key=lambda r: r.instantiation_time_seconds or float('inf'))

        print("🏆 OVERALL BEST PERFORMERS:")
        print(f"   🚀 Best speed: {best_samples_per_sec.name} ({best_samples_per_sec.samples_per_second:.2f} samples/sec)")
        print(f"   💾 Lowest memory: {lowest_memory.name} ({lowest_memory.peak_memory_mb:.2f} MB)")
        if fastest_instantiation.instantiation_time_seconds:
            print(f"   ⚡ Fastest setup: {fastest_instantiation.name} ({fastest_instantiation.instantiation_time_seconds:.3f}s)")
        print()

        # Export comprehensive results
        export_benchmark_results(results=all_results, output_prefix="dataset_comprehensive")

    print()
    print("🎉 DATASET REUSE BENCHMARKING COMPLETED!")


if __name__ == "__main__":
    print("BioNeMo Benchmarking Framework - Dataset Reuse Example")
    print("=" * 80)
        
    # Run the actual dataset reuse benchmarking
    dataset_reuse_benchmarking_example(num_epochs=1, num_runs=1) 