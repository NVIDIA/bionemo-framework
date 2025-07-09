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
import random
import string

import anndata as ad
from anndata.experimental import AnnCollection, AnnLoader
from torch.utils.data import DataLoader

# Import the benchmarking framework
from bionemo.scbenchmark import benchmark_multiple_dataloaders
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
        datasets = AnnCollection([ad.read_h5ad(data_path, backed=backed)])
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
        dataset = SingleCellMemMapDataset(data_path, adata_path)
        dataset.save()
        del dataset
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


def comprehensive_benchmarking_example():
    """Run comprehensive benchmarking with all features.

    This function demonstrates every capability of the benchmarking framework
    using AnnData dataloaders with different configurations.
    """
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARKING EXAMPLE")
    print("=" * 80)
    print("Demonstrating ALL framework features with AnnData dataloaders...")
    print()

    # Try to use real AnnData if available
    adata_path = "/home/pbinder/bionemo-framework/sub-packages/bionemo-scdl/small_samples/sample_50000_19836_0.85.h5ad"
    memmap_path = "/home/pbinder/bionemo-framework/sub-packages/bionemo-scdl/small_samples/s_memmap"

    # Create different configurations of the same dataloader
    random_prefix1 = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    random_prefix2 = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    configurations = [
        {
            "name": "SCDL Simple (64)",
            "dataloader_factory": create_scdl_factory(
                batch_size=64,
                shuffle=True,
                adata_path=adata_path,
                data_path=f"{memmap_path}_{random_prefix1}",
                num_workers=0,
            ),
            "max_time_seconds": 100.0,
            "warmup_seconds": 0.5,
            "data_path": f"{memmap_path}_{random_prefix1}",
        },
        {
            "name": "SCDL Simple (64) 8 workers",
            "dataloader_factory": create_scdl_factory(
                batch_size=64,
                shuffle=True,
                adata_path=adata_path,
                data_path=f"{memmap_path}_{random_prefix2}",
                num_workers=8,
            ),
            "max_time_seconds": 100.0,
            "warmup_seconds": 0.5,
            "data_path": f"{memmap_path}_{random_prefix2}",
        },
        {
            "name": "Anndata backed (64)",
            "dataloader_factory": create_annloader_factory(
                batch_size=64, shuffle=True, backed="r", data_path=adata_path, num_workers=0
            ),
            "max_time_seconds": 4.0,
            "warmup_seconds": 0.1,
            "data_path": adata_path,
        },
        {
            "name": "Anndata (64)",
            "dataloader_factory": create_annloader_factory(
                batch_size=64, shuffle=True, backed=False, data_path=adata_path, num_workers=0
            ),
            "max_time_seconds": 4.0,
            "warmup_seconds": 0.1,
            "data_path": adata_path,
        },
    ]

    results = benchmark_multiple_dataloaders(dataloaders=configurations, output_dir="comprehensive_anndata_results")

    print("✅ Multiple configuration comparison completed!")
    print("   Results saved to: comprehensive_anndata_results/")
    print()

    # 9. SUMMARY AND ANALYSIS
    print("8️⃣ SUMMARY AND ANALYSIS")
    print("-" * 50)

    # Use results from the multiple configuration comparison
    if results:
        # Find best performers
        best_samples_per_sec = max(results, key=lambda r: r.samples_per_second)
        fastest_instantiation = min(results, key=lambda r: r.instantiation_time_seconds)
        lowest_memory = min(results, key=lambda r: r.peak_memory_mb)

        print("🏆 BEST PERFORMERS:")
        print(f"   Best samples/second: {best_samples_per_sec.name} ({best_samples_per_sec.samples_per_second:.2f})")
        print(
            f"   Fastest instantiation: {fastest_instantiation.name} ({fastest_instantiation.instantiation_time_seconds:.4f}s)"
        )
        print(f"   Lowest memory: {lowest_memory.name} ({lowest_memory.peak_memory_mb:.2f} MB)")
        print()

        print("📊 PERFORMANCE COMPARISON:")
        print(f"{'Name':<25} {'Samples/sec':<12} {'Inst Time':<10} {'Peak Mem':<10}")
        print("-" * 60)
        for result in results:
            print(
                f"{result.name:<25} {result.samples_per_second:<12.2f} {result.instantiation_time_seconds:<10.4f} {result.peak_memory_mb:<10.2f}"
            )

    print()
    print("🎉 COMPREHENSIVE BENCHMARKING COMPLETED!")


if __name__ == "__main__":
    """Main execution function for comprehensive benchmarking.

    This function runs the complete comprehensive benchmarking example
    that demonstrates all features of the framework.
    """
    print("BioNeMo Benchmarking Framework - Comprehensive Example")
    print("=" * 80)

    comprehensive_benchmarking_example()
