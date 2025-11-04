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


import os
from datetime import datetime

from torch.utils.data import DataLoader

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch
from bionemo.scspeedtest import benchmark_dataloaders_with_configs, print_comparison


def create_scdl_conversion_from_anndata_factory(anndata_path, scdl_path):
    """Create a dataset factory that converts AnnData to SCDL.

    Args:
        anndata_path: Path to the AnnData file (.h5ad)
        scdl_path: Path to the SCDL directory

    Returns:
        Factory function that converts AnnData to SCDL
    """

    def factory():
        # Note: we could return the dataset instead of the path
        print(f"Converting AnnData to SCDL from: {anndata_path} to {scdl_path}")
        # This creates the SCDL dataset at the given path
        _ = SingleCellMemMapDataset(scdl_path, h5ad_path=anndata_path)
        return scdl_path

    return factory


def create_scdl_dataset_and_loader_factory(batch_size=64, shuffle=True, data_path=None, num_workers=0):
    """Create a SCDL dataloader factory that loads dataset each time.

    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        data_path: Path to the memmap data files
        num_workers: Number of worker processes

    Returns:
        Factory function that creates SCDL dataset and DataLoader (reload approach)
    """

    def factory(scdl_path):
        dataset = SingleCellMemMapDataset(scdl_path)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            collate_fn=collate_sparse_matrix_batch,
            num_workers=num_workers,
        )

    return factory


# =============================================================================
# BENCHMARKING EXAMPLES
# =============================================================================


def dataset_conversion_benchmarking_example(num_epochs=1, num_runs=1, adata_path=None, scdl_path=None):
    """Demonstrate dataset conversion functionality.

    This example shows how to:
    1. Convert an AnnData file to SCDL
    2. Test multiple dataloader configurations on the SCDL dataset

    Args:
        num_epochs: Number of epochs to run per configuration
        num_runs: Number of runs per configuration for statistical analysis
        adata_path: Path to the AnnData file (.h5ad)
        scdl_path: Path to the SCDL directory
    """
    # Create timestamped prefix for all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"All results will be saved to: comprehensive_benchmark_{timestamp}_detailed_breakdown.csv")
    print()

    print("Mixed Benchmarking Example: Dataset Reuse vs Reload")
    print("=" * 60)
    print(f"Testing {num_runs} run(s) each across different configs")
    print("AnnData: Dataset loaded ONCE (reuse)")
    print("SCDL: Dataset loaded PER CONFIG (reload)")
    print()

    # =============================================================================
    # EXAMPLE 2: SCDL Dataset with Multiple DataLoader Configurations
    # =============================================================================

    print("EXAMPLE 2: SCDL with Multiple DataLoader Configs (Reload Each Time)")
    print("-" * 60)

    scdl_configurations = [
        {
            "name": "SCDL_Batch32_Shuffle",
            "dataloader_factory": create_scdl_dataset_and_loader_factory(
                batch_size=64, shuffle=True, data_path=scdl_path, num_workers=0
            ),
            "num_epochs": num_epochs,
            "max_time_seconds": 1.0,
            "warmup_time_seconds": 0.0,
            "data_path": scdl_path,
            "num_runs": num_runs,
        },
    ]
    # The conversiton from anndta happens once, and then each time the dataset is loaded, the dataloader is created.
    # Alternatively, create_scdl_conversion_from_anndata_factory could be used to create the dataset only once and return the dataset.
    # Then create_scdl_dataset_and_loader_factory could be used to just create the dataloader each time.
    scdl_results = benchmark_dataloaders_with_configs(
        dataloader_configs=scdl_configurations,
        shared_dataset_factory=create_scdl_conversion_from_anndata_factory(adata_path, scdl_path),
        output_prefix=f"scdl_conversion_benchmark_{timestamp}",
    )

    print()
    print("=" * 60)

    # =============================================================================
    # ANALYSIS AND SUMMARY
    # =============================================================================

    print("ANALYSIS & COMPARISON")
    print("-" * 60)

    print_comparison(scdl_results)
    print()
    print("COMPREHENSIVE BENCHMARKING COMPLETED!")
    print(f"All results saved to: comprehensive_benchmark_{timestamp}_detailed_breakdown.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--adata-path", type=str, required=True, help="Path to the AnnData file (.h5ad)")
    parser.add_argument("--new-scdl-path", type=str, required=True, help="Path to the new SCDL directory")
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of epochs to run for each configuration (default: 1)"
    )
    parser.add_argument(
        "--num-runs", type=int, default=1, help="Number of runs to perform for each configuration (default: 1)"
    )

    args = parser.parse_args()

    # Validate paths exist
    if not os.path.exists(args.adata_path):
        print(f"Error: AnnData file not found: {args.adata_path}")
        exit(1)
    if os.path.exists(args.new_scdl_path):
        print(f"Error: SCDL directory already exists: {args.new_scdl_path}")
        exit(1)

    # Run the actual dataset reuse benchmarking
    dataset_conversion_benchmarking_example(
        num_epochs=args.num_epochs, num_runs=args.num_runs, adata_path=args.adata_path, scdl_path=args.new_scdl_path
    )
