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
"""Simple Benchmarking Examples - No Factory Functions Needed!

This example shows how to benchmark ANY dataloader directly without
writing factory functions or complex setup code.

It also demonstrates explicit use of the DataloaderProtocol for type safety.
"""

import torch
from torch.utils.data import DataLoader, Dataset

# Import the simple benchmarking interface and protocol
from bionemo.scbenchmark import DataloaderProtocol, benchmark_any_dataloader, benchmark_multiple_dataloaders_simple


# ============================================================================
# EXPLICIT PROTOCOL USAGE EXAMPLE
# ============================================================================


def run_benchmark_with_protocol(dl: DataloaderProtocol):
    """Demonstrate explicit protocol type annotation.

    This function shows how to use the DataloaderProtocol for type safety
    when working with dataloaders. The protocol ensures that any object
    passed to this function supports iteration.

    Args:
        dl: Any object that implements the DataloaderProtocol
    """
    result = benchmark_any_dataloader(dl, name="Protocol-Typed Dataloader", num_epochs=1, max_batches=5)
    print(f"[Protocol] Samples/sec: {result.samples_per_second:.2f}")


# You can use any iterable, e.g. a PyTorch DataLoader:
class ProtocolDemoDataset(Dataset):
    """Simple dataset for demonstrating protocol usage."""

    def __len__(self):
        """Return the number of samples in the dataset."""
        return 10

    def __getitem__(self, idx):
        """Get a sample at the given index."""
        return torch.randn(5)


# ============================================================================
# EXAMPLE 1: Simple PyTorch DataLoader
# ============================================================================


class SimpleDataset(Dataset):
    """A simple PyTorch dataset for demonstration purposes.

    This dataset creates random data and labels for benchmarking.
    It demonstrates how to create a basic dataset that can be
    wrapped in a DataLoader for benchmarking.
    """

    def __init__(self, size=1000):
        """Initialize the dataset with random data.

        Args:
            size: Number of samples in the dataset
        """
        self.data = torch.randn(size, 100)
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a sample at the given index.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (data, label) for the sample
        """
        return self.data[idx], self.labels[idx]


def example_simple_pytorch():
    """Example: Benchmark a simple PyTorch DataLoader.

    This example demonstrates how to benchmark a standard PyTorch DataLoader
    without any special setup. The dataloader is created normally and then
    passed directly to the benchmark function.
    """
    print("=" * 60)
    print("EXAMPLE: Simple PyTorch DataLoader")
    print("=" * 60)

    # Create your dataloader normally
    dataset = SimpleDataset(size=500)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # Benchmark it directly - instantiation is measured automatically!
    result = benchmark_any_dataloader(
        dataloader=dataloader,  # Just pass the dataloader!
        name="Simple PyTorch DataLoader",
        num_epochs=1,
        max_batches=20,
        warmup_batches=3,
    )

    print(f"✅ Benchmark completed: {result.samples_per_second:.2f} samples/sec")
    print(f"✅ Instantiation time: {result.instantiation_metrics.instantiation_time_seconds:.4f}s")
    print(f"✅ Instantiation memory: {result.instantiation_metrics.memory_delta_mb:.2f} MB")


# ============================================================================
# EXAMPLE 2: SCDL with Different Configurations
# ============================================================================


def example_scdl_configurations():
    """Example: Benchmark SCDL with different configurations.

    This example shows how to benchmark SCDL (Single Cell DataLoader)
    with various configurations including different batch sizes and
    shuffle settings. It demonstrates benchmarking multiple dataloaders
    and comparing their performance.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE: SCDL with Different Configurations")
    print("=" * 60)

    try:
        from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
        from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch

        # Create SCDL dataset
        adata_path = (
            "/home/pbinder/bionemo-framework/sub-packages/bionemo-scdl/small_samples/sample_500_19836_0.9.h5ad"
        )
        data = SingleCellMemMapDataset("example_scmm", adata_path)

        # Configuration 1: Basic SCDL
        dataloader1 = DataLoader(data, batch_size=8, shuffle=True, collate_fn=collate_sparse_matrix_batch)

        # Configuration 2: SCDL with features
        dataloader2 = DataLoader(
            data,
            batch_size=16,
            shuffle=False,  # Different shuffle setting
            collate_fn=collate_sparse_matrix_batch,
        )

        # Configuration 3: SCDL with different batch size
        dataloader3 = DataLoader(data, batch_size=32, shuffle=True, collate_fn=collate_sparse_matrix_batch)

        # Benchmark all configurations
        dataloaders = [
            {
                "name": "SCDL Basic (batch=8)",
                "dataloader": dataloader1,
                "data_path": adata_path,
                "num_epochs": 1,
                "max_batches": 15,
                "warmup_batches": 2,
            },
            {
                "name": "SCDL No Shuffle (batch=16)",
                "dataloader": dataloader2,
                "data_path": adata_path,
                "num_epochs": 1,
                "max_batches": 15,
                "warmup_batches": 2,
            },
            {
                "name": "SCDL Large Batch (batch=32)",
                "dataloader": dataloader3,
                "data_path": adata_path,
                "num_epochs": 1,
                "max_batches": 15,
                "warmup_batches": 2,
            },
        ]

        benchmark_multiple_dataloaders_simple(dataloaders=dataloaders, output_dir="scdl_benchmark_results")

        print("✅ SCDL benchmarks completed!")

    except ImportError:
        print("SCDL not available, skipping this example")


# ============================================================================
# EXAMPLE 3: AnnData Loading Benchmark
# ============================================================================


def example_anndata_loading():
    """Example: Benchmark AnnData loading.

    This example demonstrates how to benchmark the process of loading
    data with AnnData and converting it to PyTorch tensors. It shows
    how to measure the performance of data loading and preprocessing
    steps.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE: AnnData Loading Benchmark")
    print("=" * 60)

    try:
        import scanpy as sc
        from torch.utils.data import DataLoader, TensorDataset

        # Load data with AnnData
        print("Loading data with AnnData...")
        adata_path = (
            "/home/pbinder/bionemo-framework/sub-packages/bionemo-scdl/small_samples/sample_500_19836_0.9.h5ad"
        )
        adata = sc.read_h5ad(adata_path)

        # Convert to PyTorch tensors
        X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        # y = torch.tensor(adata.obs['cell_type'].cat.codes.values, dtype=torch.long)

        # Create dataset and dataloader
        dataset = TensorDataset(X)  # , y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Benchmark AnnData loading
        result = benchmark_any_dataloader(
            dataloader=dataloader,
            name="AnnData Loading",
            data_path=adata_path,
            num_epochs=1,
            max_batches=20,
            warmup_batches=3,
        )

        print(f"✅ AnnData benchmark completed: {result.samples_per_second:.2f} samples/sec")

    except ImportError:
        print("scanpy not available, skipping this example")


# ============================================================================
# EXAMPLE 4: Custom Batched Sampling
# ============================================================================


class CustomBatchedSampler:
    """Custom batched sampler for demonstration."""

    def __init__(self, dataset_size, batch_sizes):
        """Initialize the custom batched sampler.

        Args:
            dataset_size: Total size of the dataset
            batch_sizes: List of batch sizes to use
        """
        self.dataset_size = dataset_size
        self.batch_sizes = batch_sizes
        self.current_batch = 0

    def __iter__(self):
        """Return iterator for the sampler."""
        self.current_batch = 0
        return self

    def __next__(self):
        """Get the next batch of indices.

        Returns:
            List of indices for the next batch

        Raises:
            StopIteration: When all batches have been returned
        """
        if self.current_batch >= len(self.batch_sizes):
            raise StopIteration

        batch_size = self.batch_sizes[self.current_batch]
        start_idx = self.current_batch * batch_size
        end_idx = min(start_idx + batch_size, self.dataset_size)

        # Create batch data
        batch_data = torch.randn(end_idx - start_idx, 50)
        self.current_batch += 1

        return batch_data


def example_custom_batched_sampling():
    """Example: Benchmark custom batched sampling."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Custom Batched Sampling")
    print("=" * 60)

    # Create custom batched sampler with varying batch sizes
    custom_sampler = CustomBatchedSampler(
        dataset_size=1000,
        batch_sizes=[16, 32, 16, 64, 16, 32, 16],  # Varying batch sizes
    )

    # Benchmark it directly
    result = benchmark_any_dataloader(
        dataloader=custom_sampler, name="Custom Batched Sampling", num_epochs=1, max_batches=10, warmup_batches=2
    )

    print(f"✅ Custom sampling benchmark completed: {result.samples_per_second:.2f} samples/sec")


# ============================================================================
# EXAMPLE 5: Multiple Different Dataloaders
# ============================================================================


def example_multiple_dataloaders():
    """Example: Benchmark multiple different types of dataloaders."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Multiple Different Dataloaders")
    print("=" * 60)

    # Create different types of dataloaders

    # 1. PyTorch DataLoader
    dataset1 = SimpleDataset(size=300)
    dataloader1 = DataLoader(dataset1, batch_size=16, shuffle=True)

    # 2. List-based dataloader
    dataloader2 = [torch.randn(10) for _ in range(50)]

    # 3. Generator-based dataloader
    def create_generator():
        for i in range(50):
            yield torch.randn(8, 20)

    dataloader3 = create_generator()

    # 4. Custom iterator
    class CustomIterator:
        def __init__(self):
            self.data = [torch.randn(5) for _ in range(40)]
            self.current = 0

        def __iter__(self):
            self.current = 0
            return self

        def __next__(self):
            if self.current >= len(self.data):
                raise StopIteration
            result = self.data[self.current]
            self.current += 1
            return result

    dataloader4 = CustomIterator()

    # Benchmark all of them
    dataloaders = [
        {
            "name": "PyTorch DataLoader",
            "dataloader": dataloader1,
            "num_epochs": 1,
            "max_batches": 10,
            "warmup_batches": 2,
        },
        {
            "name": "List Dataloader",
            "dataloader": dataloader2,
            "num_epochs": 1,
            "max_batches": 10,
            "warmup_batches": 2,
        },
        {
            "name": "Generator Dataloader",
            "dataloader": dataloader3,
            "num_epochs": 1,
            "max_batches": 10,
            "warmup_batches": 2,
        },
        {
            "name": "Custom Iterator",
            "dataloader": dataloader4,
            "num_epochs": 1,
            "max_batches": 10,
            "warmup_batches": 2,
        },
    ]

    benchmark_multiple_dataloaders_simple(dataloaders=dataloaders, output_dir="multiple_dataloader_results")

    print("✅ Multiple dataloader benchmarks completed!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("BioNeMo SCDL Benchmarking Framework - Simple Examples")
    print("=" * 60)

    # Explicit protocol usage demo
    demo_dl = DataLoader(ProtocolDemoDataset(), batch_size=2)
    run_benchmark_with_protocol(demo_dl)

    # Run examples
    try:
        example_simple_pytorch()
        example_scdl_configurations()
        example_anndata_loading()
        example_custom_batched_sampling()
        example_multiple_dataloaders()

        print("\n" + "=" * 60)
        print("🎉 All simple examples completed successfully!")
        print("=" * 60)
        print("\nKey insight: You can benchmark ANY dataloader by just passing it directly!")
        print("No factory functions, no inheritance, no complex setup needed.")

    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        print("This might be due to missing dependencies or data files.")
        print("The framework itself should work with any iterable dataloader.")
