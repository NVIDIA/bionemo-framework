# BioNeMo SCDL Benchmarking Framework

A simple, flexible framework for benchmarking any dataloader without requiring inheritance or modifications to your existing code.

## Key Features

- **Zero Inheritance Required**: Your dataloader doesn't need to inherit from anything
- **Works with Any Iterable**: PyTorch DataLoaders, custom iterators, generators, lists, etc.
- **Type Safety with Protocols**: Uses Python Protocols for better type checking
- **Time-Based Benchmarking**: Set maximum runtime or warmup periods
- **Modular Architecture**: Core benchmarking logic is reusable and extensible
- **Comprehensive Metrics**: Disk space, memory usage, throughput, timing, **AND instantiation**
- **Always Measures Instantiation**: Automatically measures dataloader creation time and memory

## Quick Start

**Create a factory function and benchmark your dataloader!**

```python
from bionemo.scbenchmark import benchmark_dataloader

# Create a factory function that returns your dataloader
def create_my_dataloader():
    dataset = MyDataset()
    return DataLoader(dataset, batch_size=32)

# Benchmark it with instantiation measurement!
result = benchmark_dataloader(
    name="My Dataloader",
    dataloader_factory=create_my_dataloader,
    data_path="path/to/data",  # Optional: for disk measurement
    num_epochs=1,
    max_batches=100,           # Optional: limit number of batches
    max_time_seconds=30.0,     # Optional: limit runtime to 30 seconds
    warmup_batches=5,          # Optional: warmup with 5 batches
    warmup_time_seconds=2.0    # Optional: warmup for 2 seconds
)

print(f"Samples/second: {result.samples_per_second:.2f}")
print(f"Memory usage: {result.peak_memory_mb:.2f} MB")
print(f"Instantiation time: {result.instantiation_metrics.instantiation_time_seconds:.4f}s")
```

## Time-Based Benchmarking

The framework supports both batch-based and time-based limits:

```python
# Time-based benchmarking
result = benchmark_dataloader(
    name="Time-Based Benchmark",
    dataloader_factory=create_my_dataloader,
    max_time_seconds=60.0,      # Run for max 60 seconds
    warmup_time_seconds=5.0,    # Warmup for 5 seconds
    print_progress=True
)

# Hybrid limits (stops when either is reached)
result = benchmark_dataloader(
    name="Hybrid Limits",
    dataloader_factory=create_my_dataloader,
    max_batches=1000,           # OR stop after 1000 batches
    max_time_seconds=30.0,      # OR stop after 30 seconds
    warmup_batches=10
)
```

## Type Safety with Protocols

The framework uses Python Protocols for type safety without requiring inheritance:

```python
from bionemo.scbenchmark import DataloaderProtocol, DatasetProtocol

# Your dataloader just needs to be iterable
class MyCustomDataloader:
    def __iter__(self):
        # Return an iterator
        return iter([...])

    def __len__(self):  # Optional
        return 100

# Type checking works!
def benchmark_my_dataloader(dl: DataloaderProtocol):
    def create_dataloader():
        return dl
    return benchmark_dataloader("My Dataloader", create_dataloader)

# This works with any iterable
my_dl = MyCustomDataloader()
result = benchmark_my_dataloader(my_dl)
```

## Modular Architecture

The framework is built with a modular design for extensibility:

```python
from bionemo.scbenchmark import BenchmarkConfig, run_benchmark_with_config

# Create custom configuration
config = BenchmarkConfig(
    name="Custom Benchmark",
    num_epochs=2,
    max_time_seconds=45.0,
    warmup_time_seconds=3.0,
    print_progress=True
)

# Run with custom config
def create_dataloader():
    return MyDataLoader()

result = run_benchmark_with_config(create_dataloader(), config)
```

## Examples

### SCDL with Different Configurations
```python
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch

# Factory function for SCDL dataloader
def create_scdl_dataloader(batch_size=8):
    data = SingleCellMemMapDataset("my_dataset", "path/to/data.h5ad")
    return DataLoader(data, batch_size=batch_size, collate_fn=collate_sparse_matrix_batch)

# Benchmark different configurations
result1 = benchmark_dataloader(
    name="SCDL batch=8",
    dataloader_factory=lambda: create_scdl_dataloader(8),
    max_time_seconds=30.0
)

result2 = benchmark_dataloader(
    name="SCDL batch=16",
    dataloader_factory=lambda: create_scdl_dataloader(16),
    max_time_seconds=30.0
)

# Compare instantiation times
print(f"SCDL batch=8 instantiation: {result1.instantiation_metrics.instantiation_time_seconds:.4f}s")
print(f"SCDL batch=16 instantiation: {result2.instantiation_metrics.instantiation_time_seconds:.4f}s")
```

### AnnData Loading
```python
import scanpy as sc
from torch.utils.data import DataLoader, TensorDataset

def create_anndata_dataloader():
    # Load with AnnData
    adata = sc.read_h5ad("path/to/data.h5ad")
    X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
    dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=32)

# Benchmark AnnData loading with time-based warmup
result = benchmark_dataloader(
    name="AnnData Loading",
    dataloader_factory=create_anndata_dataloader,
    data_path="path/to/data.h5ad",
    max_time_seconds=60.0,
    warmup_time_seconds=5.0
)

print(f"Instantiation time: {result.instantiation_metrics.instantiation_time_seconds:.4f}s")
print(f"Instantiation memory: {result.instantiation_metrics.memory_delta_mb:.2f} MB")
```

### Custom Batched Sampling
```python
class CustomBatchedSampler:
    def __iter__(self):
        # Your custom sampling logic
        for i in range(100):
            yield torch.randn(16, 50)  # Variable batch sizes

def create_custom_sampler():
    return CustomBatchedSampler()

# Benchmark custom sampling with time limit
result = benchmark_dataloader(
    name="Custom Sampling",
    dataloader_factory=create_custom_sampler,
    max_time_seconds=45.0,
    warmup_batches=10
)
```

## Multiple Dataloader Comparison

```python
from bionemo.scbenchmark import benchmark_multiple_dataloaders

# Create factory functions for your dataloaders
def create_dataloader_a():
    dataset = DatasetA()
    return DataLoader(dataset, batch_size=16)

def create_dataloader_b():
    dataset = DatasetB()
    return DataLoader(dataset, batch_size=32)

def create_custom_iterator():
    return MyCustomIterator()

# Compare them with time-based limits
dataloaders = [
    {
        "name": "Dataloader A",
        "factory": create_dataloader_a,
        "data_path": "path/to/data_a",
        "max_time_seconds": 30.0,
        "warmup_time_seconds": 2.0
    },
    {
        "name": "Dataloader B",
        "factory": create_dataloader_b,
        "data_path": "path/to/data_b",
        "max_time_seconds": 30.0,
        "warmup_time_seconds": 2.0
    },
    {
        "name": "Custom Iterator",
        "factory": create_custom_iterator,
        "max_time_seconds": 30.0,
        "warmup_batches": 5
    }
]

results = benchmark_multiple_dataloaders(
    dataloaders=dataloaders,
    output_dir="benchmark_results"
)
```

## What Gets Measured

- **Instantiation Time**: Time to create the dataloader (measured automatically)
- **Instantiation Memory**: Memory usage during dataloader creation
- **Disk Space**: Size of data files/directories
- **Setup Time**: Time to create the dataloader
- **Warmup Time**: Time spent in warmup phase
- **Iteration Time**: Time to iterate through batches
- **Throughput**: Samples/second and batches/second
- **Memory Usage**: Peak and average memory consumption
- **GPU Memory**: If CUDA is available

## Examples

See the examples directory for comprehensive examples:

- `examples/simple_benchmark_examples.py` - Basic usage examples
- `examples/time_based_benchmarking.py` - Time-based benchmarking examples
- `examples/instantiation_benchmarking.py` - Instantiation measurement examples

## Adding Your Own Dataloader

Your dataloader just needs to be **iterable** (support `for batch in dataloader`). That's it!

### Basic Usage
```python
# Create a factory function that returns your dataloader
def create_my_dataloader():
    dataset = MyDataset()
    return DataLoader(dataset, batch_size=32)

# Benchmark it with instantiation measurement
result = benchmark_dataloader(
    name="My Dataloader",
    dataloader_factory=create_my_dataloader,
    max_time_seconds=30.0  # Optional time limit
)

# Access instantiation metrics
print(f"Instantiation time: {result.instantiation_metrics.instantiation_time_seconds:.4f}s")
print(f"Instantiation memory: {result.instantiation_metrics.memory_delta_mb:.2f} MB")
```

### Type Safety with Protocols
```python
from bionemo.scbenchmark import DataloaderProtocol

# Your dataloader implements the protocol automatically if it's iterable
class MyDataloader:
    def __iter__(self):
        return iter([...])

    def __len__(self):  # Optional
        return 100

# Type checking works!
def process_dataloader(dl: DataloaderProtocol):
    def create_dataloader():
        return dl
    return benchmark_dataloader("Typed Dataloader", create_dataloader)
```

## Installation

```bash
pip install bionemo-scbenchmark
```

## Dependencies

- torch >= 1.12.0
- psutil >= 5.8.0
- numpy >= 1.21.0

## License

Apache 2.0
