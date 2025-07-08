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



## Modular Architecture

The framework is built with a modular design for extensibility:

```python
from bionemo.scbenchmark import BenchmarkConfig, benchmark_dataloader

# Create custom configuration using factory pattern
def create_dataloader():
    return MyDataLoader()

# Run with custom parameters
result = benchmark_dataloader(
    name="Custom Benchmark",
    dataloader_factory=create_dataloader,
    num_epochs=2,
    max_time_seconds=45.0,
    warmup_time_seconds=3.0,
    print_progress=True
)
```

## Examples

See the examples directory for comprehensive examples:

- `examples/comprehensive_benchmarking.py` - **Complete reference** showing ALL features with one dataloader, including SCDL dataloaders with sequential, block, and weighted sampling
- `examples/benchmark_scdataset_paper.py` - **Paper replication** demonstrating benchmarking SCDL with SCDataset as used in the paper "scDataset: Scalable Data Loading for Deep Learning on Large-Scale Single-Cell Omics"

### Quick Comprehensive Example

```python
# Run the comprehensive example to see ALL features in action
python examples/comprehensive_benchmarking.py
```

### SCDataset Paper Example

```python
# Run the SCDataset paper benchmarking example
python examples/benchmark_scdataset_paper.py
```

This replicates the benchmarking approach from the scDataset paper, including:
- ✅ SCDataset with BioNeMo data (as used in the paper)
- ✅ PyTorch DataLoader with BioNeMo data
- ✅ SCDataset with HuggingFace datasets
- ✅ PyTorch DataLoader with HuggingFace datasets
- ✅ Different sampling strategies (sequential, block, weighted)
- ✅ Different modes (train/random vs eval/stream)
- ✅ Parameter sweep experiments
- ✅ Performance comparison and analysis

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

## SCDL Dataloader Support

The framework includes built-in support for SCDL dataloaders with different sampling strategies:

```python
from bionemo.scbenchmark import benchmark_dataloader
from bionemo.scbenchmark.scdl_samplers import SequentialSampler, BlockSampler, WeightedSampler

# Sequential sampling
def create_sequential_dataloader():
    dataset = SingleCellMemMapDataset("dataset_name", "data.h5ad")
    sampler = SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=32, sampler=sampler,
                     collate_fn=collate_sparse_matrix_batch)

result = benchmark_dataloader(
    name="SCDL Sequential",
    dataloader_factory=create_sequential_dataloader,
    max_time_seconds=30.0
)

# Block sampling
def create_block_dataloader():
    dataset = SingleCellMemMapDataset("dataset_name", "data.h5ad")
    sampler = BlockSampler(dataset, block_size=100)
    return DataLoader(dataset, batch_size=32, sampler=sampler,
                     collate_fn=collate_sparse_matrix_batch)

# Weighted sampling
def create_weighted_dataloader():
    dataset = SingleCellMemMapDataset("dataset_name", "data.h5ad")
    weights = torch.ones(len(dataset))  # Your custom weights
    sampler = WeightedSampler(dataset, weights)
    return DataLoader(dataset, batch_size=32, sampler=sampler,
                     collate_fn=collate_sparse_matrix_batch)
```

The comprehensive example shows all three sampling strategies in action!

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
