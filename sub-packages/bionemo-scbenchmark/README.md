# BioNeMo Single-Cell Benchmarking Framework

A simple, flexible framework for benchmarking any dataloader without requiring inheritance or modifications to your existing code.

## Key Features

- **Zero Inheritance Required**: Your dataloader doesn't need to inherit from anything
- **Works with Any Iterable**: PyTorch DataLoaders, custom iterators, generators, lists, etc.
- **Time-Based Benchmarking**: Set maximum runtime or warmup periods
- **Modular Architecture**: Core benchmarking logic is reusable and extensible
- **Comprehensive Metrics**: Disk space, memory usage, throughput, timing, **AND instantiation** P
- **Fine-Grained Metrics**: Provides per-epoch metrics and options to re-run metrics.

## Quick Start

### 0. Use a virtual environment

```bash
python -m venv bionemo_singlecell_benchmark

source bionemo_singlecell_benchmark/bin/activate
```

### 1. Install Package

```bash
pip install -e .
```

**For baseline comparison** (optional):
```bash
pip install anndata bionemo-scdl scDataset
```

## Quick Start

Your dataloader just needs to be **iterable** (support `for batch in dataloader`).

**Create a factory function**

Creating a factory function enables profiling of dataloader instantiation.

```python
from bionemo.scbenchmark import benchmark_dataloader

# Create a factory function that returns your dataloader
def create_my_dataloader():
    dataset = MyDataset()
    return DataLoader(dataset, batch_size=32)
```

**Benchmark your dataloader!**
```python
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
print(f"Instantiation time: {result.instantiation_metrics.instantiation_time_seconds:.4f}s")
print(f"Instantiation memory: {result.instantiation_metrics.memory_delta_mb:.2f} MB")
print(f"Samples/second: {result.samples_per_second:.2f}")
print(f"Peak memory usage: {result.peak_memory_mb:.2f} MB")
print(f"Average memory usage: {result.avg_memory_mb:.2f} MB")
print(f"Disk usage (MB): {getattr(result, 'disk_usage_mb', 'N/A')}")

```

## Examples

See the examples directory for comprehensive examples:
```python
# Run the comprehensive example to see ALL features in action
python examples/comprehensive_benchmarking.py
```

- **Complete reference** showing ALL features, including SCDL dataloaders with sequential, and weighted sampling, anndataloader testing, and SCDL wth scDataset.
