# BioNeMo Single-Cell Benchmarking Framework

A simple, flexible framework for benchmarking any dataloader without requiring inheritance or modifications to your existing code.


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

### Dataloader vs Dataset Factories

The framework supports two distinct patterns for benchmarking, each optimized for different scenarios:

**Dataloader Factory**: Creates both dataset and dataloader
```python
from torch import DataLoader

def dataloader_factory():
    dataset = load_dataset()  # Load data each time
    return DataLoader(dataset, batch_size=32)

benchmark_dataloader(
    dataloaders=dataloader_factory()
)

```
- **Use when**: Testing different datasets or when dataset loading is fast
- **Measures**: Total instantiation time (dataset + dataloader combined)

**Dataset Factory**: Loads dataset once, reused across multiple dataloader configs
```python
from torch import DataLoader

def dataset_factory():
    return load_dataset()  # Load once

def dataloader_factory(dataset):  # Receives pre-loaded dataset
    return DataLoader(dataset, batch_size=32)

# Dataset reuse mode - loads dataset once, tests multiple configs
benchmark_dataloader(
    dataset_factory=dataset_factory,
    dataloaders=[
        {"name": "Config1", "dataloader_factory": lambda ds: DataLoader(ds, batch_size=32)},
        {"name": "Config2", "dataloader_factory": lambda ds: DataLoader(ds, batch_size=64)}
    ]
)
```

- **Use when**: Testing multiple configurations on the same large dataset
- **Performance benefit**: Avoids expensive dataset reloading (e.g., 10GB+ datasets)
- **Separates metrics**: Dataset vs dataloader instantiation times tracked separately
- **Memory consideration**: Dataset stays in memory throughout all tests

**Benchmark your dataloader!**
```python
from bionemo.scbenchmark.benchmark import benchmark_dataloader

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


### Comprehensive Examples


See the examples directory for complete examples:
```bash
# Full feature demonstration
python examples/comprehensive_benchmarking.py
```

This demonstrates SCDL and AnnLoader dataloaders with a variety of sampling schemes and with sequential sampling, and with a multi-worker setting.

# scDataset profiling
```bash
python examples/scdataset_script.py --fetch-factors 1 2 4 8 16 32 64 --block-sizes 1 4 8 16 32 64
```
This is code for reproducing AnnDataset and SCDL results wrapped in the scDataset sampler.
## Key Features

- **Zero Inheritance Required**: Your dataloader doesn't need to inherit from anything
- **Works with Any Iterable**: PyTorch DataLoaders, custom iterators, generators, lists, etc.
- **Time-Based Benchmarking**: Set maximum runtime or warmup periods
- **Modular Architecture**: Core benchmarking logic is reusable and extensible
- **Comprehensive Metrics**: Disk space, memory usage, throughput, timing, **AND instantiation**
- **Fine-Grained Metrics**: Provides per-epoch metrics and options to re-run metrics
- **Real-time CSV Output**: Results written to CSV files after every individual run
- **Memory Monitoring**: Tracks peak and average PSS memory usage of processes and children
- **Flexible Stopping**: Stop by time limit, batch count, or epoch completion
- **Multiple Run Support**: Run same configuration multiple times for statistical analysis

### What Gets Measured

**Throughput & Performance:**
- Samples per second (throughput)
- Batches per second
- Total iteration time per epoch
- Warmup time and samples processed
- Instantiation Time

**Memory Usage:**
- Peak memory (MB) during benchmarking and instantiation
- Average memory (MB) throughout execution
- Memory baseline tracking for accurate delta measurements

**Storage & Resources:**
- Disk usage of data files (MB)
- Support for multiple files/directories

### Output Formats

**Real-time CSV Export:**
- Detailed per-epoch breakdown updated after every run
- All configurations consolidated into single CSV file
- Perfect for live monitoring and analysis

**JSON Results:**
- Complete benchmark metadata per configuration
- Structured data for programmatic analysis
- Individual files per configuration run


### Troubleshooting
**"TypeError: 'NoneType' object is not callable"**
- Check that your factory functions return the dataloader, not None
- Verify lambda functions in dataset reuse mode are correctly formed

**High memory usage**
- In dataset reuse mode, dataset stays in memory throughout all tests
- Consider reloading the dataset for very large datasets if memory is limited

**Slow benchmarking**
- Use `max_time_seconds` or `max_batches` to limit test duration
- Check if your dataloader factory is doing expensive operations repeatedly
- **Clearing the page cache**: With lazy loading, data may be stored in the page cache between runs. This is especially an issue with SCDL. Between runs, the page cache can be cleared with
```sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'```. If `benchmark_dataloader` or any of the example scripts are run with sudo, it will perform this between runs.
