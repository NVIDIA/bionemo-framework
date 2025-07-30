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

## Quick Start

### Download Data
```bash
wget -O cellxgene_example_25k.h5ad "https://datasets.cellxgene.cziscience.com/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad"
```

### Run python code
```python
import anndata as ad
from anndata.experimental import AnnCollection, AnnLoader
from bionemo.scbenchmark.benchmark import benchmark_single_dataloader, print_results

filepath = "cellxgene_example_25k.h5ad"

#create a dataloader factory
def anndata_factory(input_path, batch_size = 64):
    def factory():
        dataset = ad.read_h5ad(input_path, backed="r")
        return AnnLoader(dataset, num_workers = 0, collate_fn=lambda x: x[0])
    return factory

#benchmark the dataloader
result = benchmark_single_dataloader(
    dataloader_factory=anndata_factory(filepath),
    data_path=filepath,
    name="AnnLoader",
    max_time_seconds = 10
)

print_results(result)
```

### Output
```
============================================================
Benchmark: AnnLoader
Samples/sec: 622.98
Total samples: 6230
Total time: 10.000s
Dataset instantiation: 0.264s
Dataloader instantiation: 0.000s
Peak memory: 57.9 MB
Disk size: 144.6 MB
```

## Bring your own Dataloader

Your dataloader just needs to be **iterable** (support `for batch in dataloader`).

### Dataloader vs Dataset Factories

The framework supports two distinct patterns for benchmarking, each optimized for different scenarios:

**Dataloader Factory**: Creates both dataset and dataloader
```python
from torch.utils.data import DataLoader

def dataloader_factory():
    dataset = load_dataset()  # Load data each time
    return DataLoader(dataset, batch_size=32)

benchmark_single_dataloader(
    dataloader_factory=dataloader_factory,
    data_path="/path/to/data",
    name="MyBenchmark"
)
```
- **Use when**: Testing different datasets or when dataset loading is fast
- **Measures**: Total instantiation time (dataset + dataloader combined)

**Dataset Factory**: Loads dataset once, reused across multiple dataloader configs
```python
from torch.utils.data import DataLoader

def dataset_factory():
    return load_dataset()  # Load once

def dataloader_factory_32(dataset):  # Receives pre-loaded dataset
    return DataLoader(dataset, batch_size=32)

def dataloader_factory_64(dataset):  # Receives pre-loaded dataset
    return DataLoader(dataset, batch_size=64)

# Dataset reuse mode - loads dataset once, tests multiple configs
benchmark_dataloaders_with_configs(
    shared_dataset_factory=dataset_factory,
    dataloader_configs=[
        {"name": "Config1", "dataloader_factory": dataloader_factory_32, "data_path": "/path/to/data"},
        {"name": "Config2", "dataloader_factory": dataloader_factory_64, "data_path": "/path/to/data"}
    ],
    output_prefix="my_benchmark"
)
```

- **Use when**: Testing multiple configurations on the same large dataset
- **Performance benefit**: Avoids expensive dataset reloading (e.g., 10GB+ datasets)
- **Separates metrics**: Dataset vs dataloader instantiation times tracked separately
- **Memory consideration**: Dataset stays in memory throughout all tests

**Benchmark your dataloader!**
```python
from bionemo.scbenchmark import benchmark_single_dataloader

# Benchmark it with instantiation measurement!
result = benchmark_single_dataloader(
    dataloader_factory=create_my_dataloader,
    data_path="path/to/data",  # Required: for disk measurement
    name="My Dataloader",
    num_epochs=1,
    max_batches=100,           # Optional: limit number of batches
    max_time_seconds=30.0,     # Optional: limit runtime to 30 seconds
    warmup_batches=5,          # Optional: warmup with 5 batches
    warmup_time_seconds=2.0,   # Optional: warmup for 2 seconds
    output_prefix="my_dataloader_benchmark"  # CSV filename prefix
)

# Print results
print(f"Dataset instantiation time: {result.dataset_instantiation_time_seconds:.4f}s")
print(f"Dataloader instantiation time: {result.dataloader_instantiation_time_seconds:.4f}s")
print(f"Peak instantiation memory: {result.peak_memory_during_instantiation_mb:.2f} MB")
print(f"Samples/second: {result.samples_per_second:.2f}")
print(f"Peak memory usage: {result.peak_memory_mb:.2f} MB")
print(f"Average memory usage: {result.avg_memory_mb:.2f} MB")
print(f"Disk usage (MB): {result.disk_size_mb:.2f}")
```

## Examples


### Comprehensive Examples


See the examples directory for complete examples:
```bash
# Full feature demonstration with dataset reuse
python examples/comprehensive_benchmarking.py \
    --adata-path /path/to/data.h5ad \
    --scdl-path /path/to/scdl/ \
    --num-epochs 2 \
    --num-runs 3
```

This demonstrates SCDL and AnnLoader dataloaders with a variety of sampling schemes, sequential sampling, and multi-worker settings. Shows both dataset reuse and independent loading patterns.

# scDataset profiling
```bash
python examples/scdataset_script.py \
    --fetch-factors 1 2 4 8 16 32 64 \
    --block-sizes 1 4 8 16 32 64 \
    --scdl-path /path/to/scdl/ \
    --adata-path /path/to/data.h5ad
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

**CSV Export:**
- Detailed per-epoch breakdown with `{output_prefix}_detailed_breakdown.csv`
- All configurations consolidated into single CSV file
- Appends results from multiple benchmark runs
- Perfect for analysis and comparison

**Example Output Files:**
- `my_benchmark_detailed_breakdown.csv` - Main results file
- Contains all run data, epochs, memory, throughput metrics


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
