# SCDL Benchmark Tool

A standalone benchmark script for evaluating SingleCellMemMapDataset performance. No external BioNeMo benchmark framework required - everything is self-contained.

## Quick Start

### 0. Use a virtual environment

```bash
python -m venv bionemo_scdl_benchmark

source bionemo_scdl_benchmark/bin/activate
```

### 1. Install Dependencies

```bash
pip install torch pandas psutil tqdm bionemo-scdl
```

**For baseline comparison** (optional):
```bash
pip install anndata scipy
```

**Note**: If you have the BioNeMo source code, you can install bionemo-scdl locally:
```bash
cd /path/to/bionemo-framework
pip install -e sub-packages/bionemo-scdl/
```

### 2. Run Basic Benchmark

```bash
# Download example dataset and run a quick benchmark / smoke test.
python scdl_speedtest.py

# Benchmark your own AnnData dataset
python scdl_speedtest.py -i your_dataset.h5ad

# Export a detailed CSV file
python scdl_speedtest.py --csv
```

3. Deactivate your virtual environment to return to your original shell state

```bash
deactivate
```

## Usage Examples

```bash
# Basic benchmark with example dataset
python scdl_speedtest.py

# Benchmark specific dataset with sequential sampling
python scdl_speedtest.py -i my_data.h5ad -s sequential

# Generate CSV files for analysis
python scdl_speedtest.py --csv -o report.txt

# Custom parameters
python scdl_speedtest.py --batch-size 64 --max-time 60

# Baseline comparison (SCDL vs AnnData in backed mode (with lazy loading))
python scdl_speedtest.py --generate-baseline
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Dataset path (.h5ad, directory with .h5ad files, or scdl directory) | Auto-download example |
| `-o, --output` | Save report to file | Print to screen |
| `-s, --sampling-scheme` | Sampling method (shuffle/sequential/random) | shuffle |
| `--batch-size` | Batch size | 32 |
| `--max-time` | Max benchmark runtime (seconds) | 30 |
| `--warmup-time` | Warmup period (seconds) | 2 |
| `--csv` | Export detailed CSV files | False |
| `--generate-baseline` | Compare SCDL vs AnnData performance | False |

## Sample Output

```
============================================================
SCDL BENCHMARK REPORT
============================================================

Dataset: cellxgene_example_25k.h5ad
Method: SCDL
Sampling: shuffle
Epochs: 1

PERFORMANCE METRICS:
  Throughput:        20,098 samples/sec
  Instantiation:     0.066 seconds
  Avg Batch Time:    0.0016 seconds

MEMORY USAGE:
  Baseline:          446.6 MB
  Peak (Benchmark):  703.2 MB
  Dataset on Disk:   207.30 MB

DATA PROCESSED:
  Total Samples:     25,382 (25,382/epoch)
  Total Batches:     794 (794/epoch)
============================================================
SCDL version: 0.0.8
Anndata version: 0.11.4
```

## Baseline Comparison Output

When using `--generate-baseline`, you get a comprehensive comparison:

```
================================================================================
SCDL vs ANNDATA COMPARISON REPORT
================================================================================

Dataset: cellxgene_example_25k.h5ad
Sampling: shuffle

THROUGHPUT COMPARISON:
  SCDL:              22,668 samples/sec
  AnnData:           2,529 samples/sec
  Performance:       8.96x speedup with SCDL

MEMORY COMPARISON:
  SCDL Peak:         703.5 MB
  AnnData Peak:      568.8 MB
  Memory Efficiency: SCDL uses 1.24x more memory

DISK USAGE COMPARISON:
  SCDL Size:         0.20 GB
  AnnData Size:      0.14 GB
  Storage Efficiency: SCDL uses 1.43x more disk space

LOADING TIME COMPARISON:
  SCDL Conversion:   0.00 seconds (cached)
  AnnData Load:      0.25 seconds

SUMMARY:
  SCDL provides 9.0x throughput improvement
  SCDL uses 1.2x more memory
  SCDL disk usage: 0.20 GB
  AnnData disk usage: 0.14 GB
  SCDL uses 1.4x more disk space
================================================================================```
```
## CSV Export

When using `--csv`, the script generates:

- **`summary.csv`**: Overall benchmark metrics and configuration
- **`detailed_breakdown.csv`**: Per-epoch performance breakdown

Perfect for analysis in Excel, Python, R, or other data tools.

## Troubleshooting


### Dataset Issues

- **H5AD files**: Converted automatically to SCDL format (conversion time reported)
- **Large datasets**: Uses memory-mapped access for efficiency
- **Download failures**: Check internet connection and try again
- **Conversion caching**: H5AD files are converted once, then reused on subsequent runs

### Performance Tips

- **Faster throughput**: Use `--batch-size 64` or higher
- **Longer runs**: Increase `--max-time 120` for stable measurements
- **Memory profiling**: Use `--csv` to get detailed memory usage per epoch
- **Clearing the page cache**: With lazy loading, data may be stored in the page cache between runs. This is especially an issue with SCDL. Between runs, the page cache can be cleared with
```sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'```

## Example Datasets

The script automatically downloads a 25K cell example dataset from CellxGene. For other datasets:

- **10X Genomics**: Convert .h5 files to .h5ad using `scanpy.read_10x_h5()`
- **AnnData files**: Use directly with `-i dataset.h5ad`
- **Large datasets**: Pre-convert to SCDL format for faster loading

### Tahoe 100M

The Tahoe 100M dataset contains 100 Million cells.

To download the full Tahoe 100M dataset in AnnData format (1 file per plate, 14 total plates):

**Warning** This will trigger egress charges, which can be significant.

**Note**: You will need to have installed the google cloud CLI to download this dataset.

```bash
gcloud storage cp -R gs://arc-ctc-tahoe100/2025-02-25/* .
```

This dataset is 314 GB. The corresponding SCDL dataset is 1.1 TB, so ensure that you have sufficient disk space if using the entire dataset.



## Support

For issues with:
- **SCDL functionality**: Check bionemo-scdl documentation
- **Benchmark script**: Verify dependencies and dataset format
- **Performance**: Try different batch sizes and sampling schemes

The script is designed to be self-contained and user-friendly. Most issues are resolved by following the built-in error messages and installation prompts.
