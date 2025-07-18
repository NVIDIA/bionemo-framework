# SCDL Benchmark Tool

A standalone benchmark script for evaluating SingleCellMemMapDataset performance. No external BioNeMo benchmark framework required - everything is self-contained.

## Quick Start

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
python scdl_benchmark_standalone.py

# Benchmark your own AnnData dataset  
python scdl_benchmark_standalone.py -i your_dataset.h5ad

# Export a detailed CSV file
python scdl_benchmark_standalone.py --csv
```

## Usage Examples

```bash
# Basic benchmark with example dataset
python scdl_benchmark_standalone.py

# Benchmark specific dataset with sequential sampling  
python scdl_benchmark_standalone.py -i my_data.h5ad -s sequential

# Generate CSV files for analysis
python scdl_benchmark_standalone.py --csv -o report.txt

# Custom parameters
python scdl_benchmark_standalone.py --batch-size 64 --max-time 60

# Baseline comparison (SCDL vs AnnData)
python scdl_benchmark_standalone.py --generate-baseline
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Dataset path (.h5ad or scdl directory) | Auto-download example |
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
Sampling: shuffle
Epochs: 3

PERFORMANCE METRICS:
  Throughput:        10,521 samples/sec
  Instantiation:     1.573 seconds
  H5AD -> SCDL:      2.145 seconds
  Avg Batch Time:    0.0030 seconds

MEMORY USAGE:
  Baseline:          257.4 MB
  Peak (Benchmark):  0.0 MB
  Dataset on Disk:   160.3 MB

DATA PROCESSED:
  Total Samples:     50,764 (16,921/epoch)
  Total Batches:     1,588 (529/epoch)
============================================================
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
  SCDL:              10,521 samples/sec
  AnnData:           3,245 samples/sec
  Speedup:           3.24x faster with SCDL

MEMORY COMPARISON:
  SCDL Peak:         144.6 MB
  AnnData Peak:      856.2 MB
  Memory Efficiency: 5.92x more memory with AnnData

LOADING TIME COMPARISON:
  SCDL Conversion:   1.53 seconds
  AnnData Load:      4.67 seconds
  Load Time Ratio:   3.05x

SUMMARY:
  SCDL provides 3.2x throughput improvement
  SCDL uses 5.9x less memory
================================================================================
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

## Example Datasets

The script automatically downloads a 25K cell example dataset from CellxGene. For other datasets:

- **10X Genomics**: Convert .h5 files to .h5ad using `scanpy.read_10x_h5()`
- **AnnData files**: Use directly with `-i dataset.h5ad`  
- **Large datasets**: Pre-convert to SCDL format for faster loading

## Support

For issues with:
- **SCDL functionality**: Check bionemo-scdl documentation
- **Benchmark script**: Verify dependencies and dataset format
- **Performance**: Try different batch sizes and sampling schemes

The script is designed to be self-contained and user-friendly. Most issues are resolved by following the built-in error messages and installation prompts. 