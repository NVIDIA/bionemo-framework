# BioNeMo DNADL
Contains datasets for loading intervals from a genome and editing sequences using a VCF file

## Package Overview
- **Genome**: Extract sequences from a fasta reference genome using chromosome intervals.
- **GenomeIntervalDataset**: Load genome intervals from a BED file, allowing for easy access to DNA sequences within specified regions.
- **VCFDataset Loader**: Load variant information from a VCF file and extract sequences with variants for specified samples.


## Installation
```bash
pip install bionemo-dnadl
```

## Usage

### Genome
```python
from bionemo.dnadl.io.dataset import Genome
from bionemo.dnadl.tools.genome_interval import GenomeInterval

genome = Genome("path/to/genome.fa")
sequence = genome.extract_sequence(GenomeInterval("chr1", 10, 15))
print(sequence)  # Outputs the DNA sequence from chr1:10-15
```

### GenomeIntervalDataset
```python
from bionemo.dnadl.io.dataset import Genome, GenomeIntervalDataset
import polars as pl

genome = Genome("path/to/genome.fa")
bed_file_df = pl.read_csv("path/to/genome.bed", separator=" ", has_header=False, new_columns=["chrom", "start", "end"])
genome_interval_dataset = GenomeIntervalDataset(genome, bed_file_df, context_length=20, tokenizer=None)
```

### VCFDataset
```python
from bionemo.dnadl.io.dataset import Genome, GenomeIntervalDataset, VCFDataset

genome = Genome("path/to/genome.fa")
bed_file_df = pl.read_csv("path/to/genome.bed", separator=" ", has_header=False, new_columns=["chrom", "start", "end"])
genome_interval_dataset = GenomeIntervalDataset(genome=genome, tokenizer=None, bed_file_df=bed_file_df, context_length=20)
vcf_dataset = VCFDataset(
    vcf_file="path/to/variants.vcf.gz",
    genome_interval_dataset=genome_interval_dataset,
    sample_ids=["sample_1", "sample_2"],
    tokenizer=None
)
```
