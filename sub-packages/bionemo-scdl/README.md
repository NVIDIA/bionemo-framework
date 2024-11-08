# BioNemo-SCDL: Single Cell Data Loading for Scalable Training of Single Cell Foundation Models.

## Package Overview

BioNeMo-SCDL provides an independent pytorch-compatible dataset class for single cell data with a consistent API. BioNeMo-SCDL is developed and maintained by NVIDIA. This package can be run independently from BioNeMo. It improves upon simple AnnData-based dataset classes in the following ways:

- A consistent API across input formats that is promised to be consistent across package versions.
- Improved performance when loading large datasets. It allows for loading and fast iteration of large datasets.
- Ability to use datasets that are much, much larger than memory. This is because the datasets are stored in a numpy memory-mapped format.
- Additionally, conversion of large (significantly larger than memory) AnnData files into the SCDL format.
- [Future] Full support for ragged arrays (i.e., datasets with different feature counts; currently only a subset of the API functionality is supported for ragged arrays).
- [Future] Support for improved compression.

BioNeMo-SCDL's API resembles that of AnnData, so code changes are minimal.
In most places a simple swap from an attribute to a function is sufficient (i.e., swapping `data.n_obs` for `data.number_of_rows()`).

## Installation

This package can be installed with
```bash
pip install bionemo-scdl
```
## Usage

### Getting example data

Here is how to process an example dataset from CellxGene with ~25,000 cells:

Download "https://datasets.cellxgene.cziscience.com/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad" to hdf5s/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad

### Loading a single cell dataset from an H5AD file

```python
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset

data = SingleCellMemMapDataset("97e_scmm", "hdf5s/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad")

```

This creates a `SingleCellMemMapDataset` that is stored at 97e_scmm in large, memory-mapped arrays
that enables fast access of datasets larger than the available amount of RAM on a system.

If the dataset is large, the AnnData file can be lazy-loaded and then read in based on chunks of rows in a paginated manner. This can be done by setting the parameters when instantiating the `SingleCellMemMapDataset`:
- `paginated_load_cutoff`, which sets the minimal file size in megabytes at which an AnnData file will be read in in a paginated manner.
- `load_block_row_size`, which is the number of rows that are read into memory at a given time.

### Interrogating single cell datasets and exploring the API

```python

data.number_of_rows()
## 25382

data.number_of_variables()
## [34455]

data.number_of_values()
## 874536810

data.number_nonzero_values()
## 26947275

```

### Saving SCDL (Single Cell Dataloader) datasets to disk

When you open a SCDL dataset, you *must* choose a path where the backing
data structures are stored. However, these structures are not guaranteed
to be in a valid serialized state during runtime.

Calling the `save` method guarantees the on-disk object is in a valid serialized
state, at which point the current python process can exit, and the object can be
loaded by another process later.

```python

data.save()

```

### Loading SCDL datasets from a SCDL archive

When you're ready to reload a SCDL dataset, just pass the path to the serialized
data:

```python
reloaded_data = SingleCellMemMapDataset("97e_scmm")
```

### Using SCDL datasets in model training

SCDL implements the required functions of the PyTorch Dataset abstract class.
You can use PyTorch-compatible DataLoaders to load batches of data from a SCDL class.
With a batch size of 1 this can be run without a collating function. With a batch size
greater than 1, there is a collation function (`collate_sparse_matrix_batch`), that will
collate several sparse arrays into the CSR (Compressed Sparse Row) torch tensor format.

```python
from torch.utils.data import DataLoader
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch

## Mock model: you can remove this and pass the batch to your own model in actual code.
model = lambda x : x

dataloader = DataLoader(data, batch_size=8, shuffle=True, collate_fn=collate_sparse_matrix_batch)
n_epochs = 2
for e in range(n_epochs):
    for batch in dataloader:
        model(batch)
```

## Examples

The examples directory contains various examples for utilizing SCDL.

### Converting existing Cell x Gene data to SCDL

If there are multiple AnnData files, they can be converted into a single `SingleCellMemMapDataset`. If the hdf5 directory has one or more AnnData files, the `SingleCellCollection` class crawls the filesystem to recursively find AnnData files (with the h5ad extension).

To convert existing AnnData files, you can either write your own script using the SCDL API or utilize the convenience script `convert_h5ad_to_scdl`.

Here's an example:

```bash
convert_h5ad_to_scdl --data-path hdf5s --save-path example_dataset
```

## Future Work and Roadmap

SCDL is currently in public beta. In the future, expect improvements in data compression
and data loading performance.

## LICENSE

BioNemo-SCDL has an Apache 2.0 license, as found in the LICENSE file.
