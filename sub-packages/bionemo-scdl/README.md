# BioNeMo-SCDL: Single Cell Data Loading for Scalable Training of Single Cell Foundation Models.

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

#### Tokenization

A common use case for the single-cell dataloader is tokenizing data using a predefined vocabulary with a defined tokenizer function.

``` python
import numpy as np
ds = SingleCellMemMapDataset("97e_scmm")
index = 0
values, feature_ids = ds.get_row(index, return_features=True, feature_vars=["feature_id"])
assert (
            len(feature_ids) == 1
        )  # we expect feature_ids to be a list containing one np.array with the row's feature ids
gene_data, col_idxs = np.array(values[0]), np.array(values[1])
tokenizer_function = lambda x,y,z : x
tokenizer_function(
            gene_data,
            col_idxs,
            feature_ids[0],
        )
```

#### Loading directly with Pytorch-compatible Dataloaders

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

For some applications, we might want to also use the features. These can be specified with get_row(index, return_features = True). By default, all features are returned, but the features can be specified with the feature_vars argument in get_row, which corresponds to a list of the feature names to return.

```
for index in range(len(data)):
    model(data.get_row(index,return_features = True))
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

## Runtimes with SCDL

The runtime and memory usage are examined on a CellXGene Dataset with ~1.5 million rows and a size of 24 GB. On this dataset, there is a 4.9x memory speed up.

![Throughput Image](https://raw.githubusercontent.com/NVIDIA/bionemo-framework/main/sub-packages/bionemo-scdl/assets/throughput.png)

Additionally, the peak memory usage when iterating over the datasets with the SCDL dataloader is only 36.5 MB, since the whole dataset is never loaded into memory due to the numpy memomory-mapped backing.

![Memory Image](https://raw.githubusercontent.com/NVIDIA/bionemo-framework/main/sub-packages/bionemo-scdl/assets/disk_space.png)

### Using Neighbor Information in Single Cell Datasets

SCDL now supports loading and utilizing neighbor information from AnnData objects. This is particularly useful for tasks that require knowledge of cell neighborhoods, trajectory analysis, or spatial relationships.

#### Neighbor Data Structure in AnnData

The neighbor functionality expects neighbor information to be stored in the **`.obsp` (observations pairwise) attribute** of the AnnData object as a **sparse matrix**:

- **Location**: `adata.obsp[neighbor_key]` (default key is `'next_cell_ids'`)
- **Format**: Sparse matrix (scipy.sparse format, typically CSR - Compressed Sparse Row)
- **Dimensions**: `[n_cells × n_cells]` adjacency matrix
- **Values**: Weights/distances (e.g., pseudotime values, spatial distances, similarity scores)
- **Non-zero entries**: Indicate neighbor relationships

**Example - Generating Neighbor Data from Trajectory Analysis:**

```python
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix

# After computing pseudotime with your preferred method (e.g., DPT, Monocle, etc.)
# adata.obs['pseudotime'] contains pseudotime values for each cell
# Assuming you define a function create_pseudotime_neighbors() to find k nearest neighbors in pseudotime space and store as sparse matrix

# Create and store neighbor matrix
neighbor_matrix = create_pseudotime_neighbors(adata.obs['pseudotime'])
adata.obsp['next_cell_ids'] = neighbor_matrix
```

#### Loading a Dataset with Neighbor Support

```python
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset, NeighborSamplingStrategy

# Load dataset with neighbor support
data = SingleCellMemMapDataset(
    "dataset_path",
    "path/to/anndata.h5ad",
    load_neighbors=True,                               # Enable neighbor functionality
    neighbor_key='next_cell_ids',                      # Key in AnnData.obsp containing neighbor information
    neighbor_sampling_strategy=NeighborSamplingStrategy.RANDOM,  # Strategy for sampling neighbors
    fallback_to_identity=True                          # Use cell itself as neighbor when no neighbors exist
)
```

#### Accessing Neighbor Data

SCDL provides several methods to access and utilize neighbor information:

```python
# Get neighbor indices for a specific cell
neighbor_indices = data.get_neighbor_indices_for_cell(cell_index)

# Get neighbor weights (if available)
neighbor_weights = data.get_neighbor_weights_for_cell(cell_index)

# Sample a neighbor according to the configured strategy
neighbor_index = data.sample_neighbor_index(cell_index)

```

**Example Usage in Contrastive Learning:**

```python
# Contrastive Learning - Compare cells with their neighbors
for cell_index in range(len(data)):
    # Get current cell and its neighbor
    current_cell_data, _ = data.get_row(cell_index)
    neighbor_index = data.sample_neighbor_index(cell_index)
    neighbor_cell_data, _ = data.get_row(neighbor_index)

    # Use in contrastive loss
    current_embedding = model.encode(current_cell_data)
    neighbor_embedding = model.encode(neighbor_cell_data)
    contrastive_loss = compute_contrastive_loss(current_embedding, neighbor_embedding)
```

## Future Work and Roadmap

SCDL is currently in public beta. In the future, expect improvements in data compression
and data loading performance.

## LICENSE

BioNeMo-SCDL has an Apache 2.0 license, as found in the LICENSE file.
