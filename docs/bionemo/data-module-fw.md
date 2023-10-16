# Data Module

This section provides an overview BioNeMo DataModule and how to use and connect it to models. 

## Overview

The BioNeMo data module serves to centralize several of BioNeMo’s core operations, thus simplifying workflows for future model extensions. This is an interface that encapsulates steps for data processing of BioNeMo Models, allowing higher adaptability of models to several use cases.

Applications of the data module:
* Fine tuning datasets as in `bionemo/data/finetune_dataset.py`
* Processing FASTA format as in `bionemo/data/fasta_dataset.py`
* Transforms (for example, Tokenization) as in `bionemo/model/protein/downstream/sec_str_pred_data.py`

The data modules coordinate functions related to data processing in BioNeMo, including the instantiation of train, validation and test datasets as well as tokenizers, addition of collate functions, and inferring the number of global samples (up and downsampling included).

Since data modules are abstractors, these duties will depend on child classes being implemented:
* `train_dataset` 
* `val_dataset`
* `test_dataset`
* Variations for additional level of control such as `sample_train_dataset` and `adjust_train_dataloader`

## Reasons for using BioNeMo’s Data Module
The Data Module gives developers 4 core benefits:
* **Encapsulation**: centralizes all data processing steps in one place​​.
* **Shareability and Reusability**: sharing and reusing data processing steps across projects
* **Data Management**: Organizes data cleaning and preparation processes for clarity​.
* **Flexibility**: dataset-agnostic model development and allows for easy swapping of datasets​​.

Example:
extracted from `bionemo/model/protein/downstream/sec_str_pred_data.py`

```python
class SSDataModule(BioNeMoDataModule):
    def __init__(self, cfg, trainer, model):
        super().__init__(cfg, trainer)
        self.model=model
        self.tokenizers = [
            Label2IDTokenizer()
            for _ in range(len(self.cfg.labels_size))
            ]
```

The code snippet above defines the tokenization process. It takes as arguments:
* A configuration object of the model as in `examples/protein/esm1nv/conf/finetune_config.yaml`
* A trainer object containing functions that define callbacks, checkpoints as in `bionemo/molecule/megamolbart.pretrain_mlp_validation.py : setup_trainer(cgf)`
* A model object like _ESM1nvModel_ as in `examples/protein/esm1nv/pretrain.py`

## How to use BioNeMo DataModule
You may be familiar with Data Modules from other frameworks such as PyTorch Lightning. The principles are the same.

To begin, have a look into the `bionemo/core.py` classes and functions, particularly the `BioNeMoDataModule` class and its functions. You will notice that many methods within the `BioNeMoDataModule` class are abstract methods and, therefore, they must be implemented and overridden by a subclass. One such example is: 

```python
@abstractmethod
def train_dataset(self):
    """Creates a training dataset

    Returns:
        Dataset: dataset to use for training

    """
    raise NotImplementedError()
```

Next, let's implement a simple case to leverage the BioNeMo data module. A further example can be found in the [BioNeMo DataModule Example](./notebooks/protein-esm1nv-clustering.ipynb) notebook.

Let's make use of the BioNeMo DataModule to create a class that inherits the BioNeMo DataModule functions. Let's begin by importing the BioNeMo DataModule

```python
from bionemo.core import BioNeMoDataModule
```

Let's define a new class which will inherit `BioNeMoDataModule` and name it `FineTune`. Let's begin by creating the skeleton for the functions responsible for dealing with the _train_, _test_, and _validation_ sets.

```python
class FineTune(BioNeMoDataModule):
    def __init__(self, cfg, trainer):
        super().__init__(cfg, trainer) 
        # ...

    def train_dataset(self):
        # ...

    def val_dataset(self):
        # ...

    def test_dataset(self):
        # ...
```

:::{note}
Method names must match the method names in the abstract class.
::: 

We now have the basic building blocks of the DataModule and this will allow you to define core operations in the dataset. One can leverage the combination of other components from other libraries such as the `torch`, for example

```python
from torch.utils.data import Dataset
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, model, emb_batch_size, ...):
        self.model = model
        self.emb_batch_size = emb_batch_size
        #... 
    # ...
    def get_emb_batch_size(self):
        return self.emb_batch_size
    
    # ...
```

We can define the `train_dataset` function within the `FineTune` class as in the example below:

```python
class FineTune(BioNeMoDataModule):
    def __init__(self, cfg, trainer):
        super().__init__(cfg, trainer) 
        # ...

    def train_dataset(self):
        """Creates a training dataset
        Returns:
            Dataset: dataset to use for training
        """
        return DataSet(
            # ...
            model = self.model,
            emb_batch_size = self.cfg.emb_batch_size,
            # ...
            )

    def val_dataset(self):
        # ...

    def test_dataset(self):
        # ...

```

The same principle applies to the other functions. One can make customizations, add rules, exceptions and treatments to each one of them. 

For a full example, refer to the  [Generating embeddings for Protein Clustering](./notebooks/protein-esm1nv-clustering.ipynb) example notebook.
