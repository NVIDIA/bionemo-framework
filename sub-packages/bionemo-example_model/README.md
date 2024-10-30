## bionemo-example_model

#Introduction

This is a minimalist package containing an example model that makes use of bionemo2/nemo conventions. It contains the necessary models, dataloaders, datasets, and custom loss fucntions. The referenced classes and function are in `bionemo.example_model.lightning.lightning_basic`.

This tutorial demonstrates the creation of a simple MNIST model. This should be run in a bionemo contaner. For this tutorial, we will reuse elements from the bionemo-example_model package.


`Megatron` / `NeMo` modules and datasets are special derivatives of PyTorch modules and datasets that extend and accelerate the distributed training and inference capabilities of PyTorch.

Some distinctions of Megatron / NeMo are:

- `torch.nn.Module`/`LightningModule` changes into `MegatronModule`.
- Loss functions should extend the `MegatronLossReduction` module and implement a `reduce` method for aggregating loss across multiple micro-batches.
- Megatron configuration classes (e.g. `megatron.core.transformer.TransformerConfig`) are extended with a `configure_model` method that defines how model weights are initialized and loaded in a way that is compliant with training via NeMo2.
- Various modifications and extensions to common PyTorch classes, such as adding a `MegatronDataSampler` (and re-sampler such as `PRNGResampleDataset` or `MultiEpochDatasetResampler`) to your `LightningDataModule`.


# Loss Functions
First, we define a simple loss function. These should extend the MegatronLossReduction class. The output of forward and backward passes happen in parallel. There should be a forward function that calculates the loss defined. The reduce function is required.

Loss functions used here are MSELossReduction and ClassifierLossReduction. These functions return a Tensor, which contain the losses for the microbatches, and a SameSizeLossDict containing the average loss. This is a Typed Dictionary that is the return type for a loss that is computed for the entire batch, where all microbatches are the same size.

# Datasets and Datamodules
Datasets used for model training must be compatible with megatron datasets. To enable this, the ouput of a given index and epoch must be determensitic. However, we may with to have a different ordering in every epoch. To enable this, the items in the dataset should be accessible by both the epoch and the index. This can be done by accessing elements of the dataset with EpochIndex from bionemo.core.data.multi_epoch_dataset. A simple way of doing this is to wrap a dataset with IdentityMultiEpochDatasetWrapper imported from bionemo.core.data.multi_epoch_dataset. In this example, we use a custom dataset MNISTCustomDataset that wraps the getitem method of the MNIST dataset such that it return a dict instead of a Tuple or tensor.
