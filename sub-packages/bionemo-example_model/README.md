# bionemo-example_model

This is a minimalist package containing an example model that makes use of bionemo2/nemo conventions. It contains the necessary models, dataloaders, datasets, and custom loss fucntions.

This tutorial demonstrates the creation of a simple MNIST model. This should be run in a bionemo contaner. For this tutorial, we will reuse elements from the bionemo-example_model package.


`Megatron` / `NeMo` modules and datasets are special derivatives of PyTorch modules and datasets that extend and accelerate the distributed training and inference capabilities of PyTorch.

Some distinctions of Megatron / NeMo are:

- `torch.nn.Module`/`LightningModule` changes into `MegatronModule`.
- Loss functions should extend the `MegatronLossReduction` module and implement a `reduce` method for aggregating loss across multiple micro-batches.
- Megatron configuration classes (e.g. `megatron.core.transformer.TransformerConfig`) are extended with a `configure_model` method that defines how model weights are initialized and loaded in a way that is compliant with training via NeMo2.
- Various modifications and extensions to common PyTorch classes, such as adding a `MegatronDataSampler` (and re-sampler such as `PRNGResampleDataset` or `MultiEpochDatasetResampler`) to your `LightningDataModule`.



First, we define a simple loss function. These should inherit from losses in nemo.lightning.megatron_parallel and can inherit from MegatronLossReduction. The output of forward and backward passes happen in parallel. There should be a forward function that calculates the loss defined. The reduce function is required.

Loss functions we will use can be imported with:
```
from bionemo.example_model.lightning_basic import MSELossReduction, ClassifierLossReduction
```

Datasets used for model training must be compatible with megatron datasets. To enable this, the ouput of a given index and epoch must be determensitic. However, we may with to have a different ordering in every epoch. To enable this, the datasets should be accessible by both the epoch and the index. This can be done by accessing elements of the dataset in with EpochIndex from bionemo.core.data.multi_epoch_dataset.
An example is

```
from bionemo.example_model.lightning_basic import MNISTCustomDataset
```
