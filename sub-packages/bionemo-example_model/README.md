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

Datasets used for model training must be compatible with megatron datasets. To enable this, the ouput of a given index and epoch must be determensitic. However, we may wish to have a different ordering in every epoch. To enable this, the items in the dataset should be accessible by both the epoch and the index. This can be done by accessing elements of the dataset with EpochIndex from bionemo.core.data.multi_epoch_dataset. A simple way of doing this is to wrap a dataset with IdentityMultiEpochDatasetWrapper imported from bionemo.core.data.multi_epoch_dataset. In this example, we use a custom dataset MNISTCustomDataset that wraps the getitem method of the MNIST dataset such that it return a dict instead of a Tuple or tensor. The MNISTCustomDataset returns elements of type MnistItem, which is a TypedDict.


In the data module/ data loader class, it's necessary to have data_sampler method to shuffle the data and that allows the sampler to be used with megatron. This is a nemo2 peculiarity. A nemo.lightning.pytorch.plugins.MegatronDataSampler is the best choice. It sets up the capability to utilize micro-batching and gradient accumulation. It is also the place where the global batch size is constructed.

Also the sampler will not shuffle your data. So you need to wrap your dataset in a dataset shuffler that maps sequential ids to random ids in your dataset. This can be done with MultiEpochDatasetResampler from bionemo.core.data.multi_epoch_dataset.


This is implemented in the MNISTDataModule. In the setup method of the dataloader, the train, test and validation sets are MNISTCustomDataset are wrapped in the IdentityMultiEpochDatasetWrapper. These are then wrapped in the MultiEpochDatasetResampler. More information about MegatronCompatability and how to set up more complicated datasets can be found in docs.user-guide.background.megatron_datasets.md


We also define a train_dataloader, val_dataloader, and predict_dataloder methods that return the corresponding dataloaders.

# Models

Models need to be megatron modules. At the most basic level this just means:
  1. They extend MegatronModule from megatron.core.transformer.module.
  2. They need a config argument of type megatron.core.ModelParallelConfig. An easy way of implementing this is to inherit from bionemo.llm.model.config.MegatronBioNeMoTrainableModelConfig. This is a class for bionemo that supports usage with Megatron models, as NeMo2 requires. This class also inherits ModelParallelConfig.
  3. They need a self.model_type:megatron.core.transformer.enums.ModelType enum defined (ModelType.encoder_or_decoder is probably usually fine)
  4. def set_input_tensor(self, input_tensor) needs to be present. This is used in model parallelism. This function can be a stub/ placeholder function.

ExampleModelTrunk is a base model. This returns a tensor. ExampleModel is a model that extends the base model with a few linear layers and it's used for pretrainining. This returns the output of the base model and of the full model.

ExampleFineTuneModel extends the ExampleModelTrunk by adding a classification layer. This returns a tensor of logits over the 10 potential digits.

# Model Configs

The model config class is used to instatiate the model. These configs must have:
1. A configure_model method which allows the megatron strategy to lazily initialize the model after the parallel computing environment has been setup. These also handle loading starting weights for fine-tuning cases. Additionally these configs tell the trainer which loss you want to use with a matched model.
2. A get_loss_reduction_class method that defines the loss function.

Here, a base generic config ExampleGenericConfig is defined.  PretrainConfig extends this class. This defines the model class and the loss class in:
```
class PretrainConfig(ExampleGenericConfig["PretrainModel", "MSELossReduction"], iom.IOMixinWithGettersSetters):

    model_cls: Type[PretrainModel] = PretrainModel
    loss_cls: Type[MSELossReduction] = MSELossReduction

```

Similarly, ExampleFineTuneConfig extends ExampleGenericConfig for finetuning.

# Training Module
It is helfpul to have a training module that interits pl.LightningModule which organizes the model architecture, training, validation, and testing logic while abstracting away boilerplate code, enabling easier and more scalable training. This wrapper can be used for all model and loss combinations specified in the config.
Here, we define BionemoLightningModule.

In this example, training_step and predict_step define the training, validation, and prediction loops are independent of the forward method. In nemo:

1. NeMo's Strategy overrides the train_step, validation_step and prediction_step methods.
2. The strategies' training step will call the forward method of the model.
3. That forward method then calls the wrapped forward step of MegatronParallel which wraps the forward method of the model.
4. That wrapped forward step is then executed inside the Mcore scheduler, which calls the `_forward_step` method from the MegatronParallel class.
5. Which then calls the training_step, validation_step and prediction_step function here.

Additionally, during these steps, we log the validation, testing, and training loss. This is done similarly to https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html. These logs can then be exported to wandb, or other metric viewers. For more complicated tracking, it may be necessary to use pytorch callbacks: https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html.

Further a loss_reduction_class, training_loss_reduction, validation_loss_reduction, and test_loss_reduction are defined based on what's in the config. Additionally,  configure_model is definated based on the config.
