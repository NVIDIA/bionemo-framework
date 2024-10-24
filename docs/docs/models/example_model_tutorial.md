This tutorial demonstrates the creation of a simple Megatron model to classify MNIST digits within the BioNemo framework. This should be run in a bionemo contaner.

`Megatron` / `NeMo` modules and datasets are special derivatives of PyTorch modules and datasets that extend and accelerate the distributed training and inference capabilities of PyTorch.

Some distinctions of Megatron / NeMo are:

- `torch.nn.Module`/`LightningModule` changes into `MegatronModule`.
- Loss functions should extend the `MegatronLossReduction` module and implement a `reduce` method for aggregating loss across multiple micro-batches.
- Megatron configuration classes (e.g. `megatron.core.transformer.TransformerConfig`) are extended with a `configure_model` method that defines how model weights are initialized and loaded in a way that is compliant with training via NeMo2.
- Various modifications and extensions to common PyTorch classes, such as adding a `MegatronDataSampler` (and re-sampler such as `PRNGResampleDataset` or `MultiEpochDatasetResampler`) to your `LightningDataModule`.


Initially, we will define a models, losses, configs and other modules that are necessary for training. These can be put into a file (model_components.py)

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, Type, TypedDict, TypeVar

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from megatron.core import ModelParallelConfig
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from nemo.lightning import  io
from nemo.lightning.megatron_parallel import MegatronLossReduction
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.lightning.pytorch.plugins import MegatronDataSampler

from bionemo.core.data.resamplers import PRNGResampleDataset
from bionemo.llm.api import MegatronLossType
from bionemo.llm.lightning import LightningPassthroughPredictionMixin
from bionemo.llm.model.config import OVERRIDE_BIONEMO_CONFIG_DEFAULTS, MegatronBioNeMoTrainableModelConfig
from bionemo.llm.utils import iomixin_utils as iom
```

First, we define a simple loss function. These should inherit from losses in nemo.lightning.megatron_parallel and can inherit from MegatronLossReduction.  The output of forward and backwared passes happen in parallel. There should be a forward function that calculates the loss defined. The reduce function is required.
```python
class SameSizeLossDict(TypedDict):
    """This is the return type for a loss that is computed for the entire batch, where all microbatches are the same size."""

    avg: Tensor

class MSELossReduction(MegatronLossReduction):
    """A class used for calculating the loss, and for logging the reduced loss across micro batches."""

    def forward(self, batch: MnistItem, forward_out: Dict[str, Tensor]) -> Tuple[Tensor, SameSizeLossDict]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside LitAutoEncoder.

        Returns:
            A tuple containing [<loss_tensor>, ReductionT] where the loss tensor will be used for
                backpropagation and the ReductionT will be passed to the reduce method
        """
        x = batch["data"]
        x_hat = forward_out["x_hat"]
        xview = x.view(x.size(0), -1).to(x_hat.dtype)
        loss = nn.functional.mse_loss(x_hat, xview)

        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[SameSizeLossDict]) -> Tensor:
        """Works across micro-batches. (data on single gpu).
        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """
        mse_losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return mse_losses.mean()

class ClassifierLossReduction(MegatronLossReduction):
    """A class used for calculating the loss, and for logging the reduced loss across micro batches."""

    def forward(self, batch: MnistItem, forward_out: Tensor) -> Tuple[Tensor, SameSizeLossDict]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward
            forward_out: the output of the forward method

        Returns:
            A tuple containing [<loss_tensor>, ReductionT] where the loss tensor will be used for
                backpropagation and the ReductionT will be passed to the reduce method
                (which currently only works for logging.).
        """
        digits = batch["label"]
        digit_logits = forward_out
        loss = nn.functional.cross_entropy(digit_logits, digits)
        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[SameSizeLossDict]) -> Tensor:
        """Works across micro-batches. (data on single gpu).

        Note: This currently only works for logging and this loss will not be used for backpropagation.

        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """
        mse_losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return mse_losses.mean()
```

We define a wrapper for the MNIST dataset that returns a dictionary instead of a tuple or a tensor.
```python
class MnistItem(TypedDict):
    data: Tensor
    label: Tensor
    idx: int

class MNISTCustom(MNIST):
    def __getitem__(self, index: int) -> MnistItem:
        """Wraps the getitem method of the MNIST dataset such that we return a Dict
        instead of a Tuple or tensor.

        Args:
            index: The index we want to grab, an int.

        Returns:
            A dict containing the data ("x"), label ("y"), and index ("idx").
        """  # noqa: D205
        x, y = super().__getitem__(index)

        return {
            "data": x,
            "label": y,
            "idx": index,
        }
```
Datasets used for model training must be compatible with megatron datasets.
The dataset modules must have a data_sampler in it which is a nemo2 peculiarity. Also the sampler will not shuffle your data. So you need to wrap your dataset in a dataset shuffler that maps sequential ids to random ids in your dataset. This is what PRNGResampleDataset does. For further information, see: docs/user-guide/background/megatron_datasets.md. Moreover, the compatability of datasets with megatron can be checked by running bionemo-testing.megatron_dataset_compatibility.assert_dataset_compatible_with_megatron.

In the data module class, it's necessary to have data_sampler method to shuffle the data and that allows the sampler to be used with megatron. A nemo.lightning.pytorch.plugins.MegatronDataSampler is the best choice. It sets up the capability to utilize micro-batching and gradient accumulation. It is also the place where the global batch size is constructed.

```python
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 32, output_log = False) -> None:
        super().__init__()
        self.micro_batch_size = batch_size
        self.global_batch_size = batch_size
        self.max_len = 1048
        self.data_dir = data_dir

        # Wraps the datasampler with the MegatronDataSampler.
        self.data_sampler = MegatronDataSampler(
            seq_len=self.max_len,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=None,
            output_log = output_log
        )
    def setup(self, stage: str) -> None:
        """Sets up the datasets

        Args:
            stage: can be one of train / test / predict.
        """  # noqa: D415
        self.mnist_test = PRNGResampleDataset(
            MNISTCustom(self.data_dir, download=True, transform=transforms.ToTensor(), train=False), seed=43
        )
        mnist_full = MNISTCustom(self.data_dir, download=True, transform=transforms.ToTensor(), train=True)
        mnist_train, mnist_val = torch.utils.data.random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        self.mnist_train = PRNGResampleDataset(mnist_train, seed=44)
        self.mnist_val = PRNGResampleDataset(mnist_val, seed=45)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=self.micro_batch_size, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, batch_size=self.micro_batch_size, num_workers=0)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, batch_size=self.micro_batch_size, num_workers=0)
```

Models need to be megatron modules. At the most basic level this just means:
  1. They need a config argument of type megatron.core.ModelParallelConfig. An easy way of implementing this is to inherit from bionemo.llm.model.config.MegatronBioNeMoTrainableModelConfig. This is a class for bionemo that supports usage with Megatron models, as NeMo2 requires. This class also inherits ModelParallelConfig.
  2. They need a self.model_type:megatron.core.transformer.enums.ModelType enum defined (ModelType.encoder_or_decoder is probably usually fine)
  3. def set_input_tensor(self, input_tensor) needs to be present. This is used in model parallelism. This function can be a stub/ placeholder function.

Here, we define some models. ExampleModelTrunk is a base model. PretrainModel adds layers that enable pre-training. ExampleFineTuneModel is used for finetuning the previous model to classify the digits.

```python
class ExampleModelOutput(TypedDict):
    """Output for the example model implementation."""
    x_hat: Tensor
    z: Tensor


class ExampleModelTrunk(MegatronModule):
    def __init__(self, config: ModelParallelConfig) -> None:
        """Constructor of the model.

        Args:
            config: The config object is responsible for telling the strategy what model to create.
        """
        super().__init__(config)
        self.model_type: ModelType = ModelType.encoder_or_decoder
        self.linear1 = nn.Linear(28 * 28, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 3)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        z = self.linear1(x)
        z = self.relu(z)
        z = self.linear2(z)
        return z

    def set_input_tensor(self, input_tensor: Optional[Tensor]) -> None:
        pass


class PretrainModel(ExampleModelTrunk):
    def __init__(self, config: ModelParallelConfig) -> None:
        """Constructor of the model.

        Args:
            config: The config object is responsible for telling the strategy what model to create.
        """
        super().__init__(config)
        self.linear3 = nn.Linear(3, 64)
        self.relu2 = nn.ReLU()
        self.linear4 = nn.Linear(64, 28 * 28)

    def forward(self, x: Tensor) -> ExampleModelOutput:
        """Forward pass of the model.

        Args:
            x: The input data.

        Returns:
            x_hat: The result of the last linear layer of the network.
        """
        z: Tensor = super().forward(x)
        x_hat = self.linear3(z)
        x_hat = self.relu2(x_hat)
        x_hat = self.linear4(x_hat)
        return {"x_hat": x_hat, "z": z}

class ExampleFineTuneModel(ExampleModelTrunk):
    """Example of taking the example model and replacing output task."""

    def __init__(self, config: ModelParallelConfig):
        super().__init__(config)
        # 10 output digits, and use the latent output layer (z) for making predictions
        self.digit_classifier = nn.Linear(self.linear2.out_features, 10)

    def forward(self, x: Tensor) -> Tensor:
        z: Tensor = super().forward(x)
        digit_logits = self.digit_classifier(z)  # to demonstrate flexibility, in this case we return a tensor
        return digit_logits

```
The model config class is used to instatiate the model. These configs must have:
1. A configure_model method which allows the megatron strategy to lazily initialize the model after the parallel computing environment has been setup. These also handle loading starting weights for fine-tuning cases. Additionally these configs tell the trainer which loss you want to use with a matched model.
2. A get_loss_reduction_class method that defines the loss fucntion.
```python
# typevar for capturing subclasses of ExampleModelTrunk. Useful for Generic type hints as below.
ExampleModelT = TypeVar("ExampleModelT", bound=ExampleModelTrunk)


@dataclass
class ExampleGenericConfig(
    Generic[ExampleModelT, MegatronLossType], MegatronBioNeMoTrainableModelConfig[ExampleModelT, MegatronLossType]
):
    """ExampleConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

    loss_cls: Type[MegatronLossType] = MSELossReduction
    hidden_size: int = 64  # Needs to be set to avoid zero division error in megatron :(
    num_attention_heads: int = 1  # Needs to be set to avoid zero division error in megatron :(
    num_layers: int = 1  # Needs to be set to avoid zero division error in megatron :(
    override_parent_fields: List[str] = field(default_factory=lambda: OVERRIDE_BIONEMO_CONFIG_DEFAULTS + ["loss_cls"])

    def configure_model(self) -> ExampleModelT:
        """Uses model_cls and loss_cls to configure the model.

        Note: Must pass self into Model since model requires having a config object.

        Returns:
            The model object.
        """
        # 1. first load any settings that may exist in the checkpoint related to the model.
        if self.initial_ckpt_path:
            self.load_settings_from_checkpoint(self.initial_ckpt_path)
        # 2. then initialize the model
        model = self.model_cls(self)
        # 3. Load weights from the checkpoint into the model
        if self.initial_ckpt_path:
            self.update_model_from_checkpoint(model, self.initial_ckpt_path)
        return model

    def get_loss_reduction_class(self) -> Type[MegatronLossType]:
        """Use loss_cls to configure the loss, since we do not change the settings of the loss based on the config."""
        return self.loss_cls
```

These configs defines which model class to pair with which loss, since the abstractions around getting the model and loss are handled in the ExampleGenericConfig class.
```python
@dataclass
class PretrainConfig(ExampleGenericConfig["PretrainModel", "MSELossReduction"], iom.IOMixinWithGettersSetters):
    """PretrainConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

    model_cls: Type[PretrainModel] = PretrainModel
    loss_cls: Type[MSELossReduction] = MSELossReduction

@dataclass
class ExampleFineTuneConfig(
    ExampleGenericConfig["ExampleFineTuneModel", "ClassifierLossReduction"], iom.IOMixinWithGettersSetters
):
    """ExampleConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

```

It is helfpul to have a training module that interits pl.LightningModule which organizes the model architecture, training, validation, and testing logic while abstracting away boilerplate code, enabling easier and more scalable training.
This is a general training wrapper that can be re-used for all model/loss combos. In this example, training_step and predict_step define the training/prediction loop and are independent of the forward method.

This is some background on the training_step/predict_step.
1. NeMo's Strategy overrides this method.
2. The strategies' training step will call the forward method of the model.
3. That forward method then calls the wrapped forward step of MegatronParallel which wraps the forward method of the model.
4. That wrapped forward step is then executed inside the Mcore scheduler, which calls the `_forward_step` method from the MegatronParallel class.
5. Which then calls the training_step function here.

```python
class BionemoLightningModule(pl.LightningModule, io.IOMixin, LightningPassthroughPredictionMixin):
    """A very basic lightning module for the megatron strategy and the megatron-nemo2-bionemo contract."""

    def __init__(self, config: MegatronBioNeMoTrainableModelConfig):
        """Initializes the model.

        Args:
            config: a Config object necessary to construct the actual nn.Module (the thing that has the parameters).
        """
        super().__init__()
        self.config = config
        self.optim = MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=1e-4,
                optimizer="adam",
                use_distributed_optimizer=True,
                bf16=config.bf16,
                fp16=config.fp16,
                params_dtype=config.params_dtype,
            ),
        )
        # Bind the configure_optimizers method to the model
        self.optim.connect(self)

    def forward(self, batch: Dict, batch_idx: int) -> Any:
        """This forward will be called by the megatron scheduler and it will be wrapped.

        Args:
            batch: A dictionary of data.
            batch_idx: The index of the batch.

        Returns:
            The output of the model.
        """
        x = batch["data"]
        return self.module(x)

    def training_step(self, batch, batch_idx: Optional[int] = None):
        """The training step is where the loss is calculated and the backpropagation is done.

        Args:
            batch: A dictionary of data. requires `batch_idx` as default None.
            batch_idx: The index of the batch.
        """
        return self(batch, batch_idx)

    def predict_step(self, batch, batch_idx: Optional[int] = None):
        """Alias for forward_step."""
        return self(batch, batch_idx)

    def training_loss_reduction(self) -> MegatronLossReduction:
        # This is the function that takes batch['loss_mask'] and the logits output by the model and reduces the loss
        return self.loss_reduction_class()()

    def validation_loss_reduction(self) -> MegatronLossReduction:
        return self.loss_reduction_class()()

    def test_loss_reduction(self) -> MegatronLossReduction:
        return self.loss_reduction_class()()

    def configure_model(self) -> None:  # noqa: D102
        # Called lazily by the megatron strategy.
        self.module = self.config.configure_model()

    def loss_reduction_class(self) -> Type[MegatronLossReduction]:
        """Get the loss reduction class the user has specified in their config."""
        return self.config.get_loss_reduction_class()
```

It is useful to have a pytorch lightning callback defined to track the metric. A simple example is here:
```python
class MetricTracker(pl.Callback):
    def __init__(self, metrics_to_track_val: List[str], metrics_to_track_train: List[str]):
        self.metrics_to_track_val = metrics_to_track_val
        self.metrics_to_track_train = metrics_to_track_train
        self._collection_val = defaultdict(list)
        self._collection_train = defaultdict(list)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(outputs, torch.Tensor):
            self._collection_val["unnamed"].append(outputs)
        else:
            for metric in self.metrics_to_track_val:
                self._collection_val[metric].append(outputs[metric])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(outputs, torch.Tensor):
            self._collection_train["unnamed"].append(outputs)
        else:
            for metric in self.metrics_to_track_train:
                self._collection_train[metric].append(outputs[metric])

    def on_validation_epoch_end(self, trainer, pl_module):
        elogs = trainer.logged_metrics  # access it here
        self._collection_val["logged_metrics"].extend(elogs)

    def on_train_epoch_end(self, trainer, pl_module):
        elogs = trainer.logged_metrics  # access it here
        self._collection_train["logged_metrics"].extend(elogs)

    @property
    def collection_val(self) -> Dict[str, torch.Tensor | List[str]]:
        res = {k: torch.tensor(v) for k, v in self._collection_val.items() if k != "logged_metrics"}
        res["logged_metrics"] = self._collection_val["logged_metrics"]
        return res

    @property
    def collection_train(self) -> Dict[str, torch.Tensor | str]:
        res = {k: torch.tensor(v) for k, v in self._collection_train.items() if k != "logged_metrics"}
        res["logged_metrics"] = self._collection_train["logged_metrics"]
        return res
```

Now, we can run pre-training on MNIST. For each of the training and testing steps, create a sepereate python script.
Create a new file name (model_pretrain.py)

```python
import pytorch_lightning as pl
import tempfile
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import NeMoLogger, resume
from nemo.lightning.pytorch import callbacks as nl_callbacks

from bionemo.core import BIONEMO_CACHE_DIR
from bionemo.llm.utils import iomixin_utils as iom

from pytorch_lightning.loggers import TensorBoardLogger

from model_components import MNISTDataModule, BionemoLightningModule, PretrainConfig, MetricTracker
```
First, this creates the callbacks for model checkpoints.

```python
checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_last=True,
        save_on_train_epoch_end=True,
        monitor="reduced_train_loss",
        every_n_train_steps=25,
        always_save_context=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
)
```

This sets up the logger, the data module and the lightning module.
```python
temp_dir = tempfile.TemporaryDirectory()
save_dir = temp_dir/"pretrain"
name = "example"
# Setup the logger train the model
nemo_logger = NeMoLogger(
    log_dir=str(save_dir),
    name=name,
    tensorboard=TensorBoardLogger(save_dir=save_dir, name=name),
    ckpt=checkpoint_callback,
)
#Set up the data module
data_module = MNISTDataModule(data_dir=str(BIONEMO_CACHE_DIR), batch_size=128)

#Set up the training module
lightning_module = BionemoLightningModule(
    config=PretrainConfig()
)
```

Next, the megatron training strategy is specified.
```python
strategy = nl.MegatronStrategy(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    ddp="megatron",
    find_unused_parameters=True,
    always_save_context=True,
)
```

Then, set up the tracker for metrics and specify the trainer.
```python
metric_tracker = MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"])

trainer = nl.Trainer(
    accelerator="gpu",
    devices=1,
    strategy=strategy,
    limit_val_batches=5,
    val_check_interval=25,
    max_steps=500,
    max_epochs=10,
    num_nodes=1,
    log_every_n_steps=25,
    callbacks=[metric_tracker],
    plugins=nl.MegatronMixedPrecision(precision="bf16-mixed")
)
```

Next, the model is trained with these components.
```python
#This trains the model
llm.train(
        model=lightning_module,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )
```

We can view the results and look at the last created model that is checkpointed.
```python
pretrain_ckpt_dirpath = checkpoint_callback.last_model_path.replace(".ckpt", "")
print(metric_tracker.collection_train['loss'])
print(metric_tracker.collection_val['logged_metrics'])
print(pretrain_ckpt_dirpath)
```

Next, we will finetune this model as a classification task. Create a new file. Swap out the logger, and training module from the last step, then train the model. In this example, there is no digit_classifier output in the previous model but there is in this model. So we set initial_ckpt_skip_keys_with_these_prefixes to {"digit_classifier"} in the training module. Set the checkpoint in the config to be the pretrain_ckpt_dirpath obtained when running the last file. Then, we train the model.
```python
save_dir = temp_dir/"classifier"

nemo_logger2 = NeMoLogger(
    log_dir=str(save_dir),
    name=name,
    tensorboard=TensorBoardLogger(save_dir=save_dir, name=name),
    ckpt=checkpoint_callback,
)

lightning_module2 = BionemoLightningModule(
    config=ExampleFineTuneConfig(
        initial_ckpt_path=pretrain_ckpt_dirpath,
        initial_ckpt_skip_keys_with_these_prefixes={"digit_classifier"}
    )
)

llm.train(
        model=lightning_module2,
        data=data_module,
        trainer=trainer,
        log=nemo_logger2,
        resume=resume.AutoResume(
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
        ),
    )
finetune_dir = Path(checkpoint_callback.last_model_path.replace(".ckpt", "")
print(finetune_dir)
```

Next, we can change run the model on the test data. In a seperate file, copy or import the relevant imports, classes. Swap out the trainer, lightning module and data module from the previous file. Set the checkpoint in the config to be the finetune directory obtained when running the last file. Then, the results are obtained by running predict on the trainer.

```python

test_run_trainer = nl.Trainer(
    accelerator="gpu",
    devices=1,
    strategy=strategy,
    num_nodes=1,
    plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    callbacks=None,
)

lightning_module3 = BionemoLightningModule(
    config=ExampleFineTuneConfig(
        initial_ckpt_path=finetune_dir)
    )
)
new_data_module = MNISTDataModule(
    data_dir=str(BIONEMO_CACHE_DIR),
    batch_size=len(data_module.mnist_test),
    output_log=False
)

results = test_run_trainer.predict(lightning_module3, datamodule=new_data_module)
```
