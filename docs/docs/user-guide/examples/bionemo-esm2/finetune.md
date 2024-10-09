# ESM2 Fine-Tuning

This readme serves as a demo for implementing ESM2 Fine-tuning module, running a regression example and using the model for inference.

The ESM2 model is a transformer-based protein language model that has achieved state-of-the-art results in various protein-related tasks. The task head plays a crucial role in fine-tuning for a downstream task. As a part of transfer learning, a pre-trained model is often utilized to learn generic features from a large-scale dataset. However, these features might not be directly applicable to the specific task at hand. By incorporating a task head, which consists of learnable parameters, the model can adapt and specialize to the target task. The task head serves as a flexible and adaptable component that learns task-specific representations by leveraging the pre-trained features as a foundation. Through fine-tuning, the task head enables the model to learn and extract task-specific patterns, improving performance and addressing the nuances of the downstream task. It acts as a critical bridge between the pre-trained model and the specific task, enabling efficient and effective transfer of knowledge.


# Setup and Assumptions

In this tutorial, we will demonstrate how to create a fine-tune module, train a regression task head, and use the fine-tuned model for inference.

All commands should be executed inside the BioNeMo docker container, which has all ESM2 dependencies pre-installed. This tutorial assumes that a copy of the BioNeMo framework repo exists on workstation or server and has been mounted inside the container at `/workspace/bionemo2`. (**Note**: This `WORKDIR` may be `/workspaces/bionemo-framework` if you are using the VSCode Dev Container.) For more information on how to build or pull the BioNeMo2 container, refer to the [BioNeMo2 README]({{ github_url }})!

To successfully accomplish this we need to define some key classes:

1. Loss Reduction Method - To compute the supervised fine-tuning loss.
2. Fine-Tuned Model Head - Downstream task head model.
3. Fine-Tuned Model - Model that combines ESM2 with the task head model.
4. Fine-Tuning Config - Configures the fine-tuning model and loss to use in the training and inference framework.
5. Dataset - Training and inference datasets for ESM2.

## 1 - Loss Reduction Class

A class for calculating the supervised loss of the fine-tune model from targets. We inherit from Megatron Bert Masked Language Model Loss (`BERTMLMLossWithReduction`) and override the `forward()` pass to compute MSE loss of the regression head within a micro-batch. The `reduce()` method is used for computing the average over the micro-batches and is only used for logging.

```python
class RegressorLossReduction(BERTMLMLossWithReduction):
    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Union[PerTokenLossDict, SameSizeLossDict]]:

        targets = batch["labels"]  # [b, 1]
        regression_output = forward_out
        loss = torch.nn.functional.mse_loss(regression_output, targets)
        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[ReductionT]) -> torch.Tensor:
        losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return losses.mean()
```

## 2 - Fine-Tuned Model Head

An MLP class for sequence-level regression. This class inherits `MegatronModule` and uses the fine-tune config (`TransformerConfig`) to configure the regression head for the fine-tuned ESM2 model.

```python
class MegatronMLPHead(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        layer_sizes = [config.hidden_size, 256, 1]
        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])]
        )
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=config.ft_dropout)

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        ...
```

## 3 - Fine-Tuned Model

A fine-tuned ESM2 model class for token classification tasks. This class inherits from the `ESM2Model` class and adds the custom regression head `MegatronMLPHead` the we created in the previous step. Optionally one can freeze all or parts of the encoder by parsing through the model parameters in the model constructor.

```python
class ESM2FineTuneSeqModel(ESM2Model):
    def __init__(self, config, *args, post_process: bool = True, return_embeddings: bool = False, **kwargs):
        super().__init__(config, *args, post_process=post_process, return_embeddings=True, **kwargs)

        # freeze encoder parameters
        if config.encoder_frozen:
            for _, param in self.named_parameters():
                param.requires_grad = False

        if post_process:
            self.regression_head = MegatronMLPHead(config)

    def forward(self, *args, **kwargs,):
        output = super().forward(*args, **kwargs)
        ...
        regression_output = self.regression_head(embeddings)
        return regression_output
```

## 4 - Fine-Tuning Config

A `dataclass` that configures the fine-tuned ESM2 model. In this example `ESM2FineTuneSeqConfig` inherits from `ESM2GenericConfig` and adds custom arguments to setup the fine-tuned model. The `configure_model()` method of this `dataclass` is called within the `Lightning` module to call the model constructor with the `dataclass` arguments.

The common arguments among different fine-tuning tasks are

- `model_cls`: The fine-tune model class (`ESM2FineTuneSeqModel`)
- `initial_ckpt_path`: BioNeMo 2.0 ESM2 pre-trained checkpoint
- `initial_ckpt_skip_keys_with_these_prefixes`: skip keys when loading parameters from a checkpoint. Here we should not look for `regression_head` in the pre-trained checkpoint.
- `get_loss_reduction_class()`: Implements selection of the appropriate `MegatronLossReduction` class, e.g. `bionemo.esm2.model.finetune.finetune_regressor.RegressorLossReduction`.

```python

@dataclass
class ESM2FineTuneSeqConfig(ESM2GenericConfig[ESM2FineTuneSeqModel], iom.IOMixinWithGettersSetters):
    model_cls: Type[ESM2FineTuneSeqModel] = ESM2FineTuneSeqModel
    # The following checkpoint path is for nemo2 checkpoints. Config parameters not present in
    # self.override_parent_fields will be loaded from the checkpoint and override those values here.
    initial_ckpt_path: str | None = None
    # typical case is fine-tune the base biobert that doesn't have this head. If you are instead loading a checkpoint
    # that has this new head and want to keep using these weights, please drop this next line or set to []
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=lambda: ["regression_head"])

    encoder_frozen: bool = True  # freeze encoder parameters
    ft_dropout: float = 0.25  # MLP layer dropout

    def get_loss_reduction_class(self) -> Type[MegatronLossReduction]:
        return RegressorLossReduction
```

## 5 - Dataset

We will use a sample dataset for demonstration purposes. Create a dataset class by extending from ```torch.utils.data.Dataset```. For the purposes of this demo, we'll assume dataset consists of protein sequences with a target value of `len(sequence) / 100.0` as their labels.

```python
data = [
    ("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERH", 0.33),
    ...
]
```

Therefore, the custom BioNeMo dataset class will be appropriate (found in ```bionemo.esm2.model.finetune.finetune_regressor.InMemorySingleValueDataset```) as it facilitates predicting on a single value. An excerpt from the class is shown below. This example dataset expected a sequence of `Tuple` that hold `(sequence, target)` values. However, one can simply extend ```InMemorySingleValueDataset``` class in a similar way to customize your class for your data.

```python
class InMemorySingleValueDataset(Dataset):
    def __init__(
        self,
        data: Sequence[Tuple[str, float]],
        tokenizer: tokenizer.BioNeMoESMTokenizer = tokenizer.get_tokenizer(),
        seed: int = np.random.SeedSequence().entropy,
    ):
```

To coordinate the creation of training, validation and testing datasets from your data, we need to use a `datamodule` class. To do this we can directly use or extend the ```ESM2FineTuneDataModule``` class (located at ```bionemo.esm2.model.finetune.datamodule.ESM2FineTuneDataModule```) which defines helpful abstract methods that use your dataset class.

```python
dataset = InMemorySingleValueDataset(data)
data_module = ESM2FineTuneDataModule(predict_dataset=dataset)
```

# Fine-Tuning the Regressor Task Head for ESM2

Now we can put these five requirements together to fine-tune a regressor task head starting from a pre-trained ESM2 model (`pretrain_ckpt_path`). We can take advantage of a simple training loop in ```bionemo.esm2.model.fnetune.train``` and use the ```train_model()`` function to start the fine-tuning process in the following.

```python
# create a List[Tuple] with (sequence, target) values
artificial_sequence_data = [
    "TLILGWSDKLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
    "LYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
    "GRFNVWLGGNESKIRQVLKAVKEIGVSPTLFAVYEKN",
    "DELTALGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
    "KLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
    "LFGAIGNAISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP",
    "LGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
    "LYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
    "ISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP",
    "SGSKASSDSQDANQCCTSCEDNAPATSYCVECSEPLCETCVEAHQRVKYTKDHTVRSTGPAKT",
]

data = [(seq, len(seq)/100.0) for seq in artificial_sequence_data]

# we are training and validating on the same dataset for simplicity
dataset = InMemorySingleValueDataset(data)
data_module = ESM2FineTuneDataModule(train_dataset=dataset, valid_dataset=dataset)

experiment_tempdir = tempfile.TemporaryDirectory()
experiment_dir = Path(experiment_tempdir.name)
experiment_name = 'finetune_regressor'
n_steps_train = 50

# Set ESM2 fine-tuning config with encoder_frozen=True.
# To load a pre-trained model, set initial_ckpt_path=str(pretrain_ckpt_path).
pretrain_ckpt_path = '/data/esm2_650M'
config = ESM2FineTuneSeqConfig(initial_ckpt_path=str(pretrain_ckpt_path))

checkpoint, metrics, trainer = train_model(
    experiment_name=experiment_name,
    experiment_dir=experiment_dir,  # new checkpoint will land in a subdir of this
    config=config,  # same config as before since we are just continuing training
    data_module=data_module,
    n_steps_train=n_steps_train,
)
```

This example is fully implemented in ```bionemo.esm2.model.finetune.train``` and can be executed by:
```bash
python /workspace/bionemo2/sub-packages/bionemo-esm2/src/bionemo/esm2/model/finetune/train.py
```

# Fine-Tuned ESM2 Model Inference
Once we have a checkpoint we can create a config object by pointing the path in `initial_ckpt_path` and use that for inference. Since we need to load all the parameters from this checkpoint (and don't skip the head) we reset the `nitial_ckpt_skip_keys_with_these_prefixes` in this config. Now we can use the ```bionemo.esm2.model.fnetune.train.infer``` to run inference on prediction dataset.

```python
config = ESM2FineTuneSeqConfig(
    initial_ckpt_path = finetuned_checkpoint,
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=list)
)

results = infer_model(config, data_module)
```

To download a pre-trained ESM2 to run this inference script, run the command:
```
download_bionemo_data esm2/650m:2.0 --source ngc
```
and pass the path (e.g. `.../.cache/bionemo/975d29ee980fcb08c97401bbdfdcf8ce-esm2_650M_nemo2.tar.gz.untar`) as an argument into `initial_ckpt_path` above!

# Parameter-Efficient Fine-Tuning (PEFT) of ESM2

## PEFT Model Training

To fine-tune the pre-trained ESM2 regressor task head using PEFT, NeMo2 has abstracted tooling that converts vanilla model training into PEFT. For more information about LoRA fine-tuning, refer to the Appendix of this section.

To enable LoRA fine-tuning via NeMo, we define a `model_transform` for ESM2:
```python
# src/bionemo/esm2/model/finetune/peft.py
from nemo.collections.llm import fn
from nemo.collections.llm.fn.mixin import FNMixin
from nemo.collections.llm.peft.lora import LoRA
from torch import nn


class ESM2LoRA(LoRA):
    """LoRA for the BioNeMo2 ESM Model."""

    def __call__(self, model: nn.Module) -> nn.Module:
        fn.walk(model, self.selective_freeze)
        fn.walk(model, self.transform)
        return model

    def selective_freeze(self, m: nn.Module, name=None, prefix=None):
        if name in ["encoder", "embedding"]:
            FNMixin.freeze(m)
        return m
```

and make the following modifications to relevant sections of `train.py`:

```python
# src/bionemo/esm2/model/finetune/train.py
from nemo.lightning.pytorch.callbacks.model_transform import ModelTransform
from nemo.lightning.pytorch.callbacks.peft import PEFT
from bionemo.esm2.model.finetune.peft import ESM2LoRA

# Set a parameter-efficient fine-tuning strategy (PEFT).
# Options: None, ESM2LoRA()
peft = ESM2LoRA()

# Pass the ESM2LoRA model transform into the LightningModule.
module = BioBertLightningModule(config=config, tokenizer=tokenizer, optimizer=optimizer, model_transform=peft)

# Add a ESM2LoRA() callback to the PyTorch Lightning Trainer.
if peft is not None:
    callbacks.append(peft)
trainer = nl.Trainer(
    ...
    callbacks=callbacks
)
```
This allocates trainable LoRA adapter weights to all non-frozen weights in ESM2, and trains the LoRA adapter instead of base ESM2. (Specifically, because we froze our encoder in `ESM2FineTuneSeqConfig`, adapters will not be trained or applied to the encoder, and only the regressor task heads will be adapted and trained.)

## PEFT Model Inference

Currently, BioNeMo2 & NeMo2 do not support model loading and inference with PEFT (e.g. LoRA). (Thus, `inference.py` only works for non-PEFT model checkpoints!) This feature will be released soon!

## Appendix

### Low-Rank Adaptation

**Low-Rank Adaptation (LoRA)** is a parameter-efficient strategy designed to adapt large pretrained language models to downstream tasks while avoiding challenges associated with full fine-tuning. Unlike traditional fine-tuning approaches that adjust all parameters within a pretrained model, LoRA maintains the core weights of the pretrained model as frozen. Instead, it introduces trainable rank decomposition matrices, known as LoRA adapters, into each layer of the Transformer architecture. These adapters are smaller matrices that approximate the original weight matrices, thereby reducing the number of trainable parameters.

In the context of antibody sequences, where data availability may be limited, LoRA offers several advantages. By focusing on adapting these smaller adapter matrices rather than the entire model, LoRA makes fine-tuning more efficient and less susceptible to overfitting. This is particularly beneficial for tasks requiring adaptation to specific protein sequences, where preserving the learned features of the pretrained ESM-2nv model is crucial.

By integrating LoRA into BioNeMo's fine-tuning pipeline for ESM-2nv models, you can leverage the robustness of pretrained models while tailoring them to the unique characteristics of antibody sequences. This extension not only enhances model performance but also ensures adaptability and efficiency in handling specialized protein sequence data.
