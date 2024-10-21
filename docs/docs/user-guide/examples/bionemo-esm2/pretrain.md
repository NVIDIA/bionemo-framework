# ESM2 Pretraining

This tutorial serves as a demo for pretraining [ESM2](https://www.science.org/doi/abs/10.1126/science.ade2574) from scratch with [UniProt](https://www.uniprot.org/) sequences.

The ESM2 model is a transformer-based protein language model that was pretrained on masked language model (MLM) task. The objective is to recover the original amino acid types of the perturbed locations from the rest of the protein sequences. Through pretraining, ESM2 learns the evolutionary information in protein sequences similar to conservation analysis and Pott's model, and predicts the optimal mutations on any given protein sequence.

# Setup and Assumptions

In this tutorial, we will demonstrate how to create an ESM2 pretraining data module, and create and train a ESM2 model.

All commands should be executed inside the BioNeMo docker container, which has all ESM2 dependencies pre-installed. This tutorial assumes that a copy of the BioNeMo framework repo exists on workstation or server and has been mounted inside the container at `/workspace/bionemo2`.  For more information on how to build or pull the BioNeMo2 container, refer to the [Initialization Guide](../../getting-started/initialization-guide.md).

!!! note

    This `WORKDIR` may be `/workspaces/bionemo-framework` if you are using the VSCode Dev Container.

Similar to PyTorch Lightning, we have to define some key classes:

1. `MegatronStrategy` - To launch and setup parallelism for [NeMo](https://github.com/NVIDIA/NeMo/tree/main) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
2. `Trainer` - To configure training configurations and logging.
3. `ESMDataModule` - To load pretraining training and validation data with mapped UniRef90 sequences to UniRef50 clusters.
4. `ESM2Config` - To configure the ESM2 model as `BionemoLightningModule`.

## 1 - MegatronStrategy
BioNeMo2 supports data parallel (DP), tensor parallel (TP) and pipeline parallel (PP) for training large models. Instead of `DDPStrategy` in PyTorch Lightning, we use `MegatronStrategy` to launch and setup parallelism for NeMo and Megatron-LM.

```python
from nemo import lightning as nl
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size

micro_batch_size = 2
num_nodes = 1
devices = 2
accumulate_grad_batches = 1
tensor_model_parallel_size = 2
pipeline_model_parallel_size = 1

global_batch_size = infer_global_batch_size(
    micro_batch_size=micro_batch_size,
    num_nodes=num_nodes,
    devices=devices,
    accumulate_grad_batches=accumulate_grad_batches,
    tensor_model_parallel_size=tensor_model_parallel_size,
    pipeline_model_parallel_size=pipeline_model_parallel_size,
)

strategy = nl.MegatronStrategy(
    tensor_model_parallel_size=tensor_model_parallel_size,
    pipeline_model_parallel_size=pipeline_model_parallel_size,
    ddp="megatron",
    find_unused_parameters=True,
    ckpt_include_optimizer=True,
)
```

## 2 - Trainer
BioNeMo2 trainer is very similar to PyTorch Lightning trainer. We can configure the training configurations and logging.

```python
from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary
from bionemo.llm.lightning import PerplexityLoggingCallback

num_steps = 20
limit_val_batches = 2  # limit the validation epoch to 2 batches
val_check_interval = 10  # validation epoch every 10 steps
precision = "bf16-mixed"  # use bf16-mixed precision

trainer = nl.Trainer(
    devices=devices,
    max_steps=num_steps,
    accelerator="gpu",
    strategy=strategy,
    limit_val_batches=limit_val_batches,
    val_check_interval=val_check_interval,
    num_nodes=num_nodes,
    callbacks=[
        PerplexityLoggingCallback(),
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
    ],
    plugins=nl.MegatronMixedPrecision(precision=precision),  # precision is handled through plugins in BioNeMo2
)
```

Here are examples of other possible configurations.
```python
from bionemo.core.utils.dtypes import PrecisionTypes

limit_val_batches_all_data = 1.  # validate on 100% of the validation dataset
limit_val_batches_half_data = 0.5  # validate on 50% of the validation dataset
limit_val_batches_one_batch = 1  # validate on 1 batch

print(PrecisionTypes)  # show all possible precision types
```

## 3 - ESMDataModule
Before instantiating with data module, we can first download the testing ESM2 pretraining data with `download_bionemo_data`. The command line will download the data if we haven't yet, and will return the path to the testing data, which is needed to instantiate `ESMDataModule`.

```bash
download_bionemo_data esm2/testdata_esm2_pretrain:2.0 --source ngc  # test data
# download_bionemo_data esm2/fulldata_esm2_pretrain:2.0 --source ngc  # full data (~80GB)
```

On top of the path to the data directory, BioNeMo2 data module requires global and micro batch sizes to ensure that the input tensors are initialized correctly across model-parallel ranks (see [Megatron Dataset Considerations](../../background/megatron_datasets.md)).

```python
from bionemo.esm2.data.datamodule import ESMDataModule
from bionemo.esm2.data.dataset import RandomMaskStrategy
from bionemo.esm2.data.tokenizer import get_tokenizer

data_path = __your_downloaded_test_data_path__  # fill your path from the command line output

train_cluster_path = f"{data_path}/2024_03_sanity/train_clusters_sanity.parquet"
train_database_path = f"{data_path}/2024_03_sanity/train_sanity.db"
valid_cluster_path = f"{data_path}/2024_03_sanity/valid_clusters.parquet"
valid_database_path = f"{data_path}/2024_03_sanity/validation.db"

min_seq_length = None  # optional; filter sequences by minimum length if given
max_seq_length = 128  # required; default to 1024

num_dataset_workers = 1
random_mask_strategy = RandomMaskStrategy.ALL_TOKENS  # default in BioNemo2 and HuggingFace implementation

data = ESMDataModule(
    train_cluster_path=train_cluster_path,  # UniRef50 training cluster centers
    train_database_path=train_database_path,  # UniRef90 training sequences
    valid_cluster_path=valid_cluster_path,  # UniRef50 validation cluster centers
    valid_database_path=valid_database_path,  # UniRef90 validation sequences
    global_batch_size=global_batch_size,
    micro_batch_size=micro_batch_size,
    min_seq_length=min_seq_length,
    max_seq_length=max_seq_length,
    num_workers=num_dataset_workers,
    random_mask_strategy=random_mask_strategy,
)
```

!!! note "`RandomMaskStrategy`"

    When trained on MLM objective, the loss function randomly includes 15% of the tokens, within which 80% are masked, 10% are replaced with a random token, and 10% are kept unchanged. Since the vocabulary includes amino acids as well as special tokens, part of the protein sequence may be replaced by a special token. This is the default in both BioNeMo2 and HuggingFace ESM2 implementation.

    To enforce amino-acid-only replacement, users can pass `random_mask_strategy=RandomMaskStrategy.AMINO_ACID_ONLY` to `ESMDataModule`.

## 4. ESM2Config
Instead of initializing the whole model on each rank, sharded models are lazily created on the target rank with the help of a configuration object. `ESM2Config` is a dataclass that envelopes architecture parameters (such as `num_layers`) and the specification of each torch module (`ModuleSpec`) in the transformer, which are accelerated with flash and fused attentions in [TransformerEngine](https://github.com/NVIDIA/TransformerEngine). While we can initialize a model from `ESM2Config`, its setup is only completed in under `trainer.setup`, which is called on individual devices.

```python
from megatron.core.optimizer import OptimizerConfig
from nemo.lightning.pytorch.optim import MegatronOptimizerModule

from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.esm2.api import ESM2Config
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.lr_scheduler import WarmupAnnealDecayHoldScheduler
from bionemo.llm.lightning import BionemoLightningModule
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.biobert.model import BiobertSpecOption

# ESM2 650M config
num_layers = 33
hidden_size = 1280
num_attention_heads = 20
ffn_hidden_size = 4 * hidden_size

nemo1_init_path = None  # initialize from nemo1 checkpoint
restore_from_checkpoint_path = None  # initialize from nemo2 checkpoint
need_megatron_variable_seq_lengths_reductions: bool = (
    pipeline_model_parallel_size * tensor_model_parallel_size > 1 and min_seq_length != max_seq_length
)  # essential for pipeline/tensor parallel
biobert_spec_option = BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec  # accelerated esm2 with transformer engine

warmup_steps = 2000
lr = 1e-4

# Create model config
esm2_config = ESM2Config(
    seq_length=max_seq_length,
    num_layers=num_layers,
    hidden_size=hidden_size,
    num_attention_heads=num_attention_heads,
    ffn_hidden_size=ffn_hidden_size,
    params_dtype=get_autocast_dtype(precision),
    pipeline_dtype=get_autocast_dtype(precision),
    autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
    biobert_spec_option=biobert_spec_option,
    nemo1_ckpt_path=str(nemo1_init_path) if nemo1_init_path is not None else None,
    initial_ckpt_path=str(restore_from_checkpoint_path) if restore_from_checkpoint_path is not None else None,
    variable_seq_lengths=need_megatron_variable_seq_lengths_reductions,
)

# Create model instance
tokenizer = get_tokenizer()

model: BionemoLightningModule = biobert_lightning_module(
    esm2_config,
    tokenizer=tokenizer,
    optimizer=MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=lr,
            optimizer="adam",
            use_distributed_optimizer=True,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.98,
        ),
        lr_scheduler=WarmupAnnealDecayHoldScheduler(
            warmup_steps=warmup_steps, max_steps=num_steps, max_lr=lr, min_lr=lr / 10.0, anneal_percentage=0.10
        ),
    ),
)
```

!!! note "`ModuleSpec`"

    `ModelSpec` decides what torch modules are used in the transformer layers. By default, BioNeMo2 accelerates ESM2 architecture with TransformerEngine layers. Users can define their own `ModelSpec` for customized transformer layers. See [`get_biobert_spec`](https://github.com/NVIDIA/bionemo-framework/blob/main/sub-packages/bionemo-llm/src/bionemo/llm/model/biobert/transformer_specs.py#L61).


!!! note "`BionemoLightningModule`"

    Since the model is lazily initialized in the target rank, breakpoints for debugging purposes should be added after `trainer.setup`.


# Model Pretraining
To close the loop, users can make use of `llm.train` from NeMo to begin training.

```python
from typing import Optional

from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks

from bionemo.llm.utils.logger_utils import WandbLoggerOptions, setup_nemo_lightning_logger


# WANDB logging
wandb_options: Optional[WandbLoggerOptions] = (
    None
    if wandb_project is None
    else WandbLoggerOptions(
        offline=False,
        project=__your_wandb_project__,
        entity=__your_wandb_entity__,
        tags=None,
        group=None,
        id=None,
        anonymous=False,
        log_model=False,
    )
)

checkpoint_callback = nl_callbacks.ModelCheckpoint(
    save_last=True,
    monitor="val_loss",
    save_top_k=1,
    every_n_train_steps=100,
    always_save_context=True,
)

nemo_logger = setup_nemo_lightning_logger(
    root_dir=__your_result_dir__,
    name=__your_experiment_name__,
    initialize_tensorboard_logger=True,
    wandb_kwargs=wandb_options,
    ckpt_callback=checkpoint_callback,
)

llm.train(
    model=model,
    data=data,
    trainer=trainer,
    log=nemo_logger,
    resume=resume.AutoResume(
        resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
        resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
    ),
)
```

Or simply call `esm2_pretrain.py` directly.
```bash
# Enable fused attention in transformer engine for speed-up
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=0

DATA_DIR=$(download_bionemo_data esm2/testdata_esm2_pretrain:2.0 --source ngc)

python scripts/protein/esm2/esm2_pretrain.py \
    --train-cluster-path ${DATA_DIR}/2024_03_sanity/train_clusters_sanity.parquet \
    --train-database-path ${DATA_DIR}/2024_03_sanity/train_sanity.db \
    --valid-cluster-path ${DATA_DIR}/2024_03_sanity/valid_clusters.parquet \
    --valid-database-path ${DATA_DIR}/2024_03_sanity/validation.db \
    --precision="bf16-mixed" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 100 \
    --val-check-interval 25 \
    --max-seq-length 1024 \
    --limit-val-batches 2 \
    --micro-batch-size 2 \
    --num-layers 33 \
    --hidden-size 1280 \
    --num-attention-head 20 \
    --ffn-hidden-size 5120
```

!!! note "Non-critical Warnings from Command Line Runs"

    Users might experience `torch._dynamo.convert_frame` warning messages and depreciation warning on `async_grad_allreduce` from Megatron-LM. Users can safely ignore them and is non-critical to pretraining.
