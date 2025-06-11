# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
from typing import List, Optional, Sequence, Tuple, get_args

import typer
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, RichModelSummary
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule
from bionemo.esm2.model.finetune.dataset import (
    InMemoryPerTokenValueDataset,
    InMemoryProteinDataset,
    InMemorySingleValueDataset,
)
from bionemo.esm2.model.finetune.peft import ESM2LoRA
from bionemo.esm2.model.finetune.sequence_model import ESM2FineTuneSeqConfig
from bionemo.esm2.model.finetune.token_model import ESM2FineTuneTokenConfig
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.config import TorchmetricsConfig
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger


__all__: Sequence[str] = "finetune_esm2_entrypoint"


SUPPORTED_CONFIGS = {
    "ESM2FineTuneSeqConfig": ESM2FineTuneSeqConfig,
    "ESM2FineTuneTokenConfig": ESM2FineTuneTokenConfig,
}

SUPPORTED_DATASETS = {
    "InMemoryProteinDataset": InMemoryProteinDataset,
    "InMemorySingleValueDataset": InMemorySingleValueDataset,
    "InMemoryPerTokenValueDataset": InMemoryPerTokenValueDataset,
}

app = typer.Typer()


@app.command()
def finetune_esm2_entrypoint(
    train_data_path: Path = typer.Option(..., help="Path to training data CSV"),
    valid_data_path: Path = typer.Option(..., help="Path to validation data CSV"),
    num_nodes: int = 1,
    num_gpus: int = 1,
    min_seq_length: int = 1024,
    max_seq_length: int = 512,
    result_dir: Path = Path("./results"),
    num_steps: int = 500_000,
    max_epochs: int = 500_000,
    limit_val_batches: int = 1000,
    limit_test_batches: int = 1000,
    val_check_interval: int = 20,
    log_every_n_steps: int = 1,
    num_dataset_workers: int = 8,
    lr: float = 4e-4,
    micro_batch_size: int = 64,
    accumulate_grad_batches: int = 1,
    experiment_name: str = "esm2-finetune",
    resume_if_exists: bool = False,
    precision: str = "bf16-mixed",
    task_type: str = "regression",
    encoder_frozen: bool = False,
    scale_lr_layer: Optional[str] = None,
    lr_multiplier: float = 1.0,
    # single value classification / regression mlp
    mlp_ft_dropout: float = 0.25,
    mlp_hidden_size: int = 256,
    mlp_target_size: int = 1,
    # token-level classification cnn
    cnn_dropout: float = 0.25,
    cnn_hidden_size: int = 32,
    cnn_num_classes: int = 3,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_offline: bool = False,
    wandb_tags: Optional[List[str]] = None,
    wandb_group: Optional[str] = None,
    wandb_id: Optional[str] = None,
    wandb_anonymous: Optional[bool] = False,
    wandb_log_model: bool = False,
    pipeline_model_parallel_size: int = 1,
    tensor_model_parallel_size: int = 1,
    create_tensorboard_logger: bool = False,
    restore_from_checkpoint_path: Optional[Path] = None,
    save_last_checkpoint: bool = True,
    metric_to_monitor_for_checkpoints: str = "val_loss",
    save_top_k: int = 2,
    nsys_profiling: bool = False,
    nsys_start_step: int = 0,
    nsys_end_step: Optional[int] = None,
    nsys_ranks: List[int] = [0],
    dataset_class: str = "InMemorySingleValueDataset",
    config_class: str = "ESM2FineTuneSeqConfig",
    metric_tracker=None,
    overlap_grad_reduce: bool = False,  # Default to False to avoid communication issue in gradient synchronization step
    overlap_param_gather: bool = True,
    average_in_collective: bool = True,
    grad_reduce_in_fp32: bool = False,
    ckpt_async_save: bool = True,
    label_column: str = "labels",
    lora_checkpoint_path: Optional[Path] = None,
    lora_finetune: bool = False,
) -> Tuple[Path, Callback | None, nl.Trainer]:
    """Train an ESM2 model on UR data.

    Args:
        train_data_path (Path): path to train CSV
        valid_data_path (Path): path to validation CSV
        num_nodes (int): Number of nodes to run on
        num_gpus (int): number of GPUs per node
        min_seq_length (int): minimum sequence length
        max_seq_length (int): maximum sequence length
        result_dir (Path): directory to store results, logs and checkpoints
        num_steps (int): number of steps to train the model for
        max_epochs (int): number of epochs to train the model for
        limit_val_batches (int): limit the number of validation global batches to this many
        limit_test_batches (int): limit the number of test global batches to this many
        val_check_interval (int): number of steps to periodically check the validation loss
        log_every_n_steps (Optional[int]): log every n steps
        num_dataset_workers (int): number of dataset workers
        lr (float): learning rate for the optimizer
        micro_batch_size (int): micro batch size, from this and parallelism settings we infer the global batch size
        accumulate_grad_batches (int): number of batches to accumulate gradients for
        experiment_name (str): experiment name, this is the name used for the wandb run, and the sub-directory of the
            result_dir that stores the logs and checkpoints.
        resume_if_exists (bool): attempt to resume if the checkpoint exists [FIXME @skothenhill this doesn't work yet]
        precision (PrecisionTypes): Precision type for training (e.g., float16, float32)
        task_type (Literal["classification", "regression"]): Fine-tuning task type. Default is regression.
        encoder_frozen (bool): Freeze the encoder parameters. Default is False.
        scale_lr_layer (Optional[str]): layer names for which the lr is scaled by lr_multiplier
        lr_multiplier (float): lr multiplier for parameters in scale_lr_layer
        mlp_ft_dropout (float): dropout for single value classification / regression mlp
        mlp_hidden_size (int): dimension of hidden layer in mlp task head
        mlp_target_size: (int): output dimension of the mlp task head (number of classes in classification tasks). Set to 1 for regression tasks.
        cnn_dropout (float): dropout for token-level classification cnn
        cnn_hidden_size (int): hidden dimension of cnn head
        cnn_num_classes (int): number of classes in token-level classification
        wandb_entity (Optional[str]): The team posting this run (default: your username or your default team)
        wandb_project (Optional[str]): The name of the project to which this run will belong
        wandb_offline (bool): Run offline (data can be streamed later to wandb servers).
        wandb_tags (Optional[List[str]]): Tags associated with this run
        wandb_group (Optional[str]): A unique string shared by all runs in a given group
        wandb_id (Optional[str]): Sets the version, mainly used to resume a previous run
        wandb_anonymous (Optional[bool]): Enables or explicitly disables anonymous logging
        wandb_log_model (bool): Save checkpoints in wandb dir to upload on W&B servers
        pipeline_model_parallel_size (int): pipeline model parallel size
        tensor_model_parallel_size (int): tensor model parallel size
        create_tensorboard_logger (bool): create the tensorboard logger
        restore_from_checkpoint_path (Optional[str]): If set, restores the model from the directory passed in. Expects the
            checkpoint to be created by using the ModelCheckpoint class and always_save_context=True. Will override the `resume_if_exists` argument when set.
        save_last_checkpoint (bool): whether to save the last checkpoint
        metric_to_monitor_for_checkpoints (str): metric to monitor for checkpoints
        save_top_k (int): number of top checkpoints to save
        nsys_profiling (bool): whether to enable nsys profiling. Enable targeted `nsys` profiling on the training loop for a defined step range. To actually get profiling output you must run the whole program with `nsys`. For example: "
        " `nsys profile -s none -o output_report_name -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop  [regular python command here]`
        nsys_start_step (int): start step for nsys profiling
        nsys_end_step (Optional[int]): end step for nsys profiling
        nsys_ranks (List[int]): ranks for nsys profiling
        dataset_class (Type[InMemoryProteinDataset]): The dataset class for loading the data from a CSV file
        config_class (Type[BioBertConfig]): The config class for configuring the model using checkpoint provided. Model configs link model classes with losses, and handle model initialization (including from a prior checkpoint). This is how you can fine-tune a model. First train with one config class that points to one model class and loss, then implement and provide an alternative config class that points to a variant of that model and alternative loss. In the future this script should also provide similar support for picking different data modules for finetuning with different data types.
        metric_tracker: Optional callback to track metrics (used for testing)
        overlap_grad_reduce (bool): overlap gradient reduction
        overlap_param_gather (bool): overlap parameter gather
        average_in_collective (bool): average in collective
        grad_reduce_in_fp32 (bool): gradient reduction in fp32
        ckpt_async_save (bool): whether to save ckpt async. Set to False for federated learning
        label_column (str): name of label column in CSV data file. Defaults to `labels`.
        lora_checkpoint_path (Optional[str]): path to the lora checkpoint file.
        lora_finetune (bool): whether to use lora fine-tuning.
    """
    if min_seq_length is not None and dataset_class is InMemorySingleValueDataset:
        raise ValueError("Arguments --min-seq-length cannot be set when using InMemorySingleValueDataset.")
    if lora_checkpoint_path and not lora_finetune:
        raise ValueError("Arguments --lora=checkpoint-path cannot be set when not using lora-finetune.")
    if precision not in get_args(PrecisionTypes):
        raise ValueError(f"Precision {precision} not supported. Supported precisions are: {PrecisionTypes}")
    if task_type not in ["classification", "regression"]:
        raise ValueError(f"Task type {task_type} not supported. Supported task types are: classification, regression")
    if dataset_class not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset class {dataset_class} not supported. Supported dataset classes are: {SUPPORTED_DATASETS.keys()}"
        )
    if config_class not in SUPPORTED_CONFIGS:
        raise ValueError(
            f"Config class {config_class} not supported. Supported config classes are: {SUPPORTED_CONFIGS.keys()}"
        )

    config_class = SUPPORTED_CONFIGS[config_class]
    dataset_class = SUPPORTED_DATASETS[dataset_class]

    # Create the result directory if it does not exist.
    result_dir.mkdir(parents=True, exist_ok=True)

    # Setup the strategy and trainer
    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=num_nodes,
        devices=num_gpus,
        accumulate_grad_batches=accumulate_grad_batches,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

    # Convert lora_checkpoint_path to string if it's a Path object
    if lora_checkpoint_path is not None:
        lora_checkpoint_path = str(lora_checkpoint_path)

    # Initialize LoRA adapter first if needed
    peft = None
    if lora_finetune:
        peft = ESM2LoRA(peft_ckpt_path=lora_checkpoint_path)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=ckpt_async_save,
        ckpt_parallel_load=True,
        ckpt_load_strictness=StrictHandling.LOG_UNEXPECTED,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_grad_reduce=overlap_grad_reduce,
            overlap_param_gather=overlap_param_gather,
            average_in_collective=average_in_collective,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            use_distributed_optimizer=False,
        ),
    )

    # for wandb integration
    # Please refer to https://pytorch-lightning.readthedocs.io/en/0.7.6/api/lightning.pytorch.loggers.html"
    wandb_config: Optional[WandbConfig] = (
        None
        if wandb_project is None
        else WandbConfig(
            offline=wandb_offline,
            project=wandb_project,
            entity=wandb_entity,
            tags=wandb_tags,
            group=wandb_group,
            id=wandb_id,
            anonymous=wandb_anonymous,
            log_model=wandb_log_model,
        )
    )

    callbacks = [
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
        nl_callbacks.PreemptionCallback(),
    ]
    if metric_tracker is not None:
        callbacks.append(metric_tracker)
    if nsys_profiling:
        if nsys_end_step is None:
            nsys_end_step = num_steps
        callbacks.append(
            nl_callbacks.NsysCallback(
                start_step=nsys_start_step, end_step=nsys_end_step, ranks=nsys_ranks, gen_shape=True
            )
        )
    if peft is not None:
        callbacks.append(peft)

    trainer = nl.Trainer(
        devices=num_gpus,
        max_steps=num_steps,
        max_epochs=max_epochs,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_val_batches,  # This controls upsampling and downsampling
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        num_nodes=num_nodes,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(
            precision=precision,
            params_dtype=get_autocast_dtype(precision),
            pipeline_dtype=get_autocast_dtype(precision),
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            autocast_enabled=False,
        ),
    )

    tokenizer = get_tokenizer()

    # Initialize the data module.
    train_dataset = dataset_class.from_csv(train_data_path, task_type=task_type, label_column=label_column)
    valid_dataset = dataset_class.from_csv(valid_data_path, task_type=task_type, label_column=label_column)

    data_module = ESM2FineTuneDataModule(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        num_workers=num_dataset_workers,
        tokenizer=tokenizer,
    )
    # Configure the model
    train_metric = None
    is_model_parallel = tensor_model_parallel_size * pipeline_model_parallel_size > 1
    if is_model_parallel:
        valid_metric = None  # metric logging under model parallelism is not supported yet
    elif task_type == "regression":
        valid_metric = TorchmetricsConfig(class_path="MeanSquaredError", task="regression", metric_name="val_mse")
    else:
        valid_metric = TorchmetricsConfig(
            class_path="Accuracy",
            task="classification",
            kwargs={
                "task": "multiclass",
                "threshold": 0.5,
                "num_classes": data_module.train_dataset.label_tokenizer.vocab_size,
            },
            metric_name="val_acc",
        )

    config = config_class(
        task_type=task_type,
        encoder_frozen=encoder_frozen,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        initial_ckpt_path=str(restore_from_checkpoint_path),
        initial_ckpt_skip_keys_with_these_prefixes=[f"{task_type}_head"],
        train_metric=train_metric,
        valid_metric=valid_metric,
    )
    # Mapping of task-dependent config attributes to their new values
    task_dependent_attr = {
        "mlp_ft_dropout": mlp_ft_dropout,
        "mlp_hidden_size": mlp_hidden_size,
        "mlp_target_size": mlp_target_size,
        "cnn_dropout": cnn_dropout,
        "cnn_hidden_size": cnn_hidden_size,
        "cnn_num_classes": cnn_num_classes,
    }
    # Update attributes only if they exist in the config
    for attr, value in task_dependent_attr.items():
        if hasattr(config, attr):
            config.set_hparam(attr, value)

    optimizer = MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=lr,
            optimizer="adam",  # fused_adam not supported
            use_distributed_optimizer=True,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.98,
            clip_grad=1.0,
        ),
    )
    # fiddle is not serializing lambda fn
    # to bypass serialization of lambda fn scale_lr_condition as part of optimizer configuration
    if scale_lr_layer:
        optimizer.scale_lr_cond = lambda name, param: scale_lr_layer in name
        optimizer.lr_mult = lr_multiplier

    if peft is not None:
        module = biobert_lightning_module(
            config=config, tokenizer=tokenizer, optimizer=optimizer, model_transform=peft
        )
    else:
        module = biobert_lightning_module(config=config, tokenizer=tokenizer, optimizer=optimizer)

    # Configure our custom Checkpointer
    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_last=save_last_checkpoint,
        monitor=metric_to_monitor_for_checkpoints,  # "val_loss",
        save_top_k=save_top_k,
        every_n_train_steps=val_check_interval,
        always_save_context=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
        filename="checkpoint-{step}-{consumed_samples}",  # Including step and consumed_samples in the checkpoint filename prevents duplicate filenames and bugs related to this.
        save_weights_only=False,
        save_optim_on_train_end=True,
    )

    # Setup the logger and train the model
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=result_dir,
        name=experiment_name,
        initialize_tensorboard_logger=create_tensorboard_logger,
        wandb_config=wandb_config,
        ckpt_callback=checkpoint_callback,
    )
    llm.train(
        model=module,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            resume_if_exists=resume_if_exists,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )

    ckpt_path = Path(checkpoint_callback.last_model_path.replace(".ckpt", ""))
    return ckpt_path, metric_tracker, trainer


if __name__ == "__main__":
    app()
