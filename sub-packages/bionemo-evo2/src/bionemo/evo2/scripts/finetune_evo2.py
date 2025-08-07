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


import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, get_args

from lightning.pytorch.callbacks import Callback, LearningRateMonitor, RichModelSummary
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.evo2.models.finetune import (
    Evo2FineTuneDataModule,
    Evo2FineTuneSeqConfig,
    Evo2FineTuneTokenConfig,
    InMemoryPerTokenValueDataset,
    InMemorySingleValueDataset,
)
from bionemo.evo2.models.mamba import MAMBA_MODEL_OPTIONS, MambaModel
from bionemo.evo2.run.peft import Evo2LoRA
from bionemo.llm.model.config import TorchmetricsConfig
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger


__all__: Sequence[str] = "finetune_evo2_entrypoint"


# Supported configurations and datasets
SUPPORTED_CONFIGS = {
    "Evo2FineTuneSeqConfig": Evo2FineTuneSeqConfig,
    "Evo2FineTuneTokenConfig": Evo2FineTuneTokenConfig,
}

SUPPORTED_DATASETS = {
    "InMemorySingleValueDataset": InMemorySingleValueDataset,
    "InMemoryPerTokenValueDataset": InMemoryPerTokenValueDataset,
}


def get_parser():
    """Get argument parser for Evo2 fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune an Evo2 model.")

    # Required arguments
    parser.add_argument("--train-data-path", type=Path, required=True, help="Path to training data CSV")
    parser.add_argument("--valid-data-path", type=Path, required=True, help="Path to validation data CSV")

    # Model arguments
    parser.add_argument(
        "--model-type", type=str, default="hyena", choices=["hyena", "mamba"], help="Model architecture to use"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="7b",
        choices=sorted(list(HYENA_MODEL_OPTIONS.keys()) + list(MAMBA_MODEL_OPTIONS.keys())),
        help="Model size to use",
    )

    # Training arguments
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to run on")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs per node")
    parser.add_argument("--min-seq-length", type=int, default=None, help="Minimum sequence length")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--result-dir", type=Path, default=Path("./results"), help="Directory to store results")
    parser.add_argument("--num-steps", type=int, default=500_000, help="Number of steps to train")
    parser.add_argument("--max-epochs", type=int, default=500_000, help="Maximum number of epochs")
    parser.add_argument("--limit-val-batches", type=int, default=1000, help="Limit validation batches")
    parser.add_argument("--limit-test-batches", type=int, default=1000, help="Limit test batches")
    parser.add_argument("--val-check-interval", type=int, default=20, help="Validation check interval")
    parser.add_argument("--log-every-n-steps", type=int, default=1, help="Log every n steps")
    parser.add_argument("--num-dataset-workers", type=int, default=8, help="Number of dataset workers")
    parser.add_argument("--no-persistent-workers", action="store_true", help="Don't use persistent workers")
    parser.add_argument("--no-pin-memory", action="store_true", help="Don't use pin memory")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--micro-batch-size", type=int, default=8, help="Micro batch size")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--experiment-name", type=str, default="evo2-finetune", help="Experiment name")
    parser.add_argument("--resume-if-exists", action="store_true", help="Resume if checkpoint exists")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision type")
    parser.add_argument(
        "--task-type",
        type=str,
        default="regression",
        choices=["classification", "regression"],
        help="Fine-tuning task type",
    )
    parser.add_argument("--encoder-frozen", action="store_true", help="Freeze encoder parameters")

    # Model head parameters
    parser.add_argument("--mlp-ft-dropout", type=float, default=0.25, help="MLP dropout")
    parser.add_argument("--mlp-hidden-size", type=int, default=256, help="MLP hidden size")
    parser.add_argument("--mlp-target-size", type=int, default=1, help="MLP target size")
    parser.add_argument("--cnn-dropout", type=float, default=0.25, help="CNN dropout")
    parser.add_argument("--cnn-hidden-size", type=int, default=32, help="CNN hidden size")
    parser.add_argument("--cnn-num-classes", type=int, default=3, help="CNN number of classes")

    # Weights & Biases parameters
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project")
    parser.add_argument("--wandb-offline", action="store_true", help="Run W&B offline")
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=None, help="W&B tags")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B group")
    parser.add_argument("--wandb-id", type=str, default=None, help="W&B run ID")
    parser.add_argument("--wandb-anonymous", action="store_true", help="W&B anonymous mode")
    parser.add_argument("--wandb-log-model", action="store_true", help="Log model to W&B")

    # Model parallel parameters
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1, help="Pipeline model parallel size")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1, help="Tensor model parallel size")
    parser.add_argument("--context-parallel-size", type=int, default=1, help="Context parallel size")
    parser.add_argument("--sequence-parallel", action="store_true", help="Enable sequence parallelism")

    # Checkpoint parameters
    parser.add_argument("--create-tensorboard-logger", action="store_true", help="Create tensorboard logger")
    parser.add_argument("--restore-from-checkpoint-path", type=Path, required=True, help="Path to Evo2 checkpoint")
    parser.add_argument("--save-last-checkpoint", action="store_true", default=True, help="Save last checkpoint")
    parser.add_argument("--metric-to-monitor-for-checkpoints", type=str, default="val_loss", help="Metric to monitor")
    parser.add_argument("--save-top-k", type=int, default=2, help="Save top k checkpoints")

    # Dataset and config parameters
    parser.add_argument("--dataset-class", type=str, default="InMemorySingleValueDataset", help="Dataset class")
    parser.add_argument("--config-class", type=str, default="Evo2FineTuneSeqConfig", help="Config class")

    # Other parameters
    parser.add_argument("--label-column", type=str, default="labels", help="Label column name")
    parser.add_argument("--labels-mask-column", type=str, default=None, help="Labels mask column name")
    parser.add_argument("--lora-checkpoint-path", type=Path, default=None, help="LoRA checkpoint path")
    parser.add_argument("--lora-finetune", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument(
        "--disable-checkpointing",
        action="store_false",
        default=True,
        dest="create_checkpoint_callback",
        help="Disable checkpointing",
    )
    parser.add_argument("--early-stop-on-step", type=int, default=None, help="Stop training on this step")
    parser.add_argument("--create-tflops-callback", action="store_true", help="Enable tflops calculation")

    # Evo2-specific parameters
    parser.add_argument("--fp8", action="store_true", help="Use FP8 precision")
    parser.add_argument("--no-average-in-collective", action="store_true", help="No average in collective")
    parser.add_argument("--grad-reduce-in-fp32", action="store_true", help="Gradient reduction in fp32")
    parser.add_argument("--no-ckpt-async-save", action="store_true", help="Disable async checkpoint save")

    return parser


def train_model(
    train_data_path: Path,
    valid_data_path: Path,
    model_type: str = "hyena",
    model_size: str = "7b",
    num_nodes: int = 1,
    num_gpus: int = 1,
    min_seq_length: Optional[int] = None,
    max_seq_length: int = 8192,
    result_dir: Path = Path("./results"),
    num_steps: int = 500_000,
    max_epochs: int = 500_000,
    limit_val_batches: int = 1000,
    limit_test_batches: int = 1000,
    val_check_interval: int = 20,
    log_every_n_steps: int = 1,
    num_dataset_workers: int = 8,
    no_persistent_workers: bool = False,
    no_pin_memory: bool = False,
    lr: float = 4e-4,
    micro_batch_size: int = 8,
    accumulate_grad_batches: int = 1,
    experiment_name: str = "evo2-finetune",
    resume_if_exists: bool = False,
    precision: str = "bf16-mixed",
    task_type: str = "regression",
    encoder_frozen: bool = False,
    mlp_ft_dropout: float = 0.25,
    mlp_hidden_size: int = 256,
    mlp_target_size: int = 1,
    cnn_dropout: float = 0.25,
    cnn_hidden_size: int = 32,
    cnn_num_classes: int = 3,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_offline: bool = False,
    wandb_tags: Optional[List[str]] = None,
    wandb_group: Optional[str] = None,
    wandb_id: Optional[str] = None,
    wandb_anonymous: bool = False,
    wandb_log_model: bool = False,
    pipeline_model_parallel_size: int = 1,
    tensor_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    create_tensorboard_logger: bool = False,
    restore_from_checkpoint_path: Optional[Path] = None,
    save_last_checkpoint: bool = True,
    metric_to_monitor_for_checkpoints: str = "val_loss",
    save_top_k: int = 2,
    dataset_class: str = "InMemorySingleValueDataset",
    config_class: str = "Evo2FineTuneSeqConfig",
    label_column: str = "labels",
    labels_mask_column: Optional[str] = None,
    lora_checkpoint_path: Optional[Path] = None,
    lora_finetune: bool = False,
    create_checkpoint_callback: bool = True,
    early_stop_on_step: Optional[int] = None,
    create_tflops_callback: bool = False,
    fp8: bool = False,
    no_average_in_collective: bool = False,
    grad_reduce_in_fp32: bool = False,
    no_ckpt_async_save: bool = False,
) -> Tuple[Optional[Path], Callback | None, nl.Trainer]:
    """Train an Evo2 model with fine-tuning."""
    config_class = SUPPORTED_CONFIGS[config_class]
    dataset_class = SUPPORTED_DATASETS[dataset_class]

    result_dir.mkdir(parents=True, exist_ok=True)

    # Setup global batch size
    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=num_nodes,
        devices=num_gpus,
        accumulate_grad_batches=accumulate_grad_batches,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

    # Initialize LoRA if needed
    peft = None
    if lora_finetune:
        peft = Evo2LoRA(peft_ckpt_path=str(lora_checkpoint_path) if lora_checkpoint_path else None)

    # Setup strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallel,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=not no_ckpt_async_save,
        ckpt_parallel_load=True,
        ckpt_load_strictness=StrictHandling.LOG_UNEXPECTED,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=not no_average_in_collective,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            use_distributed_optimizer=False,
        ),
    )

    # WandB configuration
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

    # Setup callbacks
    callbacks = [
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
        nl_callbacks.PreemptionCallback(),
        TimingCallback(),
    ]

    if peft is not None:
        callbacks.append(peft)

    # Get tokenizer
    tokenizer = get_nmt_tokenizer("Byte-Level")

    # Load datasets
    train_dataset = dataset_class.from_csv(
        train_data_path,
        task_type=task_type,
        label_column=label_column,
        labels_mask_column=labels_mask_column,
        tokenizer=tokenizer,
    )
    valid_dataset = dataset_class.from_csv(
        valid_data_path,
        task_type=task_type,
        label_column=label_column,
        labels_mask_column=labels_mask_column,
        tokenizer=tokenizer,
    )

    # Create data module
    data_module = Evo2FineTuneDataModule(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        num_workers=num_dataset_workers,
        persistent_workers=not no_persistent_workers,
        pin_memory=not no_pin_memory,
        tokenizer=tokenizer,
    )

    # Configure model
    is_model_parallel = tensor_model_parallel_size * pipeline_model_parallel_size > 1

    # Setup metrics
    if is_model_parallel:
        valid_metric = None
    elif task_type == "regression":
        valid_metric = TorchmetricsConfig(class_path="MeanSquaredError", task="regression", metric_name="val_mse")
    elif task_type == "classification":
        num_classes = train_dataset.label_tokenizer_vocab_size or mlp_target_size
        valid_metric = TorchmetricsConfig(
            class_path="Accuracy",
            task="classification",
            kwargs={"task": "multiclass", "threshold": 0.5, "num_classes": num_classes},
            metric_name="val_acc",
        )

    # Create config
    config = config_class(
        model_type=model_type,
        model_size=model_size,
        task_type=task_type,
        encoder_frozen=encoder_frozen,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        initial_ckpt_path=str(restore_from_checkpoint_path),
        train_metric=None,
        valid_metric=valid_metric,
        mlp_ft_dropout=mlp_ft_dropout,
        mlp_hidden_size=mlp_hidden_size,
        mlp_target_size=mlp_target_size,
        cnn_dropout=cnn_dropout,
        cnn_hidden_size=cnn_hidden_size,
        cnn_num_classes=cnn_num_classes,
    )

    # Get the base model config
    if model_type == "hyena":
        base_model_config = HYENA_MODEL_OPTIONS[model_size]
    else:
        base_model_config = MAMBA_MODEL_OPTIONS[model_size]

    # Create optimizer
    optimizer = MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=lr,
            optimizer="adam",
            use_distributed_optimizer=True,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.98,
            clip_grad=1.0,
        ),
        lr_scheduler=CosineAnnealingScheduler(
            warmup_steps=100,
            constant_steps=0,
            min_lr=lr * 0.1,
        ),
    )

    # Set the weight decay condition
    optimizer.config.fp16 = fp8
    optimizer.config.bf16 = not fp8 and precision == "bf16-mixed"

    # Create the module
    if model_type == "hyena":
        module = llm.HyenaModel(
            config=base_model_config,
            tokenizer=tokenizer,
            optim=optimizer,
            model_transform=peft if peft else config.configure_model,
        )
    else:
        module = MambaModel(
            config=base_model_config,
            tokenizer=tokenizer,
            optim=optimizer,
            model_transform=peft if peft else config.configure_model,
        )

    # Setup logger
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=result_dir,
        name=experiment_name,
        initialize_tensorboard_logger=create_tensorboard_logger,
        wandb_config=wandb_config,
    )

    # Setup checkpoint callback
    checkpoint_callback = None
    auto_resume = None
    if create_checkpoint_callback:
        checkpoint_path = str(Path(nemo_logger.save_dir) / "checkpoints")
        checkpoint_callback = nl_callbacks.ModelCheckpoint(
            dirpath=checkpoint_path,
            save_last=save_last_checkpoint,
            monitor=metric_to_monitor_for_checkpoints,
            save_top_k=save_top_k,
            every_n_train_steps=val_check_interval,
            always_save_context=True,
            filename="checkpoint-{step}-{consumed_samples}",
            save_weights_only=False,
            save_optim_on_train_end=True,
        )
        callbacks.append(checkpoint_callback)

        auto_resume = resume.AutoResume(
            resume_from_directory=checkpoint_path,
            resume_if_exists=resume_if_exists,
            resume_ignore_no_checkpoint=True,
            resume_past_end=False,
        )

    if create_tflops_callback:
        data_module.global_batch_size = global_batch_size
        flop_meas_callback = FLOPsMeasurementCallback(
            config,
            data_module,
            model_type,
        )
        callbacks.append(flop_meas_callback)

    # Create trainer
    trainer = nl.Trainer(
        devices=num_gpus,
        max_steps=num_steps if early_stop_on_step is None else early_stop_on_step,
        max_epochs=max_epochs,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
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
        enable_checkpointing=create_checkpoint_callback,
    )

    # Train the model
    llm.train(
        model=module,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=auto_resume,
    )

    ckpt_path = Path(checkpoint_callback.last_model_path.replace(".ckpt", "")) if create_checkpoint_callback else None
    return ckpt_path, None, trainer


def finetune_evo2_entrypoint() -> Tuple[Path, Callback | None, nl.Trainer]:
    """Entry point for Evo2 fine-tuning."""
    parser = get_parser()
    args = parser.parse_args()

    # Validate arguments
    if args.lora_checkpoint_path and not args.lora_finetune:
        raise ValueError("Arguments --lora-checkpoint-path cannot be set when not using lora-finetune.")
    if args.precision not in get_args(PrecisionTypes):
        raise ValueError(f"Precision {args.precision} not supported. Supported precisions are: {PrecisionTypes}")
    if args.task_type not in ["classification", "regression"]:
        raise ValueError(f"Task type {args.task_type} not supported. Supported: classification, regression")
    if args.dataset_class not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset class {args.dataset_class} not supported. Supported: {SUPPORTED_DATASETS.keys()}")
    if args.config_class not in SUPPORTED_CONFIGS:
        raise ValueError(f"Config class {args.config_class} not supported. Supported: {SUPPORTED_CONFIGS.keys()}")
    if args.min_seq_length is not None and args.dataset_class == "InMemorySingleValueDataset":
        raise ValueError("Arguments --min-seq-length cannot be set when using InMemorySingleValueDataset.")

    return train_model(**vars(args))


if __name__ == "__main__":
    finetune_evo2_entrypoint()
