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
from typing import Optional, Sequence, get_args

from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.esm2.api import ESM2Config
from bionemo.esm2.data.datamodule import ESMDataModule
from bionemo.esm2.data.dataset import RandomMaskStrategy
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.lr_scheduler import WarmupAnnealDecayHoldScheduler
from bionemo.llm.lightning import PerplexityLoggingCallback
from bionemo.llm.model.biobert.lightning import BioBertLightningModule
from bionemo.llm.model.biobert.model import BiobertSpecOption
from bionemo.llm.utils.datamodule_utils import float_or_int_or_none, infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbLoggerOptions, setup_nemo_lightning_logger


__all__: Sequence[str] = ("main", "parser")


def main(
    train_cluster_path: Path,
    train_database_path: Path,
    valid_cluster_path: Path,
    valid_database_path: Path,
    num_nodes: int,
    devices: int,
    min_seq_length: Optional[int],
    max_seq_length: int,
    result_dir: Path,
    wandb_project: Optional[str],
    wandb_offline: bool,
    num_steps: int,
    warmup_steps: int,
    limit_val_batches: int,
    val_check_interval: int,
    num_dataset_workers: int,
    biobert_spec_option: BiobertSpecOption,  # TODO(@farhadrgh) clarify how to parse this.
    lr: float,
    micro_batch_size: int,
    accumulate_grad_batches: int,
    experiment_name: str,
    resume_if_exists: bool,
    precision: PrecisionTypes,
    wandb_entity: str = "clara-discovery",
    pipeline_model_parallel_size: int = 1,
    tensor_model_parallel_size: int = 1,
    create_tensorboard_logger: bool = False,
    nemo1_init_path: Optional[Path] = None,
    restore_from_checkpoint_path: Optional[str] = None,
    save_best_checkpoint: bool = True,
    save_last_checkpoint: bool = True,
    metric_to_monitor_for_checkpoints: str = "val_loss",
    save_top_k: int = 2,
    save_every_n_steps: int = 100,
    random_mask_strategy: RandomMaskStrategy = RandomMaskStrategy.ALL_TOKENS,
    num_layers: int = 33,
    hidden_size: int = 1280,
    num_attention_heads: int = 20,
    ffn_hidden_size: int = 1280 * 4,
) -> None:
    """Train an ESM2 model on UR data.

    Args:
        train_cluster_path (Path): path to train cluster partquet
        train_database_path (Path): path to train database
        valid_cluster_path (Path): path to validation cluster parquet
        valid_database_path (Path): path to validation database
        num_nodes (int): Number of nodes to run on
        devices (int): number of devices
        seq_length (int): sequence length
        result_dir (Path): directory to store results, logs and checkpoints
        wandb_project (Optional[str]): weights and biases project name
        wandb_offline (bool): if wandb should happen in offline mode
        num_steps (int): number of steps to train the model for
        limit_val_batches (int): limit the number of validation global batches to this many
        val_check_interval (int): number of steps to periodically check the validation loss and save num_dataset_workers (
       int): num dataset workers
        biobert_spec_option (BiobertSpecOption): the biobert spec option (architecture) to use for this run
        lr (float): learning rate
        micro_batch_size (int): micro batch size, from this and parallelism settings we infer the global batch size
        experiment_name (str): experiment name, this is the name used for the wandb run, and the sub-directory of the
            result_dir that stores the logs and checkpoints.
        resume_if_exists (bool): attempt to resume if the checkpoint exists [FIXME @skothenhill this doesn't work yet]
        wandb_entity (str): the group to use for the wandb run, sometimes called a team, could also be your username
        create_tensorboard_logger (bool): create the tensorboard logger
        restore_from_checkpoint_path (path): If set, restores the model from the directory passed in. Expects the
            checkpoint to be created by using the ModelCheckpoint class and always_save_context=True.
    """
    # Create the result directory if it does not exist.
    result_dir.mkdir(parents=True, exist_ok=True)

    # Setup the strategy and trainer
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

    wandb_options: Optional[WandbLoggerOptions] = (
        None
        if wandb_project is None
        else WandbLoggerOptions(
            offline=wandb_offline,
            project=wandb_project,
            entity=wandb_entity,
            log_model=False,
        )
    )
    trainer = nl.Trainer(
        devices=devices,
        max_steps=num_steps,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=limit_val_batches,  # This controls upsampling and downsampling
        val_check_interval=val_check_interval,
        num_nodes=num_nodes,
        callbacks=[
            PerplexityLoggingCallback(log_train=False, log_val=True),
            RichModelSummary(max_depth=4),
            LearningRateMonitor(),
        ],
        plugins=nl.MegatronMixedPrecision(precision=precision),
    )

    tokenizer = get_tokenizer()

    # Initialize the data module.
    data = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=train_database_path,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=valid_database_path,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        num_workers=num_dataset_workers,
        random_mask_strategy=random_mask_strategy,
    )

    # Configure the model
    need_megatron_variable_seq_lengths_reductions = (
        pipeline_model_parallel_size * tensor_model_parallel_size > 1 and min_seq_length != max_seq_length,
    )  # essential for pipeline/tensor parallel
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
        nemo1_ckpt_path=nemo1_init_path,
        variable_seq_lengths=need_megatron_variable_seq_lengths_reductions,
    )

    model = BioBertLightningModule(
        esm2_config,
        tokenizer=tokenizer,
        optimizer=MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=lr,
                optimizer="adam",  # fused_adam not supported
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

    # Configure our custom Checkpointer
    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_last=save_last_checkpoint,
        monitor=metric_to_monitor_for_checkpoints,  # "val_loss",
        save_top_k=save_top_k,
        every_n_train_steps=save_every_n_steps,
        always_save_context=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
    )

    # Setup the logger and train the model
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=result_dir,
        name=experiment_name,
        initialize_tensorboard_logger=create_tensorboard_logger,
        wandb_kwargs=wandb_options,
        ckpt_callback=checkpoint_callback,
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            resume_from_path=restore_from_checkpoint_path,  # Overrides the path found by resume_if_exists when set.
            resume_if_exists=resume_if_exists,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )


# TODO migrate to hydra config
# Parse the arguments and pull them out into local variables for ease of future refactor to a
#   config management system.
parser = argparse.ArgumentParser(description="Pretrain ESM2 with UR data.")
parser.add_argument(
    "--train-cluster-path",
    type=Path,
    required=True,
    help="Path to the train cluster data parquet file",
)
parser.add_argument(
    "--train-database-path",
    type=Path,
    required=True,
    help="Path to the train sequence database file",
)
parser.add_argument(
    "--valid-cluster-path",
    type=Path,
    required=True,
    help="Path to the valid cluster data parquet file",
)
parser.add_argument(
    "--valid-database-path",
    type=Path,
    required=True,
    help="Path to the vali sequence database file",
)
parser.add_argument(
    "--precision",
    type=str,
    choices=get_args(PrecisionTypes),
    required=False,
    default="bf16-mixed",
    help="Precision type to use for training.",
)
parser.add_argument(
    "--lr",
    type=float,
    required=False,
    default=4e-4,
    help="Learning rate for training. Default is 4e-4",
)
parser.add_argument(
    "--create-tensorboard-logger", action="store_true", default=False, help="Create a tensorboard logger."
)
# FIXME (@skothenhill) figure out how checkpointing and resumption should work with the new nemo trainer
parser.add_argument(
    "--resume-if-exists", action="store_true", default=False, help="Resume training if a checkpoint exists."
)
parser.add_argument(
    "--result-dir", type=Path, required=False, default=Path("./results"), help="Path to the result directory."
)
parser.add_argument("--experiment-name", type=str, required=False, default="esm2", help="Name of the experiment.")
parser.add_argument("--wandb-offline", action="store_true", default=False, help="Use wandb in offline mode.")
parser.add_argument(
    "--wandb-project",
    type=str,
    required=False,
    default=None,
    help="Wandb project name. Wandb will only happen if this is set.",
)
parser.add_argument(
    "--num-gpus",
    type=int,
    required=False,
    default=1,
    help="Number of GPUs to use for training. Default is 1.",
)
parser.add_argument(
    "--num-nodes",
    type=int,
    required=False,
    default=1,
    help="Number of nodes to use for training. Default is 1.",
)
parser.add_argument(
    "--num-steps",
    type=int,
    required=False,
    default=500000,
    help="Number of steps to use for training. Default is 500000.",
)
parser.add_argument(
    "--warmup-steps",
    type=int,
    required=False,
    default=2000,
    help="Number of warmup steps for WarmupAnnealDecayHold Scheduler. Default is 2000.",
)
parser.add_argument(
    "--num-dataset-workers",
    type=int,
    required=False,
    default=1,
    help="Number of workers to use for training. Default is 1.",
)
parser.add_argument(
    "--val-check-interval",
    type=int,
    required=False,
    default=10000,
    help="Number of steps between validation. Default is 10000.",
)
parser.add_argument(
    "--min-seq-length",
    type=int,
    required=False,
    help="Minimum sequence length. Sampled will be padded if less than this value.",
)
parser.add_argument(
    "--max-seq-length",
    type=int,
    required=False,
    default=1024,
    help="Maximum sequence length. Samples will be truncated if exceeds this value.",
)
parser.add_argument(
    "--limit-val-batches",
    type=float_or_int_or_none,
    required=False,
    default=2,
    help="Number of global batches used for validation if int. Fraction of validation dataset if float. Default is 2.",
)
parser.add_argument(
    "--micro-batch-size",
    type=int,
    required=False,
    default=64,
    help="Micro-batch size. Global batch size is inferred from this.",
)
parser.add_argument(
    "--pipeline-model-parallel-size",
    type=int,
    required=False,
    default=1,
    help="Pipeline model parallel size. Default is 1.",
)
parser.add_argument(
    "--tensor-model-parallel-size",
    type=int,
    required=False,
    default=1,
    help="Tensor model parallel size. Default is 1.",
)
parser.add_argument(
    "--accumulate-grad-batches",
    type=int,
    required=False,
    default=1,
    help="Gradient accumulation steps. Global batch size is inferred from this.",
)
parser.add_argument(
    "--biobert-spec-option",
    type=BiobertSpecOption,
    choices=[e.value for e in BiobertSpecOption],
    required=False,
    default=BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec.value,
    help="Biobert spec option to use for the model. Default is 'esm2_bert_layer_with_transformer_engine_spec'.",
)
parser.add_argument(
    "--nemo1-init-path",
    type=Path,
    required=False,
    help="Path to nemo1 file, if desired to load at init time.",
)
parser.add_argument(
    "--save-best-checkpoint",
    action="store_true",
    default=True,
    help="Save the best checkpoint based on the metric to monitor.",
)
parser.add_argument(
    "--save-last-checkpoint",
    action="store_true",
    default=True,
    help="Save the last checkpoint.",
)
parser.add_argument(
    "--metric-to-monitor-for-checkpoints",
    type=str,
    required=False,
    default="val_loss",
    help="The metric to monitor for checkpointing.",
)
parser.add_argument(
    "--save-top-k",
    type=int,
    required=False,
    default=2,
    help="Save the top k checkpoints.",
)
parser.add_argument(
    "--restore-from-checkpoint-path",
    type=Path,
    required=False,
    default=None,
    help="Path to the checkpoint directory to restore from. Will override `--resume-if-exists` when set.",
)

# ESM2 specific configuration (default: 650M)
parser.add_argument(
    "--random-mask-strategy",
    type=RandomMaskStrategy,
    choices=[e.value for e in RandomMaskStrategy],
    default=RandomMaskStrategy.ALL_TOKENS.value,
    help=f"""In ESM2 pretraining, 15%% of all tokens are masked and among which 10%% are replaced with a random token. This class controls the set of random tokens to choose from. Options are: '{"', '".join([e.value for e in RandomMaskStrategy])}'. Note that 'all_token' will introduce non-canonical amino acid tokens as effective mask tokens, and the resultant loss will appear lower than that from 'amino_acids_only'. Note that 'all_token' is the method used in hugging face as well as portions of fairseq.""",
)
parser.add_argument(
    "--num-layers",
    type=int,
    required=False,
    default=33,
    help="Number of layers in the model. Default is 33.",
)
parser.add_argument(
    "--hidden-size",
    type=int,
    required=False,
    default=1280,
    help="Hidden size of the model. Default is 1280.",
)
parser.add_argument(
    "--num-attention-heads",
    type=int,
    required=False,
    default=20,
    help="Number of attention heads in the model. Default is 20.",
)
parser.add_argument(
    "--ffn-hidden-size",
    type=int,
    required=False,
    default=4 * 1280,
    help="FFN hidden size of the model. Default is 4 * 1280.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(
        train_cluster_path=args.train_cluster_path,
        train_database_path=args.train_database_path,
        valid_cluster_path=args.valid_cluster_path,
        valid_database_path=args.valid_database_path,
        num_nodes=args.num_nodes,
        devices=args.num_gpus,
        min_seq_length=args.min_seq_length,
        max_seq_length=args.max_seq_length,
        result_dir=args.result_dir,
        wandb_project=args.wandb_project,
        wandb_offline=args.wandb_offline,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        num_dataset_workers=args.num_dataset_workers,
        biobert_spec_option=args.biobert_spec_option,
        lr=args.lr,
        micro_batch_size=args.micro_batch_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        experiment_name=args.experiment_name,
        resume_if_exists=args.resume_if_exists,
        nemo1_init_path=args.nemo1_init_path,
        restore_from_checkpoint_path=args.restore_from_checkpoint_path,
        save_best_checkpoint=args.save_best_checkpoint,
        save_last_checkpoint=args.save_last_checkpoint,
        metric_to_monitor_for_checkpoints=args.metric_to_monitor_for_checkpoints,
        save_top_k=args.save_top_k,
        save_every_n_steps=args.val_check_interval,
        random_mask_strategy=args.random_mask_strategy,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
    )
