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


# TODO(@mgreaves, @jstjohn, @jomitchell) Consider different abstractions for pretraining, inference, and fine-tuning and see
#  how they would address code duplication in the case of ESM2+Geneformer as well as a third hypothetical model that does
#  not share the same types/loaders, such as OpenFold. The design should be flexible enough to allow for those differeht
#  use cases and not hide too much complexity that a user would want to customize, while reducing code duplication
#  between scripts.

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Type, get_args

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.utils import logging
from nemo.utils.exp_manager import TimingCallback
from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.geneformer.api import FineTuneSeqLenBioBertConfig, GeneformerConfig
from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.biobert.model import BioBertConfig, BiobertSpecOption
from bionemo.llm.utils.datamodule_utils import float_or_int_or_none, infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbLoggerOptions, setup_nemo_lightning_logger


__all__: Sequence[str] = ("main", "parser")


def main(
    data_dir: Path,
    num_nodes: int,
    devices: int,
    seq_length: int,
    result_dir: Path,
    num_steps: int,
    limit_val_batches: int,
    val_check_interval: int,
    num_dataset_workers: int,
    biobert_spec_option: BiobertSpecOption,
    lr: float,
    micro_batch_size: int,
    accumulate_grad_batches: int,
    cosine_rampup_frac: float,
    cosine_hold_frac: float,
    experiment_name: str,
    resume_if_exists: bool,
    precision: PrecisionTypes,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_offline: bool = False,
    wandb_tags: Optional[List[str]] = None,
    wandb_group: Optional[str] = None,
    wandb_id: Optional[str] = None,
    wandb_anonymous: Optional[bool] = False,
    wandb_log_model: bool = False,
    create_tensorboard_logger: bool = False,
    nemo1_init_path: Path | None = None,
    restore_from_checkpoint_path: Path | None = None,
    save_last_checkpoint: bool = True,
    metric_to_monitor_for_checkpoints: str = "val_loss",
    save_top_k: int = 2,
    nsys_profiling: bool = False,
    nsys_start_step: int = 0,
    nsys_end_step: Optional[int] = None,
    nsys_ranks: List[int] = [0],
    config_class: Type[BioBertConfig] = GeneformerConfig,
    log_every_n_steps: int = 50,
    gc_interval: int = 0,
    aligned_megatron_ddp: bool = False,
    # TODO add datamodule class, and ability to change data step to get full support for pretraining workflows
) -> None:
    """Train a Geneformer model on single cell data.

    Args:
        data_dir (Path): Base directory for the data.
        num_nodes (int): Number of nodes to run on
        devices (int): number of devices
        seq_length (int): sequence length
        result_dir (Path): directory to store results, logs and checkpoints
        num_steps (int): number of steps to train the model for
        limit_val_batches (int): limit the number of validation global batches to this many
        val_check_interval (int): number of steps to periodically check the validation loss and save num_dataset_workers (
       int): num dataset workers
        biobert_spec_option (BiobertSpecOption): the biobert spec option (architecture) to use for this run
        lr (float): learning rate
        micro_batch_size (int): micro batch size, from this and parallelism settings we infer the global batch size
        cosine_rampup_frac (float): fraction of steps at the beginning of the run to ramp up the learning rate
        cosine_hold_frac (float): fraction of steps to hold the minimum learning rate at the end of the run
        experiment_name (str): experiment name, this is the name used for the wandb run, and the sub-directory of the
            result_dir that stores the logs and checkpoints.
        resume_if_exists (bool): attempt to resume if the checkpoint exists [FIXME @skothenhill this doesn't work yet]
        wandb_entity (str): The team posting this run (default: your username or your default team)
        wandb_project (str): The name of the project to which this run will belong.
        wandb_tags (List[str]): Tags associated with this run.
        wandb_group (str): A unique string shared by all runs in a given group
        wandb_offline (bool): Run offline (data can be streamed later to wandb servers).
        wandb_id (str): Sets the version, mainly used to resume a previous run.
        wandb_anonymous (bool): Enables or explicitly disables anonymous logging.
        wandb_log_model (bool): Save checkpoints in wandb dir to upload on W&B servers.
        create_tensorboard_logger (bool): create the tensorboard logger
        restore_from_checkpoint_path (path): If set, restores the model from the directory passed in. Expects the
            checkpoint to be created by using the ModelCheckpoint class and always_save_context=True.
        log_every_n_steps (int): log at this interval.
        gc_interval (int): if a value > 0 is provided, this will turn off automatic garbage collection and only run
            at this requested interval of train/val steps. This will likely slow down single GPU runs.
        aligned_megatron_ddp (bool): if activated, this will activate a number of communication optimizations that are
            good for clusters. This will likely slow down single node runs though.
    """
    # Create the result directory if it does not exist.
    result_dir.mkdir(parents=True, exist_ok=True)

    # Setup train/test/val data paths
    train_data_path = data_dir / "train"
    val_data_path = data_dir / "val"
    test_data_path = data_dir / "test"

    # Setup the strategy and trainer
    pipeline_model_parallel_size = 1
    tensor_model_parallel_size = 1
    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=num_nodes,
        devices=devices,
        accumulate_grad_batches=accumulate_grad_batches,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )
    if aligned_megatron_ddp:
        ddp: str | DistributedDataParallelConfig = DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,  # this should inherit from the optimizer config, but just in case...
        )
    else:
        ddp = "megatron"  # this will launch DistributedDataParallelConfig(check_for_nan_in_grad=True).

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        ddp=ddp,
        progress_interval=log_every_n_steps,
        find_unused_parameters=True,
        ckpt_include_optimizer=True,
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
    )

    # for wandb integration
    # Please refer to https://pytorch-lightning.readthedocs.io/en/0.7.6/api/pytorch_lightning.loggers.html"
    wandb_options: Optional[WandbLoggerOptions] = (
        None
        if wandb_project is None
        else WandbLoggerOptions(
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
        # Skip perplexity and disable forward output in the loss for speed
        RichModelSummary(max_depth=4),
        TimingCallback(),
        LearningRateMonitor(),
    ]

    if gc_interval > 0:
        callbacks.append(
            nl_callbacks.GarbageCollectionCallback(gc_interval_train=gc_interval, gc_interval_val=gc_interval)
        )

    if nsys_profiling:
        if nsys_end_step is None:
            nsys_end_step = num_steps
        callbacks.append(
            nl_callbacks.NsysCallback(
                start_step=nsys_start_step, end_step=nsys_end_step, ranks=nsys_ranks, gen_shape=True
            )
        )

    trainer = nl.Trainer(
        devices=devices,
        max_steps=num_steps,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=limit_val_batches,  # This controls upsampling and downsampling
        val_check_interval=val_check_interval,  # TODO(@jstjohn) Checkpoint saving is currently broken, fix and change this.
        log_every_n_steps=log_every_n_steps,
        num_nodes=num_nodes,
        callbacks=callbacks,
        use_distributed_sampler=False,
        plugins=nl.MegatronMixedPrecision(precision=precision),
    )

    preprocessor = GeneformerPreprocess(
        download_directory=train_data_path,
        medians_file_path=train_data_path / "medians.json",
        tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")

    # Configure the data module and model
    data = SingleCellDataModule(
        seq_length=seq_length,
        tokenizer=tokenizer,
        train_dataset_path=str(train_data_path),
        val_dataset_path=str(val_data_path),
        test_dataset_path=str(test_data_path),
        random_token_prob=0.02,  # changed to represent the incorrect setting we originally used.
        median_dict=median_dict,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        # persistent workers is supported when num_dataset_workers > 0
        persistent_workers=num_dataset_workers > 0,
        pin_memory=False,
        num_workers=num_dataset_workers,
    )
    geneformer_config = config_class(
        # TODO let users set different num layers/model shapes here to support bigger/smaller architectures
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=seq_length,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
        biobert_spec_option=biobert_spec_option,
        nemo1_ckpt_path=str(nemo1_init_path) if nemo1_init_path is not None else None,
        # handle checkpoint resumption here rather than auto-resume so this supports fine-tuning capabilities
        initial_ckpt_path=str(restore_from_checkpoint_path) if restore_from_checkpoint_path is not None else None,
    )

    # The lightning class owns a copy of the actual model, and a loss function, both of which are configured
    #  and lazily returned by the `geneformer_config` object defined above.
    model = biobert_lightning_module(
        geneformer_config,
        tokenizer=tokenizer,
        optimizer=MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=lr,
                # TODO(@jstjohn) try decoupled_lr
                optimizer="adam",
                use_distributed_optimizer=True,
                # Pass through fp16/bf16 settings to avoid errors around model having bf16 enabled but optimizer not.
                fp16=geneformer_config.fp16,
                bf16=geneformer_config.bf16,
            ),
            lr_scheduler=CosineAnnealingScheduler(
                max_steps=num_steps,
                # minimum learning rate is 1/100th of the initial learning rate, so eg lr=1e-3 -> min_lr=1e-5
                min_lr=lr / 100,
                warmup_steps=int(math.ceil(num_steps * cosine_rampup_frac)),
                interval="step",
                monitor="val_loss",
                constant_steps=int(math.ceil(num_steps * cosine_hold_frac)),
            ),
        ),
    )
    # Configure our custom Checkpointer
    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_last=save_last_checkpoint,
        monitor=metric_to_monitor_for_checkpoints,  # "val_loss",
        save_top_k=save_top_k,
        every_n_train_steps=val_check_interval,
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
            # TODO: uncomment this once nemo2 supports our fine-tuning workflow
            #  for now this happens inside of our config file in the configure_model step.
            # path=restore_from_checkpoint_path,
            resume_if_exists=resume_if_exists,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )


parser = argparse.ArgumentParser(description="Pretrain Geneformer with single cell data.")
parser.add_argument(
    "--data-dir",
    type=Path,
    required=True,
    help="Path to the data base directory, for example this might be "
    "/workspace/bionemo2/data/cellxgene_2023-12-15_small",
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
    default=1e-4,
    help="Learning rate for training. Default is 1e-4. With bigger global batches try 1e-3",
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
parser.add_argument(
    "--experiment-name", type=str, required=False, default="geneformer", help="Name of the experiment."
)
parser.add_argument("--wandb-entity", type=str, default=None, help="The team posting this run")
parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name ")
parser.add_argument("--wandb-tags", nargs="+", type=str, default=None, help="Tags associated with this run")
parser.add_argument(
    "--wandb-group", type=str, default=None, help="A unique string shared by all runs in a given group"
)
parser.add_argument(
    "--wandb-id", type=str, default=None, help="Sets the version, mainly used to resume a previous run"
)
parser.add_argument("--wandb-anonymous", action="store_true", help="Enable or explicitly disable anonymous logging")
parser.add_argument(
    "--wandb-log-model", action="store_true", help="Save checkpoints in wandb dir to upload on W&B servers"
)
parser.add_argument("--wandb-offline", action="store_true", help="Use wandb in offline mode")
parser.add_argument(
    "--cosine-rampup-frac",
    type=float,
    required=False,
    default=0.01,
    help="Fraction of steps in which to ramp up the learning rate. Default is 0.01.",
)
parser.add_argument(
    "--cosine-hold-frac",
    type=float,
    required=False,
    default=0.05,
    help="Fraction of final steps in which to hold the minimum LR. Default is 0.05.",
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
    default=10000,
    help="Number of steps to use for training. Default is 10000.",
)
parser.add_argument(
    "--num-dataset-workers",
    type=int,
    required=False,
    default=0,
    help="Number of steps to use for training. Default is 0.",
)
parser.add_argument(
    "--val-check-interval",
    type=int,
    required=False,
    default=10000,
    help="Number of steps to use for training. Default is 10000.",
)
parser.add_argument(
    "--log-every-n-steps",
    type=int,
    required=False,
    default=50,
    help="Number of steps between logging. Default is 50.",
)
parser.add_argument(
    "--seq-length",
    type=int,
    required=False,
    default=2048,
    help="Sequence length of cell. Default is 2048.",
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
    default=BiobertSpecOption.bert_layer_with_transformer_engine_spec.value,
    help="Biobert spec option to use for the model. Default is 'bert_layer_with_transformer_engine_spec'.",
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

# TODO consider whether nemo.run or some other method can simplify this config class lookup.
config_class_options: Dict[str, Type[BioBertConfig]] = {
    "GeneformerConfig": GeneformerConfig,
    "FineTuneSeqLenBioBertConfig": FineTuneSeqLenBioBertConfig,
}


def config_class_type(desc: str) -> Type[BioBertConfig]:
    try:
        return config_class_options[desc]
    except KeyError:
        raise argparse.ArgumentTypeError(
            f"Do not recognize key {desc}, valid options are: {config_class_options.keys()}"
        )


parser.add_argument(
    "--training-model-config-class",
    type=config_class_type,
    default="GeneformerConfig",
    help="Model configs link model classes with losses, and handle model initialization (including from a prior "
    "checkpoint). This is how you can fine-tune a model. First train with one config class that points to one model "
    "class and loss, then implement and provide an alternative config class that points to a variant of that model "
    "and alternative loss. In the future this script should also provide similar support for picking different data "
    f"modules for fine-tuning with different data types. Choices: {config_class_options.keys()}",
)
parser.add_argument(
    "--nsys-profiling",
    action="store_true",
    default=False,
    help="Enable targeted `nsys` profiling on the training loop for a defined step range. To actually get profiling output you must run the whole program with `nsys`. For example: "
    " `nsys profile -s none -o output_report_name -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop  [regular python command here]`",
)
# start, end, rank
parser.add_argument(
    "--nsys-start-step",
    type=int,
    required=False,
    default=0,
    help="Start nsys profiling after this step.",
)
parser.add_argument(
    "--nsys-end-step",
    type=int,
    required=False,
    help="End nsys profiling after this step.",
)
# rank as list of integers
parser.add_argument(
    "--nsys-ranks",
    type=int,
    nargs="+",
    required=False,
    default=[0],
    help="Enable nsys profiling for these ranks.",
)

parser.add_argument(
    "--gc-interval",
    type=int,
    required=False,
    default=0,
    help="Run garbage collection on the cluster every --gc-interval steps, 0 to disable (default). Keeping gc interval"
    " in sync this way on large cluster runs is important for training performance.",
)

parser.add_argument(
    "--aligned-megatron-ddp",
    action="store_true",
    default=False,
    help="By default param overlap/etc is disabled in megatron, this enables all of those settings. This is probably "
    "good for cluster performance.",
)

if __name__ == "__main__":
    # Parse the arguments and pull them out into local variables for ease of future refactor to a
    #   config management system.
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        num_nodes=args.num_nodes,
        devices=args.num_gpus,
        seq_length=args.seq_length,
        result_dir=args.result_dir,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        wandb_group=args.wandb_group,
        wandb_id=args.wandb_id,
        wandb_anonymous=args.wandb_anonymous,
        wandb_log_model=args.wandb_log_model,
        wandb_offline=args.wandb_offline,
        num_steps=args.num_steps,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        num_dataset_workers=args.num_dataset_workers,
        biobert_spec_option=args.biobert_spec_option,
        lr=args.lr,
        micro_batch_size=args.micro_batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        cosine_rampup_frac=args.cosine_rampup_frac,
        cosine_hold_frac=args.cosine_hold_frac,
        precision=args.precision,
        experiment_name=args.experiment_name,
        resume_if_exists=args.resume_if_exists,
        nemo1_init_path=args.nemo1_init_path,
        nsys_profiling=args.nsys_profiling,
        nsys_start_step=args.nsys_start_step,
        nsys_end_step=args.nsys_end_step,
        nsys_ranks=args.nsys_ranks,
        restore_from_checkpoint_path=args.restore_from_checkpoint_path,
        config_class=args.training_model_config_class,
        save_last_checkpoint=args.save_last_checkpoint,
        metric_to_monitor_for_checkpoints=args.metric_to_monitor_for_checkpoints,
        save_top_k=args.save_top_k,
        log_every_n_steps=args.log_every_n_steps,
        gc_interval=args.gc_interval,
        aligned_megatron_ddp=args.aligned_megatron_ddp,
    )
