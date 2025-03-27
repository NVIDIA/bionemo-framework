# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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
from typing import List, Optional

# TODO add back support for slurm resilience.
# import nvidia_resiliency_ext.ptl_resiliency as res_module
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.data import MockDataModule, PreTrainingDataModule
from nemo.collections.llm.gpt.data.megatron.hyena.config import parse_dataset_config
from nemo.collections.llm.gpt.data.megatron.hyena.evo2_dataset import Evo2Dataset, Evo2DatasetPadEodLossMask
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.utils.exp_manager import TimingCallback

# Add import for Mamba models
from bionemo.evo2.models.mamba import MAMBA_MODEL_OPTIONS, MambaModel
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger

# TODO(dorotat_nv) remove when https://github.com/NVIDIA/bionemo-framework/issues/749
from bionemo.testing.testing_callbacks import SignalAfterGivenStepCallback
from eden_dataset import FastaDataModule

torch._dynamo.config.suppress_errors = True


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse arguments for Evo2 model training."""
    parser = argparse.ArgumentParser(
        description="Train a Hyena model using NeMo 2.0.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_group = parser.add_mutually_exclusive_group(required=True)

    data_group.add_argument(
        "-d",
        "--dataset-config",
        type=str,
        help="Path to the blended / weighted training dataset configuration YAML.",
    )
    data_group.add_argument(
        "--mock-data",
        action="store_true",
        help="Train with Mock data (for testing/debugging), either set this or provide a dataset config.",
    )
    data_group.add_argument(
        "--fasta-data",
        action="store_true",
        help="Train with Fasta data, either set this or provide a dataset config.",
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Absolute path to the dataset directory. Defaults to using the absolute or relative paths (dataset_prefix) specified in the dataset config YAML.",
    )

    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to use for training, defaults to 1.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use for training, defaults to 1.")
    parser.add_argument("--seq-length", type=int, default=8192, help="Training sequence length")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Order of tensor parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--pipeline-model-parallel-size", type=int, default=1, help="Order of pipeline parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--context-parallel-size", type=int, default=1, help="Order of context parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--create-tensorboard-logger", action="store_true", default=False, help="Create a tensorboard logger."
    )
    parser.add_argument("--wandb-entity", type=str, default=None, help="The team posting this run")
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name ")
    parser.add_argument("--wandb-tags", nargs="+", type=str, default=None, help="Tags associated with this run")
    parser.add_argument(
        "--wandb-group", type=str, default=None, help="A unique string shared by all runs in a given group"
    )
    parser.add_argument(
        "--wandb-job-type",
        type=str,
        default=None,
        help="A unique string representing a type of run, which is useful when you're grouping runs together into larger experiments using group.",
    )
    parser.add_argument(
        "--wandb-id", type=str, default=None, help="Sets the version, mainly used to resume a previous run"
    )
    parser.add_argument(
        "--wandb-anonymous", action="store_true", help="Enable or explicitly disable anonymous logging"
    )
    parser.add_argument(
        "--wandb-log-model", action="store_true", help="Save checkpoints in wandb dir to upload on W&B servers"
    )
    parser.add_argument("--wandb-offline", action="store_true", help="Use wandb in offline mode")
    parser.add_argument("--sequence-parallel", action="store_true", help="Set to enable sequence parallelism.")
    parser.add_argument("--fp8", action="store_true", help="Set to enable FP8")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro-batch size for data-parallel training.")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Global batch size for training. If set to None, infer it from the TP, CP, and PP parameters.",
    )
    parser.add_argument(
        "--grad-acc-batches", type=int, default=1, help="Number of batches to accumulate gradients over."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Number of training optimizer update steps. This controls the total number of steps as well as the "
        "shape of the learning rate curve.",
        default=500000,
    )
    parser.add_argument(
        "--early-stop-on-step",
        type=int,
        help="Stop training on this step, if set. This may be useful for testing or debugging purposes.",
    )
    parser.add_argument(
        "--val-check-interval", type=int, help="Number of steps between validation measurements and model checkpoints."
    )
    parser.add_argument("--grad-reduce-in-fp32", action="store_true", default=False, help="Gradient reduce in FP32.")
    parser.add_argument(
        "--fp8-wgrad",
        action="store_true",
        default=False,
        help="Faster option that is maybe less accurate (TBD) when using fp8.",
    )
    parser.add_argument("--use-megatron-comm-overlap-llama3-8k", action="store_true", default=False)
    parser.add_argument(
        "--tp-comm-overlap-backend",
        type=str,
        choices=["nccl", "mpi", "gloo"],
        default="nccl",
        help="TP communication backend to use. Defaults to 'nccl'.",
    )
    parser.add_argument("--align-param-gather", action="store_true", default=False)
    # parser.add_argument("--straggler-detection", action="store_true", default=False)
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["hyena", "mamba"],
        default="hyena",
        help="Model architecture family to use. Choose between 'hyena' and 'mamba'.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=sorted(list(HYENA_MODEL_OPTIONS.keys()) + list(MAMBA_MODEL_OPTIONS.keys())),
        default="7b",
        help="Model size/configuration to use. Options depend on the selected model-type.",
    )
    parser.add_argument(
        "--add-bias-output",
        action="store_true",
        default=False,
        help="Add bias to the output layer to enable learning a simple prior.",
    )
    parser.add_argument(
        "--result-dir", type=Path, required=False, default=Path("./results"), help="Path to the result directory."
    )
    parser.add_argument("--experiment-name", type=str, required=False, default="evo2", help="Name of the experiment.")

    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=20,
        help="Number of validation steps",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=1,
        required=False,
        help="Number of steps between logging.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Directory to restore an initial checkpoint from. Use this for supervised fine-tuning.",
    )
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help="Adam optimizer beta1 parameter.",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.95,
        help="Adam optimizer beta2 parameter.",
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1e-8,
        help="Adam optimizer epsilon parameter. The inverse of this value (1/eps) represents the maximum adaptive learning rate per parameter.",
    )
    parser.add_argument(
        "--restore-optimizer-from-ckpt",
        action="store_true",
        help="Restore optimizer state from initial checkpoint. Defaults to False.",
    )
    parser.add_argument(
        "--no-average-in-collective",
        action="store_true",
        default=False,
        help="Avaerage optimizer state in collective rather than dividing by dp size and summing.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set random seed for training.")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers to use for data loading.")
    parser.add_argument(
        "--gc-interval",
        type=int,
        default=0,
        help="Set to a value > 0 if you want to synchronize garbage collection, will do gc every gc-interval steps.",
    )
    parser.add_argument(
        "--enable-preemption",
        action="store_true",
        default=False,
        help="Enable preemption hooks. If enabled this will save a checkpoint whenever slurm exits.",
    )
    parser.add_argument(
        "--ckpt-async-save",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ckpt-format",
        type=str,
        choices=["torch_dist", "zarr"],
        default="torch_dist",
        help="Specify checkpoint format to use. Defaults to 'torch_dist', as 'zarr' is deprecated. Only use if "
        "resuming training from a zarr checkpoint.",
    )
    parser.add_argument(
        "--eod-pad-in-loss-mask",
        action="store_true",
        default=False,
        help="Do not predict EOD/Pad tokens (typical default, but not default in original evo2).",
    )
    parser.add_argument(
        "--cross-entropy-loss-fusion",
        action="store_true",
        default=False,
        help="Use the faster, but maybe less accurate fused form of cross entropy, "
        "which also has bf16 grads internally.",
    )
    parser.add_argument(
        "--no-fp32-residual-connection",
        action="store_true",
        default=False,
        help="If set, turn off fp32 residual connections which may be faster but may impact accuracy.",
    )
    parser.add_argument(
        "--debug-ddp-parity-freq",
        type=int,
        default=0,
        help="Set to value > 0 to debug DDP weight parity between ranks.",
    )
    parser.add_argument(
        "--hybrid-override-pattern",
        type=str,
        help="Override the hybrid override pattern in the config (specifies hyena layer ordering and type).",
    )
    parser.add_argument(
        "--num-layers", type=int, help="If set, override the number of layers specified in the requested config."
    )
    parser.add_argument(
        "--create-tflops-callback",
        action="store_true",
        default=False,
        help="Enable tflops calculation callback for Hyena / Evo2. Defaults to False.",
    )
    parser.add_argument(
        "--log-parameters-and-shapes",
        action="store_true",
        default=False,
        help="Log training parameters shapes and dtypes for debugging.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Min learning rate in cosine annealing.")
    parser.add_argument("--warmup-steps", type=int, default=2500, help="Number of warmup steps in cosine annealing")
    # NSYS profiling/tooling arguments
    parser.add_argument(
        "--nsys-profiling",
        action="store_true",
        default=False,
        help="Enable targeted `nsys` profiling on the training loop for a defined step range. To actually get profiling"
        " output you must run the whole program with `nsys`. For example: "
        " `nsys profile -s none -o output_report_name -t cuda,nvtx --force-overwrite true "
        "--capture-range=cudaProfilerApi --capture-range-end=stop  [regular python command here]`",
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
    parser.add_argument(
        "--no-renormalize-loss",
        action="store_true",
        default=False,
        help="Do not renormalize the loss weights.",
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
        "--activation-checkpoint-recompute-num-layers",
        type=int,
        help="If set, override the default value set in the config.",
    )
    parser.add_argument(
        "--disable-checkpointing",
        action="store_false",
        default=True,
        dest="create_checkpoint_callback",
        help="Disable creating a ModelCheckpoint callback.",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Grad clip value. Note that when using DDP this may need to be inflated.",
    )
    parser.add_argument(
        "--seq-len-interpolation-factor",
        type=float,
        help="Adjusts the linear scaling of ROPE (Rotary Position Embedding) for context extension. "
        "Set this factor relative to your base context length e.g., for an original context length of 8192 and "
        "an extended context length of 524288, use 524288/8192 = 64.",
    )
    parser.add_argument(
        "--overlap-param-gather",
        action="store_true",
        default=False,
        help="Overlap the parameter gather with the optimizer step. This is currently disabled due to a NeMo bug "
        "when using DDP. Making this an option defaulting to False is a temporary solution until the bug is fixed.",
    )
    parser.add_argument(
        "--overlap-grad-reduce",
        action="store_true",
        default=False,
        help="Overlap the gradient reduce with the optimizer step.",
    )
    parser.add_argument(
        "--hidden-dropout",
        type=float,
        default=0.0,
        help="Dropout probability for the hyena layers",
    )
    parser.add_argument(
        "--log-num-zeros-in-grad",
        action="store_true",
        default=False,
        help="Log the number of zeros in the gradient.",
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        default=0.0,
        help="Dropout probability for the attention layers.",
    )
    recompute_group = parser.add_mutually_exclusive_group(required=False)
    recompute_group.add_argument("--no-activation-checkpointing", action="store_true", default=False)
    recompute_group.add_argument("--selective-activation-checkpointing", action="store_true", default=False)
    return parser.parse_args(args=args)


def train(args: argparse.Namespace) -> nl.Trainer:
    """Main function to run Evo2 training."""
    # Instantiate tokenizer.
    tokenizer = get_nmt_tokenizer(
        "byte-level",
    )

    # Infer global batch size.
    global_batch_size = args.global_batch_size
    if global_batch_size is None:
        global_batch_size = infer_global_batch_size(
            micro_batch_size=args.micro_batch_size,
            num_nodes=args.num_nodes,
            devices=args.devices,
            accumulate_grad_batches=args.grad_acc_batches,
            tensor_model_parallel_size=args.tensor_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            context_model_parallel_size=args.context_parallel_size,
        )
    if args.mock_data:
        data_module = MockDataModule(
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=global_batch_size,
            num_workers=args.workers,
            tokenizer=tokenizer,
        )

    elif args.fasta_data:
        data = FastaDataModule(
            fasta_file="/workspace/bionemo2/bionemo-evo2/bcr_small_ctrl_tags.fasta",
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=global_batch_size,
            num_workers=args.workers,
            tokenizer=tokenizer,
        )
    else:
        blended_dataset_config = parse_dataset_config(
            dataset_config_path=args.dataset_config, dataset_path=args.dataset_dir
        )
        dataset_cls = Evo2DatasetPadEodLossMask if args.eod_pad_in_loss_mask else Evo2Dataset
        # Instantiate pre-training module.
        data_module = PreTrainingDataModule(
            paths=blended_dataset_config,
            dataset_cls=dataset_cls,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=global_batch_size,
            seed=args.seed,
            num_workers=args.workers,
            tokenizer=tokenizer,
            eod_mask_loss=args.eod_pad_in_loss_mask,
        )

    if args.no_activation_checkpointing:
        activation_checkpointing_args = {
            "recompute_granularity": None,
            "recompute_method": None,
            "recompute_num_layers": None,
        }
    elif args.selective_activation_checkpointing:
        activation_checkpointing_args = {
            "recompute_granularity": "selective",
            "recompute_method": None,
            "recompute_num_layers": None,
        }
    else:
        if args.activation_checkpoint_recompute_num_layers is not None:
            activation_checkpointing_args = {
                "recompute_num_layers": args.activation_checkpoint_recompute_num_layers,
            }
        else:
            activation_checkpointing_args = {}

    # Retrieve model config.
    config_modifiers_init = {
        "tp_comm_overlap": args.use_megatron_comm_overlap_llama3_8k,
        "seq_length": args.seq_length,
        "hidden_dropout": args.hidden_dropout,
        "attention_dropout": args.attention_dropout,
        "to_upper": "weighted" if args.no_renormalize_loss else "normalized_weighted",
        "distribute_saved_activations": False if args.sequence_parallel else True,
        "cross_entropy_loss_fusion": args.cross_entropy_loss_fusion,
        "fp32_residual_connection": not args.no_fp32_residual_connection,
        "add_bias_output": args.add_bias_output,
        **activation_checkpointing_args,
    }
    if args.hybrid_override_pattern:
        config_modifiers_init["hybrid_override_pattern"] = args.hybrid_override_pattern
    if args.num_layers:
        config_modifiers_init["num_layers"] = args.num_layers

    # Create model based on selected model type
    if args.model_type == "hyena":
        if args.model_size not in HYENA_MODEL_OPTIONS:
            raise ValueError(f"Invalid model size for Hyena: {args.model_size}")
        model_config = HYENA_MODEL_OPTIONS[args.model_size](**config_modifiers_init)
        model = llm.HyenaModel(model_config, tokenizer=data.tokenizer)
    else:  # mamba
        if args.model_size not in MAMBA_MODEL_OPTIONS:
            raise ValueError(f"Invalid model size for Mamba: {args.model_size}")
        add_bias_output = config_modifiers_init.pop("add_bias_output")
        if add_bias_output:
            raise ValueError("Bias output is not supported for Mamba models.")
        model_config = MAMBA_MODEL_OPTIONS[args.model_size](**config_modifiers_init)
        model = MambaModel(model_config, tokenizer=data.tokenizer)

    # Setup callbacks.
    callbacks = [
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
        TimingCallback(),
    ]

    if args.early_stop_on_step:
        # Ask the trainer to stop by setting should_stop to True rather than emitting a kill signal.
        callbacks.append(
            SignalAfterGivenStepCallback(
                stop_step=args.early_stop_on_step, stop_before_step=True, use_trainer_should_stop=True
            )
        )
    if args.enable_preemption:
        callbacks.append(nl_callbacks.PreemptionCallback())
    if args.debug_ddp_parity_freq > 0:
        callbacks.append(nl_callbacks.DdpParityChecker(interval=args.debug_ddp_parity_freq))
    if args.log_parameters_and_shapes:
        callbacks.append(nl_callbacks.ParameterDebugger())
    if args.create_tflops_callback:
        # Add callback that logs the tera-FLOPS per second per GPU during training.
        flop_meas_callback = FLOPsMeasurementCallback(
            model_config,
            data,
            "hyena",
        )
        callbacks.append(flop_meas_callback)

    # TODO(@cye): Add this back when it works with 24.12.
    # if args.straggler_detection:
    #     callbacks.append(
    #         res_module.StragglerDetectionCallback(
    #             report_time_interval=300,
    #             calc_relative_gpu_perf=True,
    #             calc_individual_gpu_perf=True,
    #             num_gpu_perf_scores_to_print=5,
    #             gpu_relative_perf_threshold=0.7,
    #             gpu_individual_perf_threshold=0.7,
    #             stop_if_detected=True,
    #             enable_ptl_logging=True,
    #         )
    #     )
    if args.use_megatron_comm_overlap_llama3_8k:
        # Pick the floating point appropriate config.
        if args.fp8:
            tp_comm_overlap_cfg = userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192
        else:
            tp_comm_overlap_cfg = userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192
        callbacks.append(
            MegatronCommOverlapCallback(
                tp_comm_overlap=model_config.tp_comm_overlap,
                tp_comm_overlap_cfg=tp_comm_overlap_cfg,
                tp_comm_bootstrap_backend=args.tp_comm_overlap_backend,
                wgrad_deferral_limit=22,  # default from NeMo
                overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to an issue with checkpointing.
                align_param_gather=args.align_param_gather,
            )
        )

    if args.gc_interval > 0:
        callbacks.append(
            nl_callbacks.GarbageCollectionCallback(
                gc_interval_train=args.gc_interval, gc_interval_val=args.gc_interval
            )
        )
    if args.nsys_profiling:
        if args.nsys_end_step is None:
            nsys_end_step = args.max_steps
        else:
            nsys_end_step = args.nsys_end_step
        callbacks.append(
            nl_callbacks.NsysCallback(
                start_step=args.nsys_start_step, end_step=nsys_end_step, ranks=args.nsys_ranks, gen_shape=True
            )
        )

    loggers = []
    nemo_logger_kwargs = {}
    if (not args.no_wandb) and args.wandb_project:
        wandb_logger = WandbLogger(
            name=(
                f"evo2-size-{args.model_size}-TP{args.tensor_parallel_size}-"
                f"PP{args.pipeline_model_parallel_size}-CP{args.context_parallel_size}"
                f"-GBS{global_batch_size}-MBS{args.micro_batch_size}-SkipLossRenorm{args.no_renormalize_loss}"
                f"-NOAC{args.no_activation_checkpointing}-SELAC{args.selective_activation_checkpointing}"
                f"-ACRNL{model_config.recompute_num_layers}"
                f"-PAT{model_config.hybrid_override_pattern}"
                f"-F32R{model_config.fp32_residual_connection}"
                f"-FCE{model_config.cross_entropy_loss_fusion}"
                f"-AIC{not args.no_average_in_collective}"
                f"-PEOD{args.eod_pad_in_loss_mask}"
                f"-BO{args.add_bias_output}"
                f"-GCLP{args.clip_grad}"
                f"-HDO{args.hidden_dropout}"
                f"-ADO{args.attention_dropout}"
                f"-LR{args.lr}-MINLR{args.min_lr}-WUSTEPS{args.warmup_steps}-WD{args.wd}"
                f"-B1{args.adam_beta1}-B2{args.adam_beta2}-EPS{args.adam_eps}"
                f"-GRFP32{args.grad_reduce_in_fp32}-FP8WG{args.fp8_wgrad and args.fp8}"
                f"-OGR{args.overlap_grad_reduce}-OPG{args.overlap_param_gather}"
                f"-NODES{args.num_nodes}-FP8{args.fp8}"
            ),
            group=args.wandb_group,
            job_type=args.wandb_job_type,
            id=args.wandb_id,
            anonymous=args.wandb_anonymous,
            log_model=args.wandb_log_model,
        )
    )
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=args.result_dir,
        name=args.experiment_name,
        initialize_tensorboard_logger=args.create_tensorboard_logger,
        wandb_config=wandb_config,
    )

    if args.create_checkpoint_callback:
        checkpoint_path = str(Path(nemo_logger.save_dir) / "checkpoints")
        checkpoint_callback = ModelCheckpoint(
            every_n_train_steps=args.val_check_interval,
            dirpath=checkpoint_path,
            save_top_k=5,
            always_save_context=True,
            save_optim_on_train_end=True,
            save_context_on_train_end=True,
        )
        callbacks.append(checkpoint_callback)

        auto_resume = nl.AutoResume(
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
            resume_past_end=False,
            resume_from_directory=checkpoint_path,
            restore_config=(
                RestoreConfig(
                    path=args.ckpt_dir,
                    load_model_state=True,
                    load_optim_state=args.restore_optimizer_from_ckpt,
                )
                if args.ckpt_dir
                else None
            ),
        )
    else:
        auto_resume = None

    ddp: DistributedDataParallelConfig = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        overlap_grad_reduce=args.overlap_grad_reduce,
        overlap_param_gather=args.overlap_param_gather,  # Verify that this works using
        grad_reduce_in_fp32=args.grad_reduce_in_fp32,
        align_param_gather=args.align_param_gather,
        average_in_collective=not args.no_average_in_collective,
    )
    # Initialize Megatron Strategy and Trainer.
    strategy = nl.MegatronStrategy(
        ddp=ddp,
        tensor_model_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=args.sequence_parallel,
        ckpt_load_optimizer=True,
        ckpt_save_optimizer=True,
        ckpt_async_save=args.ckpt_async_save,
        save_ckpt_format=args.ckpt_format,
        ckpt_load_strictness="log_all",  # or rebasing to https://github.com/NVIDIA/NeMo/pull/11988/files#diff-7667eae242a8ef776bff78cd08e79bc81df4896a450f0a781f6ed317a3dfb7ffR139
    )
    trainer = nl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            grad_reduce_in_fp32=args.grad_reduce_in_fp32,
            fp8="hybrid" if args.fp8 else None,
            fp8_amax_history_len=16 if args.fp8 else 1,
            fp8_amax_compute_algo="max" if args.fp8 else "most_recent",
            fp8_wgrad=args.fp8
            and (
                args.fp8_wgrad or args.use_megatron_comm_overlap_llama3_8k
            ),  # faster and less accurate when set to True, and MUST be True if using TP communication overlap
        ),
        val_check_interval=args.val_check_interval,
        enable_checkpointing=args.create_checkpoint_callback,
    )

    # Logger setup
    nemo_logger.setup(
        trainer,
        resume_if_exists=True,
    )

    if auto_resume is not None:
        auto_resume.setup(trainer, model)

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=args.lr,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.wd,
        clip_grad=args.clip_grad,
        adam_eps=args.adam_eps,
        use_distributed_optimizer=True,
        log_num_zeros_in_grad=args.log_num_zeros_in_grad,
        bf16=True,
    )

    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
    )

    opt = MegatronOptimizerModule(opt_config, sched, no_weight_decay_cond=model_config.hyena_no_weight_decay_cond_fn)
    opt.connect(model)

    # Start training
    trainer.fit(model, data_module)
    return trainer


def main():
    """Parsing args and running evo2 training."""
    args = parse_args()
    train(args=args)


if __name__ == "__main__":
    main()
