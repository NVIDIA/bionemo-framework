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

import math
import pathlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Callable, Generic, List, Literal, Optional, Type, TypeVar

import nemo_run as run
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.utils import logging
from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary
from tokenizers import Tokenizer
from torch.nn import functional as F

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.geneformer.api import GeneformerConfig
from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.geneformer.model.finetune_token_regressor import FineTuneSeqLenBioBertConfig
from bionemo.llm.model.biobert.lightning import BioBertLightningModule
from bionemo.llm.model.biobert.model import BioBertGenericConfig, BiobertSpecOption
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbLoggerOptions, setup_nemo_lightning_logger


run.Config


@dataclass
class DataConfig:
    data_dir: str
    result_dir: str = "./results"
    seq_length: int = 2048
    num_dataset_workers: int = 0
    micro_batch_size: int = 8

    @property
    def train_data_path(self) -> str:
        return self.data_dir + "/train"

    @property
    def val_data_path(self) -> str:
        return self.data_dir + "/val"

    @property
    def test_data_path(self) -> str:
        return self.data_dir + "/test"


@run.cli.factory
@run.autoconvert
def small_data_config(
    data_dir="/workspaces/bionemo-fw-ea/data/cellxgene_2023-12-15_small/processed_data",
) -> DataConfig:
    # NOTE theoretically we could validate that this stuff exists.
    return DataConfig(data_dir=data_dir)


@run.cli.factory
@run.autoconvert
def full_geneformer_data_config(
    data_dir="/workspaces/bionemo-fw-ea/data/cellxgene_2023-12-15/processed_data",
) -> DataConfig:
    # NOTE theoretically we could validate that this stuff exists.
    return DataConfig(data_dir=data_dir)


@dataclass
class GeneformerDataArtifacts:
    tokenizer: Tokenizer  # TODO(SKH) typing isnt right
    median_dict: dict


def geneformer_preprocess_recipe(data_config: DataConfig) -> GeneformerDataArtifacts:
    preprocessor = GeneformerPreprocess(
        download_directory=pathlib.Path(data_config.train_data_path),
        medians_file_path=pathlib.Path(data_config.train_data_path + "/medians.json"),
        tokenizer_vocab_path=pathlib.Path(data_config.train_data_path + "/geneformer.vocab"),
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")
            raise ValueError("Preprocessing failed to create tokenizer and/or median dictionary.")
    return GeneformerDataArtifacts(tokenizer=tokenizer, median_dict=median_dict)


def singlecell_data_module(data_config: DataConfig, global_batch_size: int) -> SingleCellDataModule:
    geneformer_data_artifacts: GeneformerDataArtifacts = geneformer_preprocess_recipe(data_config)
    data = SingleCellDataModule(
        seq_length=data_config.seq_length,
        tokenizer=geneformer_data_artifacts.tokenizer,
        train_dataset_path=data_config.train_data_path,
        val_dataset_path=data_config.val_data_path,
        test_dataset_path=data_config.test_data_path,
        random_token_prob=0.02,  # changed to represent the incorrect setting we originally used.
        median_dict=geneformer_data_artifacts.median_dict,
        micro_batch_size=data_config.micro_batch_size,
        global_batch_size=global_batch_size,
        # persistent workers is supported when num_dataset_workers > 0
        persistent_workers=data_config.num_dataset_workers > 0,
        pin_memory=False,
        num_workers=data_config.num_dataset_workers,
    )
    return data


@dataclass
class ParallelConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    accumulate_grad_batches: int = 1
    ddp: Literal["megatron"] = "megatron"
    remove_unused_parameters: bool = True
    num_devices: int = 1
    num_nodes: int = 1


@run.cli.factory
@run.autoconvert
def simple_parallel_recipe(
    tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1, num_devices: int = 1
) -> ParallelConfig:
    # TODO validatorssssssss, make sure we get everythign right here.
    assert (
        num_devices >= tensor_model_parallel_size * pipeline_model_parallel_size
    ), "devices must be divisible by tensor_model_parallel_size * pipeline_model_parallel_size"
    return ParallelConfig(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        num_devices=num_devices,
    )


@dataclass
class TrainingConfig:
    max_steps: int
    limit_val_batches: int
    val_check_interval: int
    precision: PrecisionTypes = "bf16-mixed"
    accelerator: str = "gpu"


@run.cli.factory
@run.autoconvert
def default_trainer_config() -> TrainingConfig:
    return TrainingConfig(max_steps=55000, limit_val_batches=2, val_check_interval=100)


def setup_trainer_from_configs(parallel_config: ParallelConfig, training_config: TrainingConfig) -> nl.Trainer:
    # Because this returns a trainer, and trainer is not an argument to the entrypoint, this is not a factory.
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
        ddp="megatron",
        find_unused_parameters=True,
        ckpt_include_optimizer=True,
    )

    trainer = nl.Trainer(
        devices=parallel_config.num_devices,
        max_steps=training_config.max_steps,
        accelerator=training_config.accelerator,
        strategy=strategy,
        limit_val_batches=training_config.limit_val_batches,  # This controls upsampling and downsampling
        val_check_interval=training_config.val_check_interval,  # TODO(@jstjohn) Checkpoint saving is currently broken, fix and change this.
        num_nodes=parallel_config.num_nodes,
        callbacks=[
            RichModelSummary(max_depth=4),
            LearningRateMonitor(),
        ],
        plugins=nl.MegatronMixedPrecision(precision=training_config.precision),
    )
    return trainer


ModelConfigT = TypeVar("ModelConfigT", bound=BioBertGenericConfig)


@dataclass
class ExposedModelConfig(Generic[ModelConfigT], ABC):
    """ExposedConfigs are meant to be used as a way to expose a subset of the underlying model config.

    Due to the fact that some fields in the underlying TransformerConfig are not serializable, it must be wrapped.
    We tie each concrete ExposedModelConfig to a specific ModelConfigT, which is a subclass of BioBertGenericConfig.
    Then, we expect implementors to implement a method using the same type called `model_class`, this returns the literal
    type ModelConfigT.

    exposed_to_internal_model_config is then a universal method that unpacks the exposed config and returns the underlying model config.

    Users are expected to choose a recipe that returns the ExposedModelConfig of interest and parameterize it accordingly.
    Developers should carefully create recipes and factories that reflect common usescases, and these will be specified on the CLI.
    """

    @abstractmethod
    def model_class(self) -> Type[ModelConfigT]: ...

    def exposed_to_internal_model_config(self) -> ModelConfigT:
        # This is bad because it doesnt actually leverage any generics
        cls: Type[ModelConfigT] = self.model_class()
        return cls(**asdict(self))


@dataclass
class ExposedGeneformerConfig(ExposedModelConfig[GeneformerConfig]):
    """NeMo run does not like GeneformerConfig due to use its use of lambdas.

    So I basicaly need a method that does This -> GeneformerConfig
    then use regular recipes/factories on the parent and do this transform at the last step.
    """

    params_dtype: PrecisionTypes
    pipeline_dtype: PrecisionTypes
    autocast_dtype: PrecisionTypes
    num_layers: int = 6
    hidden_size: int = 256
    ffn_hidden_size: int = 512
    num_attention_heads: int = 4
    seq_length: int = 512
    fp32_residual_connection: bool = False
    hidden_dropout: float = 0.02
    init_method_std: float = 0.02
    kv_channels: Optional[int] = None
    apply_query_key_layer_scaling: bool = False
    make_vocab_size_divisible_by: int = 128
    masked_softmax_fusion: bool = True
    fp16_lm_cross_entropy: bool = False
    gradient_accumulation_fusion: bool = False
    layernorm_zero_centered_gamma: bool = False
    layernorm_epsilon: float = 1.0e-12
    activation_func: Callable = F.gelu
    qk_layernorm: bool = False
    apply_residual_connection_post_layernorm: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    get_attention_mask_from_fusion: bool = False
    attention_dropout: float = 0.1
    share_embeddings_and_output_weights: bool = True
    enable_autocast: bool = False
    nemo1_ckpt_path: Optional[str] = None
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_local_spec.value
    nemo1_ckpt_path: Optional[str] = None
    # NOTE: handle checkpoint resumption here rather than auto-resume so this supports fine-tuning capabilities
    initial_ckpt_path: Optional[str] = None

    def model_class(self) -> Type[GeneformerConfig]:
        return GeneformerConfig


@dataclass
class ExposedFineTuneSeqLenBioBertConfig(ExposedModelConfig[FineTuneSeqLenBioBertConfig]):
    """NOTE could use inheritence here, but the typing gets really weird and we'd rather have no red squiggles."""

    params_dtype: PrecisionTypes
    pipeline_dtype: PrecisionTypes
    autocast_dtype: PrecisionTypes
    num_layers: int = 6
    hidden_size: int = 256
    ffn_hidden_size: int = 512
    num_attention_heads: int = 4
    seq_length: int = 512
    fp32_residual_connection: bool = False
    hidden_dropout: float = 0.02
    init_method_std: float = 0.02
    kv_channels: Optional[int] = None
    apply_query_key_layer_scaling: bool = False
    make_vocab_size_divisible_by: int = 128
    masked_softmax_fusion: bool = True
    fp16_lm_cross_entropy: bool = False
    gradient_accumulation_fusion: bool = False
    layernorm_zero_centered_gamma: bool = False
    layernorm_epsilon: float = 1.0e-12
    activation_func: Callable = F.gelu
    qk_layernorm: bool = False
    apply_residual_connection_post_layernorm: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    get_attention_mask_from_fusion: bool = False
    attention_dropout: float = 0.1
    share_embeddings_and_output_weights: bool = True
    enable_autocast: bool = False
    nemo1_ckpt_path: Optional[str] = None
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_local_spec.value
    nemo1_ckpt_path: Optional[str] = None
    # NOTE: handle checkpoint resumption here rather than auto-resume so this supports fine-tuning capabilities
    initial_ckpt_path: Optional[str] = None
    # NOTE only new attribute between this config and the geneformer config.
    initial_ckpt_skip_keys_with_these_prefixes: Optional[List[str]] = None

    def __post_init__(self):
        if not self.initial_ckpt_skip_keys_with_these_prefixes:
            self.initial_ckpt_skip_keys_with_these_prefixes = ["regression_head"]

    def model_class(self) -> Type[FineTuneSeqLenBioBertConfig]:
        return FineTuneSeqLenBioBertConfig


@run.cli.factory
@run.autoconvert
def geneformer_finetuning_regression_head_recipe(
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    initial_ckpt_skip_keys_with_these_prefixes: Optional[List[str]] = None,
) -> ExposedModelConfig[FineTuneSeqLenBioBertConfig]:
    finetuning_config = ExposedFineTuneSeqLenBioBertConfig(
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
        nemo1_ckpt_path=nemo1_init_path,
        initial_ckpt_path=initial_ckpt_path,
        initial_ckpt_skip_keys_with_these_prefixes=initial_ckpt_skip_keys_with_these_prefixes,
    )
    return finetuning_config


# TODO(SKH) rename this recipe to something more understandable.
@run.cli.factory
@run.autoconvert
def geneformer10M_pretraining_recipe(
    seq_length: int = 128,
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_local_spec.value,
) -> ExposedModelConfig[GeneformerConfig]:
    """Sets up the base GeneformerConfig. Recipes on geneformer configs should choose what to expose and come with sensible defaults."""
    geneformer_config = ExposedGeneformerConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=seq_length,
        fp32_residual_connection=False,  # TODO(@jstjohn) check this
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=False,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,  # TODO(@jstjohn) check this
        fp16_lm_cross_entropy=False,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
        gradient_accumulation_fusion=False,  # THIS BREAKS STUFF, leave False
        layernorm_zero_centered_gamma=False,  # TODO(@jstjohn) check this
        layernorm_epsilon=1.0e-12,
        activation_func=F.gelu,  # TODO(@jstjohn) check this
        qk_layernorm=False,  # TODO(@jstjohn) check this
        apply_residual_connection_post_layernorm=False,  # False is new default, True was BERT pub.
        bias_activation_fusion=True,  # TODO(@jstjohn) check this
        bias_dropout_fusion=True,  # TODO(@jstjohn) check this
        get_attention_mask_from_fusion=False,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=False,  # This has to be set to True if we use the mixed precision plugin
        biobert_spec_option=biobert_spec_option,
        nemo1_ckpt_path=nemo1_init_path,
        initial_ckpt_path=initial_ckpt_path,
    )
    return geneformer_config


@dataclass
class OptimizerSchedulerConfig:
    lr: float = 1e-4
    optimizer: str = "adam"  # TODO Literal
    cosine_rampup_frac: float = 0.01
    cosine_hold_frac: float = 0.05
    interval: str = "step"  # TODO Literal
    monitor: str = "val_loss"


@run.cli.factory
@run.autoconvert
def default_adam_optimizer_with_cosine_annealing_recipe() -> OptimizerSchedulerConfig:
    """Prefers the default parameters for the Optimizer and Scheduler."""
    return OptimizerSchedulerConfig()


@run.cli.factory
@run.autoconvert
def exposed_optimizer_recipe(
    lr: float, optimizer: str, cosine_rampup_frac: float, cosine_hold_frac: float, interval: str, monitor: str
) -> OptimizerSchedulerConfig:
    """This recipe exposes all parameters to the underlying OptimizerSchedulerConfig."""
    return OptimizerSchedulerConfig(
        lr=lr,
        optimizer=optimizer,
        cosine_rampup_frac=cosine_rampup_frac,
        cosine_hold_frac=cosine_hold_frac,
        interval=interval,
        monitor=monitor,
    )


@run.cli.factory
@run.autoconvert
def optimizer_recipe_with_kwarg_defaults(
    lr: float = 1e-4,
    optimizer: str = "adam",
    cosine_rampup_frac: float = 0.01,
    cosine_hold_frac: float = 0.05,
    interval: str = "step",
    monitor: str = "val_loss",
) -> OptimizerSchedulerConfig:
    """This recipe exposes all parameters to the underlying OptimizerSchedulerConfig and provides defaults as kwargs."""
    return OptimizerSchedulerConfig(
        lr=lr,
        optimizer=optimizer,
        cosine_rampup_frac=cosine_rampup_frac,
        cosine_hold_frac=cosine_hold_frac,
        interval=interval,
        monitor=monitor,
    )


def biobert_lightning_module(
    model_config: BioBertGenericConfig, tokenizer: Tokenizer, optim_config: OptimizerSchedulerConfig, num_steps: int
) -> BioBertLightningModule:
    """Function that constructs a lightning module from the requisite configs.

    tokenizer: Tokenizer - must be the same tokenizer used by the DataModule.
    num_steps: int - must match the number of steps in the DataConfig.
    """
    model = BioBertLightningModule(
        model_config,
        tokenizer=tokenizer,
        optimizer=MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=optim_config.lr,
                optimizer=optim_config.optimizer,
                use_distributed_optimizer=True,
                # Pass through fp16/bf16 settings to avoid errors around model having bf16 enabled but optimizer not.
                # implies these configs must be coupled.
                fp16=model_config.fp16,
                bf16=model_config.bf16,
            ),
            lr_scheduler=CosineAnnealingScheduler(
                max_steps=num_steps,
                # minimum learning rate is 1/100th of the initial learning rate, so eg lr=1e-3 -> min_lr=1e-5
                min_lr=optim_config.lr / 100,
                warmup_steps=int(math.ceil(num_steps * optim_config.cosine_rampup_frac)),
                interval=optim_config.interval,
                monitor=optim_config.monitor,
                constant_steps=int(math.ceil(num_steps * optim_config.cosine_hold_frac)),
            ),
        ),
    )
    return model


@dataclass
class ExperimentConfig:
    save_every_n_steps: int
    result_dir: str
    experiment_name: str
    restore_from_checkpoint_path: Optional[str]
    resume_if_exists: bool
    wandb_options: WandbLoggerOptions = None  # TODO(SKH) if we are passing a type in here its gonna blow up.
    save_last_checkpoint: bool = True
    metric_to_monitor_for_checkpoints: str = "reduced_train_loss"  # TODO literal?
    save_top_k: int = 2
    create_tensorboard_logger: bool = False


@run.cli.factory
@run.autoconvert
def experiment_config_recipe() -> ExperimentConfig:
    return ExperimentConfig(
        save_every_n_steps=100,
        result_dir="./results",
        experiment_name="default_experiment",
        restore_from_checkpoint_path=None,
        resume_if_exists=True,
        save_last_checkpoint=True,
        metric_to_monitor_for_checkpoints="reduced_train_loss",
        save_top_k=2,
        create_tensorboard_logger=False,
    )


@dataclass
class WandbConfig:
    # NOTE(SKH) there is some duplication with WandbLoggerOptions
    project: str  # Must be set to log to wandb, this is the 'project' directory under your 'entity'
    entity: str  # Sometimes refers to team, sometimes username
    offline: bool  # If set does not log to wandb


def nemo_logger_factory(experiment_config: ExperimentConfig, wandb_config: Optional[WandbConfig]) -> nl.NeMoLogger:
    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_last=experiment_config.save_last_checkpoint,
        monitor=experiment_config.metric_to_monitor_for_checkpoints,
        save_top_k=experiment_config.save_top_k,
        every_n_train_steps=experiment_config.save_every_n_steps,
        always_save_context=True,
    )

    wandb_options: Optional[WandbLoggerOptions] = (
        None
        if wandb_config is None
        else WandbLoggerOptions(
            offline=wandb_config.offline,
            project=wandb_config.project,
            entity=wandb_config.entity,
            log_model=False,
        )
    )

    # Setup the logger and train the model
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=experiment_config.result_dir,
        name=experiment_config.experiment_name,
        initialize_tensorboard_logger=experiment_config.create_tensorboard_logger,
        wandb_kwargs=wandb_options,
        ckpt_callback=checkpoint_callback,
    )
    return nemo_logger


def pretrain_partial(
    model_config: ExposedModelConfig[ModelConfigT],
    data_config: DataConfig,
    parallel_config: ParallelConfig,
    training_config: TrainingConfig,
    optim_config: OptimizerSchedulerConfig,
    experiment_config: ExperimentConfig,
    resume_if_exists: bool = True,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_offline: bool = True,
) -> run.Partial:
    """Same as pretrain but in partial form instead of an entrypoint."""

    return run.Partial(
        pretrain,
        model_config=model_config,
        data_config=data_config,
        parallel_config=parallel_config,
        training_config=training_config,
        optim_config=optim_config,
        experiment_config=experiment_config,
        # Remaining are things that live outside a config
        resume_if_exists=resume_if_exists,
        # These could live as their own config, but they dont make sense to use factories with since theyre dependent on the environment.
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_offline=wandb_offline,
    )


@run.cli.entrypoint
def pretrain(
    model_config: ExposedModelConfig[ModelConfigT],  # noqa
    data_config: DataConfig,
    parallel_config: ParallelConfig,
    training_config: TrainingConfig,
    optim_config: OptimizerSchedulerConfig,
    experiment_config: ExperimentConfig,
    # Remaining are things that live outside a config
    resume_if_exists: bool = True,
    # These could live as their own config, but they dont make sense to use factories with since theyre dependent on the environment.
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_offline: bool = True,
    # ??? what was I doing with new_experiment title?
    new_experiment_title="asdf",
):
    model_config: ModelConfigT = model_config.exposed_to_internal_model_config()

    # Setup.
    # Create requisite directory.
    pathlib.Path(data_config.result_dir).mkdir(parents=True, exist_ok=True)

    if experiment_config.save_every_n_steps != training_config.val_check_interval:
        logging.warning("Mutating training_config.save_every_n_steps to be equal to val_check_interval.")
        experiment_config.save_every_n_steps = training_config.val_check_interval

    global_batch_size = infer_global_batch_size(
        micro_batch_size=data_config.micro_batch_size,
        num_nodes=parallel_config.num_nodes,
        devices=parallel_config.num_devices,
        accumulate_grad_batches=parallel_config.accumulate_grad_batches,
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
    )

    data: SingleCellDataModule = singlecell_data_module(data_config, global_batch_size)
    # TODO there must be a way to do this automatically.
    model_config.seq_length = data_config.seq_length
    model_config.bf16 = training_config.precision == "bf16-mixed"
    model_config.fp16 = training_config.precision == "16-mixed"

    model: BioBertLightningModule = biobert_lightning_module(
        model_config, tokenizer=data.tokenizer, optim_config=optim_config, num_steps=training_config.max_steps
    )
    trainer: nl.Trainer = setup_trainer_from_configs(parallel_config, training_config)
    nemo_logger: nl.NeMoLogger = nemo_logger_factory(
        experiment_config, wandb_config=WandbConfig(project=wandb_project, entity=wandb_entity, offline=wandb_offline)
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            resume_if_exists=False,  # To resume training a specific checkpoint simply set initial_ckpt_path in the ModelConfig.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )
