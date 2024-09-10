from dataclasses import dataclass
from typing import List
import pathlib
import nemo_run as run
from typing import Sequence, Literal
from nemo import lightning as nl
import math
from pathlib import Path
from typing import Optional, Sequence, get_args
from tokenizers import Tokenizer

from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import io, resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.utils import logging
from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary
from torch.nn import functional as F

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.geneformer.api import GeneformerConfig
from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.llm.lightning import LossLoggingCallback
from bionemo.llm.model.biobert.lightning import BioBertLightningModule
from bionemo.llm.model.biobert.model import BiobertSpecOption
from bionemo.llm.utils.datamodule_utils import float_or_int_or_none, infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbLoggerOptions, setup_nemo_lightning_logger
@dataclass
class DataConfig:
    data_dir: str
    result_dir: str = './results'
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
def small_data_config(data_dir="/workspaces/bionemo-fw-ea/data/cellxgene_2023-12-15_small/processed_data") -> DataConfig:
    # NOTE theoretically we could validate that this stuff exists.
    return DataConfig(data_dir=data_dir)

@run.cli.factory
@run.autoconvert
def full_geneformer_data_config(data_dir="/workspaces/bionemo-fw-ea/data/cellxgene_2023-12-15/processed_data") -> DataConfig:
    # NOTE theoretically we could validate that this stuff exists.
    return DataConfig(data_dir=data_dir)

@dataclass
class GeneformerDataArtifacts:
    tokenizer: Tokenizer # TODO(SKH) typing isnt right
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
def simple_parallel_recipe(tensor_model_parallel_size: int =1, pipeline_model_parallel_size: int = 1, num_devices: int = 1) -> ParallelConfig:
    # TODO validatorssssssss, make sure we get everythign right here.
    assert num_devices >= tensor_model_parallel_size * pipeline_model_parallel_size, "devices must be divisible by tensor_model_parallel_size * pipeline_model_parallel_size"
    return ParallelConfig(tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=pipeline_model_parallel_size, num_devices=num_devices)

@dataclass
class TrainingConfig:
    max_steps: int
    limit_val_batches: int
    val_check_interval: int
    precision: PrecisionTypes = 'bf16-mixed'
    accelerator: str = 'gpu'

@run.cli.factory
@run.autoconvert
def default_trainer_config() -> TrainingConfig:
    return TrainingConfig(
        max_steps=55000,
        limit_val_batches=2,
        val_check_interval=100
    )

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


@run.cli.factory
@run.autoconvert
def basic_geneformer_config_recipe(seq_length: int = 128, precision: PrecisionTypes='bf16-mixed', nemo1_init_path: Optional[str]=None, biobert_spec_option: BiobertSpecOption=BiobertSpecOption.bert_layer_local_spec.value) -> GeneformerConfig:
    # TODO seq_length must match the datamodule. We can pass in the DataConfig but I dont know how to enforce that its the same everywhere.
    #           another option is to construct this ad-hoc like we do with wandb 
    ''' Sets up the base GeneformerConfig. Recipes on geneformer configs should choose what to expose and come with sensible defaults. '''
    geneformer_config = GeneformerConfig(
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
    )
    return geneformer_config

@dataclass
class OptimizerSchedulerConfig:
    lr: float = 1e-4
    optimizer: str = 'adam' # TODO Literal
    cosine_rampup_frac: float = .01
    cosine_hold_frac: float = .05
    interval: str = 'step' # TODO Literal
    monitor: str = 'val_loss'

@run.cli.factory
@run.autoconvert
def default_adam_optimizer_with_cosine_annealing_recipe() -> OptimizerSchedulerConfig:
    ''' Prefers the default parameters for the Optimizer and Scheduler. '''
    return OptimizerSchedulerConfig()

@run.cli.factory
@run.autoconvert
def exposed_optimizer_recipe(lr: float, optimizer: str, cosine_rampup_frac: float, cosine_hold_frac: float, interval: str, monitor: str) -> OptimizerSchedulerConfig:
    ''' This recipe exposes all parameters to the underlying OptimizerSchedulerConfig. '''
    return OptimizerSchedulerConfig(lr=lr, optimizer=optimizer, cosine_rampup_frac=cosine_rampup_frac, cosine_hold_frac=cosine_hold_frac, interval=interval, monitor=monitor)

@run.cli.factory
@run.autoconvert
def optimizer_recipe_with_kwarg_defaults(lr: float = 1e-4, optimizer: str = 'adam', cosine_rampup_frac: float = .01, cosine_hold_frac: float = .05, interval: str = 'step', monitor: str = 'val_loss') -> OptimizerSchedulerConfig:
    ''' This recipe exposes all parameters to the underlying OptimizerSchedulerConfig and provides defaults as kwargs. '''
    return OptimizerSchedulerConfig(lr=lr, optimizer=optimizer, cosine_rampup_frac=cosine_rampup_frac, cosine_hold_frac=cosine_hold_frac, interval=interval, monitor=monitor)

def biobert_lightning_module(geneformer_config: GeneformerConfig, tokenizer: Tokenizer, optim_config: OptimizerSchedulerConfig, num_steps: int) -> BioBertLightningModule:
    ''' Function that constructs a lightning module from the requisite configs. 

    tokenizer: Tokenizer - must be the same tokenizer used by the DataModule.
    num_steps: int - must match the number of steps in the DataConfig. 
    '''
    model = BioBertLightningModule(
        geneformer_config,
        tokenizer=tokenizer,
        optimizer=MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=optim_config.lr,
                optimizer=optim_config.optimizer,
                use_distributed_optimizer=True,
                # Pass through fp16/bf16 settings to avoid errors around model having bf16 enabled but optimizer not.
                # implies these configs must be coupled.
                fp16=geneformer_config.fp16,
                bf16=geneformer_config.bf16,
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
    wandb_options: WandbLoggerOptions = None # TODO(SKH) if we are passing a type in here its gonna blow up.
    save_best_checkpoint: bool = False
    save_last_checkpoint: bool = True
    metric_to_monitor_for_checkpoints: str = 'reduced_train_loss' # TODO literal?
    save_top_k: int = 2
    create_tensorboard_logger: bool = False

@run.cli.factory
@run.autoconvert
def experiment_config_recipe() -> ExperimentConfig:
    return ExperimentConfig(
        save_every_n_steps=100,
        result_dir='./results',
        experiment_name='default_experiment',
        restore_from_checkpoint_path=None,
        resume_if_exists=True,
        save_best_checkpoint=False,
        save_last_checkpoint=True,
        metric_to_monitor_for_checkpoints='reduced_train_loss',
        save_top_k=2,
        create_tensorboard_logger=False,
    )

@dataclass
class WandbConfig:
    # NOTE(SKH) there is some duplication with WandbLoggerOptions
    project: str # Must be set to log to wandb, this is the 'project' directory under your 'entity'
    entity: str # Sometimes refers to team, sometimes username
    offline: bool # If set does not log to wandb

def nemo_logger_factory(experiment_config: ExperimentConfig, wandb_config: Optional[WandbConfig]) -> nl.NeMoLogger:
    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_best_model=experiment_config.save_best_checkpoint,
        save_last=experiment_config.save_last_checkpoint,
        monitor=experiment_config.metric_to_monitor_for_checkpoints,
        save_top_k=experiment_config.save_top_k,
        every_n_train_steps=experiment_config.save_every_n_steps,
        enable_nemo_ckpt_io=True,
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

@run.cli.entrypoint
def pretrain(
        geneformer_config: GeneformerConfig, 
        data_config: DataConfig, 
        parallel_config: ParallelConfig, 
        training_config: TrainingConfig, 
        optim_config: OptimizerSchedulerConfig,
        experiment_config: ExperimentConfig, 
        # Remaining are things that live outside a config
        resume_if_exists: bool = True,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_offline: bool = True,
        copy_val_check_interval_for_save_every_n_steps: bool = True
    ):
    # NOTE: any config passed into the entrypoint can be MUTATED by the CLI.

    # Setup.
    # Create requisite directory.
    pathlib.Path(data_config.result_dir).mkdir(parents=True, exist_ok=True)

    if copy_val_check_interval_for_save_every_n_steps and experiment_config.save_every_n_steps != training_config.val_check_interval:
        logging.warning("Mutating training_config.save_every_n_steps to be equal to val_check_interval.")
        experiment_config.save_every_n_steps = training_config.val_check_interval


    global_batch_size = infer_global_batch_size(micro_batch_size=data_config.micro_batch_size, 
                                                num_nodes=parallel_config.num_nodes,
                                                devices=parallel_config.num_devices, 
                                                accumulate_grad_batches=parallel_config.accumulate_grad_batches,
                                                tensor_model_parallel_size=parallel_config.tensor_model_parallel_size, 
                                                pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size)

    data: SingleCellDataModule = singlecell_data_module(data_config, global_batch_size)
    # TODO there must be a way to do this automatically.
    geneformer_config.seq_length = data_config.seq_length
    geneformer_config.bf16 = geneformer_config.params_dtype == 'bf16-mixed'
    geneformer_config.fp16 = geneformer_config.params_dtype == '16-mixed'

    model: BioBertLightningModule = biobert_lightning_module(geneformer_config, tokenizer=data.tokenizer, optim_config=optim_config, num_steps=training_config.max_steps)
    trainer: nl.Trainer = setup_trainer_from_configs(parallel_config, training_config)
    nemo_logger: nl.NeMoLogger = nemo_logger_factory(experiment_config, wandb_config=WandbConfig(project=wandb_project, entity=wandb_entity, offline=wandb_offline))

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            path=None,
            resume_if_exists=resume_if_exists,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )
