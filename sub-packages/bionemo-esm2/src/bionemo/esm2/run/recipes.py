
import importlib
from pathlib import Path
from bionemo.llm.config.config_models import ExperimentConfig, MainConfig, OptimizerSchedulerConfig, ParallelConfig, TrainingConfig
from bionemo.llm.utils.logger_utils import WandbConfig
from nemo.utils import logging
from typing import Optional
from bionemo.core.utils.dtypes import PrecisionTypes
from bionemo.esm2.run.config_models import ESM2DataConfig, ExposedESM2PretrainConfig
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption

import argparse
import json
from typing import Optional


def simple_parallel_recipe(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    num_devices: int = 1,
    accumulate_grad_batches: int = 1,
) -> ParallelConfig:
    assert (
        num_devices >= tensor_model_parallel_size * pipeline_model_parallel_size
    ), "devices must be divisible by tensor_model_parallel_size * pipeline_model_parallel_size"
    return ParallelConfig(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        num_devices=num_devices,
        accumulate_grad_batches=accumulate_grad_batches,
    )


def default_trainer_config_recipe() -> TrainingConfig:
    return TrainingConfig(max_steps=55000, limit_val_batches=2, val_check_interval=100)


def default_adam_optimizer_with_cosine_annealing_recipe() -> OptimizerSchedulerConfig:
    return OptimizerSchedulerConfig()


def experiment_config_recipe() -> ExperimentConfig:
    return ExperimentConfig(
        save_every_n_steps=100,
        result_dir="./results",
        experiment_name="default_experiment",
        restore_from_checkpoint_path=None,
        save_last_checkpoint=True,
        metric_to_monitor_for_checkpoints="reduced_train_loss",
        save_top_k=2,
        create_tensorboard_logger=False,
    )


def esm2_8m_model_config(
    seq_length: int = 2048,
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec,
    variable_seq_lengths: bool = False,
) -> ExposedESM2PretrainConfig:
    return ExposedESM2PretrainConfig(
        seq_length=seq_length,
        num_layers=6,
        hidden_size=320,
        num_attention_heads=20,
        ffn_hidden_size=4*320,
        params_dtype=precision,
        pipeline_dtype=precision,
        autocast_dtype=precision,
        biobert_spec_option=biobert_spec_option,
        nemo1_ckpt_path=str(nemo1_init_path) if nemo1_init_path is not None else None,
        # handle checkpoint resumption here rather than auto-resume so this supports fine-tuning capabilities
        initial_ckpt_path=str(initial_ckpt_path) if initial_ckpt_path is not None else None,
        variable_seq_lengths=variable_seq_lengths   
    )


def esm2_650m_config(
    seq_length: int = 2048,
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec,
    variable_seq_lengths: bool = False,
) -> ExposedESM2PretrainConfig:
    return ExposedESM2PretrainConfig(
        seq_length=seq_length,
        num_layers=33,
        hidden_size=1280,
        num_attention_heads=20,
        ffn_hidden_size=4*1280,
        params_dtype=precision,
        pipeline_dtype=precision,
        autocast_dtype=precision,
        biobert_spec_option=biobert_spec_option,
        nemo1_ckpt_path=str(nemo1_init_path) if nemo1_init_path is not None else None,
        # handle checkpoint resumption here rather than auto-resume so this supports fine-tuning capabilities
        initial_ckpt_path=str(initial_ckpt_path) if initial_ckpt_path is not None else None,
        variable_seq_lengths=variable_seq_lengths   
    )


'''
    --train-cluster-path ${TEST_DATA_DIR}/2024_03_sanity/train_clusters_sanity.parquet     \
    --train-database-path ${TEST_DATA_DIR}/2024_03_sanity/train_sanity.db     \
    --valid-cluster-path ${TEST_DATA_DIR}/2024_03_sanity/valid_clusters.parquet     \
    --valid-database-path ${TEST_DATA_DIR}/2024_03_sanity/validation.db     \
    --result-dir ./results     \
    --experiment-name test_experiment     \
    --num-gpus 1  \
    --num-nodes 1 \
    --val-check-interval 10 \
    --num-dataset-workers 1 \
    --num-steps 10 \
    --max-seq-length 128 \
    --limit-val-batches 2 \
    --micro-batch-size 2 \
    --restore-from-checkpoint-path ${ESM2_650M_CKPT}
'''

def esm2_8m_test_recipe(args):
    parallel_config = simple_parallel_recipe()
    training_config = default_trainer_config_recipe()
    # $(download_bionemo_data esm2/testdata_esm2_pretrain:2.0 --source $MY_DATA_SOURCE);

    # Find this from the test script... not sure what a sensible default is.
    data_config = ESM2DataConfig(
        min_seq_length=128,
        max_seq_length=128,
        micro_batch_size=2,
        num_dataset_workers=1,
        train_cluster_path=args.train_cluster_path,
        train_database_path=args.train_database_path,
        valid_cluster_path=args.valid_cluster_path,
        valid_database_path=args.valid_database_path,
    )
    # bionemo_model_config = geneformer_finetuning_regression_head_recipe()
    bionemo_model_config = esm2_650m_config(seq_length=data_config.max_seq_length, initial_ckpt_path=args.initial_ckpt_path)
    bionemo_model_config = esm2_8m_model_config(seq_length=data_config.max_seq_length, initial_ckpt_path=args.initial_ckpt_path)

    optim_config = default_adam_optimizer_with_cosine_annealing_recipe()
    experiment_config = experiment_config_recipe()
    wandb_config = WandbConfig(
        project="bionemo2-demo",
        entity="nvidia",
        offline=True,
        tags=[],
        group="dev",
        id="dev",
        log_model=False,
        anonymous=True,
    )
    main_config = MainConfig[ExposedESM2PretrainConfig, ESM2DataConfig](
        data_config=data_config,
        parallel_config=parallel_config,
        training_config=training_config,
        bionemo_model_config=bionemo_model_config,
        optim_config=optim_config,
        experiment_config=experiment_config,
        wandb_config=wandb_config,
    )
    return main_config

def main():
    def parse_args():
        parser = argparse.ArgumentParser(description="Create ESM2 configuration JSON.")
        parser.add_argument(
            "--recipe",
            type=str,
            choices=["test-8m"],
            required=True,
            help="Use one of the preconfigured recipes to create a template config file.",
        )

        parser.add_argument(
            "--dest",
            type=str,
            default="./esm2-recipe.json",
            required=True,
            help="Path to the JSON configuration file.",
        )

        parser.add_argument(
            '--train-cluster-path', type=Path, required=True, help='Path to the training cluster file.'
        )
        parser.add_argument(
            '--train-database-path', type=Path, required=True, help='Path to the training database file.'
        )
        parser.add_argument(
            '--valid-cluster-path', type=Path, required=True, help='Path to the validation cluster file.'
        )
        parser.add_argument(
            '--valid-database-path', type=Path, required=True, help='Path to the validation database file.'
        )

        # Extra argument.
        parser.add_argument(
            "--initial-ckpt-path",
            type=str,
            required=False,
            default=None,
            help="Path to an existing to a checkpoint directory to restore an existing checkpoint. Not compatible with all recipes.",
        )

        args = parser.parse_args()
        return args

    """Simple example for creating a JSON from recipes."""
    args = parse_args()

    if args.recipe == "test-8m":
        # Hardcoded test recipe.
        config = esm2_8m_test_recipe(args)
    elif args.recipe == "test-finetune":
        raise ValueError("Invalid recipe choice.")
        # config = finetune_test_recipe(args)
    else:
        raise ValueError("Invalid recipe choice.")

    # Serialize to JSON
    breakpoint()
    json_str = config.model_dump_json(indent=2)

    # Save to file
    with open(
        args.dest,
        "w",
    ) as f:
        f.write(json_str)
    logging.info(f"Saved configuration to {args.dest=}")

from pydantic import BaseModel, field_serializer, field_validator
from typing import Type
import torch
class MyConfig(BaseModel):
    core_attention_override: Optional[Type[torch.nn.Module]] = None

    @field_serializer('core_attention_override')
    def serialize_core_attention_override(self, value: Optional[Type[torch.nn.Module]]) -> Optional[str]:
        if value is None:
            return None
        return f"{value.__module__}.{value.__name__}"

    @field_validator('core_attention_override', mode='before')
    def validate_core_attention_override(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            module_name, class_name = value.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                if not issubclass(cls, torch.nn.Module):
                    raise ValueError(f"{cls} is not a subclass of torch.nn.Module")
                return cls
            except (ImportError, AttributeError):
                raise ValueError(f"Cannot import {value}")
        return value

if __name__ == "__main__":
    # NOTE: this is where I left off!
    config = esm2_650m_config()
    dumped = config.model_dump()
    config_again = ExposedESM2PretrainConfig(**dumped)

    assert config_again == config
    main()


# config.exposed_to_internal_bionemo_model_config()