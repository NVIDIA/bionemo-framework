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
from functools import partial
from typing import List, Optional

from nemo.utils import logging

from bionemo.core.utils.dtypes import PrecisionTypes
from bionemo.geneformer.run.config_models import (
    ExposedFineTuneSeqLenBioBertConfig,
    ExposedGeneformerPretrainConfig,
    GeneformerPretrainingDataConfig,
)
from bionemo.llm.config.config_models import (
    ExperimentConfig,
    MainConfig,
    OptimizerSchedulerConfig,
    ParallelConfig,
    TrainingConfig,
)
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.llm.utils.logger_utils import WandbConfig


"""
This script is for defining pre-configured recipes. Recipes at the minimum provide the user with a template config file.
Additionally, it may be useful to define prepackaged recipes for common usecases such as tests. Here we define a the
following recipes:

- example recipe with minimal data
- test recipe for running tests (same as above?)
- finetuning recipe with regression head based on the output of the test recipe.
- pretraining recipe on 10M sized model
- pretraining recipe on 106M sized model
"""


def geneformer_data_recipe(data_dir) -> GeneformerPretrainingDataConfig:
    """Recipe that produces the base geneformer small data configuration."""
    return GeneformerPretrainingDataConfig(data_dir=data_dir)


def full_geneformer_data_recipe(data_dir) -> GeneformerPretrainingDataConfig:
    return GeneformerPretrainingDataConfig(data_dir=data_dir)


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
    )


def geneformer_finetuning_regression_head_recipe(
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    initial_ckpt_skip_keys_with_these_prefixes: Optional[List[str]] = None,
) -> ExposedFineTuneSeqLenBioBertConfig:
    """NOTE on initial_ckpt_skip_keys_with_these_prefixes: configs define their own default with defaultfactory, so
    when we get passed None, we defer to the default. Importantly, the 'do nothing' case is different, where the input
    would be an empty list.
    """
    partial_finetuning_config = partial(
        ExposedFineTuneSeqLenBioBertConfig,
        params_dtype=precision,
        pipeline_dtype=precision,
        autocast_dtype=precision,
        nemo1_ckpt_path=nemo1_init_path,
        initial_ckpt_path=initial_ckpt_path,
        biobert_spec_option=BiobertSpecOption.bert_layer_with_transformer_engine_spec,
    )
    if initial_ckpt_skip_keys_with_these_prefixes:
        finetuning_config = partial_finetuning_config(
            initial_ckpt_skip_keys_with_these_prefixes=initial_ckpt_skip_keys_with_these_prefixes
        )
    else:
        # Use the sensible default when None is passed
        finetuning_config = partial_finetuning_config()
    return finetuning_config


def default_trainer_config_recipe() -> TrainingConfig:
    return TrainingConfig(max_steps=55000, limit_val_batches=2, val_check_interval=100)


def geneformer10m_finetune_config(
    seq_length: int = 2048,
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    biobert_spec_option=BiobertSpecOption.bert_layer_with_transformer_engine_spec,
) -> ExposedFineTuneSeqLenBioBertConfig:
    geneformer_config = ExposedFineTuneSeqLenBioBertConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=seq_length,
        fp32_residual_connection=False,
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=False,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,
        fp16_lm_cross_entropy=False,
        params_dtype=precision,
        pipeline_dtype=precision,
        autocast_dtype=precision,
        gradient_accumulation_fusion=False,
        layernorm_zero_centered_gamma=False,
        layernorm_epsilon=1.0e-12,
        activation_func="gelu",
        qk_layernorm=False,
        apply_residual_connection_post_layernorm=False,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        get_attention_mask_from_fusion=False,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=False,
        biobert_spec_option=biobert_spec_option,
        nemo1_ckpt_path=nemo1_init_path,
        initial_ckpt_path=initial_ckpt_path,
    )
    return geneformer_config


def geneformer10M_pretraining_config(
    seq_length: int = 2048,
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_with_transformer_engine_spec,
) -> ExposedGeneformerPretrainConfig:
    geneformer_config = ExposedGeneformerPretrainConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=seq_length,
        fp32_residual_connection=False,
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=False,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,
        fp16_lm_cross_entropy=False,
        params_dtype=precision,
        pipeline_dtype=precision,
        autocast_dtype=precision,
        gradient_accumulation_fusion=False,
        layernorm_zero_centered_gamma=False,
        layernorm_epsilon=1.0e-12,
        activation_func="gelu",
        qk_layernorm=False,
        apply_residual_connection_post_layernorm=False,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        get_attention_mask_from_fusion=False,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=False,
        biobert_spec_option=biobert_spec_option,
        nemo1_ckpt_path=nemo1_init_path,
        initial_ckpt_path=initial_ckpt_path,
    )
    return geneformer_config


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


def finetune_test_recipe(args) -> MainConfig[ExposedFineTuneSeqLenBioBertConfig, GeneformerPretrainingDataConfig]:
    data_path = args.data_path
    result_dir = args.result_dir

    parallel_config = ParallelConfig(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, num_devices=1, accumulate_grad_batches=2
    )
    training_config = TrainingConfig(
        max_steps=55, limit_val_batches=2, val_check_interval=10, precision="bf16-mixed", accelerator="gpu"
    )
    data_config = GeneformerPretrainingDataConfig(
        seq_length=128,
        micro_batch_size=2,
        num_dataset_workers=0,
        data_dir=data_path,
    )
    experiment_config = ExperimentConfig(
        save_every_n_steps=training_config.val_check_interval,
        result_dir=result_dir,
        experiment_name="test-experiment",
        restore_from_checkpoint_path=None,
        save_last_checkpoint=True,
        metric_to_monitor_for_checkpoints="reduced_train_loss",
        save_top_k=2,
        create_tensorboard_logger=False,
    )

    optim_config = OptimizerSchedulerConfig()
    geneformer_config = geneformer10m_finetune_config(
        seq_length=data_config.seq_length, initial_ckpt_path=args.initial_ckpt_path
    )

    return MainConfig(
        data_config=data_config,
        parallel_config=parallel_config,
        training_config=training_config,
        bionemo_model_config=geneformer_config,
        optim_config=optim_config,
        experiment_config=experiment_config,
    )


def pretrain_test_recipe(args) -> MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig]:
    data_path = args.data_path
    result_dir = args.result_dir

    parallel_config = ParallelConfig(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, num_devices=1, accumulate_grad_batches=2
    )
    training_config = TrainingConfig(
        max_steps=55, limit_val_batches=2, val_check_interval=10, precision="bf16-mixed", accelerator="gpu"
    )
    data_config = GeneformerPretrainingDataConfig(
        seq_length=128,
        micro_batch_size=2,
        num_dataset_workers=0,
        data_dir=data_path,
    )
    experiment_config = ExperimentConfig(
        save_every_n_steps=training_config.val_check_interval,
        result_dir=result_dir,
        experiment_name="test-experiment",
        restore_from_checkpoint_path=None,
        save_last_checkpoint=True,
        metric_to_monitor_for_checkpoints="reduced_train_loss",
        save_top_k=2,
        create_tensorboard_logger=False,
    )

    optim_config = OptimizerSchedulerConfig()
    geneformer_config = geneformer10M_pretraining_config(
        seq_length=data_config.seq_length, initial_ckpt_path=args.initial_ckpt_path
    )

    return MainConfig(
        data_config=data_config,
        parallel_config=parallel_config,
        training_config=training_config,
        bionemo_model_config=geneformer_config,
        optim_config=optim_config,
        experiment_config=experiment_config,
    )


def geneformer10m_pretrain_recipe(
    args,
) -> MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig]:
    data_config: GeneformerPretrainingDataConfig = geneformer_data_recipe(data_dir=args.data_dir)
    parallel_config = simple_parallel_recipe()
    training_config = default_trainer_config_recipe()
    # bionemo_model_config = geneformer_finetuning_regression_head_recipe()
    bionemo_model_config = geneformer10M_pretraining_config(initial_ckpt_path=args.initial_ckpt_path)
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
    main_config = MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig](
        data_config=data_config,
        parallel_config=parallel_config,
        training_config=training_config,
        bionemo_model_config=bionemo_model_config,
        optim_config=optim_config,
        experiment_config=experiment_config,
        wandb_config=wandb_config,
    )
    return main_config


def geneformer10m_finetune_recipe(
    args,
) -> MainConfig[ExposedFineTuneSeqLenBioBertConfig, GeneformerPretrainingDataConfig]:
    data_config: GeneformerPretrainingDataConfig = geneformer_data_recipe(data_dir=args.data_path)
    parallel_config = simple_parallel_recipe()
    training_config = default_trainer_config_recipe()
    bionemo_model_config = geneformer_finetuning_regression_head_recipe(initial_ckpt_path=args.initial_ckpt_path)
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
    main_config = MainConfig[ExposedFineTuneSeqLenBioBertConfig, GeneformerPretrainingDataConfig](
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
        parser = argparse.ArgumentParser(description="Create Geneformer configuration JSON.")
        parser.add_argument(
            "--recipe",
            type=str,
            choices=["test", "10m-pretrain", "test-finetune", "finetune"],
            required=True,
            help="Use one of the preconfigured recipes to create a template config file.",
        )

        parser.add_argument(
            "--dest",
            type=str,
            default="./geneformer-recipe.json",
            required=True,
            help="Path to the JSON configuration file.",
        )

        parser.add_argument(
            "--data-path", type=str, required=True, help="Path to the directory containing pretraining data."
        )
        parser.add_argument(
            "--result-dir", type=str, required=True, help="Path to the directory used to save results."
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

    if args.recipe == "test":
        config = pretrain_test_recipe(args)
    elif args.recipe == "10m-pretrain":
        config = geneformer10m_pretrain_recipe(args)
    elif args.recipe == "106m-pretrain":
        # config = geneformer106m_pretrain_recipe(args)
        raise NotImplementedError("106M pretraining recipe not implemented.")
    elif args.recipe == "test-finetune":
        config = finetune_test_recipe(args)
    elif args.recipe == "finetune":
        config = geneformer10m_finetune_recipe(args)
    else:
        raise ValueError("Invalid recipe choice.")

    # Serialize to JSON
    json_str = config.model_dump_json(indent=2)

    # Save to file
    with open(
        args.dest,
        "w",
    ) as f:
        f.write(json_str)

    logging.info(f"Saved configuration to {args.dest=}")


if __name__ == "__main__":
    main()
