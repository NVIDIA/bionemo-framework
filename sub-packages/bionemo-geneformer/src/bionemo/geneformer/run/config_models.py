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


# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import dataclass
from typing import List, Optional, Type

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

from bionemo.core.utils.dtypes import PrecisionTypes
from bionemo.geneformer.api import GeneformerConfig
from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.geneformer.model.finetune_token_regressor import FineTuneSeqLenBioBertConfig
from bionemo.llm.config.config_models import (
    DataConfig,
    DataModuleT,
    ExperimentConfig,
    ExposedModelConfig,
    OptimizerSchedulerConfig,
    ParallelConfig,
    TrainingConfig,
)
from bionemo.llm.model.biobert.lightning import BioBertLightningModule
from bionemo.llm.model.biobert.model import BioBertGenericConfig, BiobertSpecOption
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger


@dataclass
class GeneformerDataArtifacts:
    """Data artifacts produced by the geneformer preprocess."""

    tokenizer: Tokenizer
    median_dict: dict


class GeneformerPretrainingDataConfig(DataConfig[SingleCellDataModule]):
    """Configuration for the geneformer pre-training data module."""

    # Shadow two attributes from the parent for visibility.
    result_dir: str = "./results"
    micro_batch_size: int = 8

    data_dir: str
    seq_length: int = 2048
    num_dataset_workers: int = 0

    @property
    def train_data_path(self) -> str:
        return self.data_dir + "/train"

    @property
    def val_data_path(self) -> str:
        return self.data_dir + "/val"

    @property
    def test_data_path(self) -> str:
        return self.data_dir + "/test"

    def geneformer_preprocess(self) -> GeneformerDataArtifacts:
        """Geneformer datamodule expects certain artifacts to be present in the data directory.

        This method uses a legacy 'preprocessor' from BioNeMo 1 to acquire the associated artifacts.
        """
        preprocessor = GeneformerPreprocess(
            download_directory=pathlib.Path(self.train_data_path),
            medians_file_path=pathlib.Path(self.train_data_path + "/medians.json"),
            tokenizer_vocab_path=pathlib.Path(self.train_data_path + "/geneformer.vocab"),
        )
        result = preprocessor.preprocess()
        if "tokenizer" in result and "median_dict" in result:
            logging.info("*************** Preprocessing Finished ************")
            return GeneformerDataArtifacts(tokenizer=result["tokenizer"], median_dict=result["median_dict"])
        else:
            logging.error("Preprocessing failed.")
            raise ValueError("Preprocessing failed to create tokenizer and/or median dictionary.")

    def construct_data_module(self, global_batch_size: int) -> SingleCellDataModule:
        geneformer_data_artifacts: GeneformerDataArtifacts = geneformer_preprocess(self)
        data = SingleCellDataModule(
            seq_length=self.seq_length,
            tokenizer=geneformer_data_artifacts.tokenizer,
            train_dataset_path=self.train_data_path,
            val_dataset_path=self.val_data_path,
            test_dataset_path=self.test_data_path,
            random_token_prob=0.02,
            median_dict=geneformer_data_artifacts.median_dict,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=global_batch_size,
            persistent_workers=self.num_dataset_workers > 0,
            pin_memory=False,
            num_workers=self.num_dataset_workers,
        )
        return data


def geneformer_small_data_recipe(
    data_dir="/workspaces/bionemo-fw-ea/data/cellxgene_2023-12-15_small/processed_data",
) -> GeneformerPretrainingDataConfig:
    """Recipe that produces the base geneformer small data configuration."""
    return GeneformerPretrainingDataConfig(data_dir=data_dir)


def full_geneformer_data_recipe(
    data_dir="/workspaces/bionemo-fw-ea/data/cellxgene_2023-12-15/processed_data",
) -> GeneformerPretrainingDataConfig:
    return GeneformerPretrainingDataConfig(data_dir=data_dir)


def geneformer_preprocess(data_config: GeneformerPretrainingDataConfig) -> GeneformerDataArtifacts:
    """Geneformer datamodule expects certain artifacts to be present in the data directory.

    This method uses a legacy 'preprocessor' from BioNeMo 1 to acquire the associated artifacts.
    """
    preprocessor = GeneformerPreprocess(
        download_directory=pathlib.Path(data_config.train_data_path),
        medians_file_path=pathlib.Path(data_config.train_data_path + "/medians.json"),
        tokenizer_vocab_path=pathlib.Path(data_config.train_data_path + "/geneformer.vocab"),
    )
    result = preprocessor.preprocess()
    if "tokenizer" in result and "median_dict" in result:
        logging.info("*************** Preprocessing Finished ************")
        return GeneformerDataArtifacts(tokenizer=result["tokenizer"], median_dict=result["median_dict"])
    else:
        logging.error("Preprocessing failed.")
        raise ValueError("Preprocessing failed to create tokenizer and/or median dictionary.")


def simple_parallel_recipe(
    tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1, num_devices: int = 1
) -> ParallelConfig:
    assert (
        num_devices >= tensor_model_parallel_size * pipeline_model_parallel_size
    ), "devices must be divisible by tensor_model_parallel_size * pipeline_model_parallel_size"
    return ParallelConfig(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        num_devices=num_devices,
    )


def default_trainer_config_recipe() -> TrainingConfig:
    return TrainingConfig(max_steps=55000, limit_val_batches=2, val_check_interval=100)


def setup_trainer(parallel_config: ParallelConfig, training_config: TrainingConfig) -> nl.Trainer:
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
        limit_val_batches=training_config.limit_val_batches,
        val_check_interval=training_config.val_check_interval,
        num_nodes=parallel_config.num_nodes,
        callbacks=[
            RichModelSummary(max_depth=4),
            LearningRateMonitor(),
        ],
        plugins=nl.MegatronMixedPrecision(precision=training_config.precision),
    )
    return trainer

class ExposedGeneformerPretrainConfig(ExposedModelConfig[GeneformerConfig]):
    # Custom parameters for FineTuning
    initial_ckpt_path: Optional[str] = None
    initial_ckpt_skip_keys_with_these_prefixes: Optional[List[str]] = None

    def model_class(self) -> Type[GeneformerConfig]:
        return GeneformerConfig

class ExposedFineTuneSeqLenBioBertConfig(ExposedModelConfig[FineTuneSeqLenBioBertConfig]):
    """Config for models that fine-tune a BioBERT model from a pre-trained checkpoint.

    Parameters:
        initial_ckpt_path - path to a directory containing checkpoint files for initializing the model. This is only
            required on the first execution of the model, any restored checkpoints should skip this step.
        initial_ckpt_skip_keys_with_these_prefixes - skip any layer that contains this key during restoration. Useful
            for ignoring extra additional layers used for finetuning. Layers with these keys are then randomly initialized.
    """

    # Custom parameters for FineTuning
    initial_ckpt_path: Optional[str] = None
    initial_ckpt_skip_keys_with_these_prefixes: Optional[List[str]] = None

    def __post_init__(self):
        if not self.initial_ckpt_skip_keys_with_these_prefixes:
            self.initial_ckpt_skip_keys_with_these_prefixes = ["regression_head"]

    def model_class(self) -> Type[FineTuneSeqLenBioBertConfig]:
        return FineTuneSeqLenBioBertConfig


def biobert_lightning_module(
    bionemo_model_config: BioBertGenericConfig,
    tokenizer: Tokenizer,
    optim_config: OptimizerSchedulerConfig,
    num_steps: int,
) -> BioBertLightningModule:
    model = BioBertLightningModule(
        bionemo_model_config,
        tokenizer=tokenizer,
        optimizer=MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=optim_config.lr,
                optimizer=optim_config.optimizer,
                use_distributed_optimizer=True,
                fp16=bionemo_model_config.fp16,
                bf16=bionemo_model_config.bf16,
            ),
            lr_scheduler=CosineAnnealingScheduler(
                max_steps=num_steps,
                min_lr=optim_config.lr / 100,
                warmup_steps=int(math.ceil(num_steps * optim_config.cosine_rampup_frac)),
                interval=optim_config.interval,
                monitor=optim_config.monitor,
                constant_steps=int(math.ceil(num_steps * optim_config.cosine_hold_frac)),
            ),
        ),
    )
    return model

def nemo_logger_factory(experiment_config: ExperimentConfig, wandb_config: Optional[WandbConfig]) -> nl.NeMoLogger:
    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_last=experiment_config.save_last_checkpoint,
        monitor=experiment_config.metric_to_monitor_for_checkpoints,
        save_top_k=experiment_config.save_top_k,
        every_n_train_steps=experiment_config.save_every_n_steps,
        always_save_context=True,
    )

    nemo_logger = setup_nemo_lightning_logger(
        root_dir=experiment_config.result_dir,
        name=experiment_config.experiment_name,
        initialize_tensorboard_logger=experiment_config.create_tensorboard_logger,
        wandb_config=wandb_config,
        ckpt_callback=checkpoint_callback,
    )
    return nemo_logger


def train(
    bionemo_exposed_model_config: ExposedModelConfig,
    data_config: DataConfig[DataModuleT],
    parallel_config: ParallelConfig,
    training_config: TrainingConfig,
    optim_config: OptimizerSchedulerConfig,
    experiment_config: ExperimentConfig,
    wandb_config: Optional[WandbConfig],
    resume_if_exists: bool = True,
):
    bionemo_model_config = bionemo_exposed_model_config.exposed_to_internal_bionemo_model_config()
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

    data: DataModuleT = data_config.construct_data_module(global_batch_size)

    # TODO BioBertDataModule or BioBertTokenizer abstractions. We know all DataModuleT in this case have data.tokenizer,
    # although this constraint is not documented.
    model: BioBertLightningModule = biobert_lightning_module(
        bionemo_model_config, tokenizer=data.tokenizer, optim_config=optim_config, num_steps=training_config.max_steps
    )
    trainer: nl.Trainer = setup_trainer(parallel_config, training_config)
    nemo_logger: nl.NeMoLogger = nemo_logger_factory(experiment_config, wandb_config=wandb_config)

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            resume_if_exists=resume_if_exists,
            resume_ignore_no_checkpoint=True,
        ),
    )
