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
import os
from pathlib import Path
import pathlib
from typing import Dict, Optional, Sequence, Type, cast, get_args

import torch

from bionemo.llm.run import config_models
from bionemo.llm.model.biobert import model

from bionemo.llm.run.config_models import DataConfig, ExperimentConfig, ExposedModelConfig, OptimizerSchedulerConfig, ParallelConfig, TrainingConfig, DataModuleT
from bionemo.llm.utils.logger_utils import WandbConfig
import nemo.lightning as nl
from bionemo.llm.train import NsysConfig, setup_trainer 

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.esm2.api import ESM2Config
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule, InMemoryCSVDataset
from bionemo.llm.lightning import BionemoLightningModule, batch_collator
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.biobert.model import BioBertConfig
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.run.config_models import ExModelConfigT, ModelConfigT

__all__: Sequence[str] = ("infer",)

def infer(
    # core params
    model_config_cls: type[ExModelConfigT | ModelConfigT],
    checkpoint_path: str | pathlib.Path,
    data_config: DataConfig[DataModuleT],
    results_path: str | pathlib.Path,
    # less interesting stuff
    parallel_config: ParallelConfig,
    precision: PrecisionTypes,
    # Prediction parameters
    include_hiddens: bool = False,
    include_embeddings: bool = False,
    include_logits: bool = False,
    include_input_ids: bool = False,
):

    if isinstance(results_path, str):
        results_path = pathlib.Path(results_path)

    if os.path.isdir(results_path):
        results_path = results_path / "inference_results.pt"
    else:
        _, extension = os.path.splitext(results_path)
        results_path = results_path if extension == ".pt" else results_path / ".pt"


    # Setup global batch size
    global_batch_size = infer_global_batch_size(
        micro_batch_size=data_config.micro_batch_size,
        num_nodes=parallel_config.num_nodes,
        devices=parallel_config.num_devices,
        accumulate_grad_batches=parallel_config.accumulate_grad_batches,
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
    )
    # setup megatron strat and setup trainer
    # verified that the extra args in the trainer we made explicit are irrelevant for inference.
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
        ddp="megatron",
        find_unused_parameters=True,
        ckpt_include_optimizer=True,
        progress_interval=1,
    )
    trainer = nl.Trainer(
        devices=parallel_config.num_devices,
        accelerator="gpu",
        strategy=strategy,
        num_nodes=parallel_config.num_nodes,
        callbacks=[],
        plugins=nl.MegatronMixedPrecision(precision=precision),
    )

    # setup datamodule for inference
    # TODO this doesnt set it up for inference.
    datamodule: DataModuleT = data_config.construct_data_module(global_batch_size)

    if isinstance(model_config_cls, config_models.ExposedModelConfig):
        # Get associated type
        config_class = model_config_cls.model_class()
    elif isinstance(model_config_cls, model.BioBertConfig):
        config_class = model_config_cls
    else:
       raise ValueError("Expected either ExposedModelConfig or BioBertConfig for bionemo-model-config-cls")

    # atleast BioBertConfig
    bionemo_model_config = config_class(
        seq_length=data_config.seq_length,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
        # handle checkpoint resumption here rather than auto-resume so this supports fine-tuning capabilities
        initial_ckpt_path=str(checkpoint_path) if checkpoint_path is not None else None,
        include_embeddings=include_embeddings,
        include_hiddens=include_hiddens,
        include_input_ids=include_input_ids,
        skip_logits=not include_logits,
        initial_ckpt_skip_keys_with_these_prefixes=[],
    ) # type: ignore

    # setup lightning mod
    module: BionemoLightningModule = biobert_lightning_module(
        config=bionemo_model_config, tokenizer=datamodule.tokenizer, optimizer=None
    )
    # setup results
    pathlib.Path(data_config.result_dir).mkdir(parents=True, exist_ok=True)
    # setup lightning module

    # TODO will want to bring in Farhad's work, eventually we should rebase on his PR
    # Does this have to be a DataModule or can it be a DataLoader?
    predictions = trainer.predict(module, datamodule=datamodule, return_predictions=True)
    # TODO evaluate this section. I worry about gathering predictions in memory.
    results_dict = batch_collator(predictions)
    non_none_keys = [key for key, val in results_dict.items() if val is not None]
    print(f"Writing output {str(non_none_keys)} into {results_path}")
    torch.save(results_dict, results_path)


'''
We dont actually need configs for all this:

- parallel, trainer, etc, are all things you might want to change on the fly and have no impact on outcome
- results path, data setup: do matter but are also inference specific.

whats the desired workflow here?
bionemo-esm2-recpie --recipe 8m [arguments]
bionemo-esm2-train --conf 8m.yaml
(grab ckpt path)

bionemo-infer --checkpoint path/to/ckpt
    --model-config-t (Exposed or not is fine here)
    --data-config-t InMemoryCSVDatasetConfig
    --data-args arg1=val arg2=val arg3=val 
    ... (runtime args)

Workflow then looks like:
    - instantiate data_config from data-args
    - instantiate parallel_config from parameters
    - call infer
'''