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


from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.lightning.pytorch.optim import MegatronOptimizerModule

from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.esm2.api import ESM2Config
from bionemo.esm2.data.datamodule import ESMDataModule
from bionemo.esm2.data.dataset import RandomMaskStrategy
from bionemo.esm2.data.tokenizer import BioNeMoESMTokenizer, get_tokenizer
from bionemo.esm2.model.lr_scheduler import WarmupAnnealDecayHoldScheduler
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.biobert.testing_utils import compute_biobert_loss_singlegpu
from bionemo.testing.harnesses import stop_and_go


MODEL_PRECISION: Literal["bf16-mixed"] = "bf16-mixed"


class ESM2StopAndGoTest(stop_and_go.StopAndGoHarness):
    def __init__(
        self,
        train_cluster_path,
        train_database_path,
        valid_cluster_path,
        valid_database_path,
        val_check_interval=2,
        exp_name="esm2_stop_and_go",
    ):
        extra_metrics_dict = {"val_loss": compute_biobert_loss_singlegpu}
        super().__init__(
            extra_metrics_dict=extra_metrics_dict,
            val_check_interval=val_check_interval,
            exp_name=exp_name,
        )

        self.tokenizer: BioNeMoESMTokenizer = get_tokenizer()
        self.train_cluster_path: Path = train_cluster_path
        self.train_database_path: Path = train_database_path
        self.valid_cluster_path: Path = valid_cluster_path
        self.valid_database_path: Path = valid_database_path
        self.autocast_dtype = get_autocast_dtype(MODEL_PRECISION)

    def setup_model(
        self, mode: Literal["stop", "go"]
    ) -> tuple[pl.LightningModule, pl.LightningDataModule, nl.MegatronOptimizerModule]:
        devices = 1
        num_steps = 4
        lr = 1e-4

        # build data module
        data = ESMDataModule(
            train_cluster_path=self.train_cluster_path,
            train_database_path=self.train_database_path,
            valid_cluster_path=self.valid_cluster_path,
            valid_database_path=self.valid_database_path,
            global_batch_size=2 * int(devices),
            micro_batch_size=2,
            min_seq_length=None,
            max_seq_length=1024,
            num_workers=0,
            random_mask_strategy=RandomMaskStrategy.ALL_TOKENS,
        )

        # build optimizer
        optimizer = MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=lr,
                optimizer="adam",  # fused_adam not supported
                use_distributed_optimizer=True,
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.98,
            ),
            lr_scheduler=WarmupAnnealDecayHoldScheduler(
                warmup_steps=50, max_steps=num_steps, max_lr=lr, min_lr=lr / 10.0, anneal_percentage=0.10
            ),
        )

        # light ESM2 config
        config = ESM2Config(
            num_layers=3,
            hidden_size=128,
            params_dtype=self.autocast_dtype,
            pipeline_dtype=self.autocast_dtype,
            autocast_dtype=self.autocast_dtype,
        )
        # Build lightning module
        module = biobert_lightning_module(config=config, tokenizer=self.tokenizer, optimizer=optimizer)

        return module, data, optimizer

    def setup_trainer_and_strategy(self, mode: Literal["stop", "go"], metrics):
        devices, tp_size, pp_size = 1, 1, 1

        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            ddp="megatron",
            find_unused_parameters=True,
            ckpt_include_optimizer=True,
        )

        trainer = nl.Trainer(
            devices=devices,
            max_steps=4,
            accelerator="gpu",
            strategy=strategy,
            limit_val_batches=2,
            val_check_interval=self.val_check_interval,
            log_every_n_steps=self.val_check_interval,
            num_nodes=1,
            callbacks=self.get_callbacks(mode=mode, metrics=metrics),
            plugins=nl.MegatronMixedPrecision(precision=MODEL_PRECISION),
        )
        return trainer


def test_esm2_example(dummy_protein_dataset, dummy_parquet_train_val_inputs):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs
    ESM2StopAndGoTest(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
    )
