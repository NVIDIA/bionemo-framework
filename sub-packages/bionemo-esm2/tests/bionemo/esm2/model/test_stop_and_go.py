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
from typing_extensions import override

from bionemo.core.data.load import load
from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.esm2.api import ESM2Config
from bionemo.esm2.data.datamodule import ESMDataModule
from bionemo.esm2.data.dataset import RandomMaskStrategy
from bionemo.esm2.data.tokenizer import BioNeMoESMTokenizer, get_tokenizer
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.lr_scheduler import WarmupAnnealDecayHoldScheduler
from bionemo.testing.harnesses import stop_and_go
from bionemo.testing.harnesses.mode import Mode


MODEL_PRECISION: Literal["bf16-mixed"] = "bf16-mixed"


class TestESM2StopAndGo(stop_and_go.StopAndGoHarness):
    num_steps: int = 10
    val_check_interval: int = 4
    limit_val_batches: int = 2
    lr: float = 1e-4
    precision: Literal["16-mixed", "bf16-mixed", "32"] = get_autocast_dtype(MODEL_PRECISION)

    @override
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.data_dir = Path(cls.tempdir.name) / "data"
        cls.data_dir.mkdir(parents=True, exist_ok=True)

        # setup data
        data_dir = load("esm2/testdata_esm2_pretrain:2.0") / "2024_03_sanity"

        cls.train_cluster_path = data_dir / "train_clusters_sanity.parquet"
        cls.train_database_path = data_dir / "train_sanity.db"
        cls.valid_cluster_path = data_dir / "valid_clusters.parquet"
        cls.valid_database_path = data_dir / "validation.db"
        cls.tokenizer: BioNeMoESMTokenizer = get_tokenizer()

        # run stop and go
        cls.run_stop_and_go()

    @classmethod
    def setup_model(cls, mode: Mode) -> tuple[pl.LightningModule, pl.LightningDataModule, nl.MegatronOptimizerModule]:
        # build data module
        data = ESMDataModule(
            train_cluster_path=cls.train_cluster_path,
            train_database_path=cls.train_database_path,
            valid_cluster_path=cls.valid_cluster_path,
            valid_database_path=cls.valid_database_path,
            global_batch_size=2,
            micro_batch_size=2,
            min_seq_length=None,
            max_seq_length=1024,
            num_workers=1,
            persistent_workers=False,
            random_mask_strategy=RandomMaskStrategy.ALL_TOKENS,
        )

        # build optimizer
        optimizer = MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=cls.lr,
                optimizer="adam",  # fused_adam not supported
                use_distributed_optimizer=True,
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.98,
            ),
            lr_scheduler=WarmupAnnealDecayHoldScheduler(
                warmup_steps=50, max_steps=cls.num_steps, max_lr=cls.lr, min_lr=0.0, anneal_percentage=0.10
            ),
        )

        # light ESM2 config
        config = ESM2Config(
            num_layers=3,
            hidden_size=128,
            params_dtype=cls.precision,
            pipeline_dtype=cls.precision,
            autocast_dtype=cls.precision,
        )
        # Build lightning module
        module = biobert_lightning_module(config=config, tokenizer=cls.tokenizer, optimizer=optimizer)

        return module, data, optimizer
