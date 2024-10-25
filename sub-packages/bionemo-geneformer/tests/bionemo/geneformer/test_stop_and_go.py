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
"""
How to adapt these tests:

1) Need to hook up our own pretraining workflow. In the code below, we do this via subproc and CLI. Is this still best practice?
    a) use the structure in sub-packages/bionemo-geneformer/tests/bionemo/geneformer/test_model.py:test_geneformer_nemo1_v_nemo2_inference_golden_values
    b) might need to look at utilities for setup/teardown to make sure the distributed stuff is handled correctly.
2) Need to inject the callbacks either via CLI or by manually inserting them here.
3) How do we want this to work for other modules? Lots of code could be duplicated here which makes it a challenge.
4) is this the right set of code to do this on?

"""

import math
import pathlib
from typing import Literal

import pytorch_lightning as pl
import torch
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from nemo import lightning as nl
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from torch.nn import functional as F
from typing_extensions import override

from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.geneformer.api import GeneformerConfig
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.testing import testing_callbacks
from bionemo.testing.data.load import load
from bionemo.testing.harnesses import stop_and_go
from bionemo.testing.harnesses.mode import Mode


DATA_PATH: pathlib.Path = load("single_cell/testdata-20240506") / "cellxgene_2023-12-15_small" / "processed_data"

MODEL_PRECISION: Literal["bf16-mixed"] = "bf16-mixed"
USE_TE: bool = False  # TODO use this for high level decisions around whether we're ready to switch to TE


def geneformer_config():
    """Setups the default geneformer config taken from pretrain.py. Update as needed."""
    autocast_dtype = get_autocast_dtype(MODEL_PRECISION)
    return GeneformerConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=128,
        fp16=autocast_dtype == torch.float16,
        bf16=autocast_dtype == torch.bfloat16,
        fp32_residual_connection=False,
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=False,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,
        fp16_lm_cross_entropy=False,
        params_dtype=autocast_dtype,
        pipeline_dtype=autocast_dtype,
        autocast_dtype=autocast_dtype,
        gradient_accumulation_fusion=False,
        layernorm_zero_centered_gamma=False,
        layernorm_epsilon=1.0e-12,
        activation_func=F.gelu,
        qk_layernorm=False,
        apply_residual_connection_post_layernorm=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        get_attention_mask_from_fusion=False,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=False,
        biobert_spec_option=BiobertSpecOption.bert_layer_local_spec,
        nemo1_ckpt_path=None,
    )


def geneformer_datamodule(tokenizer, seq_length, median_dict, data_path=DATA_PATH):
    from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule

    num_dataset_workers = 0
    data = SingleCellDataModule(
        seq_length=seq_length,
        tokenizer=tokenizer,
        train_dataset_path=data_path / "train",
        val_dataset_path=data_path / "val",
        test_dataset_path=data_path / "test",
        random_token_prob=0.1,  # this is the incorrect setting we originally used.
        median_dict=median_dict,
        micro_batch_size=2,
        global_batch_size=2 * 1,  # micro batch size times divices
        # persistent workers is supported when num_dataset_workers > 0
        persistent_workers=num_dataset_workers > 0,
        pin_memory=False,
        num_workers=num_dataset_workers,
    )
    return data


class GeneformerStopAndGoTest(stop_and_go.StopAndGoHarness):
    num_steps: int = 10
    val_check_interval: int = 4
    limit_val_batches: int = 2
    lr: float = 1e-4
    precision: Literal["16-mixed", "bf16-mixed", "32"] = MODEL_PRECISION

    @override
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # setup data
        train_data_path = DATA_PATH / "train"
        preprocessor = GeneformerPreprocess(
            download_directory=train_data_path,
            medians_file_path=train_data_path / "medians.json",
            tokenizer_vocab_path=train_data_path / "geneformer.vocab",
        )
        match preprocessor.preprocess():
            case {"tokenizer": tokenizer, "median_dict": median_dict}:
                cls.tokenizer, cls.median_dict = tokenizer, median_dict
            case _:
                raise ValueError("Preprocessing must have failed.")

        # add your custom callbacks here
        # cls.stop_callbacks["YourCustomCallback"] = YourCustomCallback(mode=Mode.STOP)
        # cls.go_callbacks["YourCustomCallback"] = YourCustomCallback(mode=Mode.RESUME)

        # run stop and go
        cls.run_stop_and_go()

    @override
    @classmethod
    def setup_model(cls, mode: Mode) -> tuple[pl.LightningModule, pl.LightningDataModule, nl.MegatronOptimizerModule]:
        optim = MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=cls.lr,
                optimizer="adam",
                use_distributed_optimizer=True,
            ),
            lr_scheduler=CosineAnnealingScheduler(
                max_steps=cls.num_steps,
                min_lr=cls.lr / 100,
                warmup_steps=int(math.ceil(cls.num_steps * 0.1)),
                interval="step",
                monitor="reduced_train_loss",
                constant_steps=int(math.ceil(cls.num_steps * 0.1)),
            ),
        )
        module = biobert_lightning_module(config=geneformer_config(), tokenizer=cls.tokenizer, optimizer=optim)

        data = geneformer_datamodule(
            tokenizer=cls.tokenizer,
            seq_length=128,
            median_dict=cls.median_dict,
        )
        return module, data, optim

    def test_train_val_init_consumed_samples(self):
        """Override the parent test here because we know that train_consumed_go should be 8"""
        callback: testing_callbacks.TrainValInitConsumedSamplesStopAndGoCallback = self.resume_callbacks[
            "TrainValInitComsumedSamplesStopAndGoCallback"
        ]
        (
            (train_consumed_stop, val_consumed_stop),
            (train_consumed_go, val_consumed_go),
        ) = callback.load_stop_and_go_pickles()
        assert val_consumed_stop == 0
        assert val_consumed_go == 0
        assert train_consumed_stop == 0
        assert train_consumed_go == 8
