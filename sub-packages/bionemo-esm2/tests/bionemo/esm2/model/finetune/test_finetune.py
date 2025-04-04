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
from typing import Generator, Tuple

import pytest
import pytorch_lightning as pl
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm as nllm
from nemo.lightning import io, resume
from nemo.lightning.nemo_logger import NeMoLogger
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.callbacks.model_transform import ModelTransform
from nemo.lightning.pytorch.callbacks.peft import PEFT
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from pytorch_lightning.loggers import TensorBoardLogger

from bionemo import esm2
from bionemo.esm2.api import ESM2Config, ESM2GenericConfig
from bionemo.esm2.data.datamodule import ESMDataModule
from bionemo.esm2.data.tokenizer import BioNeMoESMTokenizer
from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule
from bionemo.esm2.model.finetune.finetune_regressor import (
    ESM2FineTuneSeqConfig,
    InMemorySingleValueDataset,
)
from bionemo.esm2.model.finetune.finetune_token_classifier import (
    ESM2FineTuneTokenConfig,
    InMemoryPerTokenValueDataset,
)
from bionemo.esm2.model.finetune.peft import ESM2LoRA
from bionemo.llm.model.biobert.lightning import BioBertLightningModule
from bionemo.testing import megatron_parallel_state_utils
from bionemo.testing.callbacks import MetricTracker
from bionemo.testing.data.load import load


bionemo2_root: Path = (
    # esm2 module's path is the most dependable --> don't expect this to change!
    Path(esm2.__file__)
    # This gets us from 'sub-packages/bionemo-esm2/src/bionemo/esm2/__init__.py' to 'sub-packages/bionemo-esm2'
    .parent.parent.parent.parent
    # From here, we want to get to the root of the repository: _before_ sub-packages/
    .parent.parent
).absolute()
assert bionemo2_root != Path("/")
nemo1_checkpoint_path: Path = load("esm2/nv_650m:1.0")


@pytest.fixture(scope="module")
def esm2_2layer_config() -> Generator[ESM2Config, None, None]:
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        yield ESM2Config(num_layers=3, hidden_size=128)


@pytest.fixture
def pretrain_data_module(dummy_protein_dataset, dummy_parquet_train_val_inputs):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs
    data_module = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
        global_batch_size=8,
        micro_batch_size=4,
        min_seq_length=None,
        max_seq_length=1024,
    )
    yield data_module


def _train_model(
    name: str,
    root_dir: Path,
    config: ESM2GenericConfig,
    data_module: pl.LightningDataModule,
    n_steps_train: int,
    tokenizer: BioNeMoESMTokenizer,
    peft: PEFT | None = None,
) -> Tuple[Path, MetricTracker, nl.Trainer]:
    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_last=True,
        save_on_train_epoch_end=True,
        monitor="reduced_train_loss",  # TODO find out how to get val_loss logged and use "val_loss",
        every_n_train_steps=n_steps_train // 2,
        always_save_context=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
    )

    # Setup the logger and train the model
    nemo_logger = NeMoLogger(
        log_dir=str(root_dir),
        name=name,
        tensorboard=TensorBoardLogger(save_dir=root_dir, name=name),
        ckpt=checkpoint_callback,
    )
    # Needed so that the trainer can find an output directory for the profiler
    # ckpt_path needs to be a string for SerDe
    optimizer = MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=5e-4,
            optimizer="adam",
            use_distributed_optimizer=True,
            fp16=config.fp16,
            bf16=config.bf16,
        )
    )
    module = BioBertLightningModule(config=config, tokenizer=tokenizer, optimizer=optimizer, model_transform=peft)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        ddp="megatron",
        find_unused_parameters=True,
        enable_nemo_ckpt_io=True,
    )
    metric_tracker = MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"])
    callbacks = [metric_tracker]
    if peft is not None:
        callbacks.append(ModelTransform())
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy=strategy,
        limit_val_batches=2,
        val_check_interval=n_steps_train // 2,
        max_steps=n_steps_train,
        num_nodes=1,
        log_every_n_steps=n_steps_train // 2,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )
    nllm.train(
        model=module,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )
    ckpt_path = Path(checkpoint_callback.last_model_path.replace(".ckpt", ""))
    return ckpt_path, metric_tracker, trainer


@pytest.mark.needs_gpu
@pytest.mark.parametrize("with_peft", [True, False])
def test_esm2_finetune_token_classifier(
    tmpdir,
    esm2_2layer_config,
    tokenizer,
    pretrain_data_module,
    dummy_data_per_token_classification_ft,
    with_peft: bool,
    n_steps_train: int = 50,
    seed: int = 42,
):
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        ckpt_path, initial_metrics, trainer = _train_model(
            name="test_experiment",
            root_dir=tmpdir / "pretrain",
            config=esm2_2layer_config,
            data_module=pretrain_data_module,
            n_steps_train=n_steps_train,
            tokenizer=tokenizer,
        )
        pretrain_requires_grad = [p.requires_grad for _, p in trainer.model.named_parameters()]
        assert all(pretrain_requires_grad), "Frozen parameters in pretraining"

        weights_ckpt = ckpt_path / "weights"
        assert weights_ckpt.exists()
        assert weights_ckpt.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt)
        assert initial_metrics.collection_train["loss"][0] > initial_metrics.collection_train["loss"][-1]

    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        if with_peft:
            peft = ESM2LoRA()
        else:
            peft = None
        esm2_finetune_config = ESM2FineTuneTokenConfig(initial_ckpt_path=str(ckpt_path))
        dataset = InMemoryPerTokenValueDataset(dummy_data_per_token_classification_ft)
        finetune_data_module = ESM2FineTuneDataModule(dataset, dataset)
        simple_ft_checkpoint, simple_ft_metrics, trainer = _train_model(
            name="finetune_new_head",
            root_dir=tmpdir / "finetune_new_head",  # new checkpoint will land in a subdir of this
            config=esm2_finetune_config,  # same config as before since we are just continuing training
            data_module=finetune_data_module,
            n_steps_train=n_steps_train,
            tokenizer=tokenizer,
            peft=peft,
        )

        weights_ckpt = simple_ft_checkpoint / "weights"
        assert weights_ckpt.exists()
        assert weights_ckpt.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt)
        assert simple_ft_metrics.collection_train["loss"][0] > simple_ft_metrics.collection_train["loss"][-1]

        if with_peft:
            assert trainer.model.model_transform is not None
            model = trainer.model[0].module.module.module
            assert all(not p.requires_grad for p in model.embedding.parameters())
            assert all(not p.requires_grad for name, p in model.encoder.named_parameters() if "adapter" not in name)
            assert all(p.requires_grad for name, p in model.encoder.named_parameters() if "adapter" in name)
            assert all(p.requires_grad for p in model.classification_head.parameters())
        else:
            encoder_requires_grad = [
                p.requires_grad for name, p in trainer.model.named_parameters() if "classification_head" not in name
            ]
            assert not all(encoder_requires_grad), "Pretrained model is not fully frozen during fine-tuning"


@pytest.mark.needs_gpu
@pytest.mark.parametrize("with_peft", [True, False])
def test_esm2_finetune_regressor(
    tmpdir,
    esm2_2layer_config,
    tokenizer,
    pretrain_data_module,
    dummy_data_single_value_regression_ft,
    with_peft: bool,
    n_steps_train: int = 50,
    seed: int = 42,
):
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        ckpt_path, initial_metrics, trainer = _train_model(
            name="test_experiment",
            root_dir=tmpdir / "pretrain",
            config=esm2_2layer_config,
            data_module=pretrain_data_module,
            n_steps_train=n_steps_train,
            tokenizer=tokenizer,
        )
        pretrain_requires_grad = [p.requires_grad for _, p in trainer.model.named_parameters()]
        assert all(pretrain_requires_grad), "Frozen parameters in pretraining"

        weights_ckpt = ckpt_path / "weights"
        assert weights_ckpt.exists()
        assert weights_ckpt.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt)
        assert initial_metrics.collection_train["loss"][0] > initial_metrics.collection_train["loss"][-1]

    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        if with_peft:
            peft = ESM2LoRA()
        else:
            peft = None
        esm2_regression_finetune_config = ESM2FineTuneSeqConfig(initial_ckpt_path=str(ckpt_path))
        dataset = InMemorySingleValueDataset(dummy_data_single_value_regression_ft)
        finetune_data_module = ESM2FineTuneDataModule(dataset, dataset)
        simple_ft_checkpoint, simple_ft_metrics, trainer = _train_model(
            name="finetune_new_head_regression",
            root_dir=tmpdir / "finetune_new_head_regression",  # new checkpoint will land in a subdir of this
            config=esm2_regression_finetune_config,  # same config as before since we are just continuing training
            data_module=finetune_data_module,
            n_steps_train=n_steps_train,
            tokenizer=tokenizer,
            peft=peft,
        )

        weights_ckpt = simple_ft_checkpoint / "weights"
        assert weights_ckpt.exists()
        assert weights_ckpt.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt)
        assert simple_ft_metrics.collection_train["loss"][0] > simple_ft_metrics.collection_train["loss"][-1]

        if with_peft:
            assert trainer.model.model_transform is not None
            model = trainer.model[0].module.module.module
            assert all(not p.requires_grad for p in model.embedding.parameters())
            assert all(not p.requires_grad for name, p in model.encoder.named_parameters() if "adapter" not in name)
            assert all(p.requires_grad for name, p in model.encoder.named_parameters() if "adapter" in name)
            assert all(p.requires_grad for p in model.regression_head.parameters())
        else:
            encoder_requires_grad = [
                p.requires_grad for name, p in trainer.model.named_parameters() if "regression_head" not in name
            ]
            assert not all(encoder_requires_grad), "Pretrained model is not fully frozen during fine-tuning"
