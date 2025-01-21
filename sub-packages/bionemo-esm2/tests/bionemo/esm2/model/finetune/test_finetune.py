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


import pytest
from nemo.lightning import io

from bionemo.core.data.load import load
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
from bionemo.esm2.model.finetune.train import train_model
from bionemo.testing import megatron_parallel_state_utils
from bionemo.testing.callbacks import MetricTracker


# To download a 8M internally pre-trained ESM2 model
pretrain_ckpt_path = load("esm2/8m:2.0")


@pytest.mark.needs_gpu
@pytest.mark.parametrize("with_peft", [True, False])
def test_esm2_finetune_token_classifier(
    tmp_path,
    tokenizer,
    dummy_data_per_token_classification_ft,
    with_peft: bool,
    n_steps_train: int = 50,
    seed: int = 42,
):
    if with_peft:
        pytest.xfail("FIXME PEFT fine-tuning not supported with fusions active")

    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        if with_peft:
            peft = ESM2LoRA()
        else:
            peft = None
        esm2_finetune_config = ESM2FineTuneTokenConfig(initial_ckpt_path=str(pretrain_ckpt_path))
        dataset = InMemoryPerTokenValueDataset(dummy_data_per_token_classification_ft, seed=seed)
        finetune_data_module = ESM2FineTuneDataModule(dataset, dataset)
        simple_ft_checkpoint, simple_ft_metrics, trainer = train_model(
            experiment_name="finetune_new_head",
            experiment_dir=tmp_path / "finetune_new_head",  # new checkpoint will land in a subdir of this
            config=esm2_finetune_config,  # same config as before since we are just continuing training
            data_module=finetune_data_module,
            n_steps_train=n_steps_train,
            metric_tracker=MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"]),
            tokenizer=tokenizer,
            peft=peft,
            _use_rich_model_summary=False,
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
    tmp_path,
    tokenizer,
    dummy_data_single_value_regression_ft,
    with_peft: bool,
    n_steps_train: int = 50,
    seed: int = 42,
):
    if with_peft:
        pytest.xfail("FIXME PEFT fine-tuning not supported")

    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        if with_peft:
            peft = ESM2LoRA()
        else:
            peft = None
        esm2_regression_finetune_config = ESM2FineTuneSeqConfig(initial_ckpt_path=str(pretrain_ckpt_path))
        dataset = InMemorySingleValueDataset(dummy_data_single_value_regression_ft, seed=seed)
        finetune_data_module = ESM2FineTuneDataModule(dataset, dataset)
        simple_ft_checkpoint, simple_ft_metrics, trainer = train_model(
            experiment_name="finetune_new_head_regression",
            experiment_dir=tmp_path / "finetune_new_head_regression",  # new checkpoint will land in a subdir of this
            config=esm2_regression_finetune_config,  # same config as before since we are just continuing training
            data_module=finetune_data_module,
            n_steps_train=n_steps_train,
            metric_tracker=MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"]),
            tokenizer=tokenizer,
            peft=peft,
            _use_rich_model_summary=False,
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
