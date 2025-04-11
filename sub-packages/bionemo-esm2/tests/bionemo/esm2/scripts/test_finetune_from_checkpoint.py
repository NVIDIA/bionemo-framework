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
from bionemo.esm2.model.finetune.dataset import InMemoryPerTokenValueDataset
from bionemo.esm2.model.finetune.token_model import ESM2FineTuneTokenConfig
from bionemo.esm2.scripts.finetune_esm2 import train_model
from bionemo.testing import megatron_parallel_state_utils
from bionemo.testing.callbacks import MetricTracker


@pytest.mark.needs_gpu
def test_esm2_resume_from_checkpoint(
    tmp_path,
    dummy_data_per_token_classification_ft,
    load_dcp,
    data_to_csv,
    with_peft=True,
    n_steps_train: int = 50,
    seed: int = 42,
):
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        # First training run
        weights_ckpt_first = "/workspaces/bionemo-framework/weights"
        # Second training run - ensure LoRA is initialized before loading checkpoint
        simple_ft_checkpoint, simple_ft_metrics_second, trainer = train_model(
            train_data_path=data_to_csv(dummy_data_per_token_classification_ft, tmp_path),
            valid_data_path=data_to_csv(dummy_data_per_token_classification_ft, tmp_path),
            experiment_name="finetune_new_head_token_classification2",
            restore_from_checkpoint_path=str(load("esm2/8m:2.0")),
            num_steps=n_steps_train,
            num_nodes=1,
            devices=1,
            min_seq_length=None,
            max_seq_length=1024,
            result_dir=tmp_path / "finetune_from_checkpoint",
            limit_val_batches=2,
            val_check_interval=n_steps_train // 2,
            log_every_n_steps=n_steps_train // 2,
            num_dataset_workers=1,
            lr=1e-5,
            scale_lr_layer="classification_head",
            lr_multiplier=1e2,
            micro_batch_size=4,
            accumulate_grad_batches=1,
            resume_if_exists=False,
            precision="bf16-mixed",
            task_type="classification",
            encoder_frozen=False,
            dataset_class=InMemoryPerTokenValueDataset,
            config_class=ESM2FineTuneTokenConfig,
            metric_tracker=MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"]),
            lora_finetune=with_peft,
            lora_checkpoint_path=str(weights_ckpt_first),
        )
        weights_ckpt_second = simple_ft_checkpoint / "weights"

        assert weights_ckpt_second.exists()
        assert weights_ckpt_second.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt_second)

        assert (
            simple_ft_metrics_second.collection_train["loss"][0]
            > simple_ft_metrics_second.collection_train["loss"][-1]
        )
        assert "val_acc" in trainer.logged_metrics
        # assert trainer.logged_metrics["val_acc"].item() <= 0.5  # TODO @farhad for a reasonable value
        assert trainer.model.model_transform is not None
        model = trainer.model[0].module.module.module
        assert all(not p.requires_grad for p in model.embedding.parameters())
        assert all(not p.requires_grad for name, p in model.encoder.named_parameters() if "adapter" not in name)
        assert all(p.requires_grad for name, p in model.encoder.named_parameters() if "adapter" in name)
        assert all(p.requires_grad for p in model.classification_head.parameters())

        weight_param_dict_second = load_dcp(weights_ckpt_second)
        weight_param_dict_first = load_dcp(weights_ckpt_first)

        for param in weight_param_dict_second.keys():
            assert any(keyword in param for keyword in {"head", "adapter", "optimizer", "output"})

        for param in weight_param_dict_first.keys():
            assert any(keyword in param for keyword in {"head", "adapter", "optimizer", "output"})
            assert weight_param_dict_first[param] != weight_param_dict_second[param]
