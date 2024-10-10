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


from functools import lru_cache

import torch
import torch.nn as nn
from nemo.utils.model_utils import import_class_by_path

import bionemo.utils
from bionemo.data.datasets.single_value_dataset import SingleValueDataModule
from bionemo.model.core import MLPModel
from bionemo.model.core.encoder_finetuning import EncoderFineTuning


class FineTuneMegaMolBART(EncoderFineTuning):
    def __init__(self, cfg, trainer):
        self.encoder_frozen = cfg.encoder_frozen
        super().__init__(cfg, trainer=trainer)
        self.batch_target_name = self.cfg.data.target_column

    def configure_optimizers(self):
        super().setup_optimization(optim_config=self.cfg.finetuning_optim)

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def build_loss_fn(self):
        return bionemo.utils.lookup_or_use(torch.nn, self.cfg.downstream_task.loss_func)

    def build_task_head(self):
        regressor = MLPModel(
            layer_sizes=[
                self.encoder_model.cfg.model.hidden_size,
                self.cfg.downstream_task.hidden_layer_size,
                self.cfg.downstream_task.n_outputs,
            ],
            dropout=0.1,
        )
        # return regressor
        task_head = nn.Sequential(regressor, nn.Flatten(start_dim=0))
        return task_head

    def setup_encoder_model(self, cfg, trainer):
        infer_class = import_class_by_path(self.cfg.encoder_cfg.infer_target)
        pretrained_model = infer_class(
            self.cfg.encoder_cfg,
            freeze=self.encoder_frozen,
            restore_path=self.cfg.restore_encoder_path,
            training=not self.cfg.encoder_frozen,
        )
        return pretrained_model

    # the lru cache is kind of a hacky way to make sure this isn't set up if
    # it is already initialized, since this function doesn't return anything
    @lru_cache
    def data_setup(self):
        if self.encoder_frozen:
            model = self.encoder_model
        else:
            model = None
        self.data_module = SingleValueDataModule(self.cfg, self.trainer, model=model)

    def on_fit_start(self):
        self.build_train_valid_test_datasets()
        return super().on_fit_start()

    def build_train_valid_test_datasets(self):
        self._train_ds = self.data_module.get_sampled_train_dataset()
        self._validation_ds = self.data_module.get_sampled_val_dataset()
        self._test_ds = self.data_module.get_sampled_test_dataset()

    def encoder_forward(self, bart_model, batch: dict):
        if self.encoder_frozen:
            enc_output = batch["embeddings"]
        else:
            enc_output = bart_model.seq_to_embeddings(batch["embeddings"])
        return enc_output

    def extract_for_task_head(self, input_tensor):
        # NOTE investigate using mixed precision to remove need for float casting; maybe use setup_trainer method
        return input_tensor.float()

    def get_target_from_batch(self, batch):
        ret = batch["target"]

        return ret.float()
