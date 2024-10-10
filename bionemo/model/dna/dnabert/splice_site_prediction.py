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

from nemo.utils.model_utils import import_class_by_path
from torch import nn
from torch.cuda.amp import autocast

from bionemo.data.dna.splice_site_dataset import SpliceSiteDataModule
from bionemo.model.core import MLPModel
from bionemo.model.core.encoder_finetuning import EncoderFineTuning
from bionemo.model.dna.dnabert.infer import DNABERTInference


class SpliceSiteBERTPredictionModel(EncoderFineTuning):
    def __init__(self, cfg, trainer):
        # commented code is old and dead
        self.encoder_frozen = cfg.encoder_frozen
        super().__init__(cfg, trainer=trainer)

        # TODO: we might want to adjust this to be a magic value or something taken from the dataset object.
        self.batch_target_name = self.cfg.target_name
        self.extract_idx = (self.cfg.seq_length - 1) // 2

    def build_loss_fn(self):
        return nn.CrossEntropyLoss()

    def configure_optimizers(self):
        # TODO do we need to configure a distributed optimizer?, similar to here:
        # https://github.com/NVIDIA/NeMo/blob/c9811f14fa1e1f990fd29f1aed1ae08e2ff6b014/nemo/collections/nlp/models/language_modeling/megatron_base_model.py#L349
        super().setup_optimization(optim_config=self.cfg.finetuning_optim)

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def build_task_head(self):
        # Task head in this case is a single layer
        return MLPModel(
            layer_sizes=[self.encoder_model.cfg.model.hidden_size, self.cfg.hidden_layer_size, self.cfg.n_outputs],
            dropout=0.1,
        )

    def get_target_from_batch(self, batch):
        return batch[self.batch_target_name]

    def setup_encoder_model(self, cfg, trainer):
        infer_class = import_class_by_path(self.cfg.encoder_cfg.infer_target)
        pretrained_model = infer_class(
            self.cfg.encoder_cfg,
            freeze=self.cfg.encoder_frozen,
            restore_path=self.cfg.restore_encoder_path,
            training=not self.cfg.encoder_frozen,
            adjust_config=False,
        )

        return pretrained_model

    def extract_for_task_head(self, input_tensor):
        return self.get_hiddens_for_idx(input_tensor, idx=self.extract_idx)

    @staticmethod
    def get_hiddens_for_idx(input_tensor, idx):
        return input_tensor[:, idx, :]

    def encoder_forward(self, bert_model: DNABERTInference, batch: dict):
        tokens = batch["text"]
        # 0 indicates padding, 1 indicates a lack of padding.
        padding_mask = batch["padding_mask"]
        with autocast(enabled=True):
            output_tensor = bert_model.model(tokens, padding_mask, token_type_ids=None, lm_labels=None)
        return output_tensor

    def forward(self, batch: dict):
        with autocast(enabled=True):
            output_tensor = self.encoder_forward(self.encoder_model, batch)
            task_input_tensor = self.extract_for_task_head(output_tensor)
            output = self.task_head(task_input_tensor)

        return output

    # the lru cache is kind of a hacky way to make sure this isn't set up if
    # it is already initialized, since this function doesn't return anything
    @lru_cache
    def data_setup(self):
        self.data_module = SpliceSiteDataModule(
            self.cfg,
            self.trainer,
            self.encoder_model,
        )

    def on_fit_start(self):
        self.build_train_valid_test_datasets()
        return super().on_fit_start()

    def build_train_valid_test_datasets(self):
        # if we want to make _train_ds optional for testing, we should be able
        # to enforce it with something like an `on_fit_start` method
        self._train_ds = self.data_module.get_sampled_train_dataset()
        val_dataset = self.data_module.get_sampled_val_dataset()
        if len(val_dataset) > 0:
            self._validation_ds = val_dataset
        test_dataset = self.data_module.get_sampled_test_dataset()
        if len(test_dataset) > 0:
            self._test_ds = test_dataset
