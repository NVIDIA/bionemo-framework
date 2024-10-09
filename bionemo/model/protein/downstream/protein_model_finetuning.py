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
from nemo.utils.model_utils import import_class_by_path
from omegaconf import OmegaConf

import bionemo.utils
from bionemo.data.datasets import (
    PerTokenValueDataModule,
    PerTokenValueDataset,
    SingleValueDataModule,
    SingleValueDataset,
)
from bionemo.model.core import ConvNet, MLPModel, PerTokenMaskedCrossEntropyLoss
from bionemo.model.core.encoder_finetuning import EncoderFineTuning


class FineTuneProteinModel(EncoderFineTuning):
    def __init__(self, cfg, trainer):
        self.encoder_frozen = cfg.encoder_frozen
        self.task_type = cfg.data.task_type

        if self.encoder_frozen and OmegaConf.select(cfg, "peft.enabled"):
            raise ValueError("Using PEFT requires encoder_frozen: False for training")

        super().__init__(cfg, trainer=trainer)

    def configure_optimizers(self):
        super().setup_optimization(optim_config=self.cfg.finetuning_optim)

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def build_loss_fn(self):
        if self.task_type in ["regression", "classification"]:
            loss = bionemo.utils.lookup_or_use(torch.nn, self.cfg.loss_func)
        elif self.task_type == "token-level-classification":
            loss = PerTokenMaskedCrossEntropyLoss()
        return loss

    def build_task_head(self):
        if self.task_type in ["regression", "classification"]:
            if len(self.cfg.data.target_sizes) != 1:
                raise ValueError("Classification and regression tasks do not support multi-head predictions")
            task_head = MLPModel(
                layer_sizes=[
                    self.encoder_model.cfg.model.hidden_size,
                    self.cfg.hidden_layer_size,
                    self.cfg.data.target_sizes[0],
                ],
                dropout=self.cfg.dropout_rate,
            )

        elif self.task_type == "token-level-classification":
            task_head = ConvNet(
                self.encoder_model.cfg.model.hidden_size,
                output_sizes=self.cfg.data.target_sizes,
                bottleneck_dim=self.cfg.cnn_dim,
                dropout_rate=self.cfg.dropout_rate,
            )
        return task_head

    def setup_encoder_model(self, cfg, trainer):
        # Uses PEFT flag to determine whether to load enconder +/- adapters
        infer_class = import_class_by_path(self.cfg.encoder_cfg.infer_target)
        pretrained_model = infer_class(
            self.cfg.encoder_cfg,
            freeze=self.cfg.encoder_frozen,
            restore_path=self.cfg.restore_encoder_path,
            training=not self.cfg.encoder_frozen,
            adjust_config=False,
        )
        return pretrained_model

    @lru_cache
    def data_setup(self):
        # TODO: address behavior for when we want peft
        if self.encoder_frozen:
            model = self.encoder_model
        else:
            model = None

        if self.task_type in ["regression", "classification"]:
            self.data_module = SingleValueDataModule(self.cfg, self.trainer, model=model)
        elif self.task_type == "token-level-classification":
            self.data_module = PerTokenValueDataModule(self.cfg, self.trainer, model=model)

    def on_fit_start(self):
        self.build_train_valid_test_datasets()
        return super().on_fit_start()

    def on_test_start(self):
        self._test_ds = self.data_module.get_sampled_test_dataset()
        return super().on_test_start()

    def build_train_valid_test_datasets(self):
        if self.trainer.limit_train_batches == 0:
            self._train_ds = None
        else:
            self._train_ds = self.data_module.get_sampled_train_dataset()
        if self.trainer.limit_val_batches == 0:
            self._validation_ds = None
        else:
            self._validation_ds = self.data_module.get_sampled_val_dataset()
        self._test_ds = self.data_module.get_sampled_test_dataset()

    def encoder_forward(self, protein_model, batch: dict):
        """
        Params:
            batch: Dictionary that conatains
                "embeddings": this is a sequence of amino acids when `encoder_frozen` is False.
        """
        if self.encoder_frozen:
            # If encoder is frozen, the Dataset returns the embedding directly
            enc_output = batch["embeddings"]
        elif self.task_type in ["regression", "classification"]:
            # contains a sequence, not an embedding
            enc_output = protein_model.seq_to_embeddings(batch["embeddings"])  # (B, D)
        else:
            # contains a sequence, not an embedding
            enc_output, _ = protein_model.seq_to_hiddens(batch["embeddings"])  # (B, S, D)
            batch_size, seq_len, emb_dim = enc_output.size()
            enc_output = torch.cat(
                [
                    enc_output,
                    torch.zeros((batch_size, (self.cfg.seq_length - seq_len), emb_dim)).to(device=enc_output.device),
                ],
                dim=1,
            )

        return enc_output

    def extract_for_task_head(self, input_tensor):
        # NOTE investigate using mixed precision to remove need for float casting; maybe use setup_trainer method
        return input_tensor.float()

    def get_target_from_batch(self, batch):
        if self._train_ds is not None:
            data = self._train_ds
        elif self._test_ds is not None:
            data = self._test_ds
        _, (labels, masks) = PerTokenValueDataset.prepare_batch(batch, data)
        padded_labels = []
        padded_masks = []
        for i in range(len(labels)):
            label = labels[i]
            mask = masks[i]
            batch_size, seq_len, n_labels = label.size()
            label = torch.cat(
                [
                    label,
                    torch.zeros((batch_size, (self.cfg.seq_length - seq_len), n_labels)).to(device=label.device),
                ],
                dim=1,
            )
            mask = torch.cat(
                [mask, torch.zeros((batch_size, (self.cfg.seq_length - seq_len))).to(device=mask.device)],
                dim=1,
            )
            padded_labels.append(label)
            padded_masks.append(mask)
        return (padded_labels, padded_masks)

    def _calc_step(self, batch, batch_idx):
        output_tensor = self.forward(batch)
        if self.task_type in ["regression", "classification"]:
            _, target = SingleValueDataset.prepare_batch(batch, self._train_ds, task=self.task_type)
        elif self.task_type == "token-level-classification":
            target = self.get_target_from_batch(batch)
        loss = self.loss_fn(output_tensor, target)
        return loss, output_tensor, target

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        custom_state_dict = super(FineTuneProteinModel, self).state_dict()

        if not self.encoder_frozen:
            custom_state_dict.update(self.encoder_model.model.state_dict())

        return custom_state_dict
