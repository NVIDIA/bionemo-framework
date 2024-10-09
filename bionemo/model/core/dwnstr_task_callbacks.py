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


import copy
from typing import List, Optional

import torch
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.model_utils import import_class_by_path
from pytorch_lightning import Callback
from torch.utils.data import DataLoader

from bionemo.data import accuracy, mse, per_token_accuracy
from bionemo.data.datasets import PerTokenValueDataModule, SingleValueDataModule
from bionemo.model.core.cnn import ConvNet, PerTokenMaskedCrossEntropyLoss
from bionemo.model.core.dwnstr_task_trainer import ValidationTrainer
from bionemo.model.core.mlp_model import MLPModel
from bionemo.model.utils import get_from_decoder_encoder_or_model


class DownstreamValidationCallback(Callback):
    def __init__(self, dset_dict, parent_cfg, plugins: Optional[List] = None):
        super().__init__()
        self.dset_dict = dset_dict
        self.plugins = plugins
        self.cfg = parent_cfg
        self.valid_cfg = None
        self.data_class = None
        self.loss_fn = None
        self.metrics = {}
        self.metrics_args = {}
        self.dwnstr_model = None

    # TODO(dorotat): this class can be improved in terms of docs and simplified in terms of what args are needed.
    # TODO(dorotat): self.plugins are not used

    def _prepare_model(self, main_model):
        main_model.freeze()
        args = {}
        if "ESM" in self.valid_cfg.infer_target:
            if main_model.enable_autocast:
                main_model.half()
            post_process = main_model.model.post_process
            main_model.model.post_process = False
            args["post_process"] = post_process
        return main_model, args

    def _release_model(self, main_model, args):
        main_model.unfreeze()
        if "ESM" in self.valid_cfg.infer_target:
            main_model.model.post_process = args["post_process"]
            main_model.float()
        return main_model

    def on_validation_epoch_end(self, trainer, main_model):
        main_model, args = self._prepare_model(main_model)
        infer_class = import_class_by_path(self.valid_cfg.infer_target)
        inference_wrapper = infer_class(self.cfg, main_model)
        new_cfg = copy.deepcopy(self.cfg.model)
        new_cfg.data = copy.deepcopy(self.valid_cfg)
        data_module = self.data_class(new_cfg, trainer, model=inference_wrapper)
        results = {}
        # Downstream task validation data and model are handled on rank 0
        train_dataset = data_module.get_sampled_train_dataset()
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.valid_cfg.batch_size, shuffle=False, pin_memory=False, num_workers=1
        )
        test_dataset = data_module.get_sampled_test_dataset()
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.valid_cfg.batch_size, shuffle=False, pin_memory=False, num_workers=1
        )
        if is_global_rank_zero():
            pt_trainer = ValidationTrainer(
                self.valid_cfg, self.dwnstr_model, self.loss_fn, self.metrics, self.metrics_args
            )
            pt_trainer.fit(train_dataset, train_dataloader)
            results = pt_trainer.test(test_dataset, test_dataloader)
        train_dataloader.dataset.free_memory()
        test_dataloader.dataset.free_memory()
        torch.distributed.barrier()
        self.log_dict(results, rank_zero_only=True, batch_size=self.valid_cfg.batch_size)
        for key, value in results.items():
            logging.info("{}: {}".format(key, value))
        main_model = self._release_model(main_model, args)


class PerTokenPredictionCallback(DownstreamValidationCallback):
    def __init__(self, dset_dict, parent_cfg, plugins: Optional[List] = None):
        super().__init__(dset_dict, parent_cfg)
        self.dset_dict = dset_dict
        self.plugins = plugins
        self.cfg = parent_cfg
        self.valid_cfg = self.cfg.model.dwnstr_task_validation.dataset
        self.data_class = PerTokenValueDataModule
        self.metrics = {}
        self.metrics_args = {}
        for idx, name in enumerate(self.valid_cfg.target_column):
            if self.valid_cfg.task_type == "token-level-classification":
                self.metrics[name + "_accuracy"] = per_token_accuracy
                self.metrics_args[name + "_accuracy"] = {"label_id": idx}
            else:
                raise ValueError(
                    "Invalid task_type was provided {}. Supported task_type: 'token-level-classification'".format(
                        self.valid_cfg.task_type
                    )
                )
        pretrain_model_hidden_size = get_from_decoder_encoder_or_model(self.cfg.model, "hidden_size")
        output_sizes = self.valid_cfg.target_sizes
        self.dwnstr_model = ConvNet(pretrain_model_hidden_size, output_sizes=output_sizes).to("cuda")
        self.loss_fn = PerTokenMaskedCrossEntropyLoss()


class SingleValuePredictionCallback(DownstreamValidationCallback):
    def __init__(self, dset_dict, parent_cfg, plugins: Optional[List] = None):
        super().__init__(dset_dict, parent_cfg)
        self.dset_dict = dset_dict
        self.plugins = plugins
        self.cfg = parent_cfg
        self.valid_cfg = self.cfg.model.dwnstr_task_validation.dataset
        self.data_class = SingleValueDataModule
        if self.valid_cfg.task_type == "classification":
            self.metrics[self.valid_cfg.target_column + "_accuracy"] = accuracy
            self.metrics_args[self.valid_cfg.target_column + "_accuracy"] = {}
            self.loss_fn = torch.nn.CrossEntropyLoss()
            num_heads = self.valid_cfg.num_classes
        elif self.valid_cfg.task_type == "regression":
            self.metrics[self.valid_cfg.target_column + "_mse"] = mse
            self.metrics_args[self.valid_cfg.target_column + "_mse"] = {}
            self.loss_fn = torch.nn.MSELoss()
            num_heads = 1
        else:
            raise ValueError("Invalid task_type was provided {}".format(self.valid_cfg.task_type))
        self.dwnstr_model = MLPModel(
            layer_sizes=[get_from_decoder_encoder_or_model(parent_cfg.model, "hidden_size"), 256, 128, num_heads]
        ).to("cuda")
