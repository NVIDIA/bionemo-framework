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


import torch
from nemo.core.optim.lr_scheduler import get_scheduler
from nemo.core.optim.optimizers import get_optimizer
from nemo.utils import logging
from omegaconf import OmegaConf
from tqdm import tqdm


class ValidationTrainer:
    def __init__(self, cfg, model, loss_fn, metrics, metrics_args=None):
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.metrics_args = metrics_args
        self.optimizer, self.scheduler = self._setup_optim()

    def update_metrics(self, result, subset, output, target):
        for name, m_fun in self.metrics.items():
            metrics = m_fun(output, target, **self.metrics_args[name])
            key = subset + "_" + name
            if key not in result.keys():
                result[subset + "_" + name] = metrics
            else:
                result[subset + "_" + name] += metrics
        return result

    def _setup_optim(self):
        params = OmegaConf.to_container(self.cfg)
        optim_params = params["optim"]
        opt_name = optim_params.pop("name")
        sched_params = optim_params.pop("sched")
        sched_name = sched_params.pop("name")
        opt_cls = get_optimizer(opt_name)
        optimizer = opt_cls(self.model.parameters(), **optim_params)
        sched_cls = get_scheduler(sched_name)
        scheduler = sched_cls(optimizer, **sched_params)
        return optimizer, scheduler

    def fit(self, train_dataset, train_dataloader):
        for epoch in range(self.cfg.num_epochs):
            self.model.train(True)
            logging.info("EPOCH {}".format(epoch + 1))
            avg_loss = self.train_one_epoch(
                self.model, train_dataset, train_dataloader, self.loss_fn, self.optimizer, self.scheduler
            )
            logging.info("Training Avg Loss: {}".format(avg_loss))

    def test(self, test_dataset, test_dataloader):
        running_vloss = 0.0
        avg_vloss = 0.0
        avg_metrics = {}
        self.model.eval()
        for n, vdata in enumerate(tqdm(test_dataloader)):
            vinputs, vlabels = test_dataset.prepare_batch(vdata, test_dataset, task=self.cfg.task_type)
            voutputs = self.model(vinputs)
            vloss = self.loss_fn(voutputs, vlabels)
            running_vloss += vloss
            avg_metrics = self.update_metrics(avg_metrics, "test", voutputs, vlabels)

        avg_vloss = running_vloss / (n + 1)
        for key in avg_metrics.keys():
            avg_metrics[key] = avg_metrics[key] / (n + 1)
        avg_metrics["dwnstr_task_val_loss"] = avg_vloss
        return avg_metrics

    def train_one_epoch(self, model, train_dataset, train_dataloader, loss_fn, optimizer, scheduler):
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_dataloader)):
            inputs, labels = train_dataset.prepare_batch(data, train_dataset, task=self.cfg.task_type)
            torch.set_grad_enabled(True)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        return running_loss / (i + 1)
