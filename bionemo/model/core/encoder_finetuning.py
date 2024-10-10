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


from abc import ABC, abstractmethod
from typing import List

import torch
from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator
from megatron.core import parallel_state
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core.classes import ModelPT
from nemo.core.classes.exportable import Exportable
from nemo.utils import logging

from bionemo.model.utils import (
    compute_consumed_samples,
    extract_consumed_samples_from_ckpt,
)


class EncoderFineTuning(ModelPT, Exportable, ABC):
    def __init__(self, cfg, trainer):
        super().__init__(cfg, trainer)
        self._save_restore_connector = NLPSaveRestoreConnector()
        self.cfg = cfg

        enc_model = self.setup_encoder_model(cfg, trainer)
        self.encoder_model = self.reconfigure_model(cfg, enc_model)

        self.init_consumed_samples = 0
        self.loss_fn = self.build_loss_fn()
        self.task_head = self.build_task_head()
        self.predict_dataset = None
        self.metrics = None

        self._reduced_lm_loss_buffer: List[torch.Tensor] = []  # support gradient accumulation

    def list_available_models(self):
        return []

    def on_train_start(self) -> None:
        super().on_train_start()
        self.init_global_step = self.trainer.global_step

    def configure_optimizers(self):
        # TODO do we need to configure a distributed optimizer?, similar to here:
        # https://github.com/NVIDIA/NeMo/blob/c9811f14fa1e1f990fd29f1aed1ae08e2ff6b014/nemo/collections/nlp/models/language_modeling/megatron_base_model.py#L349
        return super().configure_optimizers()

    @abstractmethod
    def setup_encoder_model(self, cfg, trainer):
        raise NotImplementedError()

    def reconfigure_model(self, cfg, model):
        self.to(model.device)

        if parallel_state.is_unitialized():
            logging.info("DDP is not initialized. Initializing...")

            def dummy():
                return

            if model.trainer.strategy.launcher is not None:
                model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
            model.trainer.strategy.setup_environment()

        cur_data_parallel_world_size = parallel_state.get_data_parallel_world_size()

        _reconfigure_microbatch_calculator(
            rank=0,  # This doesn't matter since it is only used for logging
            rampup_batch_size=None,
            # NOTE depends on global_batch_size being set in config, see MR !95
            global_batch_size=cfg.global_batch_size,
            micro_batch_size=cfg.micro_batch_size,
            data_parallel_size=cur_data_parallel_world_size,
        )

        return model

    def extract_for_task_head(self, input_tensor):
        return input_tensor

    @abstractmethod
    def encoder_forward(self, bert_model, batch: dict):
        pass

    @abstractmethod
    def build_loss_fn(self):
        pass

    @abstractmethod
    def build_task_head(self):
        pass

    @abstractmethod
    def get_target_from_batch(self, batch):
        pass

    # doing dataset (custom) setup in the normal set up method ensures that
    # distributed is initialized appropriately by the time the data is loaded,
    # since distributed is needed for NeMo upsampling
    # It is also nice to do the data setup here because the data are not
    # initialized unless this module specifically is trained
    def setup(self, *args, **kwargs):
        resume_checkpoint_path = self.trainer._checkpoint_connector._ckpt_path

        if resume_checkpoint_path:
            init_consumed_samples = extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0

        self.init_consumed_samples = init_consumed_samples
        self.data_setup()
        super().setup(*args, **kwargs)

    def on_fit_start(self):
        if self.trainer.limit_train_batches > 0:
            self.setup_training_data(self.cfg)
        if self.trainer.limit_val_batches > 0:
            self.setup_validation_data(self.cfg)
        if hasattr(self, "_test_ds"):
            self.setup_test_data(self.cfg)

    def on_test_start(self):
        self.setup_test_data(self.cfg)

    @abstractmethod
    def data_setup(self):
        pass

    def predict_dataloader(self):
        self.predict_dataset = self.data_module.create_dataset(self.cfg.data.predict_file)
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            shuffle=False,
            batch_size=self.cfg.micro_batch_size,
        )

    def forward(self, batch: dict):
        output_tensor = self.encoder_forward(self.encoder_model, batch)
        task_input_tensor = self.extract_for_task_head(output_tensor)
        output = self.task_head(task_input_tensor)

        return output

    def _calc_step(self, batch, batch_idx):
        output_tensor = self.forward(batch)
        target = self.get_target_from_batch(batch)
        loss = self.loss_fn(output_tensor, target)

        return loss, output_tensor, target

    def add_metrics(self, metrics, metrics_args=None):
        self.metrics = metrics
        self.metrics_args = metrics_args

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._calc_step(batch, batch_idx)
        reduced_loss = average_losses_across_data_parallel_group([loss])
        self._reduced_lm_loss_buffer.append(reduced_loss)

        if (
            len(self._reduced_lm_loss_buffer) + 1
        ) % self.trainer.accumulate_grad_batches == 0:  # batch_idx is not the appropriate choice when num_epoch > 1
            average_reduced_loss = sum(self._reduced_lm_loss_buffer) / len(self._reduced_lm_loss_buffer)
            self.log("reduced_train_loss", average_reduced_loss, prog_bar=True)
            self.log_stats()
            self._reduced_lm_loss_buffer = []

        return loss

    def log_stats(self):
        lr = self._optimizer.param_groups[0]["lr"]
        self.log("lr", lr)
        self.log("global_step", float(self.trainer.global_step), prog_bar=True, sync_dist=True)
        self.log(
            "consumed_samples",
            float(compute_consumed_samples(self, self.trainer.global_step - self.init_global_step + 1)),
            prog_bar=True,
            sync_dist=True,
        )

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        # Megatron sampler
        if hasattr(self.cfg.data, "dataloader_type") and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == "single":
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            elif self.cfg.data.dataloader_type == "cyclic":
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
        )

    def _step(self, batch, batch_idx, subset):
        loss, output, target = self._calc_step(batch, batch_idx)
        reduced_loss = average_losses_across_data_parallel_group([loss])
        result = {}
        result[subset + "_loss"] = reduced_loss
        if self.metrics is not None:
            for name, m_fun in self.metrics.items():
                metrics = m_fun(output, target, **self.metrics_args[name])
                metrics = torch.tensor(metrics).to("cuda")
                result[subset + "_" + name] = average_losses_across_data_parallel_group([metrics])
        return result

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs.append(self._step(batch, batch_idx, subset="val"))

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(self._step(batch, batch_idx, subset="test"))

    def _epoch_end(self, outputs, subset):
        if len(outputs) > 0:
            for name in outputs[0].keys():
                batch_value = [x[name] for x in outputs]
                averaged_value = torch.stack(batch_value).mean()
                self.log(name, averaged_value, sync_dist=True)

    def on_validation_epoch_end(self):
        self._epoch_end(self.validation_step_outputs, subset="val")
        self.log("global_step", float(self.trainer.global_step), prog_bar=True)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self._epoch_end(self.test_step_outputs, subset="test")
        self.test_step_outputs.clear()

    def setup_training_data(self, cfg):
        if hasattr(self, "_train_ds"):
            consumed_samples = compute_consumed_samples(self, 0)
            logging.info(
                f"Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}"
            )
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)
            self.data_module.adjust_train_dataloader(self, self._train_dl)

    def setup_validation_data(self, cfg):
        if hasattr(self, "_validation_ds"):
            consumed_samples = 0
            logging.info(
                f"Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}"
            )
            self._validation_dl = self.build_pretraining_data_loader(self._validation_ds, consumed_samples)
            self.data_module.adjust_val_dataloader(self, self._validation_dl)

    def setup_test_data(self, cfg):
        if hasattr(self, "_test_ds"):
            consumed_samples = 0
            logging.info(
                f"Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}"
            )
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)
            self.data_module.adjust_test_dataloader(self, self._test_dl)
