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


import os

from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.utils import logging
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from bionemo.data.prott5_utils import prott5_build_dataset


class ProtT5nvModel(MegatronT5Model):
    """
    Prot T5 training
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        # validate cfg
        self._validate_cfg()

    def setup_training_data(self, cfg):
        super().setup_training_data(cfg)

    def setup_validation_data(self, cfg):
        if hasattr(self, "_validation_ds"):
            if self._validation_ds is not None:
                consumed_samples = 0
                self._validation_dl = self.build_pretraining_data_loader(
                    self._validation_ds, consumed_samples, num_workers=0
                )

    def setup_test_data(self, cfg):
        if hasattr(self, "_test_ds"):
            if self._test_ds is not None:
                consumed_samples = 0
                self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples, num_workers=0)

    def build_train_valid_test_datasets(self):
        logging.info(f"Building {self.model_name} datasets.")
        global_batch_size = self._cfg.global_batch_size

        # Calculating number of times when validation stage is performed during a model's training assuming
        # trainer.max_steps is set, not trainer.max_epochs
        if self.trainer.max_steps is None:
            raise ValueError("ProtT5nvModel requires setting self.trainer.max_steps")

        if isinstance(self.trainer.val_check_interval, float):
            # if trainer.val_check_interval is a float, it represents a fraction of the training epoch
            # to check validation, hence we need to check how many times the validation stage will be executed
            val_steps = int(1 // self.trainer.val_check_interval)
        else:
            val_steps = self.trainer.max_steps // self.trainer.val_check_interval
        eval_iters = (val_steps + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches
        train_valid_test_num_samples = [
            self.trainer.max_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]

        kwargs = {
            "cfg": self._cfg,
            "trainer": self.trainer,
            "tokenizer": self.tokenizer,
            "data_impl": self._cfg.data.data_impl,
            "max_seq_length": self._cfg.data.seq_length,
            "max_seq_length_dec": self._cfg.data.seq_length_dec,
            "masked_lm_prob": self._cfg.data.masked_lm_prob,
            "short_seq_prob": self._cfg.data.short_seq_prob,
            "seed": self._cfg.seed,
            "skip_warmup": self._cfg.data.skip_warmup,
            "dataset_type": self._cfg.data.get("dataset_type", self.model_name.lower()),
            "max_ngram_size": self._cfg.data.get("max_ngram_size", 1),
            "mean_ngram_size": self._cfg.data.get("mean_ngram_size", None),
            "geometric_dist": self._cfg.data.get("geometric_dist", True),
            "permutation": self._cfg.data.get("permutation", False),
            "whole_word_masking": self._cfg.data.get("whole_word_masking", False),
            "favor_long_ngrams": self._cfg.data.get("favor_long_ngrams", False),
            "data_impl_kwargs": self._cfg.data.data_impl_kwargs.get(self._cfg.data.data_impl, {}),
        }
        # we add here index_mapping_dir to data_impl_kwargs (which are data_impl specific)
        with open_dict(kwargs["data_impl_kwargs"]):
            kwargs["data_impl_kwargs"]["index_mapping_dir"] = self._cfg.data.get("index_mapping_dir", None)

        dataset_path = self._cfg.data.dataset_path
        ds_train = self._cfg.data.dataset.train
        ds_val = self._cfg.data.dataset.val
        ds_test = self._cfg.data.dataset.test
        self._train_ds = prott5_build_dataset(
            data_prefix=os.path.join(dataset_path, "train", ds_train),
            num_samples=train_valid_test_num_samples[0],
            name="train",
            **kwargs,
        )
        self._validation_ds = prott5_build_dataset(
            data_prefix=os.path.join(dataset_path, "val", ds_val),
            num_samples=train_valid_test_num_samples[1],
            name="valid",
            **kwargs,
        )

        self._test_ds = prott5_build_dataset(
            data_prefix=os.path.join(dataset_path, "test", ds_test),
            num_samples=train_valid_test_num_samples[2],
            name="test",
            **kwargs,
        )

        logging.info(f"Length of train dataset: {len(self._train_ds)}")
        logging.info(f"Length of val dataset: {len(self._validation_ds)}")
        logging.info(f"Length of test dataset: {len(self._test_ds) if self._test_ds is not None else None}")
        logging.info(f"Finished building {self.model_name} datasets.")
        return self._train_ds, self._validation_ds, self._test_ds
