# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

from abc import abstractmethod
from functools import lru_cache

from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.utils import logging


class BioNeMoDataModule(object):
    """Base Class for BioNeMo Data Modules.

    Data Modules coordinate the data-driven functions for BioNeMo Modules:
    * Instantiating train/val/test dataset
    * Adjustments to dataloaders, such as adding collate functions
    * Instantiating tokenizers
    * Inferring the number of global samples (up/downsampling included)

    In order to perform these duties, a child class must implement:
    * `train_dataset`
    * `val_dataset`
    * `test_dataset`

    For an additional level of control, a child class might implement:
    * `sample_train_dataset`
    * `sample_val_dataset`
    * `sample_test_dataset`
    * `adjust_train_dataloader`
    * `adjust_val_dataloader`
    * `adjust_test_dataloader`

    """

    def __init__(self, cfg, trainer):
        """Initializes a BioNeMoDataModule

        Arguments:
            cfg (OmegaConf): A config object for a model
            trainer (pytorch_lightning.Trainer): Trainer of the corresponding
                model.

        """
        self.model_cfg = cfg
        self.cfg = cfg.data
        self.trainer = trainer
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self.train_num_samples, self.val_num_samples, self.test_num_samples = [
            None,
            None,
            None,
        ]

    @abstractmethod
    def train_dataset(self):
        """Creates a training dataset

        Returns:
            Dataset: dataset to use for training

        """
        raise NotImplementedError()

    @abstractmethod
    def val_dataset(self):
        """Creates a validation dataset

        Returns:
            Dataset: dataset to use for validation

        """
        raise NotImplementedError()

    @abstractmethod
    def test_dataset(self):
        """Creates a testing dataset

        Returns:
            Dataset: dataset to use for testing

        """
        raise NotImplementedError()

    def adjust_train_dataloader(self, model, dataloader):
        """Allows adjustments to the training dataloader

        This is a good place to adjust the collate function of the dataloader.

        """
        pass

    def adjust_val_dataloader(self, model, dataloader):
        """Allows adjustments to the validation dataloader

        This is a good place to adjust the collate function of the dataloader.

        """
        pass

    def adjust_test_dataloader(self, model, dataloader):
        """Allows adjustments to the testing dataloader

        This is a good place to adjust the collate function of the dataloader.

        """
        pass

    def init_num_samples(self):
        """Sets the number of samples for training, validation, and testing

        Side Effect:
            Sets:
                * self.train_num_samples
                * self.val_num_samples
                * self.test_num_samples

        """
        global_batch_size = self.get_global_batch_size()
        max_train_steps = self.get_max_train_steps()
        eval_iters = self.get_total_eval_batches()
        test_iters = self.get_total_test_batches()

        self.train_num_samples, self.val_num_samples, self.test_num_samples = [
            int(max_train_steps * global_batch_size),
            int(eval_iters * global_batch_size),
            int(test_iters * global_batch_size),
        ]

    def sample_train_dataset(self, dataset):
        """Creates a sampled version of the training dataset. Returns the
        unmodified version of input `dataset` if not overriden.

        """
        return dataset

    def sample_val_dataset(self, dataset):
        """Creates a sampled version of the validation dataset. Returns the
        unmodified version of input `dataset` if not overriden.

        """
        return dataset

    def sample_test_dataset(self, dataset):
        """Creates a sampled version of the testing dataset. Returns the
        unmodified version of input `dataset` if not overriden.

        """
        return dataset

    def get_global_batch_size(self):
        return self.model_cfg.global_batch_size

    def get_max_train_steps(self):
        return self.trainer.max_steps * self.trainer.accumulate_grad_batches

    def get_total_eval_batches(self):
        num_val_batches_per_epoch_full = len(self.get_val_dataset()) // self.get_global_batch_size()
        num_val_batches_per_epoch = min(self.trainer.limit_val_batches, num_val_batches_per_epoch_full)
        return num_val_batches_per_epoch

    def get_total_test_batches(self):
        num_test_batches_per_epoch_full = len(self.get_test_dataset()) // self.get_global_batch_size()
        return min(self.trainer.limit_test_batches, num_test_batches_per_epoch_full)

    def get_sampled_train_dataset(self):
        return self.sample_train_dataset(self.get_train_dataset())

    def get_sampled_val_dataset(self):
        return self.sample_val_dataset(self.get_val_dataset())

    def get_sampled_test_dataset(self):
        return self.sample_test_dataset(self.get_test_dataset())

    @lru_cache
    def get_train_dataset(self):
        """
        Returns:
            Dataset: The training dataset used by the model.
        """
        return self.train_dataset()

    @lru_cache
    def get_val_dataset(self):
        """
        Returns:
            Dataset:  The validation dataset used by the model.
        """
        return self.val_dataset()

    @lru_cache
    def get_test_dataset(self):
        """
        Returns:
            Dataset: The testing dataset used by the model.
        """
        return self.test_dataset()


def _assert_attr(o, attr, scope):
    if not hasattr(o, attr):
        raise AttributeError(f"Must assign '{attr}' before {scope} call")


class BioNeMoBertModel(MegatronBertModel):
    def __init__(self, cfg, trainer, *args, **kwargs):
        _assert_attr(self, 'data_module', 'BioNeMoBertModel.__init__()')
        super().__init__(cfg, trainer, *args, **kwargs)

    def setup_training_data(self, cfg):
        super().setup_training_data(cfg)
        self.data_module.adjust_train_dataloader(self, self._train_dl)

    def setup_validation_data(self, cfg):
        super().setup_validation_data(cfg)
        self.data_module.adjust_val_dataloader(self, self._validation_dl)

    def setup_test_data(self, cfg):
        super().setup_test_data(cfg)
        self.data_module.adjust_test_dataloader(self, self._test_dl)

    @classmethod
    def list_available_models(cls):
        """
        TODO add documentation

        This overrides a functionality from NeMo that lists pre-trained models.
        We don't have any yet.
        """
        return []

    # The functions after this are _highly_ similar to the code
    # in ESM model, we should regroup it:
    def build_train_valid_test_datasets(self):
        logging.info('Building Bert datasets.')

        self._train_ds = self.data_module.get_sampled_train_dataset()
        self._validation_ds = self.data_module.get_sampled_val_dataset()
        self._test_ds = self.data_module.get_sampled_test_dataset()

        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info('Finished building Bert datasets.')
        return self._train_ds, self._validation_ds, self._test_ds
