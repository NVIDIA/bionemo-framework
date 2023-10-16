# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Dict, Optional

import torch
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.core.neural_types import NeuralType
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from bionemo.data.dataloader import ProteinBertCollate
from bionemo.data.dataloader.protein_collate import ESM2BertCollate
from bionemo.data.dataset_builder_utils import build_typed_dataset
from bionemo.data.mapped_dataset import NeMoUpsampling, Uniref90ClusterMappingDataset
from bionemo.data.molecule import megamolbart_build_train_valid_test_datasets
from bionemo.model.protein.esm1nv.base import ESMnvMegatronBertModel


# TODO(trvachov): Clean up deps
try:
    from apex.transformer import tensor_parallel  # noqa: F401

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

__all__ = ["ESM1nvModel", "ESM2nvModel"]


class ESM1nvModel(ESMnvMegatronBertModel):
    """
    ESM1nv pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

    def _build_tokenizer(self):
        """
        Default tokenizer is based on available nemo tokenizers.
        Override this method to use an external tokenizer.
        All tokenizers are expected to provide compatible interface.
        """
        model_name = self._cfg.tokenizer.get('model_name')
        self.tokenizer = get_nmt_tokenizer(
            library=self._cfg.tokenizer.library,
            model_name=model_name,
            tokenizer_model=self.register_artifact("tokenizer.model", self._cfg.tokenizer.model),
            vocab_file=self.register_artifact("tokenizer.vocab_file", self._cfg.tokenizer.vocab_file),
            legacy=False,
        )
        # patch tokenizer for use with HF esm tokenizer
        if self._cfg.tokenizer.library == 'huggingface' and str(model_name).startswith('facebook/esm2'):
            self.tokenizer.tokenizer.vocab = self.tokenizer.tokenizer.get_vocab()

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        assert self._cfg.data.dataloader_type == 'single', AssertionError(
            f'Only the Megatron sequential ("single") sampler is currently supported. {self._cfg.data.dataloader_type} was chosen.'
        )

        dataloader = super().build_pretraining_data_loader(dataset=dataset, consumed_samples=consumed_samples)

        # Add collate function and unpin memory to avoid crash with CUDA misaligned address
        dataloader.pin_memory = False  # must be False with CSV dataset TODO check with binary
        pad_size_divisible_by_8 = True if self._cfg.masked_softmax_fusion else False

        dataloader.collate_fn = ProteinBertCollate(
            tokenizer=self.tokenizer,
            seq_length=self._cfg.seq_length,
            pad_size_divisible_by_8=pad_size_divisible_by_8,
            modify_percent=self._cfg.data.modify_percent,
            perturb_percent=self._cfg.data.perturb_percent,
        ).collate_fn
        return dataloader

    def setup_training_data(self, cfg):
        super().setup_training_data(cfg)

    def setup_validation_data(self, cfg):
        super().setup_validation_data(cfg)

    def setup_test_data(self, cfg):
        super().setup_test_data(cfg)

    @staticmethod
    def _build_train_valid_test_datasets(trainer, model_cfg):
        logging.info('Building Bert datasets.')
        global_batch_size = trainer.world_size * model_cfg.micro_batch_size / model_cfg.tensor_model_parallel_size
        # Compute trianing micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        max_train_steps = trainer.max_steps * trainer.accumulate_grad_batches
        eval_iters = (max_train_steps // trainer.val_check_interval + 1) * trainer.limit_val_batches
        test_iters = trainer.limit_test_batches

        train_valid_test_num_samples = {
            'train': int(max_train_steps * global_batch_size),
            'val': int(eval_iters * global_batch_size),
            'test': int(test_iters * global_batch_size),
        }

        _train_ds, _validation_ds, _test_ds = megamolbart_build_train_valid_test_datasets(
            cfg=model_cfg.data, train_valid_test_num_samples=train_valid_test_num_samples
        )

        logging.info(f'Length of train dataset: {len(_train_ds)}')
        logging.info(f'Length of val dataset: {len(_validation_ds)}')
        logging.info(f'Length of test dataset: {len(_test_ds)}')
        logging.info('Finished building Bert datasets.')
        return _train_ds, _validation_ds, _test_ds

    def build_train_valid_test_datasets(self):
        train, val, test = self._build_train_valid_test_datasets(self.trainer, self._cfg)
        self._train_ds = train
        self._validation_ds = val
        self._test_ds = test
        return self._train_ds, self._validation_ds, self._test_ds

    def validation_epoch_end(self, outputs):
        if not outputs:
            return
        averaged_loss = torch.stack(outputs).mean()
        self.log('val_loss', averaged_loss, prog_bar=True)
        self.log('val_loss_ECE', pow(2, averaged_loss))  # calculate exponential cross entropy loss for logs
        self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step - self.init_global_step))

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')
        logging.info(f'test_loss_ECE: {pow(2, averaged_loss[0])}')

    @property
    def input_names(self):
        return [
            'input_ids',
            'attention_mask',
        ]

    @property
    def output_names(self):
        return ['output']

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {'input_ids': {0: 'batch', 1: 'time'}, 'attention_mask': {0: 'batch', 1: 'time'}}

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {'output': {0: 'batch', 1: 'time', 2: 'size'}}


class ESM2nvModel(ESM1nvModel):
    """
    Extends the ESM1nv model by customizing dataset construction to mimic ESM2's implementation.

    This model introduces a dataset structure that not only retains elements from ESM1nv but also
    incorporates new fields specifically tailored for ESM2nv. One significant change involves using
    NeMoUpsampling to upsample UniRef50 cluster IDs and pre-computing samples from UniRef90 via a mapped dataset.

    The configuration files are designed to incorporate specific argument structures. A sample configuration
    demonstrating the dataset configuration can be viewed in `examples/protein/esm2nv/conf/base_config.yaml`

    Args:
        cfg (DictConfig): The configuration object, typically constructed from a YAML file.
        trainer (Trainer): The training instance associated with this model.

    Attributes:
        dataset_path (str): Parent directory containing train, test, and validation data.
        dataset (dict): Specifies the data file range for training, testing, and validation.
        data_impl (str): Implementation choice, like "csv_mmap".
        data_impl_kwargs (dict): Arguments specific to the data implementation choice.
        uf50_datapath (str): Path to the raw UniRef50 fasta file.
        uf90_datapath (str): Path to the raw UniRef90 fasta file.
        cluster_mapping_tsv (str): TSV file mapping UniRef50 cluster IDs to UniRef90 entries.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

    @staticmethod
    def _build_train_valid_test_datasets(trainer, model_cfg, keep_uf50=False):
        '''
        keep_uf50 - use this flag for testing, manually assigns the uniref50_dataset object to the resulting Uniref90ClusterMappingDataset object.
        '''
        _train_ds, _validation_ds, _test_ds = ESM1nvModel._build_train_valid_test_datasets(trainer, model_cfg)

        dataset_path = model_cfg.data.uf90.uniref90_path
        split = 'uf90_csvs'
        ds_name = model_cfg.data.uf90.dataset.get(split, None)
        filepath: str = os.path.join(dataset_path, split, ds_name)

        data_impl = model_cfg.data.uf90.get('data_impl', None)
        assert data_impl is not None, 'Config "cfg" should contain field "cfg.data_impl"'
        # NOTE train/val/test will all each into uniref90, since we are split on clusters, we now they are independent.
        #   hence, we only need one dataset object for uf90
        uniref90_dataset = build_typed_dataset(
            dataset_paths=filepath,
            data_impl=data_impl,
            cfg=model_cfg.data.uf90,
            use_upsampling=False,
            num_samples=None,
        )

        results = []
        for ds, split in zip([_train_ds, _validation_ds, _test_ds], ['train', 'val', 'test']):
            # TypeHint for intellisense
            _ds: NeMoUpsampling = ds

            index_mapping_dir = ds.index_mapping_dir

            path_root: str = os.path.join(model_cfg.data.dataset_path, split)
            # Setup the resampling
            ds = Uniref90ClusterMappingDataset(
                uniref50_dataset=_ds,
                uniref90_dataset=uniref90_dataset,
                data_prefix=split,  # used for index creation
                seed=model_cfg.seed,  # used for rng, although awkward because global statehood
                index_mapping_dir=index_mapping_dir,  # stores index
                cluster_map_starts_fn=f'{path_root}/starts.mmap',
                cluster_map_counts_fn=f'{path_root}/counts.mmap',
                name=ds.name,
            )

            if keep_uf50:
                ds.uniref50_dataset = _ds

            results.append(ds)

        [_train_ds, _validation_ds, _test_ds] = results
        logging.info(f'Length of train dataset: {len(_train_ds)}')
        logging.info(f'Length of val dataset: {len(_validation_ds)}')
        logging.info(f'Length of test dataset: {len(_test_ds)}')
        logging.info('Finished building Bert datasets.')
        return _train_ds, _validation_ds, _test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        assert self._cfg.data.dataloader_type == 'single', AssertionError(
            f'Only the Megatron sequential ("single") sampler is currently supported. {self._cfg.data.dataloader_type} was chosen.'
        )
        dataloader = super().build_pretraining_data_loader(dataset=dataset, consumed_samples=consumed_samples)

        # Add collate function and unpin memory to avoid crash with CUDA misaligned address
        dataloader.pin_memory = False

        pad_size_divisible_by_8 = True if self._cfg.masked_softmax_fusion else False

        dataloader.collate_fn = ESM2BertCollate(
            tokenizer=self.tokenizer,
            seq_length=self._cfg.seq_length,
            pad_size_divisible_by_8=pad_size_divisible_by_8,
            modify_percent=self._cfg.data.modify_percent,
            perturb_percent=self._cfg.data.perturb_percent,
        ).collate_fn
        return dataloader
